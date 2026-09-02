#!/usr/bin/env python3
"""hfvast gateway — single-file, stdlib-only OpenAI-compatible auth proxy.

Runs on the rented instance (written by the hfvast bootstrap payload):
    Internet ──▶ gateway 0.0.0.0:8000 ──▶ backend 127.0.0.1:8001

Responsibilities (spec §25):
  * /v1/* proxying with API-key auth (constant-time compare);
  * SSE streaming preserved chunk-by-chunk (never buffered);
  * /health (public) reflecting bootstrap/backend state;
  * /internal/state (authed) for lifecycle watchdogs;
  * activity tracking for idle shutdown (health checks never count);
  * backend port stays bound to localhost only.

Python 3.10+ compatible; no third-party dependencies.
"""

from __future__ import annotations

import hmac
import http.client
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

GATEWAY_PORT = int(os.environ.get("HFVAST_GATEWAY_PORT", "8000"))
BACKEND_HOST = "127.0.0.1"
BACKEND_PORT = int(os.environ.get("HFVAST_BACKEND_PORT", "8001"))
API_KEY = os.environ.get("HFVAST_GATEWAY_KEY", "")
STATE_FILE = os.environ.get("HFVAST_STATE_FILE", "/opt/hfvast/state.json")
ACTIVITY_FILE = os.environ.get("HFVAST_ACTIVITY_FILE", "/opt/hfvast/last_activity")

INFERENCE_PATHS = ("/v1/chat/completions", "/v1/completions", "/v1/responses")
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "content-encoding",
    "content-length",
}

_lock = threading.Lock()
ACTIVE_REQUESTS = 0


def _load_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def _note_activity() -> None:
    try:
        with open(ACTIVITY_FILE, "w", encoding="utf-8") as handle:
            handle.write(str(time.time()))
    except OSError:
        pass


def _last_activity() -> float:
    try:
        with open(ACTIVITY_FILE, encoding="utf-8") as handle:
            return float(handle.read().strip())
    except (OSError, ValueError):
        return 0.0


def _backend_probe() -> str:
    """Probe backend /health: 'ok' | 'loading' | 'down'."""
    try:
        conn = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=3)
        conn.request("GET", "/health")
        resp = conn.getresponse()
        resp.read()
        conn.close()
        if resp.status == 200:
            return "ok"
        if resp.status == 503:
            return "loading"
        return "down"
    except OSError:
        return "down"


def _authorized(handler: BaseHTTPRequestHandler) -> bool:
    if not API_KEY:
        return False  # fail closed: never run an unauthenticated public API
    header = handler.headers.get("Authorization", "")
    if not header.startswith("Bearer "):
        return False
    return hmac.compare_digest(header[7:].strip(), API_KEY)


class GatewayHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "hfvast-gateway/0.2"
    sys_version = ""

    def log_message(self, fmt: str, *args: object) -> None:  # never log bodies/keys
        path = getattr(self, "path", "")
        if "key" in path.lower() or "token" in path.lower():
            path = "***REDACTED***"
        with open(os.environ.get("HFVAST_GATEWAY_LOG", "/dev/null"), "a", encoding="utf-8") as log:
            log.write("%s %s\n" % (self.log_date_time_string(), fmt % args))

    def _json(self, status: int, payload: dict, drain: bool = False) -> None:
        """JSON error/info response. ``drain`` also swallows any request body so
        HTTP keep-alive framing stays intact (401s on POSTs with bodies)."""
        if drain:
            try:
                length = int(self.headers.get("Content-Length") or 0)
                while length > 0:
                    chunk = self.rfile.read(min(length, 65536))
                    if not chunk:
                        break
                    length -= len(chunk)
            except OSError:
                pass
            self.close_connection = True
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        global ACTIVE_REQUESTS
        path = self.path.split("?", 1)[0]
        if path == "/health":
            bootstrap = _load_state()
            backend = _backend_probe()
            status = (
                "ok"
                if bootstrap.get("status") == "ready" and backend == "ok"
                else ("loading" if bootstrap.get("status") != "error" and backend != "down" else "error")
            )
            self._json(
                200,
                {
                    "status": status,
                    "gateway": "ok",
                    "bootstrap": {
                        "status": bootstrap.get("status"),
                        "message": bootstrap.get("message"),
                        "bytes_done": bootstrap.get("bytes_done", 0),
                        "bytes_total": bootstrap.get("bytes_total", 0),
                    },
                    "backend": backend,
                },
            )
            return
        if path == "/internal/state":
            if not _authorized(self):
                self._json(401, {"error": "unauthorized"})
                return
            state = _load_state()
            self._json(
                200,
                {
                    "status": state.get("status"),
                    "message": state.get("message"),
                    "bytes_done": state.get("bytes_done", 0),
                    "bytes_total": state.get("bytes_total", 0),
                    "ready_since": state.get("ready_since"),
                    "last_activity": _last_activity(),
                    "active_requests": ACTIVE_REQUESTS,
                    "uptime_s": time.time() - state.get("started_at", time.time()),
                },
            )
            return
        if path.startswith("/v1/"):
            if not _authorized(self):
                self._json(401, {"error": "unauthorized"})
                return
            self._proxy()
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        global ACTIVE_REQUESTS
        path = self.path.split("?", 1)[0]
        if path.startswith("/v1/") or path in INFERENCE_PATHS:
            if not _authorized(self):
                self._json(401, {"error": "unauthorized"}, drain=True)
                return
            is_inference = any(path.startswith(p) for p in INFERENCE_PATHS)
            if is_inference:
                with _lock:
                    ACTIVE_REQUESTS += 1
            try:
                self._proxy(note_activity=is_inference)
            finally:
                if is_inference:
                    with _lock:
                        ACTIVE_REQUESTS -= 1
                        if ACTIVE_REQUESTS <= 0:
                            _note_activity()  # activity timestamped at request end
            return
        self._json(404, {"error": "not found"}, drain=True)

    def do_DELETE(self) -> None:
        self._json(404, {"error": "not found"}, drain=True)

    def _proxy(self, note_activity: bool = False) -> None:
        length = int(self.headers.get("Content-Length") or 0)
        body = self.rfile.read(length) if length else None

        headers = {
            k: v
            for k, v in self.headers.items()
            if k.lower() not in HOP_BY_HOP and k.lower() != "authorization" and k.lower() != "host"
        }
        backend_key = os.environ.get("HFVAST_BACKEND_KEY", "")
        if backend_key:
            headers["Authorization"] = "Bearer " + backend_key

        try:
            conn = http.client.HTTPConnection(BACKEND_HOST, BACKEND_PORT, timeout=600)
            conn.request(self.command, self.path, body=body, headers=headers)
            resp = conn.getresponse()
        except OSError as exc:
            self._json(502, {"error": {"message": f"backend unreachable: {exc}"}})
            return

        resp_headers = resp.getheaders()
        self.send_response(resp.status, resp.reason)
        for key, value in resp_headers:
            if key.lower() in HOP_BY_HOP:
                continue
            self.send_header(key, value)
        if note_activity and resp.status < 400:
            _note_activity()
        # content-length is always stripped (hop-by-hop) → re-chunk everything;
        # h11/httpx/requests/curl decode chunks transparently and SSE chunks are
        # forwarded immediately (never buffered).
        self.send_header("Transfer-Encoding", "chunked")
        self.end_headers()

        try:
            while True:
                chunk = resp.read(8192)
                if not chunk:
                    break
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()  # SSE: forward each chunk immediately
            self.wfile.write(b"0\r\n\r\n")
        except (BrokenPipeError, ConnectionResetError):
            pass
        finally:
            conn.close()


class GatewayServer(ThreadingHTTPServer):
    daemon_threads = True
    allow_reuse_address = True


def main() -> None:
    if not API_KEY:
        raise SystemExit("HFVAST_GATEWAY_KEY is required — refusing to serve unauthenticated")
    server = GatewayServer(("0.0.0.0", GATEWAY_PORT), GatewayHandler)
    server.serve_forever()


if __name__ == "__main__":
    main()
