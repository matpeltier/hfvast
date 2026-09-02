"""Test harness: real gateway subprocess + fake in-process backend.

Everything runs on 127.0.0.1 — no cloud, no network beyond localhost.
"""

from __future__ import annotations

import json
import socket
import subprocess
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import httpx

from hfvast.deploy.bootstrap import decode_payload, encode_payload

REPO_ROOT = Path(__file__).parents[2]


def free_port() -> int:
    with socket.socket() as sock:
        sock.bind(("127.0.0.1", 0))
        return int(sock.getsockname()[1])


class FakeBackendHandler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    healthy = True  # class attribute, flipped by tests

    def log_message(self, fmt: str, *args: object) -> None:
        pass

    def _json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.split("?")[0] == "/health":
            if FakeBackendHandler.healthy:
                self._json(200, {"status": "ok"})
            else:
                self._json(503, {"error": {"message": "Loading model"}})
            return
        self._json(404, {"error": "not found"})

    def do_POST(self) -> None:
        path = self.path.split("?")[0]
        length = int(self.headers.get("Content-Length") or 0)
        self.rfile.read(length)
        if path == "/v1/chat/completions/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            for chunk in (b'data: {"delta": "a"}\n\n', b'data: {"delta": "b"}\n\n', b"data: [DONE]\n\n"):
                self.wfile.write(b"%x\r\n%s\r\n" % (len(chunk), chunk))
                self.wfile.flush()
                time.sleep(0.05)
            self.wfile.write(b"0\r\n\r\n")
            return
        if path == "/v1/chat/completions":
            auth = self.headers.get("Authorization", "")
            self._json(200, {"choices": [{"message": {"content": f"backend ok {auth[:25]}"}}]})
            return
        self._json(404, {"error": "not found"})


class FakeBackend:
    def __init__(self) -> None:
        self.port = free_port()
        self._server = ThreadingHTTPServer(("127.0.0.1", self.port), FakeBackendHandler)
        self._thread = threading.Thread(target=self._server.serve_forever, daemon=True)

    def __enter__(self) -> FakeBackend:
        self._thread.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self._server.shutdown()


class GatewayProcess:
    """The real packaged gateway, run as a subprocess against the fake backend."""

    def __init__(
        self,
        tmp_path: Path,
        backend_port: int,
        state: dict | None = None,
        gateway_key: str = "test-gateway-key-1234567890",
    ) -> None:
        self.port = free_port()
        self.gateway_key = gateway_key
        self.state_file = tmp_path / "state.json"
        self.activity_file = tmp_path / "last_activity"
        self.state_file.write_text(json.dumps(state or {"status": "downloading", "message": "x"}))
        self.env = {
            **dict(__import__("os").environ),
            "HFVAST_GATEWAY_PORT": str(self.port),
            "HFVAST_BACKEND_PORT": str(backend_port),
            "HFVAST_GATEWAY_KEY": gateway_key,
            "HFVAST_BACKEND_KEY": "backend-key-1234567890",
            "HFVAST_STATE_FILE": str(self.state_file),
            "HFVAST_ACTIVITY_FILE": str(self.activity_file),
            "HFVAST_GATEWAY_LOG": str(tmp_path / "gateway.log"),
        }
        self._proc: subprocess.Popen[bytes] | None = None

    def start(self) -> GatewayProcess:
        # Use the packaged source directly (same file instances receive).
        gateway_src = (REPO_ROOT / "src/hfvast/runtime/gateway.py").read_text(encoding="utf-8")
        script = tmp_script(gateway_src)
        self._proc = subprocess.Popen(
            [sys.executable, str(script)],
            env=self.env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        deadline = time.time() + 10
        while time.time() < deadline:
            try:
                httpx.get(f"http://127.0.0.1:{self.port}/health", timeout=1)
                return self
            except httpx.HTTPError:
                time.sleep(0.05)
        raise RuntimeError("gateway did not start")

    def stop(self) -> None:
        if self._proc:
            self._proc.terminate()
            self._proc.wait(timeout=5)

    def set_state(self, **fields: object) -> None:
        data = json.loads(self.state_file.read_text())
        data.update(fields)
        self.state_file.write_text(json.dumps(data))


def tmp_script(source: str) -> Path:
    path = Path("/tmp") / f"hfvast-test-gateway-{free_port()}.py"
    path.write_text(source, encoding="utf-8")
    return path


def payload_roundtrip(payload: str) -> str:
    return decode_payload(encode_payload(payload))
