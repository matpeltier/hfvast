#!/usr/bin/env python3
# hfvast chunked downloader: parallel ranges over the signed CDN URL.
# HF's Xet CDN throttles single connections after ~250MB (live-verified
# 2026-09-03: 40MB/s collapsing to ~0 mid-file on every host); parallel chunks
# with per-chunk retries dodge both throttling and stalls. The user's HF token
# is used ONLY to resolve the signed URL, never on data connections.
import json
import os
import sys
import time
import urllib.request

ROOT = os.environ.get("HFVAST_ROOT", "/opt/hfvast")
MODELS_DIR = os.path.join(ROOT, "models")
STATE_FILE = os.path.join(ROOT, "state.json")
HF_TOKEN = os.environ.get("HF_TOKEN", "")
BASE_URL = os.environ.get("HFVAST_BASE_URL", "")
CHUNK = 8 * 1024 * 1024
WORKERS = 4


def set_state(**kw):
    import tempfile

    try:
        s = json.load(open(STATE_FILE))
    except Exception:
        s = {}
    s.update(kw)
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(STATE_FILE))
    with os.fdopen(fd, "w") as f:
        json.dump(s, f)
    os.replace(tmp, STATE_FILE)


def resolve(url):
    req = urllib.request.Request(url)
    if HF_TOKEN:
        req.add_header("Authorization", "Bearer " + HF_TOKEN)

    class NoRedirect(urllib.request.HTTPRedirectHandler):
        def redirect_request(self, *a, **k):
            return None

    opener = urllib.request.build_opener(NoRedirect)
    try:
        resp = opener.open(req, timeout=60)
        return url, int(resp.headers.get("Content-Length") or 0)
    except urllib.error.HTTPError as e:
        loc = e.headers.get("Location") if e.code in (301, 302, 303, 307, 308) else None
        if not loc:
            raise
        # signed URL embeds its own auth — no HF header on data requests
        return loc, int(e.headers.get("X-Linked-Size") or 0)


def fetch_range(url, start, end, dest, offset, tries=8):
    got = 0
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers={"Range": "bytes=%d-%d" % (start + got, end)})
            with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "r+b") as f:
                f.seek(offset + got)
                while True:
                    piece = resp.read(1024 * 1024)
                    if not piece:
                        break
                    f.write(piece)
                    got += len(piece)
            if got >= end - start + 1:
                return
        except OSError:
            time.sleep(min(30, 2 * (attempt + 1)))
    raise RuntimeError("chunk %d-%d failed after %d attempts" % (start, end, tries))


def download(url, size, dest):
    if size <= 0:
        req = urllib.request.Request(url)
        with urllib.request.urlopen(req, timeout=120) as resp, open(dest, "wb") as f:
            while True:
                piece = resp.read(1024 * 1024)
                if not piece:
                    break
                f.write(piece)
        return
    with open(dest, "wb") as f:
        f.truncate(size)
    jobs = [(s, min(s + CHUNK, size) - 1) for s in range(0, size, CHUNK)]
    done = {"n": 0}
    lock = __import__("threading").Lock()

    def work(job):
        fetch_range(url, job[0], job[1], dest, job[0])
        with lock:
            done["n"] += job[1] - job[0] + 1
            set_state(bytes_done=done["n"])

    from concurrent.futures import ThreadPoolExecutor

    with ThreadPoolExecutor(WORKERS) as pool:
        list(pool.map(work, jobs))


def main():
    try:
        run()
    except Exception as e:  # surface the real error to /health (state.json)
        set_state(message="download error: %s: %s" % (type(e).__name__, str(e)[:300]))
        set_state(status="error")
        sys.exit(1)


def run():
    set_state(status="downloading", message="downloader ready")
    total_all = 0
    ok = 0
    with open(os.path.join(ROOT, "files.tsv")) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rel, size = line.split("\t")
            size = int(size)
            total_all += size
    set_state(bytes_total=total_all)
    done_all = 0
    with open(os.path.join(ROOT, "files.tsv")) as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            rel, size = line.split("\t")
            size = int(size)
            name = os.path.basename(rel)
            dest = os.path.join(MODELS_DIR, name)
            set_state(message="downloading %s" % name)
            resolve_url = BASE_URL + "/" + rel
            url, hdr_size = resolve(resolve_url)
            if hdr_size and hdr_size != size:
                size = hdr_size  # trust the CDN
            download(url, size, dest)
            actual = os.path.getsize(dest)
            if actual < size:
                set_state(message="size mismatch for %s (%d < %d)" % (name, actual, size))
                set_state(status="error")
                sys.exit(1)
            done_all += size
            set_state(bytes_done=done_all)
            ok += 1
    set_state(message="all files downloaded")


if __name__ == "__main__":
    main()
