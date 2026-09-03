#!/usr/bin/env python3
"""hfvast in-container fail-safe watchdog (stdlib only, Python 3.10+).

Second line of defense against runaway bills (spec §29): the LOCAL daemon is
primary, but if the user's machine sleeps/dies, this watchdog — running on the
instance — destroys the instance itself using Vast's per-instance restricted
key (CONTAINER_API_KEY), which can only start/stop/destroy THAT instance.
The unrestricted VAST_API_KEY never enters the container.

Enforced rules:
  * idle: READY + zero active requests + (now - last_activity) >= idle_timeout
  * hard max runtime: now - start >= max_runtime (independent of activity)
"""

from __future__ import annotations

import json
import os
import time
import urllib.request

ROOT = os.environ.get("HFVAST_ROOT", "/opt/hfvast")
STATE_FILE = os.path.join(ROOT, "state.json")
ACTIVITY_FILE = os.path.join(ROOT, "last_activity")
MODELS_DIR = os.path.join(ROOT, "models")

CONTAINER_ID = os.environ.get("CONTAINER_ID", "")
CONTAINER_API_KEY = os.environ.get("CONTAINER_API_KEY", "")

IDLE_TIMEOUT_S = float(os.environ.get("HFVAST_IDLE_TIMEOUT", "1800"))
MAX_RUNTIME_S = float(os.environ.get("HFVAST_MAX_RUNTIME", "21600"))
POLL_INTERVAL_S = float(os.environ.get("HFVAST_WATCHDOG_POLL", "30"))

STARTED_AT = time.time()


def log(message: str) -> None:
    with open(os.path.join(ROOT, "watchdog.log"), "a", encoding="utf-8") as handle:
        handle.write("%s %s\n" % (time.strftime("%Y-%m-%dT%H:%M:%S"), message))


def read_state() -> dict:
    try:
        with open(STATE_FILE, encoding="utf-8") as handle:
            return json.load(handle)
    except (OSError, ValueError):
        return {}


def write_state(**fields: object) -> None:
    state = read_state()
    state.update(fields)
    with open(STATE_FILE, "w", encoding="utf-8") as handle:
        json.dump(state, handle)


def last_activity() -> float:
    try:
        with open(ACTIVITY_FILE, encoding="utf-8") as handle:
            return float(handle.read().strip())
    except (OSError, ValueError):
        return 0.0


def directory_bytes(path: str) -> int:
    total = 0
    for dirpath, _dirnames, filenames in os.walk(path):
        for name in filenames:
            try:
                total += os.path.getsize(os.path.join(dirpath, name))
            except OSError:
                continue
    return total


def destroy_instance(reason: str) -> bool:
    if not CONTAINER_ID or not CONTAINER_API_KEY:
        log("cannot self-destroy: CONTAINER_ID/CONTAINER_API_KEY missing (%s)" % reason)
        return False
    url = "https://console.vast.ai/api/v0/instances/%s/" % CONTAINER_ID
    request = urllib.request.Request(url, method="DELETE")
    request.add_header("Authorization", "Bearer " + CONTAINER_API_KEY)
    try:
        with urllib.request.urlopen(request, timeout=30) as resp:
            resp.read()
        log("self-destroyed instance (%s)" % reason)
        return True
    except OSError as exc:
        log("self-destroy failed (%s): %s" % (reason, exc))
        return False


def main() -> None:
    log(
        "watchdog started (idle=%ss max_runtime=%ss container_id=%s)"
        % (IDLE_TIMEOUT_S, MAX_RUNTIME_S, "set" if CONTAINER_ID else "MISSING")
    )
    while True:
        try:
            state = read_state()
            status = state.get("status")

            if time.time() - STARTED_AT >= MAX_RUNTIME_S:
                log("hard max runtime reached (%.0fs)" % MAX_RUNTIME_S)
                destroy_instance("max_runtime")
                return

            if status == "ready":
                activity = max(last_activity(), float(state.get("ready_since") or 0))
                idle_for = time.time() - activity
                if idle_for >= IDLE_TIMEOUT_S:
                    log("idle for %.0fs >= %.0fs" % (idle_for, IDLE_TIMEOUT_S))
                    destroy_instance("idle_timeout")
                    return
        except Exception as exc:
            log("watchdog cycle error: %r" % (exc,))
        time.sleep(POLL_INTERVAL_S)


if __name__ == "__main__":
    main()
