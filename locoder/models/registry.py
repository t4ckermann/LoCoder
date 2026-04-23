from __future__ import annotations

import json
import urllib.request
from importlib.resources import files
from pathlib import Path

REGISTRY_URL = (
    "https://raw.githubusercontent.com/locoder-ai/locoder/main/locoder/data/registry.json"
)
_USER_REGISTRY = Path("~/.locoder/registry.json").expanduser()


def load_registry() -> dict:
    if _USER_REGISTRY.exists():
        return json.loads(_USER_REGISTRY.read_text())
    data = files("locoder.data").joinpath("registry.json").read_text()
    return json.loads(data)


def lookup(name: str) -> dict | None:
    return load_registry().get(name)


def refresh_registry() -> int:
    with urllib.request.urlopen(REGISTRY_URL, timeout=15) as resp:
        payload = resp.read()
    registry = json.loads(payload)
    _USER_REGISTRY.parent.mkdir(parents=True, exist_ok=True)
    _USER_REGISTRY.write_bytes(payload)
    return len(registry)
