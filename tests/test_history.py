from __future__ import annotations

import json

from locoder.agent.history import _MAX_TURNS, _SEED_TURNS, clear, load, recent_summaries, save


def _msgs(task: str) -> list[dict]:
    return [
        {"role": "system", "content": "you are helpful"},
        {"role": "user", "content": task},
        {"role": "assistant", "content": "done"},
    ]


def test_load_empty(tmp_path):
    assert load(tmp_path) == []


def test_save_and_load_roundtrip(tmp_path, monkeypatch):
    monkeypatch.setattr("locoder.agent.history._HISTORY_DIR", tmp_path)
    msgs = _msgs("fix the bug")
    save(tmp_path, msgs)
    loaded = load(tmp_path)
    assert loaded == msgs


def test_load_seeds_last_n_turns(tmp_path, monkeypatch):
    monkeypatch.setattr("locoder.agent.history._HISTORY_DIR", tmp_path)
    for i in range(_SEED_TURNS + 5):
        save(tmp_path, _msgs(f"task {i}"))
    loaded = load(tmp_path)
    # Should contain exactly _SEED_TURNS * len(_msgs()) messages
    assert len(loaded) == _SEED_TURNS * 3


def test_trim_to_max_turns(tmp_path, monkeypatch):
    monkeypatch.setattr("locoder.agent.history._HISTORY_DIR", tmp_path)
    for i in range(_MAX_TURNS + 10):
        save(tmp_path, _msgs(f"task {i}"))
    from locoder.agent.history import _path
    lines = [l for l in _path(tmp_path).read_text().splitlines() if l.strip()]
    assert len(lines) == _MAX_TURNS


def test_clear(tmp_path, monkeypatch):
    monkeypatch.setattr("locoder.agent.history._HISTORY_DIR", tmp_path)
    save(tmp_path, _msgs("hello"))
    clear(tmp_path)
    assert load(tmp_path) == []


def test_recent_summaries(tmp_path, monkeypatch):
    monkeypatch.setattr("locoder.agent.history._HISTORY_DIR", tmp_path)
    for i in range(7):
        save(tmp_path, _msgs(f"task {i}"))
    summaries = recent_summaries(tmp_path, n=5)
    assert len(summaries) == 5
    assert summaries[-1] == "task 6"
