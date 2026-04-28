from __future__ import annotations

from locoder.models.registry import load_registry, lookup


def test_load_registry_returns_dict() -> None:
    reg = load_registry()
    assert isinstance(reg, dict)
    assert len(reg) > 0


def test_lookup_known_model() -> None:
    entry = lookup("qwen2.5-coder-7b")
    assert entry is not None
    assert "repo" in entry
    assert "default_quant" in entry
    assert "params_b" in entry


def test_lookup_unknown_returns_none() -> None:
    assert lookup("definitely-not-a-real-model") is None


def test_all_entries_have_required_keys() -> None:
    reg = load_registry()
    required = {"repo", "default_quant", "filename"}
    for name, entry in reg.items():
        missing = required - entry.keys()
        assert not missing, f"{name!r} is missing keys: {missing}"


def test_new_models_registered() -> None:
    for model in ("phi-4", "phi-3.5-mini", "codellama-7b", "codellama-13b"):
        assert lookup(model) is not None, f"{model!r} missing from registry"
