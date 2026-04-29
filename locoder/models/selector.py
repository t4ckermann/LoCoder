from __future__ import annotations

from locoder.models.registry import lookup

# (quant_name, bits_per_weight) — ordered best→worst quality
QUANT_LADDER: list[tuple[str, float]] = [
    ("q8_0", 8.50),
    ("q6_k", 6.57),
    ("q5_k_m", 5.60),
    ("q4_k_m", 4.85),
    ("q3_k_m", 3.35),
    ("q2_k", 2.50),
]
_LADDER_NAMES = [name for name, _ in QUANT_LADDER]
_BPW: dict[str, float] = dict(QUANT_LADDER)

# Reserve 15 % of available RAM for OS/KV-cache headroom
_HEADROOM = 1.15


def quant_size_gb(params_b: float, quant: str) -> float:
    """Estimated GGUF file size in GB for a model with params_b billion parameters."""
    bpw = _BPW.get(quant.lower(), 4.85)
    return params_b * bpw / 8.0 * _HEADROOM


def select_quant(model_name: str, available_gb: float, prefer: str = "q5_k_m") -> str:
    """Return the best quant that fits in available_gb for model_name.

    Walks the quality ladder down from *prefer* until a variant fits.
    Falls back to the registry's default_quant if sizing data is absent.
    """
    entry = lookup(model_name)
    if entry is None:
        return prefer

    default: str = entry["default_quant"]
    params_b: float = entry.get("params_b", 0.0)
    available: list[str] = [q.lower() for q in entry.get("available_quants", [])]

    # No sizing data → trust the registry default
    if params_b <= 0 or not available:
        return default

    prefer_lower = prefer.lower()
    try:
        start_idx = _LADDER_NAMES.index(prefer_lower)
    except ValueError:
        start_idx = _LADDER_NAMES.index("q5_k_m")

    for quant_name, bpw in QUANT_LADDER[start_idx:]:
        if quant_name not in available:
            continue
        if params_b * bpw / 8.0 * _HEADROOM <= available_gb:
            return quant_name

    # Nothing from the ladder fits — return the smallest available variant
    for quant_name, _ in reversed(QUANT_LADDER):
        if quant_name in available:
            return quant_name

    return default
