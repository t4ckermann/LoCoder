from __future__ import annotations

from pathlib import Path

from locoder.server.launcher import build_argv


def test_build_argv_required_flags() -> None:
    argv = build_argv(
        llama_server_bin="/usr/bin/llama-server",
        model_path=Path("/tmp/model.gguf"),
        port=8080,
        args={},
    )
    assert argv[0] == "/usr/bin/llama-server"
    assert "--model" in argv
    assert str(Path("/tmp/model.gguf")) in argv
    assert "--port" in argv
    assert "8080" in argv
    assert "--host" in argv
    assert "127.0.0.1" in argv


def test_build_argv_flash_attn_defaults_to_auto() -> None:
    argv = build_argv("/bin/llama-server", Path("/tmp/m.gguf"), 8080, {})
    assert "--flash-attn" in argv
    idx = argv.index("--flash-attn")
    assert argv[idx + 1] == "auto"


def test_build_argv_flash_attn_explicit_on() -> None:
    argv = build_argv("/bin/llama-server", Path("/tmp/m.gguf"), 8080, {"flash_attn": "on"})
    idx = argv.index("--flash-attn")
    assert argv[idx + 1] == "on"


def test_build_argv_threads_and_ctx() -> None:
    argv = build_argv(
        "/bin/llama-server",
        Path("/tmp/m.gguf"),
        8080,
        {"threads": 4, "ctx_size": 8192},
    )
    assert "--threads" in argv
    assert argv[argv.index("--threads") + 1] == "4"
    assert "--ctx-size" in argv
    assert argv[argv.index("--ctx-size") + 1] == "8192"


def test_build_argv_ngl_zero_included() -> None:
    argv = build_argv("/bin/llama-server", Path("/tmp/m.gguf"), 8080, {"ngl": 0})
    assert "-ngl" in argv
    assert argv[argv.index("-ngl") + 1] == "0"


def test_build_argv_unknown_key_ignored() -> None:
    # Keys not in key_map should not appear as flags
    argv = build_argv("/bin/llama-server", Path("/tmp/m.gguf"), 8080, {"planner": {"ctx_size": 1}})
    assert "--planner" not in argv
