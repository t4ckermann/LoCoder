"""Download and install a pre-built llama-server binary from llama.cpp GitHub releases."""

from __future__ import annotations

import io
import platform
import shutil
import stat
import tarfile
import urllib.request
import zipfile
from pathlib import Path

_BIN_DIR = Path("~/.locoder/bin").expanduser()
_INSTALLED_BIN = _BIN_DIR / "llama-server"
_RELEASES_API = "https://api.github.com/repos/ggerganov/llama.cpp/releases/latest"


def installed_bin() -> Path | None:
    """Return the path to the managed llama-server binary if it exists."""
    if _INSTALLED_BIN.exists():
        return _INSTALLED_BIN
    return None


def find_on_path() -> str | None:
    """Return the path to llama-server if already available on PATH."""
    return shutil.which("llama-server")


def _detect_asset_keyword() -> str:
    """Map current platform/arch to the substring in the GitHub release asset name."""
    system = platform.system().lower()
    machine = platform.machine().lower()

    if system == "darwin":
        # Apple Silicon and Intel Macs
        if machine in ("arm64", "aarch64"):
            return "macos-arm64"
        return "macos-x86_64"

    if system == "linux":
        if machine in ("aarch64", "arm64"):
            return "ubuntu-arm64"
        return "ubuntu-x64"

    if system == "windows":
        return "win-avx2-x64"

    raise RuntimeError(f"Unsupported platform: {system}/{machine}")


def _latest_release_assets() -> list[dict]:
    """Fetch the asset list from the latest llama.cpp GitHub release."""
    import json

    req = urllib.request.Request(
        _RELEASES_API,
        headers={"Accept": "application/vnd.github+json", "User-Agent": "locoder"},
    )
    with urllib.request.urlopen(req, timeout=15) as resp:
        data = json.loads(resp.read())
    return data.get("assets", [])


_ARCHIVE_EXTS = (".zip", ".tar.gz", ".tar.xz", ".tar.bz2")


def _pick_asset(assets: list[dict], keyword: str) -> dict:
    """Choose the best asset for this platform from the release asset list."""
    # Skip GPU-specific and exotic variants so we get a CPU-runnable binary by default.
    skip_keywords = {"cuda", "rocm", "vulkan", "kompute", "sycl", "openvino",
                     "opencl", "hip", "kleidiai"}

    def is_archive(name: str) -> bool:
        return any(name.endswith(ext) for ext in _ARCHIVE_EXTS)

    candidates = [
        a for a in assets
        if keyword in a["name"].lower()
        and is_archive(a["name"])
        and not any(s in a["name"].lower() for s in skip_keywords)
    ]

    if not candidates:
        # Fall back: accept anything matching the platform keyword
        candidates = [
            a for a in assets
            if keyword in a["name"].lower() and is_archive(a["name"])
        ]

    if not candidates:
        raise RuntimeError(
            f"No suitable release asset found for keyword '{keyword}'.\n"
            f"Available assets: {[a['name'] for a in assets]}"
        )

    # Prefer the smallest archive (typically the CPU-only build)
    return min(candidates, key=lambda a: a.get("size", 0))


def download_and_install(progress_callback=None) -> Path:
    """
    Download the latest llama-server binary for this platform and install it to
    ~/.locoder/bin/llama-server. Returns the installed path.

    progress_callback(downloaded_bytes, total_bytes) is called during download.
    """
    keyword = _detect_asset_keyword()
    assets = _latest_release_assets()
    asset = _pick_asset(assets, keyword)

    url: str = asset["browser_download_url"]
    total: int = asset.get("size", 0) or 0

    # Download zip into memory
    req = urllib.request.Request(url, headers={"User-Agent": "locoder"})
    buf = io.BytesIO()
    with urllib.request.urlopen(req, timeout=120) as resp:
        chunk_size = 65536
        downloaded = 0
        while True:
            chunk = resp.read(chunk_size)
            if not chunk:
                break
            buf.write(chunk)
            downloaded += len(chunk)
            if progress_callback:
                progress_callback(downloaded, total or None)

    buf.seek(0)
    asset_name: str = asset["name"]

    # Extract the full archive contents into _BIN_DIR — llama-server links
    # against companion shared libraries (libllama*.dylib / libggml*.so) that
    # live alongside it in the archive.
    _BIN_DIR.mkdir(parents=True, exist_ok=True)

    if asset_name.endswith(".zip"):
        with zipfile.ZipFile(buf) as zf:
            if "llama-server" not in [n.split("/")[-1] for n in zf.namelist()]:
                raise RuntimeError(
                    f"llama-server binary not found in zip. Contents: {zf.namelist()[:20]}"
                )
            # Extract everything flat into _BIN_DIR (strip any leading directory)
            for member in zf.infolist():
                filename = Path(member.filename).name
                if not filename:
                    continue
                dest = _BIN_DIR / filename
                with zf.open(member) as src, dest.open("wb") as dst:
                    shutil.copyfileobj(src, dst)
                # Restore executable bit from zip external_attr
                if member.external_attr >> 16 & 0o111:
                    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)
    else:
        # tar.gz / tar.xz / tar.bz2
        with tarfile.open(fileobj=buf, mode="r:*") as tf:
            members = [m for m in tf.getmembers() if m.isfile()]
            if not any(
                m.name.endswith("llama-server") or m.name.endswith("llama-server.exe")
                for m in members
            ):
                raise RuntimeError(
                    f"llama-server binary not found in archive. "
                    f"Contents: {[m.name for m in members[:20]]}"
                )
            for member in members:
                filename = Path(member.name).name
                if not filename:
                    continue
                dest = _BIN_DIR / filename
                f = tf.extractfile(member)
                if f is None:
                    continue
                with dest.open("wb") as dst:
                    shutil.copyfileobj(f, dst)
                # Restore executable bit from tar mode
                if member.mode & 0o111:
                    dest.chmod(dest.stat().st_mode | stat.S_IXUSR | stat.S_IXGRP)

    # Create missing compatibility symlinks for versioned dylibs.
    # Archives ship e.g. libllama.0.0.8902.dylib but the binary's RPATH
    # references the shorter libllama.0.dylib form.
    _create_dylib_symlinks(_BIN_DIR)

    return _INSTALLED_BIN


def _create_dylib_symlinks(bin_dir: Path) -> None:
    """
    For every versioned dylib (lib*.X.Y.Z.dylib) in bin_dir, create a
    shorter compatibility symlink (lib*.X.dylib) if it doesn't exist yet.
    """
    import re

    pattern = re.compile(r"^(lib.+?)\.(\d+)\.\d+.*\.dylib$")
    for lib in bin_dir.glob("*.dylib"):
        m = pattern.match(lib.name)
        if not m:
            continue
        short_name = f"{m.group(1)}.{m.group(2)}.dylib"
        link = bin_dir / short_name
        if not link.exists():
            link.symlink_to(lib.name)
