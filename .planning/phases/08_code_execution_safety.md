# Phase 8: Code Execution Safety (Sandbox)

Code execution is the highest-risk surface in any coding agent. Mitigation layers:

1. **Subprocess isolation**: `subprocess.run` with `timeout=30` and `check=False`. Resource capping is platform-specific:
   - **Linux/macOS**: `resource.setrlimit` (Unix-only — does not exist on Windows)
   - **Windows**: Use Job Objects via `subprocess` + `ctypes`, or rely on container isolation
   - **Cross-platform baseline**: `timeout` + `subprocess` without shell; no `shell=True`
2. **Workspace sandboxing**: Agent can only read/write files under a designated workspace directory. All paths are resolved and validated against the workspace root before any file operation.
3. **No network access** by default during code execution (can be toggled per session)
4. **Recommended path to full isolation**: Docker with `--network none --read-only --tmpfs /tmp` — works cross-platform and gives the strongest guarantee without kernel-level complexity

**Cross-platform sandboxing strategy:**

| Platform | Lightweight | Robust |
| :--- | :--- | :--- |
| Linux | `resource.setrlimit` + subprocess | Docker / gVisor |
| macOS | `resource.setrlimit` + subprocess | Docker |
| Windows | subprocess + timeout only | Docker with WSL2 |

Never pass user-controlled strings directly to `shell=True`. All tool arguments are validated against JSON Schema before execution reaches the subprocess layer.
