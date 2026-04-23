# Phase 8: Code Execution Safety (Sandbox)

Code execution is the highest-risk surface in any coding agent. Mitigation layers:

1. **Subprocess isolation**: `subprocess.run` with a configurable soft timeout and an interactive prompt on expiry (see Timeout Behaviour below). Resource capping is platform-specific:
   - **Linux/macOS**: `resource.setrlimit` (Unix-only — does not exist on Windows)
   - **Windows**: Use Job Objects via `subprocess` + `ctypes`, or rely on container isolation
   - **Cross-platform baseline**: `timeout` + `subprocess` without shell; no `shell=True`
2. **Workspace sandboxing**: Agent can only read/write files under a designated workspace directory. All paths are resolved and validated against the workspace root before any file operation.
3. **No network access** during code execution by default. Controlled via `[sandbox] allow_network` in `config.toml` — set to `true` to permit outbound connections. Default is `false`.
4. **Recommended path to full isolation**: Docker with `--network none --read-only --tmpfs /tmp` — works cross-platform and gives the strongest guarantee without kernel-level complexity

**Cross-platform sandboxing strategy:**

| Platform | Lightweight | Robust |
| :--- | :--- | :--- |
| Linux | `resource.setrlimit` + subprocess | Docker / gVisor |
| macOS | `resource.setrlimit` + subprocess | Docker |
| Windows | subprocess + timeout only | Docker with WSL2 |

## Timeout Behaviour

Rather than hard-killing the process at a fixed deadline, LoCoder uses a **soft timeout with a user prompt**:

```
Code execution is taking longer than expected (60s).
  [w] Wait another 60s
  [a] Abort — let the agent try a different approach
```

- The user can keep choosing **[w]** indefinitely to extend execution
- Choosing **[a]** sends `SIGTERM` (or `TerminateProcess` on Windows), then `SIGKILL` after 5s if the process hasn't exited
- After abort, control returns to the `[observe]` node with the result `"Execution aborted by user after Ns"` — the agent reasons about it in the next `[plan]` step and may try a different approach

**Config** (under `[sandbox]` in `config.toml`):

```toml
[sandbox]
execution_timeout = 60        # Seconds before the first user prompt appears
max_extensions = 10           # Maximum number of times the user can choose [w] (0 = unlimited)
allow_network = false         # Network access during code execution — off by default
```

`max_extensions = 0` disables the ceiling entirely — the user can wait as long as they like. A non-zero value prevents runaway processes from being kept alive indefinitely by accident.

Never pass user-controlled strings directly to `shell=True`. All tool arguments are validated against JSON Schema before execution reaches the subprocess layer.
