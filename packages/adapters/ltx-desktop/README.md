# adapter-ltx-desktop

This package is the compatibility seam for the existing `ltx-desktop` shell.

Its role is narrower than a first-party `MLXR` app:

- preserve the current desktop shell where useful
- replace direct local Torch execution with calls to the shared runtime
- keep host-specific compatibility code out of the runtime core

This package is not the home of the new first-party Mac app. The first-party app
track lives separately as a native `MLXR` client over the same runtime.
