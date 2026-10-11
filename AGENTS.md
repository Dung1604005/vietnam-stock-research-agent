# Workspace rules

- Install all project dependencies, virtual environments, package caches, build caches, and temporary project files on drive `D:` inside this workspace whenever possible.
- Do not install project packages into the system Python on drive `C:`.
- The existing system Python executable may be used only to bootstrap the workspace-local `.venv` when no D-drive Python distribution is available.
- This project is analysis-only. Never add broker integration or real trade execution.

