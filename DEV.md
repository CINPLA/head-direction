# Developer notes

This document provides guidelines for maintaining and developing this project. Following these practices ensures that the research remains reproducible and our codebase remains clean.

---

### uv
We use [uv](https://docs.astral.sh/uv/) for dependency management and environment isolation. It replaces `pip`, `conda`, and `venv`.

**Install**
```bash
# Unix (macOS/Linux)
curl -LsSf [https://astral.sh/uv/install.sh](https://astral.sh/uv/install.sh) | sh

# Windows
powershell -c "irm [https://astral.sh/uv/install.ps1](https://astral.sh/uv/install.ps1) | iex"
```

> **IMPORTANT:**  Ensure you are not in a Conda base environment when using `uv`. Deactivate conda first (`conda deactivate`) to avoid path conflicts


**Adding dependencies**

```bash
# Research libraries (torch, numpy, etc.)
uv add <package>

# Development tools (ruff, mypy, etc.)
uv add <package> --optional dev
```

**Sync & environment**

```bash
# Sync your local environment with the lockfile (all dependencies included)
uv sync --all-extras --dev

# Run scripts via the managed environment
uv run python <script.py>

# OR activate the environment traditionally
source .venv/bin/activate  # Unix
.venv\Scripts\activate     # Windows
```

**Updating uv**
```bash
uv self update
```

---

### pre-commit

We use [pre-commit](https://pre-commit.com/) as our project's gatekeeper. It acts as an automated "health check" that runs every time you try to commit code. Instead of manually checking your code, pre-commit automatically triggers Ruff (for linting and formatting) to ensure everything is perfect before it hits the repository. Our configuration is specified in the `.pre-commit-config.yml` file.

**Set up Git hooks**

To set up the hooks, run:
```bash
uv run pre-commit install
```

**What happens during a commit?**

1.  **Automated cleanup:** Ruff automatically reformats your code, sorts imports, and fixes many linting errors on the fly.
2.  **Deep logic checks:** Ruff performs deep analysis to catch logic bugs (e.g., incorrect `torch` usage or legacy `os.path` calls). If it finds an issue it cannot fix automatically, the commit is blocked so you can review the error.
3.  **The safety gate:** If any automatic changes were made, the commit will stop. This is a safety feature to ensure you see exactly what was modified. You must **stage the new changes** and commit again to finish.

**Manual run**

You can run all checks on the entire codebase at any time:

```bash
uv run pre-commit run --all-files
```

**Update hooks**

To keep our tools and rules up to date:

```bash
uv run pre-commit autoupdate
```

---

### Ruff
We use [Ruff](https://docs.astral.sh/ruff/) as an all-in-one linter, formatter, and import sorter. It is significantly faster than traditional tools like Black or Flake8.

**Common commands**
```bash
# Check for errors and sort imports
uv run ruff check

# Automatically fix what can be fixed
uv run ruff check --fix

# Format the code according to our style guide
uv run ruff format
```

> **TIP:**  If you use VS Code or Cursor, install the Ruff extension. It will highlight errors as you type and can be configured to "Format on Save."

---
