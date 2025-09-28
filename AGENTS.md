# Repository Guidelines

## Project Structure & Module Organization
- `src/yolo13_reading/` — Python package (entry module: `__init__.py`).
- `context/` and `magic-context/` — research notes, prompts, and supporting materials.
- `notes/` — ad‑hoc notes and TODOs.
- `paper-soruce/` — paper artifacts (e.g., YOLO13 references).
- Tests (if added) live under `tests/` mirroring `src/` paths.

## Build, Test, and Development Commands
- Install environments: `pixi install`
- Open dev shell (GPU/CUDA 12.6 enabled): `pixi shell -e dev`
- Quick check: `pixi run -e dev python -c "import torch; print(torch.__version__, torch.version.cuda)"`
- Run scripts: `pixi run -e dev python -m yolo13_reading`
- Add packages to dev only: edit `[tool.pixi.feature.dev.pypi-dependencies]` in `pyproject.toml`.

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indentation, type hints where practical.
- Names: modules `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Keep public APIs small; prefer simple, composable helpers in `src/yolo13_reading/`.

## Testing Guidelines
- Framework: `pytest` (recommended). Create `tests/` matching `src/` layout.
- Naming: `test_<module>.py`, functions `test_<behavior>()`.
- Run: `pixi run -e dev pytest -q` (add `pytest` to the dev feature first).
- Aim for tests alongside any new module; keep unit tests deterministic.

## Commit & Pull Request Guidelines
- Commits: imperative mood, concise scope first line (e.g., "Add parser for ...").
- Include rationale and notable trade‑offs in the body when helpful.
- PRs: clear description, linked issues, reproduction/validation steps, and screenshots if UI‑adjacent.
- Keep changes focused; avoid drive‑by refactors.

## Environment Notes
- Python: project targets `>=3.11`; Pixi dev pins `>=3.13.7,<3.14`.
- GPU: dev environment uses PyTorch CUDA 12.6 via `extra-index-urls`. If you lack CUDA, remove the `dev` feature or switch to CPU wheels.
# Repository Guidelines

## Project Structure & Module Organization
- `src/yolo13_reading/` — Python package entry (library code).
- `notes/` — paper reading notes (e.g., `notes/YOLOv13/` with `figures/`).
- `paper-soruce/yolo13/` — paper assets (LaTeX, figures); do not modify.
- `context/` and `magic-context/` — reference material and prompts.
- `tmp/` — temporary scripts (non-shipping), e.g., PDF→PNG converter.

## Build, Test, and Development Commands
- Install environments: `pixi install`
- Dev shell (CUDA 12.6, tools): `pixi shell -e dev`
- YOLOv13 shell: `pixi shell -e yolo13`
- Run tests: `pixi run -e dev pytest -q`
- Lint: `pixi run -e dev ruff check .`
- Type-check: `pixi run -e dev mypy src`
- Convert figure (PDF→PNG): `pixi run -e dev convert-figs`

## Coding Style & Naming Conventions
- Python: PEP 8, 4‑space indent, add type hints for public functions.
- Names: modules/files `snake_case.py`; classes `PascalCase`; functions/vars `snake_case`.
- Linting: use `ruff` for style; address reported issues before pushing.
- Typing: `mypy` in dev env; prefer precise types over `Any`.

## Testing Guidelines
- Framework: `pytest`. Place tests under `tests/` mirroring `src/`.
- Filenames: `tests/test_<module>.py`; test names `test_<behavior>()`.
- Keep tests deterministic; avoid network and GPU dependence unless required.
- Run via `pixi run -e dev pytest -q`.

## Commit & Pull Request Guidelines
- Commits: imperative, concise first line (e.g., "Add YOLOv13 env feature").
- Group related changes; avoid drive‑by refactors.
- PRs: include summary, rationale, linked issues, and validation steps (commands/output). Screenshots when UI/figures change.

## Agent‑Specific Instructions
- Do not modify `context/refcode/yolov13` (Git submodule). It is configured with `ignore = all`; local changes are intentionally ignored.
- Prefer Pixi tasks and envs:
  - Dev tooling lives in feature `dev`; CUDA wheels via `--index-url` equivalent (CUDA 12.6).
  - YOLOv13 usage in env `yolo13`; import via `from ultralytics import YOLO`.
- Keep default environment clean; add tools only to the `dev` feature.
