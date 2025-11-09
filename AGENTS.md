# Repository Guidelines

## Project Structure & Module Organization
- `src/data`, `src/utils`: dataset bindings (e.g., `RH20TTraceDataset`) and tensor helpers shared across training and evaluation.
- `src/models`, `src/loss`: transformer-based cross-modal encoders plus contrastive objectives; keep weight presets in `model_weight/`.
- `src/pipelines`: task-specific entry points such as `trajectoryTrain.py`, `trajectory_hypersearch.py`, and augmentation pipelines; scripts expect dataset roots under `dataset/`.
- `src/evaluation`: retrieval metrics (`evaluate_gemini`) and ranking utilities for reporting results found in `results/`.
- Top-level `scripts/` hosts quick experiments (`videomae*.py`, `test.py`) and is safe for ad-hoc prototypes; sync durable logic back into `src/`.

## Build, Test, and Development Commands
- `python -m pip install -e .` — installs `bise` with its PyTorch/Transformer stack for local development; re-run after editing `setup.py`.
- `python src/pipelines/trajectoryTrain.py --data-root <path>` — launches the default training loop; override constants such as `BATCH_SIZE` and `MODEL_PARAMS` via CLI flags when adding them.
- `python src/pipelines/trajectory_hypersearch.py --config configs/cfg23.yaml` — run sweeps before changing default hyper-parameters.
- `python scripts/test.py` — fast CUDA sanity check to ensure PyTorch sees the expected device before longer jobs.

## Coding Style & Naming Conventions
- Follow PEP 8: four-space indents, 100-character lines, and `snake_case` for functions/modules; keep classes in `PascalCase`.
- Prefer explicit tensor device/dtype handling (`tensor.to(device)` once per batch) and annotate public functions with type hints.
- Config or experiment files should follow the existing `trajectoryTrain_cfg<number>.py` naming.
- Keep inline comments English and action-oriented; summarize non-obvious tensor transforms right above the block.

## Testing Guidelines
- Add scenario-focused tests under `tests/` (create if absent) using `pytest`; mirror module names (`test_trajectory_encoder.py`) and target ≥80 % coverage for touched files.
- For ranking logic, reuse fixtures that build small `torch` tensors and assert `evaluate_gemini` metrics.
- Before opening a PR, run `pytest src tests` plus `python scripts/test.py` to catch CUDA regressions.

## Commit & Pull Request Guidelines
- Keep commits short and imperative (e.g., `use_6_keypoints`, `trajectory_augment`); squash noisy WIP commits locally.
- PRs should describe motivation, dataset shard, and metrics (include `R@1/5/10` and `mean_percentage_rank` from `results/`).
- Link issues or experiment notes, attach tensorboard screenshots when you change learning schedules, and call out compatibility risks (dataset schema, checkpoint format).

## Security & Configuration Tips
- Never commit raw RH20T subsets or API credentials; keep per-user paths in `.env` or local YAML and respect `.gitignore`.
- Parameterize `DATASET_ROOT` and checkpoint destinations to avoid hard-coded `/home/ttt` paths; prefer relative paths when scripting.
- Validate any new cloud evaluation hooks in `evaluate_gemini` with mocked responses before enabling real keys.
