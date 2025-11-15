# Dev Scratchpad

## Runtime overview
- Entry point `main.py` spins up FastAPI with `lifespan` that loads three fastai learners. Run by `python main.py` or `uvicorn main:app --host 0.0.0.0 --port 9020`.
- Async endpoints call FastAI `Learner.predict` via `run_in_threadpool`; uploads arrive as `UploadFile`.
- Upload guardrails: MIME whitelist (jpeg/png/tiff), 10 MB cap enforced via `seek` (no double read), centralized logging + exception hooks.
- Preprocessing: `imageio.v3.imread` -> drop alpha -> `exposure.rescale_intensity(..., out_range=np.float64)` -> `img_as_ubyte` -> `PILImage.create`.
- `/healthz` reports readiness (`ok` vs `degraded`) based on which learners are loaded.

## Models & data
- Models expected at `models/xtype-simple_resnet18_fp16_01`, `models/lateral_fliprot_resnet18_fp16_07`, `models/frontal_fliprot_resnet18_fp16_03` (fastai exported learners). They are treated as files (not dirs); ensure git-lfs or manual drop.
- `label_func` stub exists solely to satisfy learner export requirements (fastai dataloaders reference it).

## API surface
- `/` welcome ping.
- `/xray-info` chained classification: global type + optional flip/rotation for lateral/frontal (uses `map_fliprot_prediction`).
- `/xray-class` type-only; `/lateral-fliprot` + `/frontal-fliprot` call `classify_specific_model` directly.
- All endpoints reuse `validate_file_size` + `validate_image`; responses include raw prediction, probability, full vocab w/ probabilities.

## Operational notes
- Logging now stdout-only (single `StreamHandler`), container-friendly.
- CORS wide open.
- Dockerfile: multi-stage (`python:3.13-slim` base) installs torch CPU wheels, fastai, scikit-image, imageio, fastapi. Models copied via scratch stage; `BFD9020.html` copied for `/test` route.
- Compose file (`docker-compose.yml`) builds from repo root, publishes `9020:9020`, sets `LOG_LEVEL=INFO`.
- Tester available at `/test` (serves `BFD9020.html`); standalone file also works locally. TIFF thumbnails are rendered via `<canvas>` using vendored `/static/pako.min.js` + `/static/UTIF.js` (FastAPI mounts `StaticFiles(directory="static")`).
- No requirements.txt/pyproject—deps inferred from image setup (fastapi, fastai, torch/vision/audio, scikit-image, imageio, numpy, uvicorn).
