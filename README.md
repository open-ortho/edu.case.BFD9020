# edu.case.BFD9020
AI services backend for the BFD9000 project, starting with FastAPI-based image classification.

## API

This FastAPI service exposes lightweight endpoints for both coarse X-ray typing and flip/rotation inference on lateral and frontal ceph studies.

### Endpoints

- `GET /` – simple health/welcome message.
- `POST /xray-info` – returns type, rotation, and flip info for a single X-ray.
- `POST /xray-class` – predicts only the X-ray type.
- `POST /lateral-fliprot` – rotation/flip classification for lateral ceph images.
- `POST /frontal-fliprot` – rotation/flip classification for frontal ceph images.

## Utilities

- `BFD9020.html` – browser tester that runs all endpoints in sequence. When the API is running visit `/test`; otherwise open the file locally and point it to a remote base URL. TIFF previews are decoded client-side via vendored `pako` + `UTIF` scripts exposed from `/static`.
