# Docker images

Three image variants are provided; pick the one that matches your runtime.

| Tag         | Base                                       | Size (approx) | Backends         | Use case                                |
|-------------|--------------------------------------------|---------------|------------------|-----------------------------------------|
| `tode:cpu`  | `python:3.12-slim`                         | ~600 MB       | ONNX (CPU)       | CI, headless batch, dataset export      |
| `tode:gpu`  | `pytorch/pytorch:2.4.0-cuda12.4-cudnn9`    | ~7 GB         | Ultralytics + ONNX (GPU) | Full GUI build with NVIDIA acceleration |
| `tode:onnx` | `python:3.12-slim`                         | ~500 MB       | ONNX (CPU)       | AGPL-free deploys (no torch, no ultralytics) |

The repository root also contains a `Dockerfile` that builds the default
GUI-capable CPU image used by `docker-compose.yml`. The variants in this
folder are alternatives for specific environments.

## Build

All commands are run from the repository root.

```bash
docker build -f docker/Dockerfile-cpu  -t tode:cpu  .
docker build -f docker/Dockerfile-gpu  -t tode:gpu  .
docker build -f docker/Dockerfile-onnx -t tode:onnx .
```

## License notes

- The `tode:gpu` and root `Dockerfile` images include the `ultralytics`
  package (AGPL-3.0). Distributing or hosting a service built on those
  images requires releasing your source code under AGPL-3.0.
- The `tode:onnx` image deliberately omits `ultralytics` and `torch` so
  that the resulting image is AGPL-free.
- See `THIRD_PARTY_LICENSES.md` at the repo root for the full breakdown.
