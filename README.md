# Room Occupancy Detection

Predicts whether a room is occupied from sensor readings (temperature, humidity,
light, CO2, humidity ratio, and a timestamp), so systems like HVAC and lighting
can be switched on or off automatically.

This repo covers the engineering side: a training pipeline, an inference API,
containers, and CI. It's not trying to squeeze out the best possible accuracy.

## Layout

```
data/                     datasets (CSV)
src/occupancy/
  config.py               paths, feature lists, hyperparameter search space
  data.py                 load / split the CSVs
  features.py             datetime features + the sklearn pipeline
  train.py                XGBoost + randomized search, evaluation, saving
  predict.py              load the model and score a reading
api/main.py               FastAPI service
tests/                    unit + integration tests
docker/                   Dockerfile.train, Dockerfile.serve
docker-compose.yml        train, then serve
.github/workflows/ci.yml  lint, test, build & push images
```

## Training

Feature engineering derives `hour` and `weekday` from the timestamp and one-hot
encodes them; the numeric sensor columns pass through. Preprocessing and the
XGBoost classifier are bundled in one scikit-learn `Pipeline`, so whatever runs
in training runs identically at inference. Hyperparameters are tuned with
`RandomizedSearchCV` on ROC-AUC, and the best model is evaluated on the two
held-out test sets.

```bash
python -m venv .venv
. .venv/Scripts/activate            # PowerShell: .\.venv\Scripts\Activate.ps1
pip install -r requirements-dev.txt
export PYTHONPATH=src               # PowerShell: $env:PYTHONPATH="src"
python -m occupancy.train
```

Writes `models/occupancy_model.joblib` and `models/metrics.json`. ROC-AUC lands
around 0.99 on both test sets.

## Inference

Each `POST /predict` scores one reading, so the service is stateless and can sit
behind a queue or load balancer.

```bash
export PYTHONPATH=src
uvicorn api.main:app --reload
```

- `GET /health` reports status and whether the model is loaded.
- `POST /predict` takes `{date, Temperature, Humidity, Light, CO2, HumidityRatio}`
  and returns `{occupancy, probability}`.

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"date":"2015-02-04 17:51:00","Temperature":23.18,"Humidity":27.27,"Light":426,"CO2":721.25,"HumidityRatio":0.0047929}'
# {"occupancy":1,"probability":0.808...}
```

## Docker

Two images: one runs training and writes the model to a shared volume, the other
serves the API and reads it. Compose runs them in order.

```bash
docker compose up --build        # trains, then serves on :8000
```

## CI

`.github/workflows/ci.yml` runs on push and PR to `main`:

1. Lint with `ruff`, run `pytest`.
2. On `main`, build both images and push them to GHCR, tagged `latest` and the
   commit SHA.

Keeping training and serving as separate images means retraining and redeploying
the API are independent.

## Continuous training

The training image is the retraining job; a scheduler or a drift alarm just runs
it. A rough plan:

- Trigger on a schedule and on a drift/performance alarm.
- Log prediction requests and join delayed ground truth (badge-in logs, motion
  sensors, audits) to build new training batches.
- Monitor input drift and, once labels arrive, live ROC-AUC.
- Retrain by running the existing training image against the new data.
- Only promote a new model if it beats the current one on a frozen holdout and
  passes schema checks.
- Register models, roll out with a canary, keep the previous one for rollback.

## Tests

```bash
pytest
```
