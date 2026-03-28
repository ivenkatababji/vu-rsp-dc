# RPS vision training & ONNX export

Trains a **MobileNet V3 Small** head for **rock / paper / scissors / none** from a CSV + image directory, exports **ONNX** and a **manifest.json** that matches the web client (NCHW, ImageNet mean/std, input name `input`).

Configuration is **JSON only**: each script takes a single argument, the path to its config file. **Relative paths in JSON are resolved from the config file’s directory** (not the shell’s current working directory).

## Setup

```bash
cd train
python3 -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -r requirements-train.txt
```

## Train config

Copy `train_config.example.json` to e.g. `my_train.json` and set:

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `data_dir` | yes | — | Root folder for image files |
| `csv` | yes | — | CSV with labels |
| `filename_column` | no | `filename` | CSV column for image path or filename |
| `label_column` | no | `label` | CSV column for class name |
| `image_size` | no | `224` | Square H=W for train and ONNX |
| `epochs` | no | `20` | |
| `batch_size` | no | `16` | |
| `lr` | no | `0.0003` | AdamW learning rate |
| `val_frac` | no | `0.15` | Stratified validation fraction |
| `seed` | no | `42` | |
| `out_dir` | no | `export_rps` | Export directory for ONNX + manifest |
| `manifest_version` | no | `1.0.0` | Written into `manifest.json` |

## CSV format

- Header row with at least **`filename`** and **`label`** (or set `filename_column` / `label_column` in JSON).
- **label** must be one of: `rock`, `paper`, `scissors`, `none` (case-insensitive).
- **filename** can be a path relative to `data_dir`, or a basename under `data_dir`, or an absolute path.

Example:

```csv
filename,label
img001.jpg,rock
sub/paper2.png,paper
```

## Train and export

```bash
python train_export.py my_train.json
```

- **`image_size`**: square resolution used for training and in ONNX (must match what the browser will use). Common values: `224`, `256`, `320`.

Outputs in `out_dir`:

- `model.onnx` — deploy to the server as `ml_artifacts/vision/model.onnx`
- `manifest.json` — copy alongside it (defines width/height/labels for the client)
- `training_meta.json` — optional notes (val accuracy, counts)

## Deploy config

Copy `deploy_config.example.json` to e.g. `my_deploy.json`:

| Key | Required | Default | Meaning |
|-----|----------|---------|---------|
| `export_dir` | yes | — | Directory with `model.onnx` (and usually `manifest.json`) from training |
| `dest` | no | `../deployment/src/server/ml_artifacts/vision` (from repo `train/`) | Server vision artifacts directory |

## Deploy into the running API tree

From repo `train/` (default `dest` is `../deployment/src/server/ml_artifacts/vision`):

```bash
python deploy_model.py my_deploy.json
```

Restart the server if it is already running (so file hashes are picked up). In **Admin → Settings**, enable **Vision** under client input modes.

## Server-side default resolution (no manifest yet)

If `model.onnx` is not present, the API still advertises default input dimensions for documentation. You can set either:

- In **`deployment/src/server/server_config.json`** (optional keys):

  ```json
  "vision_input_size": 256
  ```

  or

  ```json
  "vision_input_width": 256,
  "vision_input_height": 256
  ```

- Or environment variables: `RPS_VISION_INPUT_SIZE`, or `RPS_VISION_INPUT_WIDTH` + `RPS_VISION_INPUT_HEIGHT`.

**Deployed `manifest.json` overrides** these for vision (and audio) whenever it includes an `input` block.

## End-to-end check

1. Train with the same `image_size` you intend in production.
2. Deploy ONNX + `manifest.json` via `deploy_model.py` and a deploy JSON whose `export_dir` points at the train `out_dir`.
3. Open the game, sign in, use **Photo** after enabling vision in admin.

The browser resizes the captured image to **manifest `width` × `height`** before inference.
