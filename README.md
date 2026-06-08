# bird-call-classifier

Machine learning project with the goal of training a model able to correctly classify bird species based on their call audio.

## Downloading training data

Recordings are fetched from [xeno-canto.org](https://xeno-canto.org). You need a free API key from [your account page](https://xeno-canto.org/account).

```bash
pip install -r requirements.txt
export XENO_CANTO_API_KEY="your-key-here"
python scripts/download_data.py 
# or pass the key as an argument 'python scripts/download_data.py --api-key="your-key-here"'
```

Audio is saved under `data/<Genus_species>/` (e.g. `data/Grus_grus/`). By default, up to 50 recordings are downloaded per species. Re-runs skip recordings already fetched (`downloaded_ids.json` in `data/`).

Useful options:

- `python scripts/download_data.py --verbose` — progress per recording
- `python scripts/download_data.py --max-per-species 100` — change the per-species limit
- `python scripts/download_data.py --species "Parus major" "Turdus merula"` — subset only

## Preprocessing data

Audio files need to be converted to mel-spectrograms (stored as `.npy` matrices) before they can be fed into the CNN.

To preprocess the downloaded recordings, run:
```bash
python scripts/preprocess_data.py
```
This script will read all `.mp3` and `.wav` files from the `data/` directory, pad/trim them to exactly 30 seconds, and save the resulting mel-spectrograms to the `processed_data/` directory. Note that both `data/` and `processed_data/` are excluded from version control.

## Training on Kaggle GPU

You do **not** need to rewrite the project as a notebook. The repo includes a thin Kaggle notebook that clones this project, links your uploaded dataset, and runs the existing training scripts on a GPU.

### 1. Package your data for Kaggle

Yes — you can save your dataset to Kaggle and reuse it in notebooks. Upload the **processed** spectrograms (recommended, ~670 MB) rather than raw audio (~2.6 GB).

```bash
pip install kaggle
# Configure Kaggle API credentials in ~/.kaggle/kaggle.json

python scripts/package_kaggle_dataset.py --username YOUR_KAGGLE_USERNAME
kaggle datasets create -p kaggle_upload
```

This creates a dataset with structure:

```
processed_data/
  Alauda_arvensis/
    *.npy
  ...
```

You can also package raw audio instead:

```bash
python scripts/package_kaggle_dataset.py --username YOUR_KAGGLE_USERNAME --source raw --dataset-slug bird-call-raw --title "Bird Call Raw Audio"
```

If you upload raw audio, run `python scripts/preprocess_data.py` inside the notebook before training.

### 2. Create the Kaggle notebook

1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **New Notebook**
2. **Add Data** → select your uploaded dataset (e.g. `bird-call-processed`)
3. **Settings** → **Accelerator** → **GPU**
4. Upload or copy `kaggle/train.ipynb` into the notebook
5. Set `DATASET_SLUG` in the notebook to match the folder name under `/kaggle/input/`
6. Run all cells

The notebook will:
- Clone this repo into `/kaggle/working/`
- Symlink `/kaggle/input/.../processed_data` into the project
- Install Python deps (PyTorch is already available on Kaggle)
- Run `train.py --device cuda`
- Copy checkpoints to `/kaggle/working/outputs/` so they persist as notebook output

### 3. Train both batching strategies

Locally:

```bash
python train_all_modes.py
```

On Kaggle, uncomment the last cell in `kaggle/train.ipynb` or run strategies one at a time by changing `BATCHING_STRATEGY`.

### Optional: push notebook via Kaggle API

Edit `kaggle/kernel-metadata.json` with your username and dataset slug, then:

```bash
kaggle kernels push -p kaggle
```
