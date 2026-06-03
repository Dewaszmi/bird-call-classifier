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
