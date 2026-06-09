# bird-call-classifier

## Machine learning project with the goal of training a model able to correctly classify bird species based on their call audio.

## Overview

The project consists of a pipeline of downloading data in the form of short audio clips of different animal species, preprocessing it to a mel-spectrogram format, and using it to train a VGG-based classification model. The model can then be evaluated via command line or by accessing the web UI.

Raw training data has the form of short audio clips (up to 30 seconds in length by default) of recorded calls and sounds of different species.

The sample set is defined in bird_species.tsv, consisting of 17 classes of polish birds observable near the city of Poznań.

## Downloading training data

Recordings can be fetched from [xeno-canto.org](https://xeno-canto.org) website. This requires a valid user account, you can retrieve your free API key by accessing [the account page](https://xeno-canto.org/account), after that you need to pass your key as an argument to the script.

```bash
pip install -r requirements.txt
python scripts/download_data.py --api-key="YOUR-API-KEY"
```

Downloaded species/genus are taken from the first column of bird_species.tsv by default (other columns are for user clarity and are not parsed by code). Therefore, it's possible to change the set of species by editing the first column of the file before issuing the download.

Audio is saved under `data/<Genus_species>/` (e.g. `data/Corvus_cornix/`). By default, up to 200 recordings are downloaded per species. Re-runs skip recordings already fetched (`downloaded_ids.json` in `data/`).

Useful options:

- `python scripts/download_data.py --verbose` — progress per recording
- `python scripts/download_data.py --max-per-species 50` — change the per-species limit
- `python scripts/download_data.py --species "Parus major" "Turdus merula"` — subset only

#### Kaggle dataset (pre-defined species only)

Because downloading raw data from xeno-canto servers might take a while (around 30-45 minutes for the sample dataset), the default set of recordings (17 classes of polish birds observable near the city of Poznań, 200 recordings per class), is available for direct download at Kaggle website here: https://www.kaggle.com/datasets/xnyzcyz/polish-birds-call-audio

## Preprocessing data

After downloading the dataset, the raw audio files need to be pre-processed and converted to mel-spectrograms (stored as \.npy matrices under `processed_data/`) before they can be fed into the model infrastructure.

To preprocess the downloaded recordings, run:
```bash
python scripts/preprocess_data.py
```

## Training the model

After downloading and preprocessing the data, you can train the model via:

```bash
python scripts/train.py --run-name [NAME]
```
which trains the model on default hyperparameters and batching strategies (explained more in the report). Trained models are saved to `checkpoints/{RUN_NAME}/`, with timestamp as the default name.

#### Tensorboard implementation

Tensorboard implentation is available for easy monitoring and comparison of different training runs, to access it, source the virtual environment and run:

```bash
tensorboard --logdir runs
```

you can then access the panel at `127.0.0.1:6006`.

## Running the inference

Trained models can be evaluated on external audio, returning a probability distribution of top 3 best matching classes. The inference can be ran either either in the available web app, accessed by running
```bash
python scripts/web_app.py
```

which runs the GUI at `127.0.0.1:5000`, or directly via CLI, with `python scripts/predict.py [AUDIO_FILE]`

The files are evaluated by default with the best available model (the one stored under `checkpoints/best.pt`).

## Additional info

More detailed technical information is accessible at report.pdf. Project documentation and presentation materials are in `docs/`.
