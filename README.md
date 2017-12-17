# Setup

1. Install conda/miniconda
2. Run `conda env create -f environment.yml`
3. Activate environment (see terminal output for command)
4. Run `python genetic-gaming/run.py -config=genetic-gaming/config/flappybird.json -single_process`

## Saving and Restoring
To simplify reproducing bugs or agent accomplishments, it's possible to save and restore all configuration parameters and learned network weights. This can be achieved using the `-save_to=SAVE_DIR` flag and the `-restore_from=SAVE_DIR` flag, respectively.
