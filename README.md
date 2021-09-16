# Project Name

###### Semester project in LTS4, autumn 2021

## How to run

We have implemented argument parsing with the `click` library. 

For reference, here are the options (find them by running `python src/test.py --help`).
```bash
-> % python src/run.py --help
Usage: run.py [OPTIONS]

Options:
  --model [ConvNet|MLP]           Model to evaluate.
  --siamese / --no-siamese        Use a siamese version of the model.
  --epochs INTEGER                Number of training epochs.
  --lr FLOAT                      Learning rate.
  --decay FLOAT                   Optimizer weight decay.
  --gamma FLOAT                   Auxiliary contribution.
  --trials INTEGER                Number of trials to run.
  --seed INTEGER                  Seed for randomness.
  --batch-size INTEGER            Batch size for training.
  --standardize / --dont-standardize
                                  Standardize train and test data with train
                                  data statistics.
  --make-figs / --no-figs         Create figures for the trial.
  --clear-figs / --keep-figs      Clear the figures directory of all its
                                  contents.
  --verbose INTEGER               Print out info for debugging purposes.
  --help                          Show this message and exit.

```

## Structure

### test

The `test.py` script run is able to run training trials on any of our evaluated models. The default parameters run our best performing model. For a different trial procedure, one could run, for instance:

```bash
python src/test.py --model MLP --no-siamese --epochs 20 --lr 1e-3 --decay 5e-3 --gamma 0.75 --trials 4
```

This procedure automatically saves training and testing plots to a `Proj1-extended/figures/` subdirectory named by the current timestamp.

### train

This module implements the `train` method, used for training a given model.

### utils

This module implements training and testing set standardisation and is also able to pull a dataset using predefined functions in [`dlc_practical_prologue.py`](dlc_practical_prologue.py).

### metrics

This module implements the custom `TrainingMetrics` and `TestingMetrics` classes used to track model performance. Above that, it automatically creates plots about selected training and testing metrics.

### hyperopt

This module implements grid-search hyper-parameter optimization. It has its own argument parsing and calls the `test.py` run function. The parameters of this script can be displayed using `python src/hyperopt.py --help`.

### models

The models package contains all the models evaluated during the course of this project.

### notebooks

Some notebooks have been added as they were used to generate plots. They are not essential but remain in our submission for completeness.

## Authors

* DHAENE, Arnaud
