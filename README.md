# Graph analytics for immunotherapeutic response prediction for metastatic melanoma patients

[![flake8 Actions Status](https://github.com/arnauddhaene/graphmel/actions/workflows/lint.yml/badge.svg)](https://github.com/arnauddhaene/graphmel/actions)

###### Semester project in LTS4, autumn 2021

## Structure

### EDA (Exploratory Data Analysis)

A semi-interactive dashboard was developed with Streamlit in `eda.py`. To launch it, run the following command from the base directory:

```bash
streamlit run src/eda.py -- --data filepath
```

If you are on macOS, `filepath` will most likely be `/Volumes/lts4-immuno/data_2021-09-20` (VPN connection to EPFL and smb connection to fileserver are necessary conditions for this to work).

### MLflow

The modeling experiments and their runs are tracked with MLflow. To run the UI, use the following command:

```bash
mlflow ui
```

#### Sharing the experimentation results

For the moment, the `mlruns` is included in the `.gitignore` as the artifacts are quite large and most coding is done on my laptop.

For this reason, I will use `ngrok` to share my localhost port via http during meetings or code review sessions. For reference, the command is:

```bash
cd && ./ngrok http 5000
```

### Running the model

To run the model, use the following command:

```bash
python src/run.py
```

with any of the optional parameters described below.

```
Usage: run.py [OPTIONS]

Options:
  --filepath TEXT               Filepath where most recent data is stored.
  --connectivity [fully|organ]  Graph connectivity choice.
  --epochs INTEGER              Number of training epochs.
  --lr FLOAT                    Learning rate.
  --decay FLOAT                 Optimizer weight decay.
  --hidden-dim INTEGER          GNN hidden dimensions.
  --batch-size INTEGER          Batch size for training.
  --experiment-name TEXT        Assign run to experiment.
  --verbose INTEGER             Print out info for debugging purposes.
  --help                        Show this message and exit.
```

## Authors

* DHAENE, Arnaud
