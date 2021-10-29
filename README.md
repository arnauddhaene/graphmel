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

### Graph Representation Dashboard

To test out different lesion-connection methodologies, a dashboard was developed. To launch it, run the following command from the base directory:

```bash
streamlit run src/representation.py
```

### Hyper-parameter optimization

Raytune is set up to run a hyper-parameter optimization in `hpopt.py`. You can extend or constrain the hyper-parameter options in the `tune.run` call. To run the optimization:

```bash
python src/hpopt.py
```

with any of the optional parameters described below.

```bash
Usage: hpopt.py [OPTIONS]

Options:
  --model [GNN|GAT|GIN|DiffPool]  Model architecture choice.
  --connectivity [fully|organ|wasserstein]
                                  Graph connectivity choice.
  --epochs INTEGER                Number of training epochs.
  --test-size FLOAT               Test set size in ratio.
  --seed INTEGER                  Random seed.
  --cv INTEGER                    Cross-validation splits.
  --verbose INTEGER               Print out info for debugging purposes.
  --help                          Show this message and exit.
```


### MLflow

The modeling experiments and their runs are tracked with MLflow. To run the UI, use the following command:

```bash
mlflow ui
```

If you happen to quit your Terminal window before killing MLflow (as I have done too many times). Use the following commands:

```bash
lsof -i :5000
kill -9 <PID>
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
