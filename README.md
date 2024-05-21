# Offline Reinforcement Learning with Classification

The repository organisation is inspired by [CORL](https://github.com/corl-team/CORL) and  [ReBRAC](https://github.com/DT6A/ReBRAC/tree/public-release) repositories.
## Dependencies & Docker setup
To set up a python environment (with dev-tools of your taste, in our workflow, we use conda and python 3.8), just install all the requirements:

```commandline
python3 install -r requirements.txt
```

However, in this setup, you must install mujoco210 binaries by hand. Sometimes this is not super straightforward, but this recipe can help:
```commandline
mkdir -p /root/.mujoco \
    && wget https://mujoco.org/download/mujoco210-linux-x86_64.tar.gz -O mujoco.tar.gz \
    && tar -xf mujoco.tar.gz -C /root/.mujoco \
    && rm mujoco.tar.gz
export LD_LIBRARY_PATH=/root/.mujoco/mujoco210/bin:${LD_LIBRARY_PATH}
```
You may also need to install additional dependencies for mujoco_py. 
We recommend following the official guide from [mujoco_py](https://github.com/openai/mujoco-py).

### Docker

We also provide a more straightforward way with a dockerfile that is already set up to work. All you have to do is build and run it :)
```commandline
docker build -t clorl .
```
To run, mount current directory:
```commandline
docker run -it \
    --gpus=all \
    --rm \
    --volume "<PATH_TO_THE_REPO>:/workspace/" \
    --name clorl \
    clorl bash
```

## How to reproduce experiments

### Training

Configs for reproducing results of original algorithms are stored in the `configs/<algorithm_name>/<task_type>`. All avaialable hyperparameters are listed in the `src/algorithms/<algorithm_name>.py`. Implemented algorithms are: `rebrac`, `iql`, `lb-sac`.

Configs for reproducing results of algorithms with classification are stored in `configs/<algorithm_name>-ce/<task_type>`, `configs/<algorithm_name>-ce-ct/<task_type>`, `configs/<algorithm_name>-ce-at/<task_type>`. The notation (the same in the paper): `ce` denotes the replacement of MSE with cross-entropy, `ce-at` denotes cross-entropy with tuned algorithm parameters, `ce-ct` denotes cross-entropy with tuned classification parameter. All available hyperparameters are listed in the `src/algorithms/<algorithm_name>_cl.py`. Implemented algorithms are: `rebrac`, `iql`, `lb-sac`.

For example, to start ReBRAC+classification training process with D4RL `halfcheetah-medium-v2` dataset, run the following:
```commandline
PYTHONPATH=. python3 src/algorithms/rebrac_cl.py --config_path="configs/rebrac-ce/halfcheetah/medium_expert_v2.yaml"
```

### Targeted Reproduction

[//]: # (For better transparency and replication, we release all the experiments in the form of [Weights & Biases reports]&#40;https://wandb.ai/tlab/ReBRAC/reportlist&#41;.)

If you want to replicate results from our work, you can use the configs for [Weights & Biases Sweeps](https://docs.wandb.ai/guides/sweeps/quickstart) provided in the `configs/sweeps`.

| Paper element          | Sweeps path (we omit the common prefix `configs/sweeps/`)                                                                             |
|------------------------|---------------------------------------------------------------------------------------------------------------------------------------|
| Tables 1, 2, 3         | `eval/<algorithm_name>.yaml`, `eval/<algorithm_name>-ce.yaml`, `eval/<algorithm_name>-ce-at.yaml`, `eval/<algorithm_name>-ce-ct.yaml` |
| Figure 2               | All sweeps from `expand`                                                                                                       |
| Figure 3               | All sweeps from `network_sizes`                                                                                                |
| Hyperparameters tuning | All sweeps from `tuning`                                                                                                              |

### Reliable Reports

We also provide a script and binary data for reconstructing the graphs and tables from our paper: `plotting/plotting.py`. We repacked the results into .pickle files, so you can re-use them for further research and head-to-head comparisons.
