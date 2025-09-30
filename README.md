# data_swarm

Repository for [Data Swarms: Optimizable Generation of Synthetic Evaluation Data](https://bunsenfeng.github.io/).

## Quick Start

#### Initialization

Create a conda environment for Data Swarms.
```
conda env create -f d_swarm.yml
conda activate d_swarm
```

Log into huggingface (for model access).
```
huggingface-cli login
```

Download initial experts.
```
cd initial_experts
python expert_init.py
cd ..
```

Setup Gemini access by providing your Google Cloud project ID at line 20 in `utils.py`.

#### Execute your first Data Swarms search.

Let's run Data Swarms on the Alpaca instruction following task and the difficult objective. `data_swarm.sh` provides the starter script for this.

Before running, how many GPUs do you have (and what are the GPU ids?). Change `-g` in line 23 of `data_swarm.sh`: by default five GPUs with ids `0,1,2,3`, but you can change to `0`, `0,1`, `0,2,4,5` or any combination you'd like. It is recommended to have ~40GB memory per GPU and the code will auto double batch size for ~80GB GPUs.

Run it!
```
bash data_swarm.sh
```

You might be prompted to log into WandB and it is highly recommended. There will be a directory in `search/` that starts with `alpaca...`, all the logs, models, and results will be stored there. You can check out `search/alpaca_.../log.txt` to see current progress. Check out the logs on the wandb website too. Yes, it might be slow.

#### Other Objectives

Choose one `-t` from `alpaca`, `gsm8k`, `nlgraph`, `truthfulqa`, `wikidyk`; Choose one `-o` from `difficult`, `separate`, `novel`, `stable`, and additionally the pair `(alpaca, realistic)`.

#### Dual Swarms

```
bash dual_swarm.sh
```

Please keep `-o` as `difficult` for the adversarial setting.

## Changing Hyperparameters and Settings

Do `python data_swarm.py -h` to see a list of all possible hyperparameters and settings. Additionally look at the comments for hyperparameters in `data_swarm.py`. We already included the default settings in the four `data_swarm.sh` starter scripts, but feel free to play around different settings.

## Citation

If Data Swarms is helpful to you:

```
@article{feng2025data,
  title={Data Swarms: Optimizable Generation of Synthetic Evaluation Data},
  author={Feng, Shangbin and Wang, Yike and Shi, Weijia and Tsvetkov, Yulia},
  journal={arXiv preprint arXiv:2506.00741},
  year={2025}
}
```
