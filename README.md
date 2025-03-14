# Disentangled World Models: Learning to Transfer Semantic Knowledge from Distracting Videos for Reinforcement Learning

## Getting Strated
DisWM is implemented and tested on Ubuntu 20.04 with python == 3.7, PyTorch == 1.13.1:

1) Create an environment
```bash
conda create -n diswm python=3.7
conda activate diswm
```
2) Install dependencies
```bash
pip install -r requirements.txt
```

3) Install Distracting Control Suite and dmc2gym. The `distracting_control` folder contains the [Distracting Control Suite](https://github.com/sahandrez/distracting_control) code with modification to create disjoint colour sets. The `dmc2gym` folder contains the [dmc2gym](https://github.com/denisyarats/dmc2gym) code revised to use the distracting_control wrappers.

4) Collect distracting video datasets with DreamerV2 on DMC/MuJoCo Pusher. 

## Train DisWM on DMC / MuJoCo Pusher
1. Pretrain the video prediction model with collected videos on DMC:  
```bash
python dreamer.py --configs defaults dmc2gym \
    --device 'cuda:0' --task dmc2gym_reacher_easy \
    --logdir $log_directory \
    --seed 0 --beta_vae_pretrain True --beta_vae True \
    --pretrain_action_num $source_task_action_number \
    --pretrain_datasets_path $dataset_directory/train_eps
```
Put pretrained checkpoints into `checkpoints` folder. 

2. (a) Finetune the disentangled world model on DMC:  
```bash
python dreamer.py --device 'cuda:0' --task gymnasium_Pusher-v5 \
    --logdir $log_directory \
    --pretrain_checkpoint_path ./checkpoints \
    --seed 0 --configs defaults dmc2gym --traverse True --beta_vae True \
    --method_name 'diswm' --cross_domain True \
    --distillation True --color_distractor True
```

2. (b) Finetune the disentangled world model on MuJoCo Pusher:  
```bash
python dreamer.py --device 'cuda:0' --task gymnasium_Pusher-v5 \
    --logdir $log_directory \
    --pretrain_checkpoint_path ./checkpoints \
    --seed 0 --configs defaults pusher --traverse True --beta_vae True \
    --method_name 'diswm' --cross_domain True \
    --distillation True --color_distractor True
```

## Acknowledgement
The codes refer to the implemention of [dreamer-torch](https://github.com/jsikyoon/dreamer-torch). Thanks for the authors！



