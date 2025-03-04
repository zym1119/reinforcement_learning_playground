# reinforcement_learning_playground
My playground for learning RL

# Environment
https://nio.feishu.cn/docx/LAG0dvuzVo4TBzxBDGic6Rmsnxf

# Usage

train:
```bash
python train.py --model {model_type} --lr {lr} --episodes {num of episodes}
```

test:
```bash
python test.py --model {model_type} -c {ckpt path}
```

# Implement your own method

1. Define you model in `models` folder
2. Implement your own Trainer and Inferer based on BaseTrainer or BaseInferer