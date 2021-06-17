# DAT
Source code of Distribution aware test selection metric

## Requirements
    - Tensorflow 2.3
    - Keras 2.4.3

## project structure
```
├── models                        # pretrained OOD detector
├── art_attack.py                 # adv examples generation
├── data_preparation.py           # img transformation
├── DAT_metric.py                 # DAT metrics
├── transform_retrain.py          # baseline metrics and MNIST, Fashion-MNIST, CIFAR-10 retraining
├── wilds_retrain.py              # WILDs retraining
```

We reuse part of the code from [MCP](https://github.com/actionabletest/MCP)
