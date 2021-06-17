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

## Others

- WILDs data can be downloaded from https://wilds.stanford.edu/

- WILDs pretrained OOD detector model [wilds](https://drive.google.com/drive/folders/1t16my17GlCP8dDMjKcOMgnNCyGmZErH1)

- We reuse part of the code from [MCP](https://github.com/actionabletest/MCP)
