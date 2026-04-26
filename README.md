# Machine Unlearning under Retain-Forget Entanglement

Reference implementation for the ICLR 2026 paper
[*Machine Unlearning under Retain-Forget Entanglement*](https://arxiv.org/abs/2603.26569)


## Setup

```bash
conda create -n unlearning python=3.10
conda activate unlearning
pip install -r requirements.txt
```

## Start

```bash
TINY_ROOT=/path/to/tiny-imagenet-200 \
CELEBA_ROOT=/path/to/celeba_root \
bash scripts/reproduce_all.sh
```




## Citation

```bibtex
@inproceedings{cheng2026unlearning,
  title={Machine Unlearning under Retain-Forget Entanglement},
  author={Cheng, Jingpu and Liu, Ping and Li, Qianxiao and Zhang, Chi},
  booktitle={International Conference on Learning Representations (ICLR)},
  year={2026},
}
```
