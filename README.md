# Domain Robust Text Representation

The Implementation for

Li, Yitong, Timothy Baldwin and Trevor Cohn (2018) What's in a Domain? Learning Domain-Robust Text Representations Using Adversarial Training , In Proceedings of the 2018 Conference of the North American Chapter of the Association for Computational Linguistics – Human Language Technologies (NAACL HLT 2018), New Orleans, USA.

## Data

1. Multi-Domain Sentiment Dataset (Blitzer et al., 2007);
2. Language identification data (Lui and Baldwin (2011)).

## Requirements

- python 2.7
- Tensorflow 1.1+
- numpy
- scipy

## Models

1. Baseline: ood_BDEK_adv_baseline.py
2. Cond: ood_BDEK_adv_cond.py
3. Gen: ood_BDEK_adv_gen.py

as described in the paper.

## running the code

### Multi-domain sentiment task
```bash
python ood_BDEK_adv_*.py [parameters]
```

### main parameters:
```text
  --num_train_epochs: normal training epochs w\o adversarial supervision;
  --num_tune_epochs: adversarial learning epochs;
  --adv_lambda: hyper-parameter lambda for adversarial loss.
```

### examples:
Baseline w\o adversarial:
```bash
python ood_BDEK_adv_baseline.py --num_train_epochs 50
```
Baseline w adversarial loss:
```bash
python ood_BDEK_adv_baseline.py --num_tune_epochs 50 --adv_lambda 1e-3
```
Gen model with pre-train:
```bash
python ood_BDEK_adv_gen.py --num_train_epochs 50 --num_tune_epochs 50
```
