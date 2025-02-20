# Continuous Diffusion for Mixed-Type Tabular Data

![Banner](https://github.com/muellermarkus/cdtd_simple/blob/main/images/cdtd_overview.png)


This is an easy-to-use wrapper around the training and sampling procedures of the CDTD model.

Please see `main.ipynb` for an illustration of the CDTD on the adult dataset.

Paper: https://arxiv.org/abs/2312.10431 (published in ICLR 2025)

Full Replication Code: https://github.com/muellermarkus/cdtd

## Install Instructions

Initialize virtual environment in Python 3.10, e.g., `python3.10 -m venv .venv` on Linux.

Activate environment via `source .venv/bin/activate` (on Linux).

Install packages via `pip install -r requirements.txt`.


## Citation

```
@inproceedings{mueller2025cdtd,
  title = {Continuous {{Diffusion}} for {{Mixed-Type Tabular Data}}},
  author = {Mueller, Markus and Gruber, Kathrin and Fok, Dennis},
  booktitle = {The Thirteenth International Conference on Learning Representations},
  year = {2025},
  url = {https://openreview.net/pdf?id=QPtoBPn4lZ}
}
```