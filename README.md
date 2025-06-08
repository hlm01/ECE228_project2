# Graph Attention Networks for Physicochemical Property Prediction
## ECE228, Group 19, Track #1
## Henry Lin, Omkar Sathe, Stephen Wilcox

In our project we build upon the [model](https://github.com/OpenDrugAI/AttentiveFP) [Attentive FP](https://sci-hub.st/10.1021/acs.jmedchem.9b00959)---specifically we build upon [the implementation by pytorch](https://github.com/pyg-team/pytorch_geometric/blob/master/examples/attentive_fp.py) as the original has compatability issues with newer systems.

The code requires torch torchvision torchaudio torch-scatter torch-sparse torch-geometric rdkit pubchempy tqdm optuna

To install the necessary dependencies, run:

```bash
pip install torch torchvision torchaudio torch-scatter torch-sparse torch-geometric rdkit pubchempy tqdm optuna
```

You can run the code in main.ipynb