# ⚡ RL2Grid: Benchmarking RL for Power Grid Operations

**RL2Grid** is a **realistic and standardized reinforcement learning benchmark** for power grid operations, developed in close collaboration with major Transmission System Operators (TSOs). It builds upon [Grid2Op](https://github.com/rte-france/Grid2Op) and extends the widely-used [CleanRL](https://github.com/vwxyzjn/cleanrl) framework to provide:

- ✅ Standardized **environments**, **state/action spaces**, and **reward structures**  
- ♻️ Realistic **transition dynamics** incorporating stochastic grid events and human heuristics  
- ⚠️ **Safe RL tasks** via constrained MDPs, with load shedding and thermal overload constraints  
- 🧪 Extensive **baselines** including DQN, PPO, SAC, TD3, and Lagrangian PPO  
- 📊 Integration with [Weights & Biases (wandb)](https://wandb.ai/home) for experiment tracking  
- 🧠 Designed to foster **algorithmic innovation and safe control** in power grids

---

## 🔧 Installation

First, ensure you have [Miniconda](https://docs.anaconda.com/free/miniconda/) installed.

```bash
# Step 1: Clone the repository
git clone https://github.com/emarche/RL2Grid.git
cd RL2Grid

# Step 2: Create the environment
conda env create -f conda_env.yml

# Step 3: Activate the environment
conda activate rl2grid

# Step 4: Install RL2Grid
pip install .
```

---

## 🚀 Quick Start

To run training on a predefined task (remember to set up the correct entity and project for wandb in the `main.py` script):

```bash
python main.py --task bus14 --algo ppo --track --wandb_project rl2grid-benchmarks
```

Available arguments include task difficulty, action type (topology/redispatch), reward weights, constraint types, and more. Check `main.py` and `alg/<algorithm>/config.py` for the full configuration space.

---

## 🧪 Benchmark Environments

RL2Grid supports **39 distinct tasks** across discrete (topological) and continuous (redispatch/curtailment) settings. The main grid variations include:

| Grid ID          | Action Type         | Contingencies            | Batteries | Constraints | Difficulty Levels |
|------------------|---------------------|---------------------------|-----------|-------------|-------------------|
| bus14            | Topology, Redispatch | Maintenance               | No        | Optional    | 0-1                 |
| bus36-MO-v0      | Topology, Redispatch | Maintenance + Opponent    | No        | Optional    | 0–4               |
| bus118-MOB-v0    | Topology, Redispatch | Maintenance + Opponent + B | Yes       | Optional    | 0–4               |

Full environment specs and task variants are detailed in the paper.

---

## 🧠 Built-In Heuristics

To bridge human expertise with RL training, RL2Grid embeds **two human-informed heuristics**:

- `idle`: suppresses agent actions during safe periods
- `recovery`: gradually restores topology toward safe defaults when the grid is stabilized

Heuristic guidance can be toggled via command-line arguments (see `env/config.py`).

---

## ✅ Safe RL Support

RL2Grid natively supports **CMDP-style safety constraints**, including:

- **Load Shedding & Islanding (LSI)** – penalizes disconnected grid regions or unmet demand
- **Thermal Line Overloads (TLO)** – penalizes line overloads and disconnections

These constraints can be incorporated using Lagrangian methods (e.g., LagrPPO).

---

## 📈 Baseline Results

RL2Grid includes implementations and benchmark results for:

- **Discrete (topological)**: DQN, PPO, SAC (+ heuristic variants)
- **Continuous (redispatch)**: PPO, SAC, TD3
- **Constrained**: Lagrangian PPO (LSI, TLO tasks)

Performance is measured via normalized **grid survival rate**, overload penalties, topology modifications, and cost metrics.

---

## 📚 Documentation

- [📄 Paper](https://scholar.google.com/citations?view_op=view_citation&hl=en&user=9V1_SGkAAAAJ&sortby=pubdate&citation_for_view=9V1_SGkAAAAJ:4DMP91E08xMC)  
- [🧠 Grid2Op documentation](https://grid2op.readthedocs.io/)  
- [📊 ChroniX2Grid time-series generator](https://github.com/BDonnot/ChroniX2Grid)  

---

## 🌍 Environmental Impact

We are committed to responsible research. Experiments were run with carbon offsets purchased via [Treedom](https://www.treedom.net) and estimated via [MLCO2](https://mlco2.github.io/impact).

---

## 📬 Citation

If you use RL2Grid, please cite:

```bibtex
@misc{rl2grid,
      title={RL2Grid: Benchmarking Reinforcement Learning in Power Grid Operations}, 
      author={Enrico Marchesini and Benjamin Donnot and Constance Crozier and Ian Dytham and Christian Merz and Lars Schewe and Nico Westerbeck and Cathy Wu and Antoine Marot and Priya L. Donti},
      year={2025},
      eprint={2503.23101},
      archivePrefix={arXiv},
      primaryClass={cs.LG},
      url={https://arxiv.org/abs/2503.23101}, 
}
```

---

## 🤝 Acknowledgments

This project was developed in collaboration with:

- RTE France  
- 50Hertz  
- National Grid ESO  
- MIT, Georgia Tech, University of Edinburgh  
