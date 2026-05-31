# TinyWorlds Autoresearch

Minimal Karpathy-style autoresearch harness for action-conditioned visual world models.

## Files

- `program.md` - human-written research instructions for the agent.
- `setup.py` - fixed infrastructure: MiniGrid dataset generation, dataloading, and architecture-agnostic eval.
- `train.py` - single editable genome: model, optimizer, hyperparameters, and training loop.
- `requirements.txt` - Python dependencies.

## Setup

```bash
python3 -m pip install -r requirements.txt
python3 setup.py
```

The default dataset is generated locally at `data/minigrid_empty8x8_actions.h5`.

## Train

```bash
python3 train.py
```

Each run trains for a fixed 300-second budget by default. For smoke tests:

```bash
TW_TIME_BUDGET=2 python3 train.py
```

`DEPTH` in `train.py` is the main scale knob. It derives patch size, context length, width, heads, batch size, and learning rate.

