# tiny-diffusion

A character-level language diffusion model for text generation. The model is a modified version of the [nanochat gpt](https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
) implementation and is trained on Tiny Shakespeare! It is only 10.7 million parameters, so you can try it out locally!

![Demo](animations/animation.gif)



## Installation

```bash
# Clone the repository
git clone <repository-url>
cd tiny-diffusion

# Install dependencies (Python 3.10+)
uv sync
```



## Quick Start

The file `training.py` puts the weights in `weights/diffusion_model.pt`. The sample and animation files load the model from this file.

### Train Your Own Model
Currently, the weights are already provided for you! It took me around half an hour to train this model for 20,000 steps on 4xA100s. But if you want to retrain the model again, run:

```bash
# Train from scratch on Shakespeare
uv run training.py

# Training will save checkpoints to weights/diffusion_model.pt
```

### Generate Text
To generate a continuous stream of output (currently 30 context lengths), run:

```bash
# Generate samples using the pre-trained model
uv run sample.py
```

### Visualize the Diffusion Process
To see the diffusion process as a nice animation, run:

```bash
# Watch the denoising process step-by-step
uv run animations/diffusion-process.py

# See Game of Life-inspired sampling (fun little experiment)
uv run animations/game-of-life.py
```

### Track Training with Weights & Biases
Enable experiment tracking by exporting `USE_WANDB=1` (and optionally `WANDB_PROJECT`/`WANDB_RUN_NAME`) before running `uv run training.py`. Router histograms, losses, learning rate, and periodic samples will be logged automatically.


## Default Config

- **Parameters**: 10.7 million
- **Layers**: 6
- **Attention Heads**: 6
- **Embedding Dim**: 384
- **Sequence Length**: 256 characters
- **Diffusion Steps**: 128

## MoE Architecture Overview

The optional Mixture-of-Experts path replaces the dense feed-forward with routers that dispatch tokens to a small pool of experts. With 16 layers and 3 experts, each block looks like the diagram below:

```
            Input Tokens
                  |
      +-------------------------+
      | Token + Time Embedding  |
      +-------------------------+
                  |
  +-----------------------------------------------------+
  |        16 x Diffusion Transformer Blocks            |
  |                                                     |
  |  +--------------+    +----------------------------+ |
  |  | Attention +  |--->| Router (top-k)             | |
  |  | RMSNorm      |    | + 3 parallel experts (E0..)| |
  |  +--------------+    +------------+---------------+ |
  |                               | | |                |
  |             Weighted sum / residual merge          |
  +-----------------------------------------------------+
                  |
          Output logits
```

Routers emit gate histograms during training, making it easy to track how balanced the three experts are layer-by-layer.



## File Structure

```
tiny-diffusion/
├── model.py                    # Core diffusion transformer
├── training.py                 # Training script
├── sample.py                   # Text generation
├── data/
│   └── tiny_shakespeare.txt    # Training data
├── weights/
│   └── diffusion_model.pt      # Pre-trained weights
└── animations/
    ├── diffusion-process.py    # Denoising visualization
    └── game-of-life.py         # Game of Life sampling
```



## License
MIT
