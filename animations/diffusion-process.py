"""
Visualize the diffusion sampling process step by step
Shows each denoising step with masks and intermediate results
Supports both single-shot and continuous block generation
"""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from model import (
    DiffusionTransformer,
    DiffusionConfig,
    encode_text,
    decode_tokens,
)
from tokenizer import get_tokenizer, tokenizer_vocab_size


def load_model(checkpoint_path, device, tokenizer):
    """Load a trained model from checkpoint"""
    vocab_size = tokenizer_vocab_size(tokenizer)
    config = DiffusionConfig(
        vocab_size=vocab_size, mask_token_id=tokenizer.mask_token_id
    )
    model = DiffusionTransformer(config).to(device)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model.eval()
    return model


def load_initial_context(data_path, context_len, tokenizer):
    """Load the first context_len characters from dataset"""
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()[:context_len]
    tokens = encode_text(text, tokenizer)
    return tokens


def generate_with_visualization(
    model,
    num_blocks=5,
    temperature=1.0,
    dataset_tokens=None,
    confidence_threshold=0.95,
):
    """
    Generate samples and visualize each denoising step

    Args:
        model: The trained diffusion model
        num_blocks: Number of blocks to generate
        temperature: Sampling temperature
        dataset_tokens: Dataset tokens for initial context (if context_len > 0)
        confidence_threshold: Confidence threshold for decoding
    """
    device = model.get_device()
    seq_len = model.config.sequence_len
    context_len = model.config.context_len

    print(f"Generating {num_blocks} blocks using confidence-aware decoding...")
    print(f"Context length: {context_len}")
    print()

    all_frames = []
    all_masks = []
    all_block_indices = []
    completed_blocks_text = []

    prev_context = None

    for block_idx in range(num_blocks):
        print(f"Pre-calculating block {block_idx + 1}/{num_blocks}...")

        context_tokens = None
        if context_len > 0:
            if block_idx == 0 and dataset_tokens is not None:
                context_tokens = dataset_tokens[:context_len].unsqueeze(0).to(device)
            elif block_idx > 0 and prev_context is not None:
                context_tokens = prev_context.unsqueeze(0)

        x = torch.full(
            (1, seq_len), model.config.mask_token_id, dtype=torch.long, device=device
        )

        if context_tokens is not None:
            x[:, :context_len] = context_tokens

        mask = torch.zeros(seq_len, dtype=torch.bool)
        mask[context_len:] = True
        all_frames.append(x[0].cpu().clone())
        all_masks.append(mask.cpu())
        all_block_indices.append(block_idx)

        masked_positions = torch.ones(1, seq_len, dtype=torch.bool, device=device)
        if context_tokens is not None:
            masked_positions[:, :context_len] = False

        with torch.no_grad():
            step = 0
            while masked_positions.any():
                t_batch = torch.full((1,), step, device=device, dtype=torch.long)
                t_batch = torch.clamp(t_batch, 0, model.config.diffusion_steps - 1)

                logits = model.forward(x, t_batch)
                probs = F.softmax(logits / temperature, dim=-1)
                confidences, predicted_tokens = torch.max(probs, dim=-1)

                above_threshold = (
                    confidences >= confidence_threshold
                ) & masked_positions

                if not above_threshold.any():
                    masked_confidences = confidences.clone()
                    masked_confidences[~masked_positions] = -float("inf")
                    best_idx = torch.argmax(masked_confidences[0])
                    above_threshold[0, best_idx] = True

                x = torch.where(above_threshold, predicted_tokens, x)
                masked_positions = masked_positions & ~above_threshold

                all_frames.append(x[0].cpu().clone())
                all_masks.append(masked_positions[0].cpu())
                all_block_indices.append(block_idx)
                step += 1

        if context_len > 0:
            prev_context = x[0, -context_len:]

        chars_per_row = 64
        num_rows = (seq_len + chars_per_row - 1) // chars_per_row
        block_lines = []
        for row_idx in range(num_rows):
            row_start = row_idx * chars_per_row
            row_end = min(row_start + chars_per_row, seq_len)
            row_text = ""
            for idx in range(row_start, row_end):
                char = decode_tokens([x[0, idx].item()])
                if char == "\n":
                    char = "↵"
                row_text += char
            block_lines.append(row_text)
        completed_blocks_text.append(block_lines)

    print("Done! Now showing animation...\n")

    # Setup matplotlib
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # Calculate number of rows based on sequence length
    chars_per_row = 64
    num_rows = (seq_len + chars_per_row - 1) // chars_per_row

    # Create text object
    # Use left alignment but position it to center the 64-char block
    text_obj = ax.text(
        0.5,
        0.5,
        "",
        ha="center",
        va="center",
        fontsize=8,
        family="monospace",
        fontweight="normal",
        linespacing=1.2,
        multialignment="left",
    )

    title = fig.suptitle("", fontsize=12)

    def init():
        """Initialize animation"""
        return [text_obj, title]

    def update(frame_idx):
        """Update function for animation"""
        if frame_idx >= len(all_frames):
            return [text_obj, title]

        frame_tokens = all_frames[frame_idx]
        mask = all_masks[frame_idx]
        block_idx = all_block_indices[frame_idx]

        max_visible_blocks = 6

        start_block = max(0, block_idx - max_visible_blocks + 1)

        # Collect all text as one continuous stream
        continuous_text = ""

        # Add completed blocks
        for prev_block_idx in range(start_block, block_idx):
            block_text_lines = completed_blocks_text[prev_block_idx]
            block_text = "".join(block_text_lines)

            if prev_block_idx == start_block:
                # First visible block - show everything
                continuous_text += block_text
            else:
                # Skip context_len characters
                continuous_text += block_text[context_len:]

        # Add current block being generated
        start_offset = 0 if block_idx == start_block else context_len
        for idx in range(start_offset, seq_len):
            if mask[idx]:
                continuous_text += "█"
            else:
                char = decode_tokens([frame_tokens[idx]])
                if char == "\n":
                    char = "↵"
                continuous_text += char

        # Wrap at 64 characters per line
        all_lines = []
        for i in range(0, len(continuous_text), chars_per_row):
            all_lines.append(continuous_text[i : i + chars_per_row])

        text_obj.set_text("\n".join(all_lines))
        text_obj.set_color("black")

        num_masked = mask.sum().item()
        title_text = (
            f"Block {block_idx + 1}/{num_blocks} - Remaining: {num_masked} tokens"
        )

        title.set_text(title_text)

        return [text_obj, title]

    # Create animation
    anim = FuncAnimation(
        fig,
        update,
        init_func=init,
        frames=len(all_frames),
        interval=10,
        blit=False,
        repeat=True,
    )

    plt.tight_layout()
    plt.show()


def main():
    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}\n")

    tokenizer = get_tokenizer()

    # Load model
    checkpoint_path = "weights/diffusion_model.pt"
    print(f"Loading model from {checkpoint_path}...")
    model = load_model(checkpoint_path, device, tokenizer)
    print("Model loaded!\n")

    # Load dataset tokens for initial context if context_len > 0
    dataset_tokens = None
    if model.config.context_len > 0:
        print("Loading initial context from dataset...")
        dataset_tokens = load_initial_context(
            "data/tiny_shakespeare.txt", model.config.context_len, tokenizer
        )
        print(f"Loaded {len(dataset_tokens)} tokens\n")

    # Generate with visualization
    generate_with_visualization(
        model,
        num_blocks=6,
        temperature=1.0,
        dataset_tokens=dataset_tokens,
        confidence_threshold=0.9,
    )


if __name__ == "__main__":
    main()
