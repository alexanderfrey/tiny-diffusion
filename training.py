"""
Training script for character-level discrete diffusion model
"""

import math

import torch
import torch.nn.functional as F
from tqdm import tqdm
from model import (
    DiffusionTransformer,
    DiffusionConfig,
    encode_text,
    decode_tokens,
)
from sample import get_random_context
from tokenizer import get_tokenizer, tokenizer_vocab_size


class MaskedDiffusionSchedule:
    """
    Masked diffusion schedule for discrete diffusion.
    At each timestep, we have a probability of masking a token with [MASK].
    """

    def __init__(self, num_timesteps, mask_token_id, context_len=0):
        self.num_timesteps = num_timesteps
        self.mask_token_id = mask_token_id
        self.context_len = context_len

        # Use a cosine schedule so mid-range steps get more weight than the ends
        steps = torch.arange(1, num_timesteps + 1, dtype=torch.float32)
        normalized = steps / num_timesteps
        mask_probs = torch.sin(normalized * math.pi / 2).pow(2)
        self.mask_probs = mask_probs.clamp(min=1.0 / (2 * num_timesteps), max=1.0)

        # Precompute timestep sampling weights (Beta(2,2)-like) to emphasize mid-range
        centers = (torch.arange(num_timesteps, dtype=torch.float32) + 0.5) / num_timesteps
        weights = torch.sin(math.pi * centers).pow(2)
        weights = weights + 1e-3  # avoid zero probability at the boundaries
        self.sample_weights = weights / weights.sum()

    def to(self, device):
        self.mask_probs = self.mask_probs.to(device)
        self.sample_weights = self.sample_weights.to(device)
        return self

    def sample_timesteps(self, batch_size, device):
        """Draw diffusion steps with preference for the informative mid-range."""
        if self.sample_weights.device != device:
            self.sample_weights = self.sample_weights.to(device)
        return torch.multinomial(
            self.sample_weights, num_samples=batch_size, replacement=True
        )

    def add_masks(self, x_0, t):
        """
        Add masks to tokens x_0 at timestep
        Args:
            x_0: Clean tokens, shape (B, T)
            t: Timestep indices, shape (B,)
        Returns:
            x_t: Masked tokens at timestep t
        """
        B, T = x_0.shape
        device = x_0.device

        # Keep schedule tensor on the same device as data
        if self.mask_probs.device != device:
            self.mask_probs = self.mask_probs.to(device)

        # Get masking probability for each sample
        mask_prob = self.mask_probs[t]  # (B,)

        # Create mask: which tokens to replace with [MASK]
        mask = torch.rand(B, T, device=device) < mask_prob.unsqueeze(1)  # (B, T)

        # Never mask the first context_len tokens
        protected = min(self.context_len, T)
        if protected > 0:
            mask[:, :protected] = False

        # Ensure each sequence has at least one masked (non-context) position
        valid_start = protected
        if valid_start >= T:
            raise ValueError(
                "context_len must be smaller than sequence_len to allow masking tokens."
            )
        no_mask_rows = mask[:, valid_start:].sum(dim=1) == 0
        if no_mask_rows.any():
            rows = torch.nonzero(no_mask_rows, as_tuple=False).squeeze(1)
            rand_positions = torch.randint(
                valid_start, T, (rows.size(0),), device=device
            )
            mask[rows, rand_positions] = True

        # Replace masked positions with mask token
        mask_tokens = torch.full_like(x_0, self.mask_token_id)
        x_t = torch.where(mask, mask_tokens, x_0)

        return x_t

    def get_mask_prob(self, t):
        """Get the masking probability for timestep t"""
        return self.mask_probs[t].item()


def get_data_loader(data_path, batch_size, seq_len, device, tokenizer):
    """
    Simple data loader for text data
    Args:
        data_path: Path to text file
        batch_size: Batch size
        seq_len: Sequence length
        device: Device to load data on
    """
    # Read the text file
    with open(data_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Convert to tokens
    tokens = encode_text(text, tokenizer)

    # Determine max starting index for sampling
    max_start = len(tokens) - seq_len
    if max_start <= 0:
        raise ValueError("Dataset is too small for the requested sequence length.")
    arange_seq = torch.arange(seq_len, dtype=torch.long)

    # Generator function
    def data_generator():
        while True:
            start_indices = torch.randint(
                0, max_start + 1, (batch_size,), dtype=torch.long
            )
            positions = start_indices.unsqueeze(1) + arange_seq
            batch = tokens[positions].to(device)
            yield batch

    return data_generator()


def train_step(model, x_0, mask_schedule, optimizer):
    """
    Single training step
    Args:
        model: DiffusionTransformer model
        x_0: Clean tokens, shape (B, T)
        mask_schedule: Mask schedule object
        optimizer: Optimizer
    Returns:
        loss: Training loss
    """
    B, _ = x_0.shape
    device = x_0.device

    # Sample timesteps with mid-range emphasis
    t = mask_schedule.sample_timesteps(B, device=device)

    # Add mask to get x_t
    x_t = mask_schedule.add_masks(x_0, t)

    # Forward pass: predict the original tokens
    logits = model(x_t, t)  # (B, T, vocab_size)

    # Compute loss only on masked positions
    mask = x_t == mask_schedule.mask_token_id  # (B, T)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), x_0.view(-1), reduction="none"
    )
    masked_loss = loss.view(B, -1) * mask.float()
    denom = mask.sum().clamp_min(1)
    loss = masked_loss.sum() / denom  # Average over masked positions only

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return loss.item()


def train(
    model,
    data_loader,
    mask_schedule,
    optimizer,
    num_steps=10000,
    sample_interval=500,
    dataset_tokens=None,
):
    """
    Main training loop
    """
    model.train()

    pbar = tqdm(range(num_steps), desc="Training")
    for step in pbar:
        # Get batch
        x_0 = next(data_loader)

        # Training step
        loss = train_step(model, x_0, mask_schedule, optimizer)

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        # Sample generation
        if (step + 1) % sample_interval == 0:
            model.eval()
            with torch.no_grad():
                # Get random context if context_len > 0
                context_tokens = None
                if model.config.context_len > 0 and dataset_tokens is not None:
                    context_tokens = get_random_context(
                        dataset_tokens, model.config.context_len, batch_size=1
                    )

                samples = model.sample(
                    batch_size=1,
                    seq_len=model.config.sequence_len,
                    num_steps=None,
                    temperature=1.0,
                    device=model.get_device(),
                    context_tokens=context_tokens,
                    method="confidence",
                    confidence_threshold=0.95,
                )
                # Decode samples to text
                text = decode_tokens(samples[0])
                tqdm.write(f"\n--- Sample at step {step + 1} ---")
                tqdm.write(text)
                tqdm.write("--- End sample ---\n")
            model.train()


def main():
    # Hyperparameters
    batch_size = 64
    max_iters = 20000
    eval_interval = 500
    learning_rate = 3e-4

    data_path = "data/tiny_shakespeare.txt"
    tokenizer = get_tokenizer()
    vocab_size = tokenizer_vocab_size(tokenizer)
    config = DiffusionConfig(
        vocab_size=vocab_size, mask_token_id=tokenizer.mask_token_id
    )
    print(f"Sequence_len: {config.sequence_len}")
    print(f"Diffusion_steps: {config.diffusion_steps}")
    print(f"Context_len: {config.context_len}")

    # Device
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print(f"Using device: {device}")

    # Model
    model = DiffusionTransformer(config).to(device)
    model.init_weights()

    num_params = sum(p.numel() for p in model.parameters())
    print(f"Number of parameters: {num_params:,}")

    # Masked diffusion schedule
    mask_schedule = MaskedDiffusionSchedule(
        num_timesteps=config.diffusion_steps,
        mask_token_id=config.mask_token_id,
        context_len=config.context_len,
    ).to(device)

    # Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=0.01
    )

    # Data loader
    data_loader = get_data_loader(
        data_path=data_path,
        batch_size=batch_size,
        seq_len=config.sequence_len,
        device=device,
        tokenizer=tokenizer,
    )

    # Load dataset tokens for context sampling
    dataset_tokens = None
    if config.context_len > 0:
        with open(data_path, "r", encoding="utf-8") as f:
            text = f.read()
        dataset_tokens = encode_text(text, tokenizer)

    # Train
    print("Starting training...\n")
    train(
        model=model,
        data_loader=data_loader,
        mask_schedule=mask_schedule,
        optimizer=optimizer,
        num_steps=max_iters,
        sample_interval=eval_interval,
        dataset_tokens=dataset_tokens,
    )

    # Save model
    import os

    os.makedirs("weights", exist_ok=True)
    torch.save(model.state_dict(), "weights/diffusion_model.pt")
    print("Model saved to weights/diffusion_model.pt")


if __name__ == "__main__":
    main()
