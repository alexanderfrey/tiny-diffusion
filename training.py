"""
Training script for character-level discrete diffusion model
"""

import math
import os

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from tqdm import tqdm

try:
    import wandb
except ImportError:  # pragma: no cover - fallback if dependency missing
    wandb = None


def _wandb_enabled():
    flag = os.environ.get("USE_WANDB", "")
    return flag.lower() in ("1", "true", "yes")

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


@torch.no_grad()
def update_ema(model, ema_model, decay):
    """Exponential moving average of model parameters."""
    for ema_param, param in zip(ema_model.parameters(), model.parameters()):
        ema_param.data.mul_(decay).add_(param.data, alpha=1.0 - decay)


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


def train_step(model, x_0, mask_schedule, optimizer, grad_clip=None):
    """
    Single training step
    Args:
        model: DiffusionTransformer model
        x_0: Clean tokens, shape (B, T)
        mask_schedule: Mask schedule object
        optimizer: Optimizer
        grad_clip: Optional gradient clipping value
    Returns:
        loss: Training loss
        router_stats: List of per-layer MoE histograms or None
        aux_loss_value: Scalar MoE aux loss (float) or None
    """
    B, _ = x_0.shape
    device = x_0.device

    # Sample timesteps with mid-range emphasis
    t = mask_schedule.sample_timesteps(B, device=device)

    # Add mask to get x_t
    x_t = mask_schedule.add_masks(x_0, t)

    # Forward pass: predict the original tokens
    logits, aux_loss, router_stats = model(
        x_t, t, return_aux=True, return_router_stats=True
    )  # (B, T, vocab_size)

    # Compute loss only on masked positions
    mask = x_t == mask_schedule.mask_token_id  # (B, T)
    loss = F.cross_entropy(
        logits.view(-1, logits.size(-1)), x_0.view(-1), reduction="none"
    )
    masked_loss = loss.view(B, -1) * mask.float()
    tokens_per_sample = mask.view(B, -1).sum(dim=1).clamp_min(1)
    per_example = masked_loss.sum(dim=1) / tokens_per_sample.float()

    if mask_schedule.mask_probs.device != device:
        mask_schedule.mask_probs = mask_schedule.mask_probs.to(device)
    timestep_probs = mask_schedule.mask_probs[t].clamp(min=1e-3)
    weights = (1.0 / timestep_probs).detach()
    weights = weights / weights.mean()
    loss = (per_example * weights).mean()
    if aux_loss is not None and model.config.moe_aux_loss_weight > 0:
        loss = loss + model.config.moe_aux_loss_weight * aux_loss

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    if grad_clip is not None:
        clip_grad_norm_(model.parameters(), grad_clip)
    optimizer.step()

    aux_loss_value = aux_loss.item() if aux_loss is not None else None
    return loss.item(), router_stats, aux_loss_value


def log_router_histograms(router_stats, step=None, wandb_run=None):
    if router_stats is None:
        return
    lines = []
    wandb_log = {}
    for layer_idx, hist in enumerate(router_stats):
        if hist is None:
            continue
        counts = hist.detach().to("cpu")
        counts_list = [int(x) for x in counts.tolist()]
        total = sum(counts_list)
        if total > 0:
            frac_list = [c / total for c in counts_list]
        else:
            frac_list = [0.0 for _ in counts_list]
        count_str = ", ".join(str(c) for c in counts_list)
        frac_str = ", ".join(f"{f:.2f}" for f in frac_list)
        lines.append(f"  Layer {layer_idx:02d}: counts [{count_str}] | frac [{frac_str}]")
        if wandb_run is not None:
            for expert_idx, (count_val, frac_val) in enumerate(zip(counts_list, frac_list)):
                wandb_log[
                    f"router/layer_{layer_idx}/expert_{expert_idx}_count"
                ] = count_val
                wandb_log[
                    f"router/layer_{layer_idx}/expert_{expert_idx}_frac"
                ] = frac_val
    if not lines:
        return
    header = f"Router histogram @ step {step}" if step is not None else "Router histogram"
    tqdm.write("\n".join([header] + lines))
    if wandb_run is not None and wandb_log:
        wandb_run.log(wandb_log, step=step)


def train(
    model,
    data_loader,
    mask_schedule,
    optimizer,
    num_steps=10000,
    sample_interval=500,
    dataset_tokens=None,
    scheduler=None,
    ema_model=None,
    ema_decay=0.999,
    grad_clip=None,
    router_log_interval=500,
    wandb_run=None,
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
        loss, router_stats, aux_loss_value = train_step(
            model, x_0, mask_schedule, optimizer, grad_clip=grad_clip
        )

        if scheduler is not None:
            scheduler.step()
        if ema_model is not None:
            update_ema(model, ema_model, ema_decay)

        # Update progress bar
        pbar.set_postfix({"loss": f"{loss:.4f}"})

        if wandb_run is not None:
            log_payload = {
                "train/loss": loss,
                "train/lr": optimizer.param_groups[0]["lr"],
            }
            if aux_loss_value is not None:
                log_payload["train/moe_aux_loss"] = aux_loss_value
            wandb_run.log(log_payload, step=step + 1)

        if (
            router_stats is not None
            and router_log_interval is not None
            and router_log_interval > 0
            and (step + 1) % router_log_interval == 0
        ):
            log_router_histograms(router_stats, step + 1, wandb_run=wandb_run)

        # Sample generation
        if (step + 1) % sample_interval == 0:
            eval_model = ema_model if ema_model is not None else model
            was_training = eval_model.training
            eval_model.eval()
            with torch.no_grad():
                # Get random context if context_len > 0
                context_tokens = None
                if eval_model.config.context_len > 0 and dataset_tokens is not None:
                    context_tokens = get_random_context(
                        dataset_tokens, eval_model.config.context_len, batch_size=1
                    )

                samples = eval_model.sample(
                    batch_size=1,
                    seq_len=eval_model.config.sequence_len,
                    num_steps=None,
                    temperature=1.0,
                    device=eval_model.get_device(),
                    context_tokens=context_tokens,
                    method="confidence",
                    confidence_threshold=0.95,
                )
                # Decode samples to text
                text = decode_tokens(samples[0])
                tqdm.write(f"\n--- Sample at step {step + 1} ---")
                tqdm.write(text)
                tqdm.write("--- End sample ---\n")
                if wandb_run is not None:
                    formatted = text.replace("\n", "<br>")
                    wandb_run.log(
                        {"samples/text": wandb.Html(formatted)}, step=step + 1
                    )
            if ema_model is None and was_training:
                eval_model.train()


def main():
    # Hyperparameters
    batch_size = 64
    max_iters = 20000
    eval_interval = 500
    learning_rate = 3e-4
    warmup_iters = max(1000, max_iters // 10)
    ema_decay = 0.999
    grad_clip = 1.0

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
    ema_model = DiffusionTransformer(config).to(device)
    ema_model.load_state_dict(model.state_dict())
    ema_model.eval()
    for param in ema_model.parameters():
        param.requires_grad_(False)

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

    def lr_lambda(step):
        if step < warmup_iters:
            return float(step + 1) / max(1, warmup_iters)
        progress = min(
            float(step - warmup_iters) / max(1, max_iters - warmup_iters), 1.0
        )
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lr_lambda)

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

    wandb_run = None
    if _wandb_enabled():
        if wandb is None:
            raise ImportError(
                "wandb is not installed but USE_WANDB is enabled. "
                "Install wandb or unset USE_WANDB."
            )
        wandb_config = {
            "batch_size": batch_size,
            "learning_rate": learning_rate,
            "max_iters": max_iters,
            "warmup_iters": warmup_iters,
            "ema_decay": ema_decay,
            "grad_clip": grad_clip,
            "sequence_len": config.sequence_len,
            "diffusion_steps": config.diffusion_steps,
            "context_len": config.context_len,
            "n_layer": config.n_layer,
            "n_head": config.n_head,
            "n_embd": config.n_embd,
            "num_experts": config.num_experts,
            "moe_top_k": config.moe_top_k,
            "moe_capacity_factor": config.moe_capacity_factor,
            "moe_router_jitter": config.moe_router_jitter,
            "moe_aux_loss_weight": config.moe_aux_loss_weight,
            "device": str(device),
        }
        wandb_project = os.environ.get("WANDB_PROJECT", "tiny-diffusion")
        wandb_run = wandb.init(
            project=wandb_project,
            name=os.environ.get("WANDB_RUN_NAME"),
            config=wandb_config,
        )

    try:
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
            scheduler=scheduler,
            ema_model=ema_model,
            ema_decay=ema_decay,
            grad_clip=grad_clip,
            router_log_interval=eval_interval,
            wandb_run=wandb_run,
        )
    finally:
        if wandb_run is not None:
            wandb_run.finish()

    # Save model
    import os

    os.makedirs("weights", exist_ok=True)
    target_model = ema_model if ema_model is not None else model
    torch.save(target_model.state_dict(), "weights/diffusion_model.pt")
    print("EMA model saved to weights/diffusion_model.pt")


if __name__ == "__main__":
    main()
