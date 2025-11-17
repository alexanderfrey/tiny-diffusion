# Heavily modified version of nanochat gpt.py to do diffusion
# https://github.com/karpathy/nanochat/blob/master/nanochat/gpt.py
#
# Config is based on hyperparameters from Karpathy's "Let's build GPT" video
# https://github.com/karpathy/ng-video-lecture/blob/master/gpt.py
#
# Tokenizer is simple ascii mapping

"""
Simple Character-Level Discrete Diffusion Transformer
Major changes from nanochat/gpt.py:
- Bidirectional attention instead of Causal (no kvcache)
- Time step conditioning added (time embeddings)
- Replace autoregressive generation with topk and confidence-aware parallel decoding
- Removed MQA/GQA (n_kv_head), simplified to standard multi-head attention
- Removed optimizer setup, FLOPs estimation, and embedding dtype casting
"""

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F
from tokenizer import get_tokenizer


@dataclass
class DiffusionConfig:
    sequence_len: int = 256
    vocab_size: int = 128  # Full ASCII (0-127), where 0 is reserved for mask
    mask_token_id: int = 0  # NUL character used as [MASK] token
    n_layer: int = 6
    n_head: int = 6
    n_embd: int = 384
    diffusion_steps: int = 128
    context_len: int = 16  # Number of prefix tokens that are never masked
    num_experts: int = 0  # 0 disables MoE blocks
    moe_top_k: int = 1
    moe_capacity_factor: float = 1.25
    moe_router_jitter: float = 0.0
    moe_aux_loss_weight: float = 0.01


def _functional_rms_norm(x, eps=1e-8):
    return x * torch.rsqrt(x.pow(2).mean(dim=-1, keepdim=True) + eps)


class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-8):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x):
        return _functional_rms_norm(x, self.eps) * self.weight


def apply_rotary_emb(x, cos, sin):
    assert x.ndim == 4  # multihead attention
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # split up last time into two halves
    y1 = x1 * cos + x2 * sin  # rotate pairs of dims
    y2 = x1 * (-sin) + x2 * cos
    out = torch.cat([y1, y2], 3)  # re-assemble
    out = out.to(x.dtype)  # ensure input/output dtypes match
    return out


class BidirectionalAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.n_head = config.n_head
        self.n_embd = config.n_embd
        self.head_dim = self.n_embd // self.n_head
        assert self.n_embd % self.n_head == 0
        self.c_q = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_k = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_v = nn.Linear(self.n_embd, self.n_embd, bias=False)
        self.c_proj = nn.Linear(self.n_embd, self.n_embd, bias=False)

    def forward(self, x, cos_sin):
        B, T, C = x.size()

        # Project the input to get queries, keys, and values
        q = self.c_q(x).view(B, T, self.n_head, self.head_dim)
        k = self.c_k(x).view(B, T, self.n_head, self.head_dim)
        v = self.c_v(x).view(B, T, self.n_head, self.head_dim)

        # Apply Rotary Embeddings to queries and keys
        cos, sin = cos_sin
        q, k = apply_rotary_emb(q, cos, sin), apply_rotary_emb(k, cos, sin)
        q, k = _functional_rms_norm(q), _functional_rms_norm(k)  # QK norm
        q, k, v = (
            q.transpose(1, 2),
            k.transpose(1, 2),
            v.transpose(1, 2),
        )  # (B, T, H, D) -> (B, H, T, D)

        # Bidirectional attention - no causal masking
        y = F.scaled_dot_product_attention(q, k, v, is_causal=False)

        # Re-assemble the heads and project back
        y = y.transpose(1, 2).contiguous().view(B, T, -1)
        y = self.c_proj(y)
        return y


class MLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.c_fc = nn.Linear(config.n_embd, 4 * config.n_embd, bias=False)
        self.c_proj = nn.Linear(4 * config.n_embd, config.n_embd, bias=False)

    def forward(self, x):
        x = self.c_fc(x)
        x = F.relu(x).square()
        x = self.c_proj(x)
        return x


class MoEMLP(nn.Module):
    """
    Switch-style mixture-of-experts feed-forward layer using top-k routing.
    """

    def __init__(self, config: DiffusionConfig):
        super().__init__()
        if config.num_experts <= 0:
            raise ValueError("MoEMLP requires num_experts > 0")
        self.model_dim = config.n_embd
        self.hidden_dim = 4 * config.n_embd
        self.num_experts = config.num_experts
        self.top_k = max(1, min(config.moe_top_k, self.num_experts))
        self.capacity_factor = config.moe_capacity_factor
        self.router_jitter = config.moe_router_jitter

        self.router = nn.Linear(self.model_dim, self.num_experts, bias=False)

        self.w1 = nn.Parameter(
            torch.empty(self.num_experts, self.model_dim, self.hidden_dim)
        )
        self.w2 = nn.Parameter(
            torch.empty(self.num_experts, self.hidden_dim, self.model_dim)
        )
        self.reset_parameters()

    def reset_parameters(self):
        fan_in_fc = float(self.model_dim)
        fan_out_fc = float(self.hidden_dim)
        std_fc = 1.0 / math.sqrt(fan_in_fc) * min(1.0, math.sqrt(fan_out_fc / fan_in_fc))
        torch.nn.init.normal_(self.w1, mean=0.0, std=std_fc)

        fan_in_proj = float(self.hidden_dim)
        fan_out_proj = float(self.model_dim)
        std_proj = 1.0 / math.sqrt(fan_in_proj) * min(
            1.0, math.sqrt(fan_out_proj / fan_in_proj)
        )
        torch.nn.init.normal_(self.w2, mean=0.0, std=std_proj)
        torch.nn.init.normal_(self.router.weight, mean=0.0, std=1.0 / math.sqrt(self.model_dim))

    def _compute_capacity(self, num_tokens: int):
        if num_tokens == 0:
            return 1
        approx = math.ceil((num_tokens * self.top_k) / max(1, self.num_experts))
        capacity = int(math.ceil(self.capacity_factor * approx))
        return max(1, capacity)

    def forward(self, x):
        B, T, C = x.shape
        tokens = x.reshape(-1, C)
        num_tokens = tokens.size(0)
        if num_tokens == 0:
            zeros = x.new_zeros(B, T, C)
            hist = x.new_zeros(self.num_experts)
            return zeros, x.new_zeros(()), hist

        router_logits = self.router(tokens)
        if self.training and self.router_jitter > 0:
            noise = torch.randn_like(router_logits)
            router_logits = router_logits + noise * self.router_jitter
        router_probs = F.softmax(router_logits, dim=-1)

        topk_vals, topk_idx = torch.topk(router_probs, self.top_k, dim=-1)
        topk_sum = topk_vals.sum(dim=-1, keepdim=True)
        topk_probs = topk_vals / (topk_sum + 1e-9)

        token_indices = torch.arange(num_tokens, device=x.device).unsqueeze(1).expand(
            -1, self.top_k
        )
        token_indices = token_indices.reshape(-1)
        expert_indices = topk_idx.reshape(-1)
        expert_gates = topk_probs.reshape(-1)

        capacity = self._compute_capacity(num_tokens)

        # Determine slot for each (token, expert) pair and drop overflow
        one_hot = F.one_hot(expert_indices, num_classes=self.num_experts)
        expert_cumsum = torch.cumsum(one_hot, dim=0) - 1
        expert_positions = expert_cumsum[
            torch.arange(expert_indices.size(0), device=x.device), expert_indices
        ]
        within_capacity = expert_positions < capacity

        if not within_capacity.any():
            aux_loss = router_probs.new_zeros(())
            hist = x.new_zeros(self.num_experts)
            return x.new_zeros(B, T, C), aux_loss, hist

        token_indices = token_indices[within_capacity]
        expert_indices = expert_indices[within_capacity]
        expert_positions = expert_positions[within_capacity]
        expert_gates = expert_gates[within_capacity]

        expert_inputs = torch.zeros(
            self.num_experts,
            capacity,
            C,
            dtype=tokens.dtype,
            device=tokens.device,
        )
        expert_inputs[expert_indices, expert_positions] = tokens[token_indices]

        hidden = torch.einsum("ecd,edh->ech", expert_inputs, self.w1)
        hidden = F.relu(hidden).square()
        expert_outputs = torch.einsum("ech,eho->eco", hidden, self.w2)

        gathered = expert_outputs[expert_indices, expert_positions]
        combined = torch.zeros_like(tokens)
        combined.index_add_(
            0, token_indices, gathered * expert_gates.unsqueeze(-1)
        )
        combined = combined.view(B, T, C)

        # Load balancing loss
        importance = router_probs.mean(dim=0)
        load = torch.bincount(expert_indices, minlength=self.num_experts).float()
        load_sum = load.sum()
        if load_sum > 0:
            load = load / load_sum
        else:
            load = load * 0.0
        aux_loss = (importance * load).sum() * self.num_experts
        gate_hist = torch.bincount(
            expert_indices, minlength=self.num_experts
        ).to(tokens.dtype)
        return combined, aux_loss, gate_hist


class Block(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attn = BidirectionalAttention(config)
        self.use_moe = config.num_experts > 0
        self.mlp = MoEMLP(config) if self.use_moe else MLP(config)
        self.attn_norm = RMSNorm(config.n_embd)
        self.mlp_norm = RMSNorm(config.n_embd)

    def forward(self, x, cos_sin):
        x = x + self.attn(self.attn_norm(x), cos_sin)
        if self.use_moe:
            mlp_out, aux_loss, router_hist = self.mlp(self.mlp_norm(x))
        else:
            mlp_out = self.mlp(self.mlp_norm(x))
            aux_loss = None
            router_hist = None
        x = x + mlp_out
        return x, aux_loss, router_hist


class DiffusionTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        # Token and time embeddings (include mask token in vocab)
        self.token_emb = nn.Embedding(config.vocab_size, config.n_embd)
        self.time_emb = nn.Embedding(config.diffusion_steps, config.n_embd)

        # Transformer blocks
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])

        # Output head to predict denoised tokens
        self.output_head = nn.Linear(config.n_embd, config.vocab_size, bias=False)

        # Normalization layers
        self.input_norm = RMSNorm(config.n_embd)
        self.final_norm = RMSNorm(config.n_embd)

        # Rotary embeddings
        self.rotary_seq_len = config.sequence_len * 2
        head_dim = config.n_embd // config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.register_buffer("cos", cos, persistent=False)
        self.register_buffer("sin", sin, persistent=False)
        mask_schedule = self._build_mask_schedule(config.diffusion_steps)
        self.register_buffer("mask_schedule", mask_schedule, persistent=False)

    def init_weights(self):
        self.apply(self._init_weights)
        # Init the rotary embeddings
        head_dim = self.config.n_embd // self.config.n_head
        cos, sin = self._precompute_rotary_embeddings(self.rotary_seq_len, head_dim)
        self.cos, self.sin = cos, sin

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            # https://arxiv.org/pdf/2310.17813
            fan_out = module.weight.size(0)
            fan_in = module.weight.size(1)
            std = 1.0 / math.sqrt(fan_in) * min(1.0, math.sqrt(fan_out / fan_in))
            torch.nn.init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=1.0)

    def _precompute_rotary_embeddings(self, seq_len, head_dim, base=10000, device=None):
        if device is None:
            device = self.token_emb.weight.device
        channel_range = torch.arange(0, head_dim, 2, dtype=torch.float32, device=device)
        inv_freq = 1.0 / (base ** (channel_range / head_dim))
        t = torch.arange(seq_len, dtype=torch.float32, device=device)
        freqs = torch.outer(t, inv_freq)
        cos, sin = freqs.cos(), freqs.sin()
        cos, sin = (
            cos[None, :, None, :],
            sin[None, :, None, :],
        )  # add batch and head dims
        return cos, sin

    def _build_mask_schedule(self, num_timesteps):
        steps = torch.arange(1, num_timesteps + 1, dtype=torch.float32)
        normalized = steps / num_timesteps
        mask_probs = torch.sin(normalized * math.pi / 2).pow(2)
        mask_probs = mask_probs.clamp(min=1.0 / (2 * num_timesteps), max=1.0)
        mask_probs[0] = 0.0  # ensure final step fully decodes
        return mask_probs

    def get_device(self):
        return self.token_emb.weight.device

    def forward(self, x_t, t, return_aux=False, return_router_stats=False):
        """
        Forward pass for diffusion model
        Args:
            x_t: Noisy tokens at timestep t, shape (B, T)
            t: Timestep indices, shape (B,)
            return_aux: If True, also return auxiliary MoE loss for training
            return_router_stats: If True, return per-layer router histograms
        Returns:
            logits: Predicted token logits, shape (B, T, vocab_size)
            aux_loss (optional): Auxiliary MoE loss scalar
            router_stats (optional): List of per-layer expert histograms
        """
        B, T = x_t.size()

        # Get embeddings
        x = self.token_emb(x_t)  # (B, T, n_embd)
        t_emb = self.time_emb(t)  # (B, n_embd)

        # Add time embedding to all positions
        x = x + t_emb.unsqueeze(1)  # broadcast time embedding across sequence
        x = self.input_norm(x)

        # Get rotary embeddings
        assert T <= self.cos.size(1)
        cos_sin = (self.cos[:, :T], self.sin[:, :T])

        # Forward through transformer blocks
        total_aux_loss = None
        router_stats = [] if return_router_stats else None
        has_router_stats = False
        for block in self.blocks:
            x, block_aux, block_hist = block(x, cos_sin)
            if block_aux is not None:
                total_aux_loss = (
                    block_aux if total_aux_loss is None else total_aux_loss + block_aux
                )
            if router_stats is not None:
                router_stats.append(block_hist)
                if block_hist is not None:
                    has_router_stats = True
        x = self.final_norm(x)

        # Predict denoised tokens
        logits = self.output_head(x)  # (B, T, vocab_size)
        outputs = [logits]
        if return_aux:
            if total_aux_loss is None:
                total_aux_loss = logits.new_zeros(())
            outputs.append(total_aux_loss)
        if return_router_stats:
            if router_stats is None or not has_router_stats:
                router_stats = None
            outputs.append(router_stats)
        if len(outputs) == 1:
            return outputs[0]
        return tuple(outputs)

    @torch.inference_mode()
    def sample_topk(
        self,
        batch_size,
        seq_len,
        k=None,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
    ):
        """
        Generate samples using top-K-style parallel decoding.
        At each diffusion timestep, release as many tokens as required by the
        training mask schedule, optionally capping updates with `k`.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            k: Number of tokens to decode per step
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if device is None:
            device = self.get_device()
        timesteps = self._get_sampling_schedule(num_steps, device)
        x, masked_positions, initial_masked = self._init_sampling_state(
            batch_size, seq_len, device, context_tokens
        )

        # Decode step by step
        for step in timesteps:
            # Check if all tokens are decoded
            if not masked_positions.any():
                break

            # Create timestep (use actual diffusion timestep)
            t_val = step.long()
            t_batch = torch.full(
                (batch_size,), t_val.item(), device=device, dtype=torch.long
            )

            # Predict tokens
            logits = self.forward(x, t_batch)

            # Sample tokens stochastically while measuring peak confidence
            temp = max(temperature, 1e-4)
            probs = F.softmax(logits / temp, dim=-1)
            confidences, _ = torch.max(probs, dim=-1)  # (B, T)
            predicted_tokens = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), num_samples=1
            ).view(batch_size, seq_len)

            decode_counts = self._decode_counts_from_schedule(
                masked_positions, initial_masked, t_val, cap=k
            )

            masked_confidences = confidences.masked_fill(
                ~masked_positions, -float("inf")
            )

            # Update the scheduled number of positions
            for b in range(batch_size):
                need = int(decode_counts[b].item())
                if need <= 0:
                    continue
                available = int(masked_positions[b].sum().item())
                if available == 0:
                    continue
                need = min(need, available)
                _, topk_idx = torch.topk(masked_confidences[b], k=need)
                x[b, topk_idx] = predicted_tokens[b, topk_idx]
                masked_positions[b, topk_idx] = False

        return x

    @torch.inference_mode()
    def sample_confidence(
        self,
        batch_size,
        seq_len,
        confidence_threshold=0.95,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
    ):
        """
        Generate samples using confidence-aware parallel decoding (Fast-dLLM).
        Each timestep decodes the number of tokens prescribed by the mask
        schedule while prioritizing logits above the provided threshold.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            confidence_threshold: Threshold τ for token acceptance
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if device is None:
            device = self.get_device()
        timesteps = self._get_sampling_schedule(num_steps, device)
        x, masked_positions, initial_masked = self._init_sampling_state(
            batch_size, seq_len, device, context_tokens
        )

        # Decode step by step
        for step in timesteps:
            # Check if all tokens are decoded
            if not masked_positions.any():
                break

            # Create timestep (use actual diffusion timestep)
            t_val = step.long()
            t_batch = torch.full(
                (batch_size,), t_val.item(), device=device, dtype=torch.long
            )

            # Predict tokens
            logits = self.forward(x, t_batch)

            # Sample tokens stochastically while measuring peak confidence
            temp = max(temperature, 1e-4)
            probs = F.softmax(logits / temp, dim=-1)
            confidences, _ = torch.max(probs, dim=-1)  # (B, T)
            predicted_tokens = torch.multinomial(
                probs.reshape(-1, probs.size(-1)), num_samples=1
            ).view(batch_size, seq_len)

            decode_counts = self._decode_counts_from_schedule(
                masked_positions, initial_masked, t_val
            )

            # Select positions above threshold (only among masked positions)
            candidates = (confidences >= confidence_threshold) & masked_positions
            selected = candidates.clone()
            selected_counts = selected.sum(dim=1)
            remaining = (decode_counts - selected_counts).clamp(min=0)

            if remaining.any():
                masked_confidences = confidences.masked_fill(
                    ~masked_positions, -float("inf")
                )
                for b in range(batch_size):
                    need = int(remaining[b].item())
                    if need <= 0:
                        continue
                    available = int(masked_positions[b].sum().item())
                    if available == 0:
                        continue
                    # Exclude already-selected positions
                    mask = masked_positions[b] & ~selected[b]
                    if mask.sum() == 0:
                        continue
                    need = min(need, int(mask.sum().item()))
                    masked_view = masked_confidences[b].clone()
                    masked_view[~mask] = -float("inf")
                    _, extra_idx = torch.topk(masked_view, k=need)
                    selected[b, extra_idx] = True

            # Update positions according to combined selection
            x = torch.where(selected, predicted_tokens, x)
            masked_positions = masked_positions & ~selected

        return x

    @torch.inference_mode()
    def sample(
        self,
        batch_size,
        seq_len,
        num_steps=None,
        temperature=1.0,
        device=None,
        context_tokens=None,
        method="confidence",
        k=None,
        confidence_threshold=0.95,
    ):
        """
        Generate samples using parallel decoding methods.

        Args:
            batch_size: Number of samples to generate
            seq_len: Length of sequences to generate
            num_steps: Maximum number of denoising steps
            temperature: Sampling temperature
            device: Device to generate on
            context_tokens: Optional context tokens for conditioning, shape (batch_size, context_len)
            method: Decoding method - 'topk' or 'confidence'
            k: Optional cap on tokens revealed per step (for 'topk' method)
            confidence_threshold: Confidence threshold τ (for 'confidence' method)
        Returns:
            samples: Generated token sequences, shape (batch_size, seq_len)
        """
        if method == "topk":
            return self.sample_topk(
                batch_size,
                seq_len,
                k,
                num_steps,
                temperature,
                device,
                context_tokens,
            )
        elif method == "confidence":
            return self.sample_confidence(
                batch_size,
                seq_len,
                confidence_threshold,
                num_steps,
                temperature,
                device,
                context_tokens,
            )
        else:
            raise ValueError(f"Unknown sampling method: {method}")

    def _get_sampling_schedule(self, num_steps, device):
        """
        Create a schedule of diffusion timesteps to visit during sampling.
        We always start from the noisiest step and move toward 0 so the model
        sees the same conditioning distribution as in training.
        """
        max_steps = self.config.diffusion_steps
        if num_steps is None or num_steps >= max_steps:
            timesteps = torch.arange(
                max_steps - 1, -1, -1, device=device, dtype=torch.long
            )
        else:
            # Downsample the schedule if the user wants fewer steps
            lin = torch.linspace(
                max_steps - 1, 0, steps=num_steps, device=device, dtype=torch.float32
            )
            timesteps = lin.round().clamp_(0, max_steps - 1).long()
            timesteps = torch.unique_consecutive(timesteps, dim=0)
        return timesteps

    def _init_sampling_state(self, batch_size, seq_len, device, context_tokens):
        x = torch.full(
            (batch_size, seq_len),
            self.config.mask_token_id,
            dtype=torch.long,
            device=device,
        )
        masked_positions = torch.ones(
            batch_size, seq_len, dtype=torch.bool, device=device
        )
        context_len = 0
        if context_tokens is not None:
            context_len = min(context_tokens.size(1), seq_len)
            x[:, :context_len] = context_tokens[:, :context_len].to(device)
            masked_positions[:, :context_len] = False
        initial_masked = masked_positions.sum(dim=1)
        return x, masked_positions, initial_masked

    def _decode_counts_from_schedule(
        self, masked_positions, initial_masked_counts, timestep, cap=None
    ):
        if masked_positions.numel() == 0:
            return torch.zeros(0, dtype=torch.long, device=initial_masked_counts.device)
        mask_ratio = self.mask_schedule[timestep]
        desired = (mask_ratio * initial_masked_counts.float()).round().long()
        desired = torch.minimum(desired, initial_masked_counts)
        current = masked_positions.sum(dim=1)
        decode_needed = (current - desired).clamp(min=0)
        decode_needed = torch.minimum(decode_needed, current)
        needs_progress = (current > 0) & (decode_needed == 0)
        decode_needed = torch.where(
            needs_progress, torch.ones_like(decode_needed), decode_needed
        )
        if cap is not None:
            cap_tensor = torch.full_like(decode_needed, cap)
            decode_needed = torch.minimum(decode_needed, cap_tensor)
        return decode_needed


def encode_text(text, tokenizer=None):
    """Convert text to vocab indices using the shared GPT-2 tokenizer."""
    tok = tokenizer or get_tokenizer()
    token_ids = tok.encode(text, add_special_tokens=False)
    return torch.tensor(token_ids, dtype=torch.long)


def decode_tokens(tokens, tokenizer=None):
    """Convert vocab indices back to text using the shared GPT-2 tokenizer."""
    tok = tokenizer or get_tokenizer()
    if isinstance(tokens, torch.Tensor):
        token_seq = tokens.tolist()
    else:
        token_seq = list(tokens)
    return tok.decode(token_seq, skip_special_tokens=True)
