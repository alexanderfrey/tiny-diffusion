"""
Utility helpers for loading a shared GPT-2 tokenizer from HuggingFace Transformers.
We add a dedicated mask token so the diffusion model can reuse it during training
and sampling. The tokenizer object is cached globally so repeated calls are cheap.
"""

from __future__ import annotations

from typing import Optional

from transformers import GPT2TokenizerFast

_TOKENIZER: Optional[GPT2TokenizerFast] = None


def get_tokenizer(model_name: str = "gpt2", mask_token: str = "<mask>") -> GPT2TokenizerFast:
    """
    Load (and cache) a GPT-2 tokenizer, ensuring a mask token exists.
    """
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER

    tokenizer = GPT2TokenizerFast.from_pretrained(model_name)
    if tokenizer.mask_token is None:
        special_tokens = {}
        if mask_token:
            special_tokens["mask_token"] = mask_token
        if special_tokens:
            tokenizer.add_special_tokens(special_tokens)

    if tokenizer.mask_token_id is None:
        raise ValueError(
            "GPT-2 tokenizer is missing a mask token. "
            "Please provide a valid mask_token string."
        )

    _TOKENIZER = tokenizer
    return _TOKENIZER


def tokenizer_vocab_size(tokenizer: Optional[GPT2TokenizerFast] = None) -> int:
    """
    Return the total size of the tokenizer vocabulary, including any added tokens.
    """
    tok = tokenizer or get_tokenizer()
    return tok.vocab_size + len(tok.get_added_vocab())
