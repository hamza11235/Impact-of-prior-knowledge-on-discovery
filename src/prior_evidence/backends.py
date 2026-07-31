"""Model backends used by the feasibility gate."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class GenerationResult:
    """Generated text plus the backend's termination metadata."""

    text: str
    finish_reason: str | None
    generation_tokens: int | None


class GenerationBackend(Protocol):
    """Minimal generation interface kept independent of MLX."""

    model_id: str

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> GenerationResult:
        """Generate one response for a chat prompt."""


class MLXBackend:
    """Lazy-loading MLX backend for Apple Silicon."""

    def __init__(
        self,
        model_id: str,
        *,
        adapter_path: str | None = None,
        enable_thinking: bool = False,
        top_p: float = 0.0,
        top_k: int = 0,
    ) -> None:
        self.model_id = model_id
        self.adapter_path = adapter_path
        self.enable_thinking = enable_thinking
        self.top_p = top_p
        self.top_k = top_k
        self._model = None
        self._tokenizer = None
        self._lora_modules = None
        self._baseline_lora_scales: tuple[float, ...] | None = None

    def _load(self) -> None:
        if self._model is not None:
            return
        try:
            from mlx_lm import load
        except ImportError as exc:
            raise RuntimeError(
                'MLX support is not installed. Run: python -m pip install -e ".[mac]"'
            ) from exc
        self._model, self._tokenizer = load(
            self.model_id,
            adapter_path=self.adapter_path,
        )

    def set_adapter_multiplier(self, multiplier: float) -> tuple[float, ...]:
        """Scale a loaded LoRA relative to the values in its adapter config."""

        if multiplier < 0:
            raise ValueError("adapter multiplier must be non-negative")
        if self.adapter_path is None:
            raise RuntimeError("an adapter path is required for LoRA scaling")
        self._load()
        assert self._model is not None

        if self._lora_modules is None:
            try:
                from mlx_lm.tuner.lora import (
                    LoRAEmbedding,
                    LoRALinear,
                    LoRASwitchLinear,
                )
            except ImportError as exc:
                raise RuntimeError("the installed mlx-lm package lacks LoRA support") from exc
            lora_types = (LoRALinear, LoRAEmbedding, LoRASwitchLinear)
            self._lora_modules = tuple(
                module
                for _, module in self._model.named_modules()
                if isinstance(module, lora_types)
            )
            if not self._lora_modules:
                raise RuntimeError("the loaded model contains no LoRA modules")
            self._baseline_lora_scales = tuple(
                float(module.scale) for module in self._lora_modules
            )

        assert self._baseline_lora_scales is not None
        for module, baseline in zip(
            self._lora_modules,
            self._baseline_lora_scales,
            strict=True,
        ):
            module.scale = baseline * multiplier
        return tuple(
            baseline * multiplier for baseline in self._baseline_lora_scales
        )

    def generate(
        self,
        messages: list[dict[str, str]],
        *,
        max_tokens: int,
        temperature: float,
        seed: int,
    ) -> str:
        self._load()
        assert self._model is not None
        assert self._tokenizer is not None

        try:
            import mlx.core as mx
            from mlx_lm import stream_generate
            from mlx_lm.sample_utils import make_sampler
        except ImportError as exc:
            raise RuntimeError("The installed mlx-lm package is incomplete") from exc

        mx.random.seed(seed)
        template_options = {
            "tokenize": False,
            "add_generation_prompt": True,
            # Qwen3 defaults to thinking unless this is explicitly false.
            "enable_thinking": self.enable_thinking,
        }
        prompt = self._tokenizer.apply_chat_template(messages, **template_options)
        sampler = make_sampler(
            temp=temperature,
            top_p=self.top_p,
            top_k=self.top_k,
        )
        text_parts: list[str] = []
        finish_reason: str | None = None
        generation_tokens: int | None = None
        for response in stream_generate(
            self._model,
            self._tokenizer,
            prompt=prompt,
            max_tokens=max_tokens,
            sampler=sampler,
        ):
            text_parts.append(response.text)
            if response.finish_reason is not None:
                finish_reason = response.finish_reason
                generation_tokens = response.generation_tokens

        return GenerationResult(
            text="".join(text_parts),
            finish_reason=finish_reason,
            generation_tokens=generation_tokens,
        )
