from dataclasses import dataclass
from typing import Any, List, Optional, Tuple

import torch
from transformers.modeling_outputs import ModelOutput
from transformers.models.idefics2.configuration_idefics2 import Idefics2PerceiverConfig
from transformers.models.idefics2.modeling_idefics2 import Idefics2PerceiverResampler
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VLForConditionalGeneration
from transformers.processing_utils import Unpack


@dataclass
class TRASEROutput(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[List[torch.FloatTensor]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    rope_deltas: Optional[torch.LongTensor] = None


class TRASER(Qwen2_5_VLForConditionalGeneration):
    """Qwen2.5-VL extended with two Perceiver Resamplers for trajectory-aligned tokens.

    - Temporal-Window Resampler (TWR, `perceiver_resampler`): compresses one object's
      visual tokens inside each temporal window into a fixed number of latents.
    - Object-Trajectory Resampler (OTR, `second_perceiver_resampler`): compresses all
      of one object's visual tokens across the whole video into a global summary.
    """

    def __init__(self, config: Qwen2_5_VLConfig, **kwargs):
        super().__init__(config)
        # Store resampler hyperparameters passed via from_pretrained(**kwargs) on the
        # config so that they are serialized with the checkpoint.
        for k, v in kwargs.items():
            if not hasattr(config, k):
                setattr(config, k, v)

        self.config = config
        self._build_perceiver(dtype=config.torch_dtype, attn_impl=config._attn_implementation)
        self.post_init()

    def _build_perceiver(self, dtype: torch.dtype, attn_impl: str) -> None:
        hidden_size = int(getattr(self.config, "hidden_size", 2048))
        n_latents = int(getattr(self.config, "temporal_resampler_n_latents", 32))
        depth = int(getattr(self.config, "resampler_depth", 3))

        perceiver_cfg = Idefics2PerceiverConfig(
            hidden_size=hidden_size,
            resampler_n_latents=n_latents,
            resampler_depth=depth,
            _attn_implementation=attn_impl,
            torch_dtype=dtype,
        )
        self.perceiver_resampler = Idefics2PerceiverResampler(perceiver_cfg)

        if getattr(self.config, "object_resampler", True):
            second_n_latents = int(getattr(self.config, "object_resampler_n_latents", 32))

            second_perceiver_cfg = Idefics2PerceiverConfig(
                hidden_size=hidden_size,
                resampler_n_latents=second_n_latents,
                resampler_depth=depth,
                _attn_implementation=attn_impl,
                torch_dtype=dtype,
            )
            self.second_perceiver_resampler = Idefics2PerceiverResampler(second_perceiver_cfg)

    def prepare_inputs_for_generation(
        self,
        input_ids,
        past_key_values=None,
        attention_mask=None,
        inputs_embeds=None,
        cache_position=None,
        position_ids=None,
        use_cache=True,
        pixel_values=None,
        pixel_values_videos=None,
        image_grid_thw=None,
        video_grid_thw=None,
        second_per_grid_ts=None,
        **kwargs,
    ):
        model_inputs = super().prepare_inputs_for_generation(
            input_ids,
            past_key_values=past_key_values,
            attention_mask=attention_mask,
            inputs_embeds=inputs_embeds,
            cache_position=cache_position,
            position_ids=position_ids,
            pixel_values=pixel_values,
            pixel_values_videos=pixel_values_videos,
            image_grid_thw=image_grid_thw,
            video_grid_thw=video_grid_thw,
            second_per_grid_ts=second_per_grid_ts,
            use_cache=use_cache,
            **kwargs,
        )

        model_inputs["position_ids"] = position_ids
        if cache_position is not None and cache_position[0] != 0:
            model_inputs["pixel_values"] = None
            model_inputs["pixel_values_videos"] = None
            model_inputs["position_ids"] = None
        return model_inputs

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        rope_deltas: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[Any],
    ) -> TRASEROutput:

        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states

        if rope_deltas is not None:
            self.model.rope_deltas = rope_deltas  # [B, 1]

        # Prefill: the caller provides the rearranged embeddings and position ids directly.
        is_prefill = (inputs_embeds is not None) and (
            past_key_values is None or (hasattr(past_key_values, "get_seq_length") and past_key_values.get_seq_length() == 0)
        )

        if is_prefill:
            outputs = self.model.language_model(
                input_ids=None,
                inputs_embeds=inputs_embeds,
                position_ids=position_ids,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                return_dict=True,
            )
        else:
            # Decode: 1D positions continued from the prefill via rope_deltas.
            inputs_embeds = self.model.get_input_embeddings()(input_ids)
            batch_size, seq_length, _ = inputs_embeds.shape
            delta = (
                (cache_position[0] + self.model.rope_deltas).to(inputs_embeds.device)
                if cache_position is not None
                else 0
            )
            pos = torch.arange(seq_length, device=inputs_embeds.device).view(1, -1).expand(batch_size, -1)
            if cache_position is not None:
                delta = delta.repeat_interleave(max(1, batch_size // delta.shape[0]), dim=0)
            pos = pos.add(delta).unsqueeze(0).expand(3, -1, -1)

            outputs = self.model.language_model(
                input_ids=None,
                position_ids=pos,
                attention_mask=attention_mask,
                past_key_values=past_key_values,
                inputs_embeds=inputs_embeds,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                cache_position=cache_position,
                **kwargs,
            )

        hidden_states = outputs.last_hidden_state
        logits = self.lm_head(hidden_states)

        loss = None
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=logits.size(-1))

        return TRASEROutput(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
            rope_deltas=self.model.rope_deltas,
        )
