from transformers import Trainer
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import (
    Qwen2_5_VisionTransformerPretrainedModel,
    Qwen2_5_VLModel,
)
from transformers.pytorch_utils import ALL_LAYERNORM_LAYERS
from transformers.trainer import get_parameter_names


def print_trainable_parameters_visual(self) -> None:
    """Print the trainable status of the vision attention blocks and the merger module."""
    trainable_blocks = []
    non_trainable_blocks = []

    for block_idx, block in enumerate(self.blocks):
        is_trainable = all(param.requires_grad for param in block.parameters())
        if is_trainable:
            trainable_blocks.append(block_idx)
        else:
            non_trainable_blocks.append(block_idx)

    is_merger_trainable = any(param.requires_grad for param in self.merger.parameters())

    print("Vision Module - Attention Blocks:")
    print(f"Trainable Block Indices: {trainable_blocks if trainable_blocks else 'None'}")
    print(f"Non-Trainable Block Indices: {non_trainable_blocks if non_trainable_blocks else 'None'}")
    print(f"Merger Module Trainable: {is_merger_trainable}")


def print_trainable_parameters(self) -> None:
    """Print the trainable status of the LLM embeddings and decoder layers."""
    is_embed_trainable = any(param.requires_grad for param in self.embed_tokens.parameters())
    print(f"LLM Module - Embed Tokens Trainable: {is_embed_trainable}")

    trainable_layers = []
    non_trainable_layers = []

    for layer_idx, layer in enumerate(self.layers):
        is_trainable = any(param.requires_grad for param in layer.parameters())
        if is_trainable:
            trainable_layers.append(layer_idx)
        else:
            non_trainable_layers.append(layer_idx)

    print(f"LLM Module - Trainable Layer Indices: {trainable_layers if trainable_layers else 'None'}")
    print(f"LLM Module - Non-Trainable Layer Indices: {non_trainable_layers if non_trainable_layers else 'None'}")


def create_optimizer(self):
    """Build the optimizer with separate learning rates for the multimodal projector
    (``mm_projector_lr``) and the perceiver resamplers (``resampler_lr``)."""
    opt_model = self.model

    if self.optimizer is None:
        decay_parameters = get_parameter_names(opt_model, ALL_LAYERNORM_LAYERS)
        decay_parameters = [name for name in decay_parameters if "bias" not in name]
        decay_parameters = [name for name in decay_parameters if "embed_tokens" not in name]

        use_projector_lr = self.args.mm_projector_lr is not None and self.args.mm_projector_lr != 0
        use_resampler_lr = self.args.resampler_lr is not None

        projector_parameters = (
            [name for name, _ in opt_model.named_parameters() if "merger" in name]
            if use_projector_lr else []
        )
        resampler_parameters = (
            [name for name, _ in opt_model.named_parameters() if "resampler" in name]
            if use_resampler_lr else []
        )

        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n in decay_parameters
                        and n not in projector_parameters
                        and n not in resampler_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": self.args.weight_decay,
            },
            {
                "params": [
                    p
                    for n, p in opt_model.named_parameters()
                    if (
                        n not in decay_parameters
                        and n not in projector_parameters
                        and n not in resampler_parameters
                        and p.requires_grad
                    )
                ],
                "weight_decay": 0.0,
            },
        ]
        if use_projector_lr:
            optimizer_grouped_parameters += [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": self.args.weight_decay,
                    "lr": self.args.mm_projector_lr,
                },
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n not in decay_parameters and n in projector_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.mm_projector_lr,
                },
            ]
        if use_resampler_lr:
            optimizer_grouped_parameters += [
                {
                    "params": [
                        p
                        for n, p in opt_model.named_parameters()
                        if (n in resampler_parameters and p.requires_grad)
                    ],
                    "weight_decay": 0.0,
                    "lr": self.args.resampler_lr,
                },
            ]

        optimizer_cls, optimizer_kwargs = Trainer.get_optimizer_cls_and_kwargs(self.args)
        self.optimizer = optimizer_cls(optimizer_grouped_parameters, **optimizer_kwargs)

    return self.optimizer


# Apply monkey patches.
Trainer.create_optimizer = create_optimizer

Qwen2_5_VisionTransformerPretrainedModel.print_trainable_parameters = (
    print_trainable_parameters_visual
)
Qwen2_5_VLModel.print_trainable_parameters = print_trainable_parameters
