"""TRASER-F: TRASER + an object-centric latent forecaster.

The forecaster's hyperparameters live on the model config (like the resampler ones), so a
saved checkpoint reloads with ``TRASERForecast.from_pretrained(path)``. Starting from the
released TRASER weights, the forecaster is missing from the checkpoint and gets initialized
by ``init_forecaster`` (call it after ``from_pretrained``: Hugging Face's ``_init_weights``
would otherwise overwrite the zero-initialized output layers).
"""

import torch

from ..train.modeling_traser import TRASER
from .dynamics import ObjectLatentForecaster

FORECASTER_DEFAULTS = dict(
    forecaster_d_model=512,
    forecaster_layers=4,
    forecaster_decoder_layers=2,
    forecaster_heads=8,
    forecaster_max_windows=32,
    forecaster_max_objects=64,
    forecaster_box_dim=4,
    forecaster_dropout=0.1,
)


class TRASERForecast(TRASER):
    def __init__(self, config, **kwargs):
        super().__init__(config, **kwargs)
        for k, v in FORECASTER_DEFAULTS.items():
            if not hasattr(self.config, k):
                setattr(self.config, k, v)
        c = self.config
        hidden = getattr(c, "hidden_size", None) or c.text_config.hidden_size
        self.forecaster = ObjectLatentForecaster(
            hidden_size=int(hidden),
            n_latents=int(getattr(c, "temporal_resampler_n_latents", 32)),
            d_model=c.forecaster_d_model, n_layers=c.forecaster_layers,
            n_decoder_layers=c.forecaster_decoder_layers, n_heads=c.forecaster_heads,
            max_windows=c.forecaster_max_windows, max_objects=c.forecaster_max_objects,
            box_dim=c.forecaster_box_dim, dropout=c.forecaster_dropout,
        )


def init_forecaster(model: TRASERForecast) -> None:
    """Re-run the forecaster's own init (gathering ZeRO-3 shards when needed)."""
    params = list(model.forecaster.parameters())
    try:
        import deepspeed
        if any(hasattr(p, "ds_id") for p in params):
            with deepspeed.zero.GatheredParameters(params, modifier_rank=0):
                if torch.distributed.get_rank() == 0:
                    model.forecaster.reset_parameters()
            return
    except ImportError:
        pass
    model.forecaster.reset_parameters()
