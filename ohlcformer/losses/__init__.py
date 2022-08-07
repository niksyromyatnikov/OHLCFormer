from .builder import (
    register_loss,
    LossBuilder
)

from .loss import (
    Loss,
    get_metric_direction
)

from .direction_loss import (
    MaskedDirectionLoss,
)

from .mae_loss import (
    MaskedMAELoss,
)

from .mse_loss import (
    MaskedMSELoss,
)

from .rmse_loss import (
    MaskedRMSELoss,
)

default_loss = [
    {"loss_type": "masked_mse_loss", "weighted": False, "reduction": "sum", "name": "loss"},
    {"loss_type": "masked_rmse_loss", "weighted": False, "reduction": "sum", "name": "rmse_loss"},
    {"loss_type": "masked_mae_loss", "weighted": False, "reduction": "sum", "name": "mae_loss"},
    {"loss_type": "masked_direction_loss", "weighted": False, "reduction": "sum", "name": "mask_direction_loss"}
]
