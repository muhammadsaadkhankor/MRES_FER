"""Multi-task objective for the micro and macro heads."""

from __future__ import annotations

from torch import Tensor, nn

from mres_fer.config import LossConfig
from mres_fer.models.mres_fer import ModelOutput


class MicroMacroLoss(nn.Module):
    """Weighted sum of the two cross entropies.

    Clips without a micro annotation carry ``ignore_index`` and contribute nothing to
    the micro term, which lets macro-only datasets be mixed in.
    """

    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config
        self.micro_criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing, ignore_index=config.ignore_index
        )
        self.macro_criterion = nn.CrossEntropyLoss(
            label_smoothing=config.label_smoothing, ignore_index=config.ignore_index
        )

    def forward(
        self, output: ModelOutput, micro_labels: Tensor, macro_labels: Tensor
    ) -> dict[str, Tensor]:
        macro_loss = self.macro_criterion(output.macro_logits, macro_labels)
        has_micro = (micro_labels != self.config.ignore_index).any()
        if has_micro:
            micro_loss = self.micro_criterion(output.micro_logits, micro_labels)
        else:
            micro_loss = macro_loss.new_zeros(())
        total = self.config.macro_weight * macro_loss + self.config.micro_weight * micro_loss
        return {"loss": total, "macro_loss": macro_loss, "micro_loss": micro_loss}
