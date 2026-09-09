"""Multi-task objective for the micro and macro heads."""

from __future__ import annotations

from torch import Tensor, nn

from mres_fer.config import LossConfig
from mres_fer.models.mres_fer import ModelOutput


class MicroMacroLoss(nn.Module):
    """Weighted sum of the two cross entropies.

    Clips without a micro (or macro) annotation carry ``ignore_index`` and contribute
    nothing to that term, which lets mixed corpora and single-task stages share the loss.
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
        zero = output.macro_logits.new_zeros(())
        macro_loss = (
            self.macro_criterion(output.macro_logits, macro_labels)
            if (macro_labels != self.config.ignore_index).any()
            else zero
        )
        micro_loss = (
            self.micro_criterion(output.micro_logits, micro_labels)
            if (micro_labels != self.config.ignore_index).any()
            else zero
        )
        total = self.config.macro_weight * macro_loss + self.config.micro_weight * micro_loss
        return {"loss": total, "macro_loss": macro_loss, "micro_loss": micro_loss}
