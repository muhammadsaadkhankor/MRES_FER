"""Model components of the micro-to-macro expression pipeline."""

from mres_fer.models.appearance import AppearanceBranch
from mres_fer.models.fusion import GatedFusion
from mres_fer.models.heads import ClipHead, MicroGuidedMacroHead
from mres_fer.models.motion import FlowEncoder
from mres_fer.models.mres_fer import ModelOutput, MresFer, build_model
from mres_fer.models.temporal import TemporalTransformer

__all__ = [
    "AppearanceBranch",
    "ClipHead",
    "FlowEncoder",
    "GatedFusion",
    "MicroGuidedMacroHead",
    "ModelOutput",
    "MresFer",
    "TemporalTransformer",
    "build_model",
]
