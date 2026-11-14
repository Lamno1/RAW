"""CAMO-Build measure-aware point-cloud research components."""

from .assignment import build_full_cloud_assignment, select_spatial_anchors
from .losses import BalancedCEWithLovasz, LovaszSoftmaxLoss, StaticBalancedCrossEntropy
from .measure import estimate_surface_mass
from .metrics import SegmentationMeter
from .model import CAMOPrototype1
from .operator import LocalMeasureAwareOperator
from .transport import (
    SparseTransportPlan,
    geometry_transport_cost,
    pushforward_axial_normals,
    semi_relaxed_sparse_transport,
)

__all__ = [
    "BalancedCEWithLovasz",
    "CAMOPrototype1",
    "LocalMeasureAwareOperator",
    "LovaszSoftmaxLoss",
    "SegmentationMeter",
    "SparseTransportPlan",
    "StaticBalancedCrossEntropy",
    "build_full_cloud_assignment",
    "estimate_surface_mass",
    "geometry_transport_cost",
    "pushforward_axial_normals",
    "select_spatial_anchors",
    "semi_relaxed_sparse_transport",
]
