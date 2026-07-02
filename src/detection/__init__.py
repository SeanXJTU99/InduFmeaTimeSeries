"""Anomaly detection subpackage: physics-informed detection, adaptive baselines, feature engineering."""

from src.detection.physics_informed_detector import PhysicsInformedDetector
from src.detection.adaptive_baseline import AdaptiveBaseline, EWMAKDEBaseline
from src.detection.feature_engineering import FeatureEngineer, compute_physics_features

__all__ = [
    "PhysicsInformedDetector",
    "AdaptiveBaseline",
    "EWMAKDEBaseline",
    "FeatureEngineer",
    "compute_physics_features",
]
