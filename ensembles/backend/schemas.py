from pathlib import Path
from typing import Any, Dict, List

from typing import Optional
import numpy as np
from pydantic import BaseModel, field_validator


class ExistingExperimentsResponse(BaseModel):
    """
    Response model for existing experiments.

    Attributes:
        location (Path): The directory path where the experiments are stored.
        experiment_names (list[str]): A list of names of the existing experiments. Defaults to an empty list.
        abs_paths (list[Path]): A list of absolute paths to the experiment directories. Defaults to an empty list.
    """

    location: Path
    experiment_names: List[str] = []
    abs_paths: List[Path] = []


class ExperimentConfig(BaseModel):
    name: str
    ml_model: str  
    n_estimators: int
    max_depth: int
    max_features: Any  
    target_column: str
    learning_rate: float = 0.1

    @field_validator('max_features', mode='before')
    @classmethod
    def parse_max_features(cls, v):
        return v


class ConvergenceHistoryResponse(BaseModel):
    train: List[float]
    val: Optional[List[float]] = None


class PredictRequest(BaseModel):
    data: List[Dict[str, Any]]


class PredictResponse(BaseModel):
    predictions: List[float]
