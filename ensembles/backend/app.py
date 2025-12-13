from pathlib import Path

from fastapi import (
    FastAPI,
    Body,
    Query,
    Path as PathParam,
    File,
    UploadFile,
    HTTPException,
)

from sklearn.model_selection import train_test_split
from pydantic import TypeAdapter
from typing import Annotated
import pandas as pd
import json

from .schemas import (
    ExistingExperimentsResponse,
    ExperimentConfig,
    PredictRequest,
    PredictResponse,
    ConvergenceHistoryResponse,
)

from ..boosting import GradientBoostingMSE
from ..random_forest import RandomForestMSE
from ..utils import ConvergenceHistory

RANDOM_SEED = 42

app = FastAPI()


def get_runs_dir() -> Path:
    return Path.cwd() / "runs"


@app.get("/existing_experiments/", response_model=ExistingExperimentsResponse)
async def existing_experiments() -> ExistingExperimentsResponse:
    """
    Get information about existing experiments.

    This endpoint scans the directory where experiments are stored and returns a list of
    existing experiments along with their absolute paths. Each experiment is stored as
    a directory in the host filesystem.

    Returns:
        ExistingExperimentsResponse: A response containing the location of the experiments
        directory, absolute paths of the experiment directories, and the names of the experiments.
    """
    path = get_runs_dir()
    response = ExistingExperimentsResponse(location=path)
    if not path.exists():
        return response
    response.abs_paths = [obj for obj in path.iterdir() if obj.is_dir()]
    response.experiment_names = [filepath.stem for filepath in response.abs_paths]
    return response


def get_experiment_dir(experiment_name: str) -> Path:
    return get_runs_dir() / experiment_name

def get_model_config_path(experiment_name: str) -> Path:
    return get_experiment_dir(experiment_name) / "config.json"

def get_train_data_path(experiment_name: str) -> Path:
    return get_experiment_dir(experiment_name) / "train.csv"

def get_model_dir(experiment_name: str) -> Path:
    return get_experiment_dir(experiment_name) / "model"

def get_history_path(experiment_name: str) -> Path:
    return get_experiment_dir(experiment_name) / "history.json"


@app.post("/register_experiment/")
async def register_experiment(
    config: Annotated[ExperimentConfig, Body()]
):
    exp_dir = get_experiment_dir(config.name)

    exp_dir.mkdir(parents=True)

    with open(get_model_config_path(config.name), "w") as f:
        f.write(config.model_dump_json(indent=4))

    return {"message": f"Experiment '{config.name}' registered successfully."}


@app.post("/upload_train_data/")
async def upload_train_data(
    experiment_name: Annotated[str, Query()],
    file: Annotated[UploadFile, File(...)]
):
    train_path = get_train_data_path(experiment_name)

    contents = await file.read()
    with open(train_path, "wb") as f:
        f.write(contents)

    return {"message": f"File for '{experiment_name}' uploaded successfully."}



@app.get("/experiment_config/{experiment_name}", response_model=ExperimentConfig)
async def get_experiment_config(
    experiment_name: Annotated[str, PathParam()]
):
    with open(get_model_config_path(experiment_name), "r") as f:
        return ExperimentConfig(**json.load(f))


@app.get("/needs_training/")
async def needs_training(
    experiment_name: Annotated[str, Query()]
):
    model_dir = get_model_dir(experiment_name)
    return {"response": not model_dir.exists()}


@app.post("/train_model/")
async def train_model(
    experiment_name: Annotated[str, Query()]
):
    with open(get_model_config_path(experiment_name), "r") as f:
        config = ExperimentConfig(**json.load(f))

    train_data = pd.read_csv(get_train_data_path(experiment_name))

    y = train_data[config.target_column]
    X = train_data.drop(columns=[config.target_column])

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_SEED
    )

    max_features_value = None if config.max_features == "all" else config.max_features

    tree_params = {
        "max_depth": config.max_depth,
        "max_features": max_features_value,
    }
    tree_params = {k: v for k, v in tree_params.items() if v is not None}

    if config.ml_model == "Random Forest":
        model = RandomForestMSE(n_estimators=config.n_estimators, tree_params=tree_params)

    elif config.ml_model == "Gradient Boosting":
        model = GradientBoostingMSE(
            n_estimators=config.n_estimators,
            tree_params=tree_params,
            learning_rate=config.learning_rate,
        )

    history = model.fit(
        X_train.values, y_train.values,
        X_val.values, y_val.values,
        trace=True #, patience=10
    )

    model.dump(get_model_dir(experiment_name))

    if history and history.get("train") and len(history["train"]) > 0:
        with open(get_history_path(experiment_name), "w") as f:
            hist_bytes = TypeAdapter(ConvergenceHistory).dump_json(history)
            f.write(hist_bytes.decode("utf-8"))

    return {"message": f"Model for experiment '{experiment_name}' trained successfully."}


@app.get("/convergence_history/", response_model=ConvergenceHistoryResponse)
async def get_convergence_history(
    experiment_name: Annotated[str, Query()]
):
    path = get_history_path(experiment_name)

    with open(path, "r") as f:
        content = f.read()

        if not content.strip():
            return ConvergenceHistoryResponse(train=[0.0], val=[0.0])

        history = TypeAdapter(ConvergenceHistory).validate_json(content)
        return ConvergenceHistoryResponse(train=history["train"], val=history["val"])


@app.post("/predict/", response_model=PredictResponse)
async def predict(
    experiment_name: Annotated[str, Query()],
    request: Annotated[PredictRequest, Body()]
):
    with open(get_model_config_path(experiment_name), "r") as f:
        config = ExperimentConfig(**json.load(f))

    if config.ml_model == "Random Forest":
        model = RandomForestMSE.load(get_model_dir(experiment_name))
    elif config.ml_model == "Gradient Boosting":
        model = GradientBoostingMSE.load(get_model_dir(experiment_name))

    test_df = pd.DataFrame(request.data)

    predictions = model.predict(test_df.values)

    final_predictions = predictions.tolist() if hasattr(predictions, 'tolist') else list(predictions)

    return PredictResponse(predictions=final_predictions)
