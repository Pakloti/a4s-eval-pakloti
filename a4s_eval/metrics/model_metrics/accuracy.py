from datetime import datetime
import numpy as np
import pandas as pd
import torch

from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.data_model.measure import Measure
from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.service.model_functional import FunctionalModel


@model_metric(name="accuracy")
def accuracy(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> list[Measure]:
    
    """
    Compute accuracy = number of correct predictions / total number of predictions.
    """

    df = dataset.data
    
    # Identify label and features from DataShape
    label_col = datashape.target.name
    feature_cols = [f.name for f in datashape.features]

    # Extract X (features) and y (true labels)
    X = df[feature_cols]
    y_true = df[label_col]

    # 🔧 Convert DataFrame → NumPy array → Torch Tensor
    # Model expects float32 tensors, not pandas DataFrames.
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    X_tensor = torch.from_numpy(X_np)

    # Use the functional model to predict
    y_pred = functional_model.predict(X_tensor)

    # Ensure arrays for consistent comparison
    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    # Compute accuracy
    correct = np.sum(y_pred == y_true)
    total = len(y_true)
    accuracy_value = correct / total if total > 0 else 0.0

    # Record the timestamp
    current_time = datetime.now()

    # Return result as a list of Measure objects
    return [Measure(name="accuracy", score=accuracy_value, time=current_time)]