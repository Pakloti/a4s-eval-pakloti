from datetime import datetime
import numpy as np
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
    Calculate model accuracy of the dataset.
    Accuracy = number of correct predictions / total number of predictions.
    
    Argumentss:
        datashape: describes which column is the target and which are features.
        model: the trained model object.
        dataset: contains the data as a pandas DataFrame.
        functional_model: standardized interface to make predictions.
    """

    df = dataset.data
    
    # Identify label and features from DataShape
    label_col = datashape.target.name
    feature_cols = [f.name for f in datashape.features]

    X = df[feature_cols]
    y_true = df[label_col]

    #Convert DataFrame to NumPy array to Torch Tensor
    X_np = X.to_numpy(dtype=np.float32, copy=False)
    X_tensor = torch.from_numpy(X_np)
    y_pred = functional_model.predict(X_tensor)

    y_pred = np.asarray(y_pred).flatten()
    y_true = np.asarray(y_true).flatten()

    # Calculate accuracy
    correct = np.sum(y_pred == y_true)
    total = len(y_true)
    accuracy_value = correct / total if total > 0 else 0.0

    # Timestamp
    current_time = datetime.now()
    
    return [Measure(name="accuracy", score=accuracy_value, time=current_time)]