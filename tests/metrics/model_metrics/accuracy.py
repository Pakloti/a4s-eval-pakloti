from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.evaluation import Dataset, DataShape
from a4s_eval.metric_registries.model_metric_registry import Model, ModelMetric, FunctionalModel

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
    label_col = "charged_off"
    y_true = df[label_col]
    X= df.drop(columns=[label_col])
    
    if hasattr(model, "predict"):
        y_pred = model.predict(X)
    else:
        y_pred = functional_model.predict(X)
        
        
    correct=sum(int(p == t) for p,t in zip(y_pred, y_true))
    total=len(y_true)
    acc = (correct / total) if total else 0.0
    
    return [Measure(name="accuracy", value=acc, unit="ratio")]    