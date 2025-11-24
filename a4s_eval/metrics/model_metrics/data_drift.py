from datetime import datetime

from a4s_eval.metric_registries.model_metric_registry import model_metric
from a4s_eval.data_model.measure import Measure
from a4s_eval.data_model.evaluation import DataShape, Dataset, Model
from a4s_eval.service.model_functional import FunctionalModel

# Evidently version 0.4.28 used because API change in the latest versions.
from evidently.report import Report
from evidently.metric_preset import DataDriftPreset


@model_metric(name="data_drift")
def data_drift(
    datashape: DataShape,
    model: Model,
    dataset: Dataset,
    functional_model: FunctionalModel,
) -> list[Measure]:
    """
    Data drift metric using Evidently's DataDriftPreset.

    Compute data drift between the model's training dataset (reference)
    and the current dataset using Evidently's DataDriftPreset.

    Returns:
    A single Measure containing the share of drifted columns.
    """

    reference_df = model.dataset.data
    current_df = dataset.data

    # build evidently rapport
    report = Report(metrics=[DataDriftPreset()])
    report.run(reference_data=reference_df, current_data=current_df)
    results = report.as_dict()

    result_dict = results["metrics"][0]["result"]

    # Check if dataset is considered as drifted or not
    drift_score = float(result_dict["share_of_drifted_columns"])

    return [
        Measure(
            name="data_drift",
            score=drift_score,
            time=datetime.now(),
        )
    ]
