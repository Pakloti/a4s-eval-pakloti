from datetime import datetime
from uuid import uuid4
import pandas as pd

from a4s_eval.data_model.evaluation import (
    Dataset,
    DataShape,
    Feature,
    FeatureType,
    Model,
)
from a4s_eval.service.model_functional import FunctionalModel
from a4s_eval.metrics.model_metrics.data_drift import data_drift


def test_data_drift_metric_multiple_runs():
    reference_df = pd.DataFrame(
        {
            "x": [1, 2, 3, 4],
            "y": [10, 20, 30, 40],
        }
    )

    current_df = pd.DataFrame(
        {
            "x": [1, 1, 1, 1],
            "y": [10, 20, 30, 40],
        }
    )

    features = []
    for col in reference_df.columns:
        features.append(
            Feature(
                pid=uuid4(),
                name=col,
                feature_type=FeatureType.FLOAT,
                min_value=float(reference_df[col].min()),
                max_value=float(reference_df[col].max()),
            )
        )

    reference_dataset = Dataset(
        pid=uuid4(),
        name="ref",
        data=reference_df,
        shape=DataShape(
            features=features,
            target=None,
            date=None,
        ),
        created_at=datetime.now(),
    )

    current_dataset = Dataset(
        pid=uuid4(),
        name="cur",
        data=current_df,
        shape=DataShape(
            features=features,
            target=None,
            date=None,
        ),
        created_at=datetime.now(),
    )

    dummy_model = Model(
        pid=uuid4(),
        model=None,
        dataset=reference_dataset,
    )

    functional_model = FunctionalModel(
        predict=lambda x: x,
        predict_proba=lambda x: x,
        predict_with_grad=lambda x: (x, x),
    )

    results = []
    for _ in range(20):
        out = data_drift(
            datashape=reference_dataset.shape,
            model=dummy_model,
            dataset=current_dataset,
            functional_model=functional_model,
        )
        results.extend(out)

    assert len(results) == 20
    for measure in results:
        assert measure.name == "data_drift"
        assert 0.0 <= measure.score <= 1.0
