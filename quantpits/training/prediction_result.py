"""Persist an already computed static prediction with Qlib's signal format."""

import pandas as pd


def save_prediction_result(prediction, dataset, recorder):
    """Save one inference result and native raw labels, without calling a model.

    Standard SignalRecord configurations are satisfied by these artifacts;
    their analysis dependents continue to load pred.pkl and label.pkl normally.
    """
    from qlib.data.dataset import DatasetH
    from qlib.workflow.record_temp import SignalRecord

    if isinstance(prediction, pd.Series):
        prediction = prediction.to_frame("score")
    elif not isinstance(prediction, pd.DataFrame):
        raise TypeError("prediction must be a pandas Series or DataFrame")

    signal = SignalRecord(dataset=dataset, recorder=recorder)
    signal.save(**{"pred.pkl": prediction})
    if isinstance(dataset, DatasetH):
        signal.save(**{"label.pkl": signal.generate_label(dataset)})
