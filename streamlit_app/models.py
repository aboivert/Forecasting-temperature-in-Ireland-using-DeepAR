from statsmodels.tsa.ar_model import AutoReg

from sklearn.metrics import mean_squared_error

from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
# from gluonts.evaluation import Evaluator
from gluonts.evaluation.backtest import make_evaluation_predictions

from datetime import datetime

import pandas as pd

import numpy as np
import mxnet as mx
np.random.seed(7)
mx.random.seed(7)


def forecast_deepar(data):
    computation_time = datetime.now()
    training_data = ListDataset([{"start": str(data[:800].index[0]), "target": data[:800].temp,
                                  "feat_dynamic_real": data[:800][["wetb", "dewpt", "vappr", "rhum"]]}], freq="d")
    estimator = DeepAREstimator(freq="d",
                                prediction_length=36,
                                context_length=50,
                                trainer=Trainer(epochs=10))
    predictor = estimator.train(training_data=training_data)
    forecast_data = ListDataset(
        [{"start": str(data[-200:].index[0]), "target": data[-200:].temp,
          "feat_dynamic_real": data[-200:][["wetb", "dewpt", "vappr", "rhum"]]}],
        freq="d"
    )
    forecast_it, ts_it = make_evaluation_predictions(forecast_data, predictor=predictor)
    forecast = list(forecast_it)
    tss = list(ts_it)
    # evaluator = Evaluator(quantiles=[0.5])
    # agg_metrics, item_metrics = evaluator(iter(tss), iter(forecast), num_series=len(forecast_data))
    # summary = agg_metrics
    summary = 0
    total_fc_dar = np.zeros(36)
    for i in range(len(forecast[0].samples)):
        total_fc_dar += forecast[0].samples[i]
    total_fc_dar = total_fc_dar / len(forecast[0].samples)
    pred = forecast
    mse = mean_squared_error(data.temp[-36:], total_fc_dar)
    computation_time = datetime.now() - computation_time
    computation_time = (pd.to_timedelta(computation_time).seconds * 1e6 +
                        pd.to_timedelta(computation_time).microseconds)/1e6
    return pred, summary, mse, computation_time


def forecast_ar(data):
    computation_time = datetime.now()
    res = AutoReg(data[800:964].temp, lags=None, exog=data[800:964][["wetb", "dewpt", "vappr", "rhum"]]).fit()
    pred = res.predict(start=data.index[-36], end=data.index[-1], exog_oos=data[-36:][["wetb", "dewpt", "vappr",
                                                                                       "rhum"]])
    summary = res.summary()
    mse = mean_squared_error(data.temp[-36:], pred)
    computation_time = datetime.now() - computation_time
    computation_time = (pd.to_timedelta(computation_time).seconds * 1e6 +
                        pd.to_timedelta(computation_time).microseconds)/1e6
    return pred, summary, mse, computation_time
