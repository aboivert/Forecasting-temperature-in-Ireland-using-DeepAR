import pandas as pd
import numpy as np
from datetime import datetime

from statsmodels.tsa.ar_model import AutoReg

from gluonts.model.deepar import DeepAREstimator
from gluonts.mx.trainer import Trainer
from gluonts.dataset.common import ListDataset
from gluonts.evaluation.backtest import make_evaluation_predictions

from sklearn.metrics import mean_squared_error

from streamlit_app import dataset_import as data

import mxnet as mx
np.random.seed(7)
mx.random.seed(7)

stations = {'phoenix_park': '175', 'mace_head': '275', 'oak_park': '375', 'shannon_airport': '518',
            'dublin_airport': '532', 'ballyhaise': '675', 'sherkinisland': '775',
            'mullingar': '875', 'roches_point': '1075', 'newport': '1175', 'markree': '1275',
            'dunsany': '1375', 'gurteen': '1475', 'malin_head': '1575', 'johnstownii': '1775',
            'finner': '2075', 'claremorris': '2175', 'valentia_observatory': '2275',
            'belmullet': '2375', 'casement': '3723', 'cork_airport': '3904'}

metrics = pd.DataFrame(columns=['mse_deepar', 'computation_time_deepar'])


for bs in [32, 64, 128]:
    for nl in [2,3]:
        for nc in [20,40]:
            for ep in [10, 20]:
                for cl in [36, 50, 100, 200]:
                    metrics = pd.DataFrame(columns=['mse_deepar', 'computation_time_deepar'])
                    for station in stations.keys():

                        print(station)
                        df = data.data_import("stations/" + stations[station] + "_" + station + ".csv")
                        computation_time_deepar = datetime.now()
                        training_data = ListDataset([{"start": str(df[:800].index[0]), "target": df[:800].temp,
                                                      "feat_dynamic_real": df[:800][["wetb", "dewpt", "vappr", "rhum"]]}], freq="d")
                        estimator = DeepAREstimator(freq="d",
                                                    prediction_length=36,
                                                    context_length=cl, batch_size = bs, num_layers = nl, num_cells= nc,
                                                    trainer=Trainer(epochs=ep))
                        predictor = estimator.train(training_data=training_data)
                        forecast_data = ListDataset(
                            [{"start": str(df[-200:].index[0]), "target": df[-200:].temp,
                              "feat_dynamic_real": df[-200:][["wetb", "dewpt", "vappr", "rhum"]]}],
                            freq="d"
                        )
                        forecast_it, ts_it = make_evaluation_predictions(forecast_data, predictor=predictor)
                        forecast = list(forecast_it)
                        total_fc_dar = np.zeros(36)
                        for i in range(len(forecast[0].samples)):
                            total_fc_dar += forecast[0].samples[i]
                        total_fc_dar = total_fc_dar / len(forecast[0].samples)

                        mse_deepar = mean_squared_error(df.temp[-36:], total_fc_dar)
                        computation_time_deepar = datetime.now() - computation_time_deepar

                        new_row = {"mse_deepar": mse_deepar,
                                   "computation_time_deepar": computation_time_deepar}
                        metrics = metrics.append(new_row, ignore_index=True)

                        print(metrics)

                        metrics.to_csv("bs"+str(bs)+"nl"+str(nl)+"nc"+str(nc)+"ep"+str(ep)+"cl"+str(cl)+".csv")


metrics_comp = pd.DataFrame(columns=['batch_size','num_layers','num_cells','epoch','context_length','mean_mse','mean_time'])
for bs in [32, 64, 128]:
    for nl in [2,3]:
        for nc in [20,40]:
            for ep in [10, 20]:
                for cl in [36, 50, 100, 200]:
                    metrics = pd.read_csv("bs"+str(bs)+"nl"+str(nl)+"nc"+str(nc)+"ep"+str(ep)+"cl"+str(cl)+".csv")
                    mean_mse_deepar = metrics.mse_deepar.mean()
                    mean_time_deepar = 0
                    for i in range(len(metrics)):
                        mean_time_deepar += pd.to_timedelta(metrics.computation_time_deepar[i]).seconds * 1e6 + pd.to_timedelta(
                            metrics.computation_time_deepar[i]).microseconds
                    mean_time_deepar /= (1e6 * len(metrics))
                    #print(" --- New configuration ---")
                    #print("Parameters")
                    #print("Batch_size : "+str(bs))
                    #print("Num_layers : "+str(nl))
                    #print("Num_cells : "+str(nc))
                    #print("Epochs : "+str(ep))
                    #print("Context_length : "+str(cl))
                    #print("Mean MSE DeepAR : " + str(mean_mse_deepar))
                    #print("Mean computation time DeepAR : " + str(mean_time_deepar))
                    #print(" ------------ ")
                    new_row = {'batch_size':bs,'num_layers':nl,'num_cells':nc,'epoch':ep,'context_length':cl,'mean_mse':mean_mse_deepar,'mean_time':mean_time_deepar}
                    metrics_comp = metrics_comp.append(new_row, ignore_index=True)
                    metrics_comp.to_csv("metrics.csv")
                    #print(metrics_comp)