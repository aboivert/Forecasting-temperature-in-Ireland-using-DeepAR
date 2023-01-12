import streamlit as st

import dataset_import as data
import models
import plot_functions as plot

stations = {'phoenix_park': '175', 'mace_head': '275', 'oak_park': '375', 'shannon_airport': '518',
            'dublin_airport': '532', 'ballyhaise': '675', 'sherkinisland': '775',
            'mullingar': '875', 'roches_point': '1075', 'newport': '1175', 'markree': '1275',
            'dunsany': '1375', 'gurteen': '1475', 'malin_head': '1575', 'johnstownii': '1775',
            'finner': '2075', 'claremorris': '2175', 'valentia_observatory': '2275',
            'belmullet': '2375', 'casement': '3723', 'cork_airport': '3904'}

station = st.selectbox('Choose a station', (stations.keys()))
df = data.data_import("../stations/" + stations[station] + "_" + station + ".csv")
model = st.selectbox("Choose a model", ("AR", "DeepAR"))
if model == "AR":
    pred_ar, summary_ar, mse_ar, computation_time_ar = models.forecast_ar(df)
    st.pyplot(plot.plot_forecast(df, pred_ar, "AR"))
if model == "DeepAR":
    pred_deepar, summary_deepar,  mse_deepar, computation_time_deepar = models.forecast_deepar(df)
    st.pyplot(plot.plot_forecast(df, pred_deepar, "DeepAR"))
