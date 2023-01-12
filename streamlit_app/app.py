import pandas as pd
import streamlit as st

import dataset_import as data
import models
import plot_functions as plot

st.set_page_config(page_title="Temperature forecasting", page_icon="⛅", layout="centered",
                   initial_sidebar_state="collapsed", menu_items={
        'Get Help': 'https://github.com',
        'Report a bug': "https://github.com",
        'About': "App designed to forecast temperature of a station in Ireland using DeepAR and AR models. More infos"
                 "on the Git."
    })

st.title("Temperature forecasting using DeepAR")
stations = {'phoenix_park': '175', 'mace_head': '275', 'oak_park': '375', 'shannon_airport': '518',
            'dublin_airport': '532', 'ballyhaise': '675', 'sherkinisland': '775',
            'mullingar': '875', 'roches_point': '1075', 'newport': '1175', 'markree': '1275',
            'dunsany': '1375', 'gurteen': '1475', 'malin_head': '1575', 'johnstownii': '1775',
            'finner': '2075', 'claremorris': '2175', 'valentia_observatory': '2275',
            'belmullet': '2375', 'casement': '3723', 'cork_airport': '3904'}

station = st.selectbox('Choose a station', (stations.keys()))
df = data.data_import("stations/" + stations[station] + "_" + station + ".csv")
pred_deepar, summary_deepar, mse_deepar, computation_time_deepar = models.forecast_deepar(df)
pred_ar, summary_ar, mse_ar, computation_time_ar = models.forecast_ar(df)
st.pyplot(plot.final_plot(df, pred_ar, pred_deepar))
st.dataframe(pd.DataFrame(data=[[mse_ar, mse_deepar, computation_time_ar, computation_time_deepar]],
                          columns=["MSE AR", "MSE DeepAR", "Computation time AR (in seconds)",
                                   "Computation time DeepAR (in seconds)"]))
