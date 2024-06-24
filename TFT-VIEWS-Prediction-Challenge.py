import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="TFT - Prediction Competition", page_icon="dove_of_peace")

logo_link = 'kompzkfe_logo.png'
with st.sidebar:
    st.logo(logo_link, link="https://www.unibw.de/ciss-en/kompz-kfe/")
    #st.markdown("[![KompZ KFE](app/static/kompzkfe_logo.png)](https://www.unibw.de/ciss-en/kompz-kfe/)", unsafe_allow_html=True)
    #st.title("Sections")
    #st.page_link("pages/country-month-predictions.py", label="Country Month Predictions")
    #st.page_link("pages/priogrid-month-predictions.py", label="PRIO-Grid Month Predictions")


st.title('Taking time seriously: Predicting conflict fatalities using temporal fusion transformers')

st.markdown(" **Authors**: <a href='https://jwalterskirchen.github.io/' style='text-decoration: none; color: black; font-weight: bold;'>Julian Walterskirchen</a> and <a href='https://www.unibw.de/ciss-en/kompz-kfe' style='text-decoration: none; color: black; font-weight: bold;'>Sonja Häffner</a> and <a href='https://www.unibw.de/ciss/kompzkfe/personen/christian-oswald-m-a-1' style='text-decoration: none; color: black; font-weight: bold;'>Christian Oswald</a> and <a href='https://www.uni-bremen.de/en/institut-fuer-interkulturelle-und-internationale-studien/personen/wissenschaftliche-mitarbeiterinnen/marco-nicola-binetti' style='text-decoration: none; color: black; font-weight: bold;'>Marco N. Binetti</a>", unsafe_allow_html=True)

st.markdown(f'''
**Abstract:**
Previous conflict forecasting efforts identified three areas for improvement: the importance of spatiotemporal dependencies and nonlinearities and the need 
to exploit the latent information contained in conflict variables further, a lack of interpretability in return for high accuracy of complex algorithms, and 
the need to quantify uncertainty around point forecasts. We predict fatalities with temporal fusion transformer models which have several desirable features 
for conflict forecasting and tackle all these points. First, they can produce multi-horizon forecasts and probabilistic predictions through quantile regression. 
This offers a flexible and non-parametric approach to estimate prediction uncertainty. Second, they can incorporate time-invariant covariates, known future inputs,
 and other exogenous time series which allows to identify globally important variables, persistent temporal patterns, and significant events for our prediction 
 problem. This mechanism makes them suitable to model both long-term and short-term dependencies. Third, this approach puts a strong focus on interpretability 
 such that we can investigate temporal dynamics more thoroughly via temporal self-attention decoders. Our models outperform an award-winning early warning
  system on several metrics and test windows and are thus a valuable addition to the forecaster's toolkit.
''')

with st.expander("VIEWS Prediction Challenge 2023/24"):
    st.write('''
        The models and results presented here document our contribution to the VIEWS Prediction Challenge 2023/2024. Financial support for the Prediction Challenge was provided by the German Ministry for Foreign Affairs.
        For more information on the Prediction Challenge please see Hegre et al. (Forthcoming) and https://viewsforecasting.org/research/prediction-challenge-2023 .
    ''')
    st.image('views_logo.png')
    
with st.expander("Temporal Fusion Transformer background"):
    st.write('''
        Temporal fusion transformers (TFTs) are attention-based deep neural networks designed to provide both good performance and interpretability.
        The main characteristic differentiating them from standard transformer models is that they specifically utilize the self-attention mechanism to identify complex temporal patterns in multiple time series.
        TFTs have additional desirable features for the purpose of conflict forecasting. First, a model can be trained on multiple multivariate time series. 
        Second, TFTs provide multi-horizon forecasts with prediction intervals. Third, they support different types of features such as time-variant and -invariant exogenous variables. 
        Fourth, they provide interpretability via variable importance, seasonality, and extreme event detection  (Lim, Arık, Loeff & Pfister 2021). 
        Lastly, TFTs, and deep learning models more generally, have been shown to outperform statistical, machine learning and other deep learning models, including for example gradient boosted trees and Deep Space-State Models
        (Elsayed, Thyssens, Rashed, Jomaa & Schmidt-Thieme 2021, Lim et al. 2021, Makridakis, Spiliotis, Assimakopoulos, Semenoglou, Mulder & Nikolopoulos 2023).
    ''')

col1, col2 = st.columns(2, gap='small')

with col1:
    st.subheader('CRPS overview - Country-Month')
    st.image('CRPS_overview.jpg')

with col2:
    st.subheader('CRPS overview - PRIOGrid-Month')
    st.image('CRPS_overview_pgm.jpg')


