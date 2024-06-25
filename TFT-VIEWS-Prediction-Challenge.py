import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

st.set_page_config(layout="wide", page_title="TFT - Prediction Competition", page_icon="earth_americas")

logo_link = 'kompzkfe_logo.png'
with st.sidebar:
    st.logo(logo_link, link="https://www.unibw.de/ciss-en/kompz-kfe/")
    #st.markdown("[![KompZ KFE](app/static/kompzkfe_logo.png)](https://www.unibw.de/ciss-en/kompz-kfe/)", unsafe_allow_html=True)
    #st.title("Sections")
    #st.page_link("pages/country-month-predictions.py", label="Country Month Predictions")
    #st.page_link("pages/priogrid-month-predictions.py", label="PRIO-Grid Month Predictions")


st.title('Taking time seriously: Predicting conflict fatalities using temporal fusion transformers')

st.markdown(" **Authors**: <a href='https://jwalterskirchen.github.io/' style='text-decoration: none; color: black; font-weight: bold;'>Julian Walterskirchen</a> and <a href='https://www.unibw.de/ciss-en/kompz-kfe' style='text-decoration: none; color: black; font-weight: bold;'>Sonja Häffner</a> and <a href='https://www.unibw.de/ciss/kompzkfe/personen/christian-oswald-m-a-1' style='text-decoration: none; color: black; font-weight: bold;'>Christian Oswald</a> and <a href='https://www.uni-bremen.de/en/institut-fuer-interkulturelle-und-internationale-studien/personen/wissenschaftliche-mitarbeiterinnen/marco-nicola-binetti' style='text-decoration: none; color: black; font-weight: bold;'>Marco N. Binetti</a>", unsafe_allow_html=True)


st.markdown('''
     On this website, you can explore the Temporal Fusion Transformer (TFT) predictions for the historical test windows (2018-2023) and the true future forecast (July 2024-June 2025)
    that we submitted to the VIEWS Prediction Challenge (https://viewsforecasting.org/research/prediction-challenge-2023) in more detail. We provide
    evaluation metrics (CRPS, IGN, MIS) for all historical forecasts at both the country-month level and the PRIOGrid-level. Furthermore, model performance can be compared to the results
    achieved by several VIEWS benchmark models. For the true future forecasts, users can investigate our probabilistic predictions by adjusting the prediction interval level to obtain lower- and upper-bound predictions.
    Further details about the models, benchmarks, and evaluation metrics can be found in our preprint article at XYZ.com.
    ''')

with st.expander("Abstract"):
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
    st.image('header.png', width=600)
    
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

with st.expander("Bibliography"):
    st.markdown(
        """
        - Akiba, Takuya, Shotaro Sano, Toshihiko Yanase, Takeru Ohta & Masanori Koyama. 2019.
            Optuna: A Next-generation Hyperparameter Optimization Framework. In Proceedings
            of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data
            Mining. KDD ’19 New York, NY, USA: Association for Computing Machinery pp. 2623–
            2631.
        - Elsayed, Shereen, Daniela Thyssens, Ahmed Rashed, Hadi Samer Jomaa & Lars Schmidt-
            Thieme. 2021. “Do We Really Need Deep Learning Models for Time Series Forecasting?”.
        - Gleditsch, Kristian Skrede. 2022. “One without the Other? Prediction and Policy in International
            Studies.” International Studies Quarterly 66(3):sqac036.
        - Hegre, H˚avard & others. Forthcoming. “The 2023/24 VIEWS Prediction Competition.”
            Journal of Peace Research XXX.
        - Hegre, H˚avard, Paola Vesco & Michael Colaresi. 2022. “Lessons from an Escalation Prediction
            Competition.” International Interactions 48(4):521–554.
        - Huber, Peter J. 1964. “Robust Estimation of a Location Parameter.” The Annals of Mathematical
            Statistics 35(1):73–101.
        - Lim, Bryan, Sercan ¨ O. Arık, Nicolas Loeff & Tomas Pfister. 2021. “Temporal Fusion Transformers
            for Interpretable Multi-Horizon Time Series Forecasting.” International Journal
            of Forecasting 37(4):1748–1764.
        - Makridakis, Spyros, Evangelos Spiliotis, Vassilios Assimakopoulos, Artemios-Anargyros Semenoglou,
            Gary Mulder & Konstantinos Nikolopoulos. 2023. “Statistical, Machine
            Learning and Deep Learning Forecasting Methods: Comparisons and Ways Forward.”
            Journal of the Operational Research Society 74(3):840–859.
        - Olivares, Kin G., Cristian Chall´u, Federico Garza, Max Mergenthaler Canseco & Artur
            Dubrawski. 2022. “NeuralForecast: User Friendly State-of-the-Art Neural Forecasting
            Models.” PyCon Salt Lake City, Utah, US 2022.
        - Salinas, David, Valentin Flunkert, Jan Gasthaus & Tim Januschowski. 2020. “DeepAR:
            Probabilistic Forecasting with Autoregressive Recurrent Networks.” International Journal
            of Forecasting 36(3):1181–1191.
        - Vaswani, Ashish, Noam Shazeer, Niki Parmar, Jakob Uszkoreit, Llion Jones, Aidan N Gomez,
            Lukasz Kaiser & Illia Polosukhin. 2017. Attention Is All You Need. In Advances in
            Neural Information Processing Systems. Vol. 30 Curran Associates, Inc.
        """
        )
