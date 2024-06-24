import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime

st.set_page_config(layout="wide", page_title="TFT - Prediction Competition", page_icon="earth_asia", initial_sidebar_state="collapsed")

logo_link = 'kompzkfe_logo.png'
with st.sidebar:
    st.logo(logo_link, link="https://www.unibw.de/ciss-en/kompz-kfe/")


st.sidebar.header("Country-month Predictions")

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

tab1, tab2 = st.tabs(["Historical Forecast", "True Future Forecast"])


@st.cache_data
def load_geodataset(data):
    return pd.read_parquet(data)

@st.cache_data
def read_csv(data):
    return pd.read_csv(data)
@st.cache_data
def get_filtereddf(filter, metric):
    df_filtered = geo_df[(geo_df['Model'] == filter) & (geo_df['Metric'] == metric)]
    return df_filtered

@st.cache_data
def get_figures(df, filter):
    fig = px.choropleth(df,
                        featureidkey="properties.ISO_A3",
                        locations="isoab",
                        color=color,
                        hover_name="isoname",
                        color_continuous_scale=colorscale,
                        range_color=range,
                        projection="orthographic",
                        width=800,
                        height=600,
                        animation_frame='Year',
                        hover_data={'Year':False,
                        f'{color}': ':.2f',
                        'isoab': False},
                        title=f"{metric} for {filter}")
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig


@st.cache_data
def predictionfig_mean(df):
    fig = px.choropleth(df,
                        featureidkey="properties.ISO_A3",
                        locations="isoab",
                        color='outcome',
                        hover_name="isoname",
                        color_continuous_scale=colorscale,
                        range_color=[0,200],
                        projection="orthographic",
                        width=800,
                        height=600,
                        animation_frame='Date',
                        hover_data={'Date':False,
                        'outcome': ':.2f',
                        'isoab': False},
                        labels={
                            'outcome': 'Predicted Fatalities',
                            'Date': 'Year & Month'
                        },
                        title=f"True future forecast")
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig

@st.cache_data
def predictionfig_hi(df):
    fig = px.choropleth(df,
                        featureidkey="properties.ISO_A3",
                        locations="isoab",
                        color='outcome',
                        hover_name="isoname",
                        color_continuous_scale=colorscale,
                        range_color=[0,200],
                        projection="orthographic",
                        width=800,
                        height=600,
                        animation_frame='Date',
                        hover_data={'Date':False,
                        'outcome': ':.2f',
                        'isoab': False},
                        labels={
                            'outcome': 'Predicted Fatalities',
                            'Date': 'Year & Month'
                        },
                        title=f"True future forecast")
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig

@st.cache_data
def predictionfig_lo(df):
    fig = px.choropleth(df,
                        featureidkey="properties.ISO_A3",
                        locations="isoab",
                        color='outcome',
                        hover_name="isoname",
                        color_continuous_scale=colorscale,
                        range_color=[0,200],
                        projection="orthographic",
                        width=800,
                        height=600,
                        animation_frame='Date',
                        hover_data={'Date':False,
                        'outcome': ':.2f',
                        'isoab': False},
                        labels={
                            'outcome': 'Predicted Fatalities',
                            'Date': 'Year & Month'
                        },
                        title=f"True future forecast")
    fig.update_geos(fitbounds="locations")
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig

colorscale = 'Reds'
geo_df = load_geodataset('data/results_df.parquet')
predictions = load_geodataset('data/predictions_2024.parquet')
countries = read_csv('data/countries.csv')
months = read_csv('data/month_ids.csv')

months = months.drop(columns=["Unnamed: 4", "Unnamed: 5", "Unnamed: 6"])

missing_months = pd.DataFrame({
    'Date': ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01', '2025-06-01'],
    'Month': [1, 2, 3, 4, 5, 6],
    'Year': [2025, 2025, 2025, 2025, 2025, 2025],
    'month_id': [541, 542, 543, 544, 545, 546]
})


months['Date'] = months['Date'].apply(lambda x: datetime.strptime(x, '%b-%y'))

months = pd.concat([months, missing_months], ignore_index=True)
months['Date'] = pd.to_datetime(months['Date']).dt.date

predictions = predictions.merge(months, how='left', on='month_id')

with tab1:
    with st.expander("Evaluation Metrics:"):
        st.write('''
            We evaluate model performance with three metrics: the Continuous Rank Probability Score (CRPS), the log or ignorance score (IGN), and Mean Interval Scores (MIS).
            CRPS measures accuracy and can be thought of as the mean absolute error equivalent for predictive distributions. Values get closer to 0 if the prediction distribution has low variance and is centered around actual values. 
            IGN is the log of the predictive density evaluated at the actual observation and complements CRPS. 
            It is less concerned with the uncertainty around a prediction or the distance between prediction and observation, but with the probability attributed to the actual event. 
            Lastly, MIS strikes a balance between having fairly narrow prediction intervals and good coverage rate.
            It focuses on the most likely values, penalizes increasing prediction interval size and rewards coverage.
            ''')
    with st.expander("VIEWS Benchmark Models:"):
        st.write('''
            We evaluate model performance with three metrics: the Continuous Rank Probability Score (CRPS), the log or ignorance score (IGN), and Mean Interval Scores (MIS).
            CRPS measures accuracy and can be thought of as the mean absolute error equivalent for predictive distributions. Values get closer to 0 if the prediction distribution has low variance and is centered around actual values. 
            IGN is the log of the predictive density evaluated at the actual observation and complements CRPS. 
            It is less concerned with the uncertainty around a prediction or the distance between prediction and observation, but with the probability attributed to the actual event. 
            Lastly, MIS strikes a balance between having fairly narrow prediction intervals and good coverage rate.
            It focuses on the most likely values, penalizes increasing prediction interval size and rewards coverage.
            ''')

    metric = st.radio('Select evaluation metric', ('CRPS', 'IGN', 'MIS'), captions=('Continuous Rank Probability Score', 'Ignorance Score', 'Mean Interval Score'),
                  horizontal=True)

    if metric == 'CRPS':
        color = 'crps'
        range = [0,100]
    elif metric == 'IGN':
        color = 'ign'
        range = [0,3]
    elif metric == 'MIS':
        color = 'mis'
        range = [0,1000]

    col1, col2, col3 = st.columns(3, gap='small')

    with col1:
        model = st.selectbox(
            "Select a Model:",
            ("TFT", "Boot", "Poisson", "Zero", "Conflictology"), key='model1')

        df_filtered = get_filtereddf(model, metric)
        fig = get_figures(df_filtered, model)

        st.plotly_chart(fig, use_container_width=False)

    with col2:
        model = st.selectbox(
            "Select a Model:",
            ("TFT", "Boot", "Poisson", "Zero", "Conflictology"), key='model2', index=1)

        df_filtered = get_filtereddf(model, metric)
        fig = get_figures(df_filtered, model)

        st.plotly_chart(fig, use_container_width=False)

    with col3:
        model = st.selectbox(
            "Select a Model:",
            ("TFT", "Boot", "Poisson", "Zero", "Conflictology"), key='model3', index=2)

        df_filtered = get_filtereddf(model, metric)
        fig = get_figures(df_filtered, model)

        st.plotly_chart(fig, use_container_width=False)


with tab2:

    ci = st.radio('Select Prediction Interval', (60, 70, 80, 90, 99), captions=('[40,60]', '[30,70]', '[20,80]', '[10,90]', '[1,99]'),
                  horizontal=True, index=3)

    predictions_mean = predictions.groupby(['country_id','Date'])['outcome'].mean()
    predictions_mean = predictions_mean.reset_index()

    predictions_mean = predictions_mean.merge(countries, how='left',
                                  left_on='country_id',
                                  right_on='id')


    predictions_hi = predictions.groupby(['country_id','Date'])[['outcome']].agg(lambda g: np.percentile(g, ci))
    predictions_hi = predictions_hi.reset_index()

    predictions_hi = predictions_hi.merge(countries, how='left',
                                  left_on='country_id',
                                  right_on='id')


    predictions_lo = predictions.groupby(['country_id','Date'])[['outcome']].agg(lambda g: np.percentile(g, (100-ci)))
    predictions_lo = predictions_lo.reset_index()

    predictions_lo = predictions_lo.merge(countries, how='left',
                                  left_on='country_id',
                                  right_on='id')


    col4, col5, col6 = st.columns(3, gap='small')

    with col4:
        st.markdown('#### Lower Bound Prediction')
        fig = predictionfig_lo(predictions_lo)
        st.plotly_chart(fig, use_container_width=False)

    with col5:
        st.markdown('#### Mean Prediction')
        fig = predictionfig_mean(predictions_mean)
        st.plotly_chart(fig, use_container_width=False)

    with col6:
        st.markdown('#### Upper Bound Prediction')
        fig = predictionfig_hi(predictions_hi)
        st.plotly_chart(fig, use_container_width=False)


