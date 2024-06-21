import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime, date
import geopandas as gpd
#from pathlib import Path

st.set_page_config(layout="wide", page_title="TFT - Prediction Competition", page_icon="dove_of_peace")

logo_link = 'kompzkfe_logo.png'
with st.sidebar:
    st.logo(logo_link, link="https://www.unibw.de/ciss-en/kompz-kfe/")

st.sidebar.header("PRIOGrid-month Predictions")

st.title('Taking time seriously: Predicting conflict fatalities using temporal fusion transformers')

st.markdown(" **Authors**: <a href='https://jwalterskirchen.github.io/' style='text-decoration: none; color: black; font-weight: bold;'>Julian Walterskirchen</a> and <a href='https://www.unibw.de/ciss-en/kompz-kfe/team/sonja-haffner-m-sc' style='text-decoration: none; color: black; font-weight: bold;'>Sonja Häffner</a> and <a href='https://www.unibw.de/ciss/kompzkfe/personen/christian-oswald-m-a-1' style='text-decoration: none; color: black; font-weight: bold;'>Christian Oswald</a> and <a href='https://www.uni-bremen.de/en/institut-fuer-interkulturelle-und-internationale-studien/personen/wissenschaftliche-mitarbeiterinnen/marco-nicola-binetti' style='text-decoration: none; color: black; font-weight: bold;'>Marco N. Binetti</a>", unsafe_allow_html=True)

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
    return pd.read_parquet(data, engine='pyarrow')

@st.cache_data
def read_pgm_predictions(data):
    return gpd.read_parquet(data)

@st.cache_data
def read_csv(data):
    return pd.read_csv(data)

@st.cache_data
def read_geodf(data):
    return gpd.read_parquet(data)
@st.cache_data
def get_filtereddf(filter, metric, filteryear, color, threshold):
    df_filtered = geo_df[(geo_df['Model'] == filter) & (geo_df['Metric'] == metric) & (geo_df['Year'] == filteryear) & (geo_df[color] >= threshold)]
    return df_filtered

@st.cache_data
def get_historical(_df, filter):
    fig = px.choropleth(_df,
                        geojson=_df['geometry'],
                        locations=_df.index,
                        color=color,
                        hover_name=None,
                        color_continuous_scale=colorscale,
                        range_color=range,
                        projection="orthographic",
                        width=800,
                        height=600,
                        #animation_frame='Year',
                        hover_data={'Year':False,
                        f'{color}': ':.2f'},
                        title=f"{metric} for {filter}")
    fig.update_geos(fitbounds="locations", showcountries=True)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig


@st.cache_data
def predictionfig_mean(_df):
    fig = px.choropleth(_df,
                        geojson=_df['geometry'],
                        locations=_df.index,
                        color='outcome',
                        #hover_name="isoname",
                        color_continuous_scale=colorscale,
                        range_color=range_mean,
                        projection="orthographic",
                        width=800,
                        height=600,
                        #animation_frame='Date',
                        hover_data={'Date':False,
                        'outcome': ':.5f'},
                        labels={
                            'outcome': 'Predicted Fatalities',
                            'Date': 'Year & Month'
                        },
                        title=f"True future forecast")
    fig.update_geos(fitbounds="locations", showcountries=True)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig

@st.cache_data
def predictionfig_hi(_df):
    fig = px.choropleth(_df,
                        geojson=_df['geometry'],
                        locations=_df.index,
                        color='outcome',
                        #hover_name="isoname",
                        color_continuous_scale=colorscale,
                        range_color=range_hi,
                        projection="orthographic",
                        width=800,
                        height=600,
                        #animation_frame='Date',
                        hover_data={'Date':False,
                        'outcome': ':.5f'},
                        labels={
                            'outcome': 'Predicted Fatalities',
                            'Date': 'Year & Month'
                        },
                        title=f"True future forecast")
    fig.update_geos(fitbounds="locations", showcountries=True)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig

@st.cache_data
def predictionfig_lo(_df):
    fig = px.choropleth(_df,
                        geojson=_df['geometry'],
                        locations=_df.index,
                        color='outcome',
                        #hover_name="isoname",
                        color_continuous_scale=colorscale,
                        range_color=range_lo,
                        projection="orthographic",
                        width=800,
                        height=600,
                        #animation_frame='Date',
                        hover_data={'Date':False,'outcome': ':.5f'},
                        labels={
                            'outcome': 'Predicted Fatalities',
                            'Date': 'Year & Month'
                        },
                        title=f"True future forecast")
    fig.update_geos(fitbounds="locations", showcountries=True)
    fig.update_layout(margin=dict(t=0, b=0, l=0, r=0), height=400)
    return fig

geo_df = read_geodf('data/geo_df_pgm.parquet')
colorscale = 'Reds'
#countries = read_csv('countries.csv')
#predictions = load_geodataset('data/predictions_2024pgm.parquet')
#months = read_csv('data/month_ids.csv')

#months = months.drop(columns=["Unnamed: 4", "Unnamed: 5", "Unnamed: 6"])

#missing_months = pd.DataFrame({
#    'Date': ['2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01', '2025-06-01'],
#    'Month': [1, 2, 3, 4, 5, 6],
#    'Year': [2025, 2025, 2025, 2025, 2025, 2025],
#   'month_id': [541, 542, 543, 544, 545, 546]
#})


#months['Date'] = months['Date'].apply(lambda x: datetime.strptime(x, '%b-%y'))

#months = pd.concat([months, missing_months], ignore_index=True)
#months['Date'] = pd.to_datetime(months['Date']).dt.date

#predictions = predictions.merge(months, how='left', on='month_id')

priogrid = gpd.read_file(
        "shapefiles/priogrid.shp", engine="pyogrio"
    )

with tab1:
    filteryear = st.radio('Select Year', range(2018,2024), index=0, horizontal=True)
    metric = st.radio('Select evaluation metric', ('CRPS', 'IGN', 'MIS'), captions=('Continuous Rank Probability Score', 'Ignorance Score', 'Mean Interval Score'),
                  horizontal=True)

    if metric == 'CRPS':
        color = 'crps'
        range = [0.02,3]
        threshold = 0.02
    elif metric == 'IGN':
        color = 'ign'
        range = [0.03,3]
        threshold = 0.83
    elif metric == 'MIS':
        color = 'mis'
        range = [0.09,30]
        threshold = 0.09

    col1, col2, col3 = st.columns(3, gap='small')

    with col1:
        model = st.selectbox(
            "Select a Model:",
            ("TFT", "Boot", "Poisson", "Zero", "Conflictology" , "Conflictology N"), key='model1')

        df_filtered = get_filtereddf(model, metric, filteryear, color, threshold)
        fig = get_historical(df_filtered, model)

        st.plotly_chart(fig, use_container_width=False)

    with col2:
        model = st.selectbox(
            "Select a Model:",
            ("TFT", "Boot", "Poisson", "Zero", "Conflictology", "Conflictology N"), key='model2', index=1)

        df_filtered = get_filtereddf(model, metric, filteryear, color, threshold)
        fig = get_historical(df_filtered, model)

        st.plotly_chart(fig, use_container_width=False)

    with col3:
        model = st.selectbox(
            "Select a Model:",
            ("TFT", "Boot", "Poisson", "Zero", "Conflictology", "Conflictology N"), key='model3', index=2)

        df_filtered = get_filtereddf(model, metric, filteryear, color, threshold)
        fig = get_historical(df_filtered, model)

        st.plotly_chart(fig, use_container_width=False)


with tab2:
    filterdate = st.radio('Select Month', ('2024-07-01', '2024-08-01', '2024-09-01', '2024-10-01', '2024-11-01', '2024-12-01',
        '2025-01-01', '2025-02-01', '2025-03-01', '2025-04-01', '2025-05-01', '2025-06-01'), index=0, horizontal=True)

    ci = st.radio('Select Prediction Interval', (60, 70, 80, 90, 99), captions=('[40,60]', '[30,70]', '[20,80]', '[10,90]', '[1,99]'),
                  horizontal=True, index=3)

    if ci == 60:
        range_mean = [0, 3]
        range_lo = [0, 3]
        range_hi = [0, 3]
        threshold_mean = 0.04
        threshold_lo = 0.01
        threshold_hi = 0.04
    elif ci == 70:
        range_mean = [0, 3]
        range_lo = [0, 3]
        range_hi = [0, 3]
        threshold_mean = 0.04
        threshold_lo = 0.01
        threshold_hi = 0.04
    elif ci == 80:
        range_mean = [0, 3]
        range_lo = [0, 3]
        range_hi = [0, 3]
        threshold_mean = 0.04
        threshold_lo = 0.02
        threshold_hi = 0.04
    elif ci == 90:
        range_mean = [0, 3]
        range_lo = [0, 3]
        range_hi = [0, 3]
        threshold_mean = 0.04
        threshold_lo = 0
        threshold_hi = 0.09
    elif ci == 99:
        range_mean = [0, 3]
        range_lo = [0, 3]
        range_hi = [0, 3]
        threshold_mean = 0.04
        threshold_lo = 0
        threshold_hi = 0.09
    
    predictions_lo = read_pgm_predictions(f'data/pgm24_lo_{ci}_{filterdate}.parquet')
    predictions_lo = predictions_lo[(predictions_lo['outcome'] >= threshold_lo)]
    predictions_mean = read_pgm_predictions(f'data/pgm24_mean_{filterdate}.parquet')
    predictions_mean = predictions_mean[(predictions_mean['outcome'] >= threshold_mean)]
    predictions_hi = read_pgm_predictions(f'data/pgm24_hi_{ci}_{filterdate}.parquet')
    predictions_hi = predictions_hi[(predictions_hi['outcome'] >= threshold_hi)]

    col4, col5, col6 = st.columns(3, gap='small')

    with col4:
        if predictions_lo['outcome'].max() == 0:
            st.markdown('#### Lower Bound Prediction')
            st.write('Lower Bound Predictions are all 0 for this Prediction Interval!')
        else:
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
