import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import matplotlib.pyplot as plt
import fetch

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")
line_colors = ['#1f77b4', '#ff7f0e', '#2ca02c']

st.title("Stock Prediction Website")
col1, col2 = st.columns(2)

with col1:
	with st.spinner("Fetching data..."):
		fetch.show_companies()

@st.cache_data
def load_data(ticker):
	data = yf.download(ticker, START, TODAY)
	data.reset_index(inplace=True)
	return data

@st.cache_data
def get_company_info(ticker):
    company_info = yf.Ticker(ticker).info
    return {
        "name": company_info.get('longName', 'N/A'),
        "industry": company_info.get('industry', 'N/A'),
        "sector": company_info.get('sector', 'N/A'),
        "country": company_info.get('country', 'N/A'),
        "market_cap": company_info.get('marketCap', 'N/A')
    }

with col2:
	selected_stock = st.text_input("Enter Stock Symbol", 'GOOG')
	n_years = st.slider("Years of prediction", 1, 4)
	period = n_years * 365
	data = load_data(selected_stock)
	company_info = get_company_info(selected_stock)

	st.divider()
	st.subheader("Company Details")
	st.write("**Company Name:**", company_info["name"])
	st.write("**Industry:**", company_info["industry"])
	st.write("**Sector:**", company_info["sector"])
	st.write("**Country:**", company_info["country"])
	st.write("**Market Cap:**", company_info["market_cap"])

st.subheader('Raw Data of {}'.format(selected_stock))
st.write(data.head())
st.subheader("Summaries of {} Data".format(selected_stock))
st.write(data.describe())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open', line=dict(color=line_colors[0])))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close', line=dict(color=line_colors[1]), opacity=0.7))
	fig.layout.update(title_text='Time Series Data',
				   xaxis_title='Date',
				   yaxis_title='Price',
				   xaxis_rangeslider_visible=True)
	st.plotly_chart(fig)

#The 100-day moving average (100-day MA)


ma100 = data['Close'].rolling(window=100).mean()

st.subheader("Closing Price vs Time Chart with 100MA")

ma100_explanation =  st.button("What is MA100?")

if ma100_explanation:
	st.write("""The 100-day moving average (100-day MA) is a technical analysis indicator used to help smooth out price data by creating a constantly updated average price over the last 100 days.""")
	st.write("The Formula is:")
	st.latex(r'''
100-day MA = \frac{\sum_{i=1}^{100} P_i}{100}
		''')

fig = go.Figure()
fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_open', line=dict(color=line_colors[0])))
fig.add_trace(go.Scatter(x=data['Date'], y=ma100, name='MA100', line=dict(color=line_colors[1])))
fig.update_layout(title='Closing Price vs Time Chart with 100MA',
                  xaxis_title='Date',
                  yaxis_title='Price',
                  xaxis_rangeslider_visible=True)
st.plotly_chart(fig)

plot_raw_data()

# Forecasting
df_train = data[['Date', 'Close']]
df_train = df_train.rename(columns={"Date": "ds", "Close": "y"})

model = Prophet()
model.fit(df_train)
future = model.make_future_dataframe(periods=period)
forecast = model.predict(future)

st.subheader("Forecast Data Table")
st.write(forecast.tail())

st.divider()

st.subheader("Forecast data")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.divider()

st.subheader("Forecast components")
fig2 = model.plot_components(forecast)
st.write(fig2)
