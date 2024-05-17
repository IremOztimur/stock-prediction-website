import streamlit as st
from datetime import date

import yfinance as yf
from prophet import Prophet
from prophet.plot import plot_plotly
from plotly import graph_objs as go
import pandas as pd
import fetch

START = "2018-01-01"
TODAY = date.today().strftime("%Y-%m-%d")

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

with col2:
	selected_stock = st.text_input("Enter Stock Symbol", 'GOOG')
	n_years = st.slider("Years of prediction", 1, 4)
	period = n_years * 365
	data = load_data(selected_stock)
	st.divider()
	company_info = yf.Ticker(selected_stock).info
	st.subheader("Company Details")
	st.write("**Company Name:**", company_info.get('longName', 'N/A'))
	st.write("**Industry:**", company_info.get('industry', 'N/A'))
	st.write("**Sector:**", company_info.get('sector', 'N/A'))
	st.write("**Country:**", company_info.get('country', 'N/A'))
	st.write("**Market Cap:**", company_info.get('marketCap', 'N/A'))

st.subheader('Raw Data of {}'.format(selected_stock))
st.write(data.head())
st.subheader("Summaries of {} Data".format(selected_stock))
st.write(data.describe())

def plot_raw_data():
	fig = go.Figure()
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Open'], name='stock_open'))
	fig.add_trace(go.Scatter(x=data['Date'], y=data['Close'], name='stock_close', line=dict(color='red')))
	fig.layout.update(title_text='Time Series Data', xaxis_rangeslider_visible=True)
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

st.write("Forecast data")
fig1 = plot_plotly(model, forecast)
st.plotly_chart(fig1)

st.write("Forecast components")
fig2 = model.plot_components(forecast)
st.write(fig2)
