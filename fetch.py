import streamlit as st
import yfinance as yf
import pandas as pd
import requests
from bs4 import BeautifulSoup

# Fetch S&P 500 tickers from Wikipedia
@st.cache_data
def fetch_sp500_tickers():
    url = 'https://en.wikipedia.org/wiki/List_of_S%26P_500_companies'
    response = requests.get(url)
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table', {'id': 'constituents'})

    tickers = []
    for row in table.findAll('tr')[1:]:
        ticker = row.findAll('td')[0].text.strip()
        tickers.append(ticker)

    return tickers

def fetch_company_names(tickers):
    company_data = []
    for ticker in tickers:
        ticker_info = yf.Ticker(ticker).info
        company_data.append({"Ticker": ticker, "Company Name": ticker_info.get('shortName', 'N/A')})
    return pd.DataFrame(company_data)

def show_companies():
	sp500_tickers = fetch_sp500_tickers()
	company_df = fetch_company_names(sp500_tickers[:50])
	st.subheader("Stocks")
	st.write(company_df)
