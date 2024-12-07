import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import requests

st.set_page_config(layout="wide")
st.title("Stock Weekly Variation Heatmap")

# Data source functions
def descargar_datos_yfinance(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data
    except Exception as e:
        st.error(f"Error downloading data from yfinance for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_analisistecnico(ticker, start_date, end_date):
    try:
        # Ensure dates are in datetime.date format
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        # Rest of the function remains the same...

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'ChyrpSession': '0e2b2109d60de6da45154b542afb5768',
            'i18next': 'es',
            'PHPSESSID': '5b8da4e0d96ab5149f4973232931f033',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'dnt': '1',
            'referer': 'https://analisistecnico.com.ar/',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://analisistecnico.com.ar/services/datafeed/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if not all(key in data for key in ['t', 'c', 'o', 'h', 'l', 'v']):
                st.error(f"Incomplete data received for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c'],
                'Open': data['o'],
                'High': data['h'],
                'Low': data['l'],
                'Volume': data['v']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df[['Close']]  # Return only Close column for consistency
        else:
            st.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error downloading data from analisistecnico for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_iol(ticker, start_date, end_date):
    try:
        # Ensure dates are in datetime.date format
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'intencionApertura': '0',
            '__RequestVerificationToken': 'DTGdEz0miQYq1kY8y4XItWgHI9HrWQwXms6xnwndhugh0_zJxYQvnLiJxNk4b14NmVEmYGhdfSCCh8wuR0ZhVQ-oJzo1',
            'isLogged': '1',
            'uid': '1107644',
        }

        headers = {
            'accept': '*/*',
            'content-type': 'text/plain',
            'referer': 'https://iol.invertironline.com',
            'user-agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        symbol = ticker.replace('.BA', '')

        params = {
            'symbolName': symbol,
            'exchange': 'BCBA',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
            'resolution': 'D',
        }

        response = requests.get(
            'https://iol.invertironline.com/api/cotizaciones/history',
            params=params,
            cookies=cookies,
            headers=headers,
        )

        if response.status_code == 200:
            data = response.json()
            if data.get('status') != 'ok' or 'bars' not in data:
                st.error(f"Error in API response for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame(data['bars'])
            df['Date'] = pd.to_datetime(df['time'], unit='s')
            df['Close'] = df['close']
            df = df[['Date', 'Close']]
            df.set_index('Date', inplace=True)
            df = df.sort_index().drop_duplicates()
            return df
        else:
            st.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error downloading data from IOL for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_byma(ticker, start_date, end_date):
    try:
        # Ensure dates are in datetime.date format
        if isinstance(start_date, str):
            start_date = datetime.strptime(start_date, '%Y-%m-%d').date()
        elif isinstance(start_date, datetime):
            start_date = start_date.date()

        if isinstance(end_date, str):
            end_date = datetime.strptime(end_date, '%Y-%m-%d').date()
        elif isinstance(end_date, datetime):
            end_date = end_date.date()

        from_timestamp = int(datetime.combine(start_date, datetime.min.time()).timestamp())
        to_timestamp = int(datetime.combine(end_date, datetime.max.time()).timestamp())

        cookies = {
            'JSESSIONID': '5080400C87813D22F6CAF0D3F2D70338',
            '_fbp': 'fb.2.1728347943669.954945632708052302',
        }

        headers = {
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'de-DE,de;q=0.9,es-AR;q=0.8,es;q=0.7,en-DE;q=0.6,en;q=0.5,en-US;q=0.4',
            'Connection': 'keep-alive',
            'DNT': '1',
            'Referer': 'https://open.bymadata.com.ar/',
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/131.0.0.0 Safari/537.36',
        }

        # Remove .BA and add 24HS for BYMA format
        symbol = ticker.replace('.BA', '')
        if not symbol.endswith(' 24HS'):
            symbol = f"{symbol} 24HS"

        params = {
            'symbol': symbol,
            'resolution': 'D',
            'from': str(from_timestamp),
            'to': str(to_timestamp),
        }

        response = requests.get(
            'https://open.bymadata.com.ar/vanoms-be-core/rest/api/bymadata/free/chart/historical-series/history',
            params=params,
            cookies=cookies,
            headers=headers,
            verify=False
        )

        if response.status_code == 200:
            data = response.json()
            if not all(key in data for key in ['t', 'c']):
                st.error(f"Incomplete data received for {ticker}")
                return pd.DataFrame()

            df = pd.DataFrame({
                'Date': pd.to_datetime(data['t'], unit='s'),
                'Close': data['c']
            })
            df = df.sort_values('Date').drop_duplicates(subset=['Date'])
            df.set_index('Date', inplace=True)
            return df
        else:
            st.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()

    except Exception as e:
        st.error(f"Error downloading data from ByMA Data for {ticker}: {e}")
        return pd.DataFrame()

@st.cache_data(ttl=86400)
def fetch_stock_data(ticker, start_date, end_date, source='YFinance'):
    try:
        if source == 'YFinance':
            return descargar_datos_yfinance(ticker, start_date, end_date)
        elif source == 'AnálisisTécnico.com.ar':
            return descargar_datos_analisistecnico(ticker, start_date, end_date)
        elif source == 'IOL (Invertir Online)':
            return descargar_datos_iol(ticker, start_date, end_date)
        elif source == 'ByMA Data':
            return descargar_datos_byma(ticker, start_date, end_date)
        else:
            st.error(f"Unknown data source: {source}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error downloading data for {ticker} from {source}: {e}")
        return pd.DataFrame()
def calculate_weekly_variation(data):
    # Check if data is empty
    if data.empty:
        raise ValueError("No data available for the specified ticker and time range")

    # Ensure we have the Close column
    if 'Close' not in data.columns and not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Data does not contain required 'Close' column")

    # Extract just the 'Close' prices
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'].iloc[:, 0]  # Take first column of 'Close' level
    else:
        close_prices = data['Close']

    # Rest of the function remains the same...

    # Resample to weekly data and calculate variations
    weekly_data = close_prices.resample('W').last()  # Resample to the last day of each week

    # Get the last closing price of the previous year
    try:
        previous_year_last_day = close_prices.loc[:weekly_data.index[0] - pd.offsets.Week(1)].iloc[-1]
    except IndexError:
        # If no previous year's data is available, set the first week's change to 0%
        previous_year_last_day = None

    # Calculate percentage change
    weekly_variation = weekly_data.pct_change()

    # Set the first week's percentage change based on the previous year's last day
    if previous_year_last_day is not None:
        weekly_variation.iloc[0] = (weekly_data.iloc[0] - previous_year_last_day) / previous_year_last_day
    else:
        weekly_variation.iloc[0] = 0  # Default to 0% if no previous year's data is available

    return weekly_variation

def prepare_comparison_data(tickers, year, source):
    # Initialize an empty DataFrame to store weekly variations for all tickers
    comparison_data = pd.DataFrame()

    for ticker in tickers:
        # Fetch data for the entire year and the last week of the previous year
        start_date = f"{year - 1}-12-25"  # Start from the last week of the previous year
        end_date = f"{year}-12-31"
        stock_data = fetch_stock_data(ticker, start_date, end_date, source)

        # Calculate weekly variation
        weekly_variation = calculate_weekly_variation(stock_data)

        # Filter for the selected year and add to the comparison DataFrame
        comparison_data[ticker] = weekly_variation.loc[f"{year}-01-01":f"{year}-12-31"]

    # Ensure the index is consistent (weeks)
    comparison_data.index = comparison_data.index.strftime('Semana %U')  # Convert to week numbers

    return comparison_data

def plot_comparison_heatmap(data, title):
    # Clear any existing plots
    plt.clf()

    # Create figure with higher DPI and specific size
    fig = plt.figure(figsize=(10, 20), dpi=300)  # Adjusted size for vertical layout
    ax = plt.gca()

    # Create custom colormap (red to white to green)
    custom_cmap = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)

    # Find the maximum absolute value for symmetric color scaling
    max_abs_val = max(abs(data.min().min()), abs(data.max().max()))

    # Create the heatmap
    sns.heatmap(data,
                cmap=custom_cmap,
                center=0,
                vmin=-max_abs_val,
                vmax=max_abs_val,
                annot=True,
                fmt='.1%',
                annot_kws={'size': 8, 'weight': 'bold', 'family': 'Arial'},
                cbar_kws={'label': 'Weekly Variation', 'shrink': 0.8},
                square=False,
                ax=ax)

    # Customize the plot
    plt.title(title, pad=20, fontsize=16, weight='bold', family='Arial')
    ax.set_xlabel('Ticker', fontsize=12, family='Arial', weight='bold')
    ax.set_ylabel('Week Number', fontsize=12, family='Arial', weight='bold')

    # Create a secondary x-axis at the top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Get the tick positions and labels from the bottom axis
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(data.columns, rotation=45, ha='left')

    # Rotate bottom labels
    ax.set_xticklabels(data.columns, rotation=45, ha='right')

    # Customize tick labels size
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', which='major', labelsize=10)

    # Add quarter labels on the right side
    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    quarter_positions = [6.5, 19.5, 32.5, 45.5]
    ax3.set_yticks(quarter_positions)
    ax3.set_yticklabels(['Q1', 'Q2', 'Q3', 'Q4'],
                        fontsize=12,
                        weight='bold',
                        family='Arial')
    ax3.tick_params(length=0)

    # Add thick horizontal lines to separate quarters
    quarter_boundaries = [13, 26, 39]
    for boundary in quarter_boundaries:
        ax.hlines(y=boundary, xmin=0, xmax=data.shape[1],
                  colors='black', linestyles='solid', linewidth=2)

    # Add watermark
    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, weight='bold', family='Arial')

    # Adjust layout
    plt.tight_layout()

    return fig




def calculate_monthly_variation(data):
    # Extract just the 'Close' prices
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'].iloc[:, 0]
    else:
        close_prices = data['Close']

    # Convert to monthly data and calculate variations
    monthly_data = close_prices.resample('M').last()

    # Check if December data from the previous year exists
    try:
        previous_december = close_prices.loc[:monthly_data.index[0] - pd.offsets.MonthBegin(1)].iloc[-1]
    except IndexError:
        previous_december = None

    # Calculate percentage change
    monthly_variation = monthly_data.pct_change()

    # Set January's percentage change based on the previous December's value
    if previous_december is not None:
        monthly_variation.iloc[0] = (monthly_data.iloc[0] - previous_december) / previous_december
    else:
        monthly_variation.iloc[0] = 0

    return monthly_variation
def prepare_monthly_comparison_data(tickers, year, source):
    # Initialize an empty DataFrame to store monthly variations for all tickers
    comparison_data = pd.DataFrame()

    for ticker in tickers:
        # Fetch data for the entire year and the previous December
        start_date = f"{year - 1}-12-01"
        end_date = f"{year}-12-31"
        stock_data = fetch_stock_data(ticker, start_date, end_date, source)

        # Calculate monthly variation
        monthly_variation = calculate_monthly_variation(stock_data)

        # Filter for the selected year and add to the comparison DataFrame
        comparison_data[ticker] = monthly_variation.loc[f"{year}-01-01":f"{year}-12-31"]

    # Ensure the index is consistent (months)
    comparison_data.index = comparison_data.index.strftime('%b')

    return comparison_data

def plot_monthly_comparison_heatmap(data, title):
    # Clear any existing plots
    plt.clf()

    # Create figure with higher DPI and specific size
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()

    # Create custom colormap (red to white to green)
    custom_cmap = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)

    # Find the maximum absolute value for symmetric color scaling
    max_abs_val = max(abs(data.min().min()), abs(data.max().max()))

    # Create the heatmap
    sns.heatmap(data,
                cmap=custom_cmap,
                center=0,
                vmin=-max_abs_val,
                vmax=max_abs_val,
                annot=True,
                fmt='.1%',
                annot_kws={'size': 8, 'weight': 'bold', 'family': 'Arial'},
                cbar_kws={'label': 'Variación Mensual', 'shrink': 0.8},
                square=False,
                ax=ax)

    # Customize the plot
    plt.title(title, pad=20, fontsize=16, weight='bold', family='Arial')
    ax.set_xlabel('Ticker', fontsize=12, family='Arial', weight='bold')
    ax.set_ylabel('Mes', fontsize=12, family='Arial', weight='bold')

    # Create a secondary x-axis at the top
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())

    # Get the tick positions and labels from the bottom axis
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(data.columns, rotation=45, ha='left')

    # Rotate bottom labels
    ax.set_xticklabels(data.columns, rotation=45, ha='right')

    # Customize tick labels size
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', which='major', labelsize=10)

    # Add watermark
    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, weight='bold', family='Arial')

    # Adjust layout
    plt.tight_layout()

    return fig


def main():
    # Add data source selection
    data_sources = ['YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data']
    selected_source = st.sidebar.selectbox('Seleccionar Fuente de Datos', data_sources)

    # Add mode selection
    mode = st.radio("Selecciona el modo",
                    ["Un Ticker, Múltiples Años",
                     "Múltiples Tickers, Un Año (Cambios Semanales)",
                     "Múltiples Tickers, Un Año (Cambios Mensuales)"])

    if mode == "Un Ticker, Múltiples Años":
        with st.sidebar:
            ticker = st.text_input("Introduce el Ticker de la Acción", value="AAPL")
            start_date = st.date_input("Fecha de Inicio", value=pd.to_datetime("2017-01-01"))
            end_date = st.date_input("Fecha de Fin", value=pd.to_datetime("2019-12-31"))
            confirm_data = st.button("Confirmar Datos")

        if confirm_data:
            try:
                with st.spinner('Obteniendo y procesando datos...'):
                    stock_data = fetch_stock_data(ticker, start_date, end_date, selected_source)
                    weekly_df = calculate_weekly_variation(stock_data).to_frame(name='Variación')
                    weekly_df['Año'] = weekly_df.index.year
                    weekly_df['Semana'] = weekly_df.index.isocalendar().week
                    heatmap_data = weekly_df.pivot(index='Semana', columns='Año', values='Variación')

                    fig = plot_comparison_heatmap(heatmap_data, f'Heatmap de Variación Semanal para {ticker}')
                    st.pyplot(fig, dpi=300)

            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")
                st.info("Por favor, verifica si el símbolo del ticker es válido y si el rango de fechas es apropiado.")

    elif mode == "Múltiples Tickers, Un Año (Cambios Semanales)":
        with st.sidebar:
            tickers = st.text_input("Introduce los Tickers de las Acciones (separados por comas)", value="AAPL, MSFT, GOOGL")
            year = st.number_input("Selecciona el Año", min_value=2000, max_value=2024, value=2020, step=1)
            confirm_data = st.button("Confirmar Datos")

        if confirm_data:
            try:
                with st.spinner('Obteniendo y procesando datos...'):
                    ticker_list = [ticker.strip().upper() for ticker in tickers.split(",")]
                    comparison_data = prepare_comparison_data(ticker_list, year, selected_source)
                    fig = plot_comparison_heatmap(comparison_data, f'Comparación de Variación Semanal para {year}')
                    st.pyplot(fig, dpi=300)

            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")
                st.info("Por favor, verifica si los tickers son válidos y si el año es apropiado.")

    elif mode == "Múltiples Tickers, Un Año (Cambios Mensuales)":
        with st.sidebar:
            tickers = st.text_input("Introduce los Tickers de las Acciones (separados por comas)", value="AAPL, MSFT, GOOGL")
            year = st.number_input("Selecciona el Año", min_value=2000, max_value=2024, value=2020, step=1)
            confirm_data = st.button("Confirmar Datos")

        if confirm_data:
            try:
                with st.spinner('Obteniendo y procesando datos...'):
                    ticker_list = [ticker.strip().upper() for ticker in tickers.split(",")]
                    monthly_comparison_data = prepare_monthly_comparison_data(ticker_list, year, selected_source)
                    fig = plot_monthly_comparison_heatmap(monthly_comparison_data, f'Comparación de Variación Mensual para {year}')
                    st.pyplot(fig, dpi=300)

            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")
                st.info("Por favor, verifica si los tickers son válidos y si el año es apropiado.")

if __name__ == "__main__":
    main()
