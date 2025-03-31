import streamlit as st
import yfinance as yf
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from datetime import datetime
import requests
import re
from operator import truediv, mul

st.set_page_config(layout="wide")
st.title("Stock Weekly Variation Heatmap")

# Expression Parsing and Evaluation
def parse_expression(expression):
    """Parse a financial expression into components."""
    expression = expression.strip()
    
    # Base case: single ticker or constant
    if '/' not in expression and '*' not in expression and '(' not in expression:
        # Check if it's a ticker with a constant (e.g., GGAL*10)
        match = re.match(r'([A-Za-z.^]+)\*(\d+)', expression)
        if match:
            ticker, constant = match.groups()
            return {'type': 'operation', 'op': '*', 'left': {'type': 'ticker', 'value': ticker}, 
                    'right': {'type': 'constant', 'value': float(constant)}}
        return {'type': 'ticker', 'value': expression}
    
    # Handle parentheses: find the outermost pair
    if '(' in expression:
        # Find the outermost parentheses
        depth = 0
        start = -1
        for i, char in enumerate(expression):
            if char == '(':
                if depth == 0:
                    start = i
                depth += 1
            elif char == ')':
                depth -= 1
                if depth == 0:
                    # Found matching closing parenthesis
                    inner_expr = expression[start + 1:i]
                    inner = parse_expression(inner_expr)
                    # Replace the parenthetical expression with a placeholder
                    placeholder = f"[{inner['value']}]"
                    new_expression = expression[:start] + placeholder + expression[i + 1:]
                    return parse_expression(new_expression)
                elif depth < 0:
                    raise ValueError(f"Unmatched parenthesis in expression: {expression}")
        if depth > 0:
            raise ValueError(f"Unmatched parenthesis in expression: {expression}")
    
    # Find the last operator outside of parentheses to split on (right-to-left precedence)
    depth = 0
    split_op = None
    split_pos = -1
    for i in range(len(expression) - 1, -1, -1):
        char = expression[i]
        if char == ')':
            depth += 1
        elif char == '(':
            depth -= 1
        elif depth == 0 and char in ['/', '*']:
            split_op = char
            split_pos = i
            break
    
    if split_op:
        left_part = expression[:split_pos]
        right_part = expression[split_pos + 1:]
        if not left_part or not right_part:
            raise ValueError(f"Invalid expression: {expression}")
        left = parse_expression(left_part)
        right = parse_expression(right_part)
        return {'type': 'operation', 'op': split_op, 'left': left, 'right': right}
    
    raise ValueError(f"Unable to parse expression: {expression}")

def evaluate_expression(parsed, start_date, end_date, source):
    """Evaluate the parsed expression with stock data."""
    if parsed['type'] == 'ticker':
        data = fetch_stock_data(parsed['value'], start_date, end_date, source)
        if data.empty:
            raise ValueError(f"No data for ticker {parsed['value']}")
        return data['Close']
    elif parsed['type'] == 'constant':
        return parsed['value']
    elif parsed['type'] == 'operation':
        left_data = evaluate_expression(parsed['left'], start_date, end_date, source)
        right_data = evaluate_expression(parsed['right'], start_date, end_date, source)
        
        if isinstance(left_data, pd.Series) and isinstance(right_data, pd.Series):
            common_index = left_data.index.intersection(right_data.index)
            left_data = left_data.reindex(common_index)
            right_data = right_data.reindex(common_index)
        
        if parsed['op'] == '/':
            return truediv(left_data, right_data)
        elif parsed['op'] == '*':
            return mul(left_data, right_data)

# Data Source Functions
def descargar_datos_yfinance(ticker, start, end):
    try:
        stock_data = yf.download(ticker, start=start, end=end)
        return stock_data
    except Exception as e:
        st.error(f"Error downloading data from yfinance for {ticker}: {e}")
        return pd.DataFrame()

def calculate_ticker_ratio(data1, data2):
    if data1.empty or data2.empty:
        raise ValueError("One or both datasets are empty")
    common_dates = data1.index.intersection(data2.index)
    if len(common_dates) == 0:
        raise ValueError("No overlapping dates between the two tickers")
    data1 = data1.reindex(common_dates)
    data2 = data2.reindex(common_dates)
    if isinstance(data1.columns, pd.MultiIndex):
        close1 = data1['Close'].iloc[:, 0]
    else:
        close1 = data1['Close']
    if isinstance(data2.columns, pd.MultiIndex):
        close2 = data2['Close'].iloc[:, 0]
    else:
        close2 = data2['Close']
    ratio = pd.DataFrame({'Close': close1 / close2}, index=common_dates)
    return ratio

def descargar_datos_analisistecnico(ticker, start_date, end_date):
    try:
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
            return df[['Close']]
        else:
            st.error(f"Error fetching data for {ticker}: Status code {response.status_code}")
            return pd.DataFrame()
    except Exception as e:
        st.error(f"Error downloading data from analisistecnico for {ticker}: {e}")
        return pd.DataFrame()

def descargar_datos_iol(ticker, start_date, end_date):
    try:
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
    if data.empty:
        raise ValueError("No data available for the specified ticker and time range")
    if 'Close' not in data.columns and not isinstance(data.columns, pd.MultiIndex):
        raise ValueError("Data does not contain required 'Close' column")
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'].iloc[:, 0]
    else:
        close_prices = data['Close']
    weekly_data = close_prices.resample('W').last()
    try:
        previous_year_last_day = close_prices.loc[:weekly_data.index[0] - pd.offsets.Week(1)].iloc[-1]
    except IndexError:
        previous_year_last_day = None
    weekly_variation = weekly_data.pct_change()
    if previous_year_last_day is not None:
        weekly_variation.iloc[0] = (weekly_data.iloc[0] - previous_year_last_day) / previous_year_last_day
    else:
        weekly_variation.iloc[0] = 0
    return weekly_variation

def prepare_comparison_data(tickers_or_expressions, year, source):
    comparison_data = pd.DataFrame()
    start_date = f"{year - 1}-12-25"
    end_date = f"{year}-12-31"
    for expr in tickers_or_expressions:
        parsed = parse_expression(expr)
        try:
            result_series = evaluate_expression(parsed, start_date, end_date, source)
            stock_data = pd.DataFrame({'Close': result_series})
            weekly_variation = calculate_weekly_variation(stock_data)
            comparison_data[expr] = weekly_variation.loc[f"{year}-01-01":f"{year}-12-31"]
        except Exception as e:
            st.error(f"Error processing {expr}: {e}")
            comparison_data[expr] = pd.Series(dtype=float)
    comparison_data.index = comparison_data.index.strftime('Semana %U')
    return comparison_data

def plot_comparison_heatmap(data, title):
    plt.clf()
    fig = plt.figure(figsize=(10, 20), dpi=300)
    ax = plt.gca()
    custom_cmap = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)
    max_abs_val = max(abs(data.min().min()), abs(data.max().max()))
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
    plt.title(title, pad=20, fontsize=16, weight='bold', family='Arial')
    ax.set_xlabel('Ticker', fontsize=12, family='Arial', weight='bold')
    ax.set_ylabel('Week Number', fontsize=12, family='Arial', weight='bold')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(data.columns, rotation=45, ha='left')
    ax.set_xticklabels(data.columns, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', which='major', labelsize=10)
    ax3 = ax.twinx()
    ax3.set_ylim(ax.get_ylim())
    quarter_positions = [6.5, 19.5, 32.5, 45.5]
    ax3.set_yticks(quarter_positions)
    ax3.set_yticklabels(['Q1', 'Q2', 'Q3', 'Q4'],
                        fontsize=12,
                        weight='bold',
                        family='Arial')
    ax3.tick_params(length=0)
    quarter_boundaries = [13, 26, 39]
    for boundary in quarter_boundaries:
        ax.hlines(y=boundary, xmin=0, xmax=data.shape[1],
                  colors='black', linestyles='solid', linewidth=2)
    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, weight='bold', family='Arial')
    plt.tight_layout()
    return fig

def calculate_monthly_variation(data):
    if isinstance(data.columns, pd.MultiIndex):
        close_prices = data['Close'].iloc[:, 0]
    else:
        close_prices = data['Close']
    monthly_data = close_prices.resample('M').last()
    try:
        previous_december = close_prices.loc[:monthly_data.index[0] - pd.offsets.MonthBegin(1)].iloc[-1]
    except IndexError:
        previous_december = None
    monthly_variation = monthly_data.pct_change()
    if previous_december is not None:
        monthly_variation.iloc[0] = (monthly_data.iloc[0] - previous_december) / previous_december
    else:
        monthly_variation.iloc[0] = 0
    return monthly_variation

def prepare_monthly_comparison_data(tickers_or_expressions, year, source):
    comparison_data = pd.DataFrame()
    start_date = f"{year - 1}-12-01"
    end_date = f"{year}-12-31"
    for expr in tickers_or_expressions:
        parsed = parse_expression(expr)
        try:
            result_series = evaluate_expression(parsed, start_date, end_date, source)
            stock_data = pd.DataFrame({'Close': result_series})
            monthly_variation = calculate_monthly_variation(stock_data)
            comparison_data[expr] = monthly_variation.loc[f"{year}-01-01":f"{year}-12-31"]
        except Exception as e:
            st.error(f"Error processing {expr}: {e}")
            comparison_data[expr] = pd.Series(dtype=float)
    comparison_data.index = comparison_data.index.strftime('%b')
    return comparison_data

def plot_monthly_comparison_heatmap(data, title):
    plt.clf()
    fig = plt.figure(figsize=(10, 8), dpi=300)
    ax = plt.gca()
    custom_cmap = sns.diverging_palette(h_neg=10, h_pos=130, s=99, l=55, sep=3, as_cmap=True)
    max_abs_val = max(abs(data.min().min()), abs(data.max().max()))
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
    plt.title(title, pad=20, fontsize=16, weight='bold', family='Arial')
    ax.set_xlabel('Ticker', fontsize=12, family='Arial', weight='bold')
    ax.set_ylabel('Mes', fontsize=12, family='Arial', weight='bold')
    ax2 = ax.twiny()
    ax2.set_xlim(ax.get_xlim())
    ax2.set_xticks(ax.get_xticks())
    ax2.set_xticklabels(data.columns, rotation=45, ha='left')
    ax.set_xticklabels(data.columns, rotation=45, ha='right')
    ax.tick_params(axis='both', which='major', labelsize=10)
    ax2.tick_params(axis='x', which='major', labelsize=10)
    fig.text(0.5, 0.5, "MTaurus - X: @MTaurus_ok", fontsize=12, color='gray',
             ha='center', va='center', alpha=0.5, weight='bold', family='Arial')
    plt.tight_layout()
    return fig

def main():
    data_sources = ['YFinance', 'AnálisisTécnico.com.ar', 'IOL (Invertir Online)', 'ByMA Data']
    selected_source = st.sidebar.selectbox('Seleccionar Fuente de Datos', data_sources)
    mode = st.radio("Selecciona el modo",
                    ["Un Ticker/Ratio, Múltiples Años",
                     "Múltiples Tickers/Ratios, Un Año (Cambios Semanales)",
                     "Múltiples Tickers/Ratios, Un Año (Cambios Mensuales)"])

    if mode == "Un Ticker/Ratio, Múltiples Años":
        with st.sidebar:
            ticker_input = st.text_input("Introduce el Ticker o Expresión (ej: AAPL o METR.BA/(YPFD.BA/YPF))", value="AAPL")
            start_date = st.date_input("Fecha de Inicio", value=pd.to_datetime("2017-01-01"))
            end_date = st.date_input("Fecha de Fin", value=pd.to_datetime("2019-12-31"))
            confirm_data = st.button("Confirmar Datos")

        if confirm_data:
            try:
                with st.spinner('Obteniendo y procesando datos...'):
                    parsed = parse_expression(ticker_input)
                    result_series = evaluate_expression(parsed, start_date, end_date, selected_source)
                    stock_data = pd.DataFrame({'Close': result_series})
                    title = f'Heatmap de Variación Semanal para {ticker_input}'
                    weekly_df = calculate_weekly_variation(stock_data).to_frame(name='Variación')
                    weekly_df['Año'] = weekly_df.index.year
                    weekly_df['Semana'] = weekly_df.index.isocalendar().week
                    heatmap_data = weekly_df.pivot(index='Semana', columns='Año', values='Variación')
                    fig = plot_comparison_heatmap(heatmap_data, title)
                    st.pyplot(fig, dpi=300)
            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")
                st.info("Por favor, verifica si la expresión es válida y si el rango de fechas es apropiado.")

    elif mode in ["Múltiples Tickers/Ratios, Un Año (Cambios Semanales)", 
                  "Múltiples Tickers/Ratios, Un Año (Cambios Mensuales)"]:
        with st.sidebar:
            tickers = st.text_input(
                "Introduce los Tickers o Expresiones (separados por comas, ej: AAPL, ^MERV/(GGAL*10))", 
                value="AAPL, AAPL/MSFT"
            )
            year = st.number_input("Selecciona el Año", min_value=2000, max_value=2025, value=2020, step=1)
            confirm_data = st.button("Confirmar Datos")

        if confirm_data:
            try:
                with st.spinner('Obteniendo y procesando datos...'):
                    ticker_list = [ticker.strip() for ticker in tickers.split(",")]
                    if mode == "Múltiples Tickers/Ratios, Un Año (Cambios Semanales)":
                        comparison_data = prepare_comparison_data(ticker_list, year, selected_source)
                        fig = plot_comparison_heatmap(comparison_data, f'Comparación de Variación Semanal para {year}')
                    else:
                        monthly_comparison_data = prepare_monthly_comparison_data(ticker_list, year, selected_source)
                        fig = plot_monthly_comparison_heatmap(monthly_comparison_data, f'Comparación de Variación Mensual para {year}')
                    st.pyplot(fig, dpi=300)
            except Exception as e:
                st.error(f"Ocurrió un error: {str(e)}")
                st.info("Por favor, verifica si las expresiones son válidas y si el año es apropiado.")

if __name__ == "__main__":
    main()
