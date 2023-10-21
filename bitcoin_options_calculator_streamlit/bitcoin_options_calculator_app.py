# IMPORT KEY LIBRARIES 

import streamlit as st
import yfinance as yf
import datetime as dt
from datetime import datetime
from datetime import date
import pandas as pd
import numpy as np
from scipy.stats import norm
from scipy.optimize import brentq
import plotly.graph_objects as go
import plotly.io as pio
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import requests

# define custom CSS

st.markdown("""
<style>
.reportview-container .main .block-container {
    margin: 0 auto;
}
</style>
""", unsafe_allow_html=True)

# Title of the app
st.title("Bitcoin option strategy calculator")

# DEFINE KEY DICTIONARIES FOR MAPPING 

# Define the strategies dictionary
strategies = {
    'Bull Put Spread': {'sentiment': 'Bull', 'volatility': 'High'},
    'Short Put': {'sentiment': 'Bull', 'volatility': 'High'},
    'Put Broken Wing Butterfly': {'sentiment': 'Bull', 'volatility': 'High'},
    'Custom Naked Put': {'sentiment': 'Bull', 'volatility': 'High'},
    'Bull Call Spread': {'sentiment': 'Bull', 'volatility': 'Low'},
    'Call Calendar Spread': {'sentiment': 'Bull', 'volatility': 'Low'},
    'Call Backspread': {'sentiment': 'Bull', 'volatility': 'Low'},
    'Put Diagonal Spread': {'sentiment': 'Bull', 'volatility': 'Low'},
    'Short Straddle': {'sentiment': 'Neutral', 'volatility': 'High'},
    'Short Strangle': {'sentiment': 'Neutral', 'volatility': 'High'},
    'Short Iron Condor': {'sentiment': 'Neutral', 'volatility': 'High'},
    'Short Iron Butterfly': {'sentiment': 'Neutral', 'volatility': 'High'},
    'Bear Call Spread': {'sentiment': 'Bear', 'volatility': 'High'},
    'Short Call': {'sentiment': 'Bear', 'volatility': 'High'},
    'Call Broken Wing Butterfly': {'sentiment': 'Bear', 'volatility': 'High'},
    'Custom Naked Call': {'sentiment': 'Bear', 'volatility': 'High'},
    'Bear Put Spread': {'sentiment': 'Bear', 'volatility': 'Low'},
    'Put Calendar Spread': {'sentiment': 'Bear', 'volatility': 'Low'},
    'Put Backspread': {'sentiment': 'Bear', 'volatility': 'Low'},
    'Call Diagonal Spread': {'sentiment': 'Bear', 'volatility': 'Low'},
    'Long Put': {'sentiment': 'Bear', 'volatility': 'Low'}
}

#Define the strategy mapping dictionary

strategy_mapping = {
    "Bear Put Spread":
        [{"leg":1, "direction":"Long", "type":"Put", "size":1, "description":"Buy 1 ITM Put; Sell 1 OTM Put at Lower Strike Price. The combination of options should result in a net debit."},
        {"leg":2, "direction":"Short", "type":"Put", "size":1, "description":""}],

    "Long Call":
        [{"leg":1, "direction":"Long", "type":"Call", "size":1, "description":""}],

    "Long Put":
        [{"leg":1, "direction":"Long", "type":"Put", "size":1, "description":""}],

    "Short Call":
        [{"leg":1, "direction":"Short", "type":"Call", "size":1, "description":""}],

    "Short Put":
        [{"leg":1, "direction":"Short", "type":"Put", "size":1, "description":""}],

    "Bull Call Spread":
        [{"leg":1, "direction":"Long", "type":"Call", "size":1, "description":"Buy 1 ITM Call; Sell 1 OTM Call at Higher Strike Price. The combination of options should result in a net debit."},
        {"leg":2, "direction":"Short", "type":"Call", "size":1, "description":""}],

    "Bull Put Spread":
        [{"leg":1, "direction":"Long", "type":"Put", "size":1, "description":"Sell 1 OTM Put; Buy 1 OTM Put at Lower Strike Price. The combination of options should result in a net overall credit."},
        {"leg":2, "direction":"Short", "type":"Put", "size":1, "description":""}],

    "Short Naked Put":
        [{"leg":1, "direction":"Short", "type":"Put", "size":1, "description":"Sell 1 OTM Put below the market for a credit."}],

    "Put Broken Wing Butterfly":
        [{"leg":1, "direction":"Long", "type":"Put", "size":1, "description":"Buy 1 ITM Put; Sell 2 OTM Puts near the ATM options; Skip Strike and Buy 1 OTM Put. If possible the trade is done for a net credit."},
        {"leg":2, "direction":"Short", "type":"Put (x2)", "size":2, "description":""},
        {"leg":3, "direction":"Long", "type":"Put (skip 1 strike from put)", "size":1, "description":""}],

    "Custom Naked Put":
        [{"leg":1, "direction":"Short", "type":"Put", "size":1, "description":"Sell 1 OTM Put; Sell 1 OTM Call; Buy 1 OTM Call at Higher Strike Price. If possible the trade is done for a net credit wider than call spread width."},
        {"leg":2, "direction":"Short", "type":"Call", "size":1, "description":""},
        {"leg":3, "direction":"Long", "type":"Call (credit should be > call spread)", "size":1, "description":""}],

    "Call Calendar Spread":
        [{"leg":1, "direction":"Short", "type":"Call", "size":1, "description":"Sell 1 OTM Call in the Front Month; Buy 1 OTM Call in the Back Month at the Same Strike Price. The combination of options should result in a net debit."},
        {"leg":2, "direction":"Long", "type":"Call", "size":1, "description":""}],

    "Call Backspread":
        [{"leg":1, "direction":"Short", "type":"Call", "size":1, "description":"Sell 1 ATM Call; Buy 2 OTM Calls at Higher Strike Price. The combination of options should result in a net debit."},
        {"leg":2, "direction":"Long", "type":"Call (x2)", "size":2, "description":""}],

    "Put Diagonal Spread":
        [{"leg":1 , "direction":"Short", "type":"Put", "size":1, "description":"Sell 1 OTM Put in the Front Month; Buy 1 OTM Put in the Back Month at a Lower Strike Price. The combination of options should result in a net debit."},
        {"leg":2, "direction":"Long", "type":"Put", "size":1, "description":""}],

    "Short Straddle":
        [{"leg":1, "direction":"Short", "type":"Put", "size":1, "description":"Sell 1 ATM Put; Sell 1 ATM Call at Same Strike Price. The result of both sales is a net credit."},
        {"leg":2, "direction":"Short", "type":"Call", "size":1, "description":""}],
    "Short Strangle":    
        [{"leg":1, "direction":"Short", "type":"Put", "size":1, "description":"Sell 1 OTM Put; Sell 1 OTM Call at Far Out Strike Prices. The result of both sales is a net credit."},
        {"leg":2, "direction":"Short", "type":"Call", "size":1, "description":""}],

    "Short Iron Condor":
        [{"leg":1, "direction":"Long", "type":"Put", "size":1, "description":"Buy 1 OTM Put; Sell 1 OTM Put at Higher Strike; Sell 1 OTM Call; Buy 1 OTM Call at Higher Strike. The combination should yield a net credit."},
        {"leg":2, "direction":"Short", "type":"Put", "size":1, "description":""},
        {"leg":3, "direction":"Short", "type":"Call", "size":1, "description":""},
        {"leg":4, "direction":"Long", "type":"Call", "size":1, "description":""}],

    "Short Iron Butterfly":
        [{"leg":1, "direction":"Long", "type":"Put", "size":1, "description":"Buy 1 OTM Put at Lower Strike;Sell 1 ATM Put; Sell 1 ATM Call; Buy 1 OTM Call at Higher Strike. The combination should yield a net credit."},
        {"leg":2, "direction":"Short", "type":"Put", "size":1, "description":""},
        {"leg":3, "direction":"Short", "type":"Call", "size":1, "description":""},
        {"leg":4, "direction":"Long", "type":"Call", "size":1, "description":""}],

    "Bear Call Spread":
        [{"leg":1, "direction":"Short", "type":"Call", "size":1, "description":"Sell 1 OTM Call; Buy 1 OTM Call at Higher Strike Price. The combination of options should result in a net overall credit."},
        {"leg":2, "direction":"Long", "type":"Call", "size":1, "description":""}],

    "Call Broken Wing Butterfly":
        [{"leg":1, "direction":"Long", "type":"Call", "size":1, "description":"Buy 1 ITM Call; Sell 2 OTM Calls near the ATM options; Skip Strike and Buy 1 OTM Call. If possible the trade is done for a net credit."},
        {"leg":2, "direction":"Short", "type":"Call (x2)", "size":2, "description":""},
        {"leg":3, "direction":"Long", "type":"Call (skip 1 strike from previous)", "size":1, "description":""}],

    "Custom Naked Call":
        [{"leg":1, "direction":"Short", "type":"Call", "size":1, "description":"Sell 1 OTM Call; Sell 1 OTM Put; Buy 1 OTM Put at Lower Strike Price. If possible the trade is done for a net credit wider than call spread width."},
        {"leg":2, "direction":"Short", "type":"Put", "size":1, "description":""},
        {"leg":3, "direction":"Long", "type":"Put", "size":1, "description":""}],
    
    "Put Calendar Spread":
        [{"leg":1, "direction":"Short", "type":"Put", "size":1, "description":"Sell 1 OTM Put in the Front Month; Buy 1 OTM Put in the Back Month at the Same Strike Price. The combination of options should result in a net debit."},
        {"leg":2 , "direction":"Long", "type":"Put", "size":1, "description":""}],

    "Put Backspread":
        [{"leg":1, "direction":"Short", "type":"Put", "size":1, "description":"Sell 1 ATM Put; Buy 2 OTM Puts at Lower Strike Price. The combination of options should result in a net debit."},
        {"leg":2, "direction":"Long", "type":"Put (x2)", "size":2, "description":""}],

    "Call Diagonal Spread":
        [{"leg":1, "direction":"Short", "type":"Call", "size":1, "description":"Sell 1 OTM Call in the Front Month; Buy 1 OTM Call in the Back Month at a Higher Strike Price. The combination of options should result in a net debit."},
        {"leg":2, "direction":"Long", "type":"Call", "size":1, "description":""}],

    "Synthetic Long":
        [{"leg":1, "direction":"Long", "type":"Call", "size":1, "description":"Buy 1 ATM Call option and Sell 1 ATM Put option"},
        {"leg":2, "direction":"Short", "type":"Put", "size":1, "description":""}]
}

# IMPORT BITCOIN PRICE DATA

# Define the ticker symbol
tickerSymbol = 'BTC-USD'

# Get data for this ticker
tickerData = yf.Ticker(tickerSymbol)

# define start and today 
start = dt.date.today() - dt.timedelta(days=365)
today = dt.date.today()

# Get the historical prices for this ticker
tickerDf = tickerData.history(period='1d', start=start, end=today)

# See your data
tickerDf.tail(2)

# Reset the index
df = tickerDf.reset_index()

# Calculate daily returns
df['Return'] = df['Close'].pct_change()

# Calculate the logarithm of 'Return'
df['Log_Return'] = np.log(df['Return']*100)

# Calculate the 'return_v2' as the division of 'Close[i]' by 'Close[i-1]'
df['return_v2'] = df['Close'] / df['Close'].shift(1)

# Calculate the logarithm of 'Return'
df['Log_Return'] = np.log(df['return_v2'])

# calculate squared difference between log return and mean of log returns 

mean_log_returns = df['Log_Return'].mean()

df['log_return_squared_diff'] = (df['Log_Return']- mean_log_returns)**2

# calculate annual volatility

log_return_squared_diff_sum = df['log_return_squared_diff'].sum()

volatility = np.sqrt(log_return_squared_diff_sum/363)

# calculate 7-day volatility 

volatility_7d = volatility * np.sqrt(7)

# calculate 10-day volatility 

volatility_10d = volatility * np.sqrt(10)

# calculate 14-day volatility 

volatility_14d = volatility * np.sqrt(14)

# calculate 365-day volatility 

annualized_volatility = volatility * np.sqrt(365)

# Convert the "Date" column to datetime format
df['Date'] = pd.to_datetime(df['Date']).dt.date

# IMPORT RISK FREE RATE

from datetime import datetime, timedelta

# Define the ticker symbol
ticker = '^IRX'

# Set the start and end dates
# define start and today 
start = dt.date.today() - dt.timedelta(days=365)
today = dt.date.today()

# Fetch the historical data
risk_free_df = yf.download(ticker, start=start, end=today)

# Reset the index
risk_free_df = risk_free_df.reset_index()

#keep only "close"

risk_free_df = risk_free_df[['Date', 'Close']]

#rename columns
risk_free_df = risk_free_df.rename(columns={"Close": "risk_free_rate"})

# Convert the "Date" column to datetime format
risk_free_df['Date'] = pd.to_datetime(risk_free_df['Date']).dt.date

risk_free_rate = risk_free_df['risk_free_rate'].iloc[-1:]

risk_free_rate = risk_free_rate.values

risk_free_rate = risk_free_rate[0]

#####
#  ASK FOR USER INPUT TO START CALCULATION
#####


# User input for the strategy   
strategy = st.selectbox('Enter your chosen strategy:', list(strategies.keys()))

# User input for the expiry date
expiry_date_input = st.date_input('Enter the expiry date:', value=datetime.now() + timedelta(days=30))
expiry_date = expiry_date_input
days_to_expiry = (expiry_date - date.today()).days

# User input for the second expiry date for calendar/diagonal spreads
expiry_date2 = np.nan
days_to_expiry2 = None
if 'Calendar' in strategy or 'Diagonal' in strategy:
    expiry_date2_input = st.date_input('Enter the second expiry date for calendar/diagonal spreads:', value=datetime.now() + timedelta(days=60))
    expiry_date2 = expiry_date2_input
    days_to_expiry2 = (expiry_date2 - date.today()).days

# Pulling dividend yield
try:
    dividend_yield = df['Dividends'].sum()/df['Close'].iloc[-1]
except Exception as e:
    st.write("Error fetching dividend yield: ", e)
    dividend_yield = 0.0

# Pulling 3-month Treasury bill rate
try:
    risk_free_rate = risk_free_rate
except Exception as e:
    st.write("Error fetching 3-month Treasury bill rate: ", e)
    risk_free_rate = st.number_input('Enter the risk-free interest rate:', value=0.0, format='%f')

# User input for the contract size
contract_size = st.number_input('Enter the contract size (number of shares/units):', value=1, format='%d')

# User input for the strike prices
user_input = st.text_input('Enter the strike price/s in USD (separated by commas when entering more than one strike):', value=30000)
strike_prices = [int(price) for price in user_input.split(',') if price.strip().isdigit()]

# User input for the transaction costs
transaction_costs = st.number_input('Enter the transaction costs in USD:', value=0.0, format='%f')

# You can add a button to trigger the calculations or continue to next steps
if st.button('Produce P&L chart'):

    ##############
    # DEFINE A FUNCTION TO BUILD THE INSTRUMENT NAMES TO SEARCH DERIBIT  

    # define the function to build the instrument names to search for in Deribit 

    def generate_instrument_names(strategy, expiry_date, expiry_date2, strike_prices):
        # Validate input
        assert strategy in strategy_mapping, f"Unknown strategy: {strategy}"
        assert len(strike_prices) >= len(strategy_mapping[strategy]), f"Mismatch in number of legs for strategy {strategy}"
        
        # Format expiry dates
        expiry_date_str = expiry_date.strftime('%d%b%y').upper()
        
        # Initialize expiry_date2_str to None
        expiry_date2_str = None
        
        # Only format expiry_date2 if it's provided and the strategy is a calendar or diagonal spread
        if expiry_date2 and ('Calendar' in strategy or 'Diagonal' in strategy):
            expiry_date2_str = expiry_date2.strftime('%d%b%y').upper()
        
        # Generate instrument names
        instrument_names = []
        for i, leg in enumerate(strategy_mapping[strategy]):
            leg_type = leg["type"].split(" ")[0]  # Extract the option type (Call or Put)
            strike_price = strike_prices[i]
            # Use expiry_date2 for the second leg if provided, else use expiry_date
            expiry_str = expiry_date2_str if i == 1 and expiry_date2_str else expiry_date_str
            instrument_name = f'BTC-{expiry_str}-{strike_price}-{leg_type[0]}'
            instrument_names.append(instrument_name)
        
        return instrument_names

    #generate instrument names 

    instrument_names = generate_instrument_names(strategy, expiry_date, expiry_date2, strike_prices)

    # check the latest prices for the relevant contracts in deribit 

    from datetime import date, datetime as dt, timedelta
    import time

    def datetime_to_timestamp(datetime_obj): 
        """Converts a datetime object to a Unix timestamp in milliseconds."""
        return int(dt.timestamp(datetime_obj)*1000)

    def trades_by_instruments(instrument_names: list, count: int = 10000) -> dict:
        """Returns trade data for specified instruments over the last day.

        Args:
            instrument_names (list): The names of the instruments, e.g. ['BTC-10NOV23-30000-C', ...].
            count (int, optional): The maximum number of trades to retrieve per request. Defaults to 10000.

        Returns:
            dict: A dictionary where keys are instrument names and values are dataframes of trade data for the specified instrument over the last day.
        """

        # Validate input arguments
        assert isinstance(instrument_names, list), "instrument_names must be a list"
        assert all(isinstance(name, str) for name in instrument_names), "All instrument names must be strings"
        
        trades_data_dict = {}
        url = 'https://history.deribit.com/api/v2/public/get_last_trades_by_instrument'

        # Set date range to be today minus 1 day to today
        end_date = dt.now().date()
        start_date = end_date - timedelta(days=1)
        
        # Loop through each instrument name
        for instrument_name in instrument_names:
            trades_list = []
            
            params = {
                "instrument_name": instrument_name,
                "count": count,
                "start_timestamp": datetime_to_timestamp(dt.combine(start_date, dt.min.time())),
                "end_timestamp": datetime_to_timestamp(dt.combine(end_date, dt.max.time()))
            }
            # Use a session object to make requests to the API endpoint
            with requests.Session() as session:
                response = session.get(url, params=params)
                response_data = response.json()
                trades_list.extend(response_data["result"]["trades"])
            
            # Create a pandas dataframe from the trade data
            trades_data = pd.DataFrame(trades_list)
            if len(trades_data) > 0:
                trades_data["date_time"] = pd.to_datetime(trades_data["timestamp"], unit='ms')
            trades_data_dict[instrument_name] = trades_data
            
            # Delay between requests to avoid overwhelming the API
            time.sleep(2)  # 2 second delay, adjust as needed

        return trades_data_dict

    # pull the latest contracts

    trades_data_dict = trades_by_instruments(instrument_names)

    # create a dataframe with the latest transactions for the relevant contracts

    # Add 'instrument_name' column to each DataFrame and store them in a list
    dataframes = [df.assign(instrument_name=name) for name, df in trades_data_dict.items()]

    # Concatenate all DataFrames into one DataFrame
    consolidated_data = pd.concat(dataframes, ignore_index=True)

    # Sort the consolidated DataFrame by 'date_time' in ascending order
    trades_data = consolidated_data.sort_values(by='date_time', ascending=True).reset_index(drop=True)

    # get the latest option prices from Deribit

    option_prices_deribit = []

    for instrument_name in instrument_names:
        # Filter the data for the current instrument
        instrument_data = trades_data[trades_data['instrument_name'] == instrument_name]
        
        # Ensure there is data for the current instrument
        if not instrument_data.empty:
            # Get the latest trade data for the current instrument
            latest_trade_data = instrument_data.iloc[-1]
            
            # Calculate the option price in USD
            option_price_usd = latest_trade_data['price'] * latest_trade_data['index_price']
            
            # Append the option price to the option_prices_deribit list
            option_prices_deribit.append(option_price_usd)


    ######
    # CREATE P&L TABLE AND CHARTS WITH OPTION PRICES FROM DERIBIT 
    ######

    # define strike prices from UI

    option_prices = option_prices_deribit

    # define your high_value, low_value and increment_steps
    high_value = trades_data['index_price'].iloc[-1] * (1 + annualized_volatility)
    low_value = max(trades_data['index_price'].iloc[-1] * (1 - annualized_volatility), 0.01)
    range_diff = high_value - low_value

    if range_diff < 500:
        increment_steps = 0.2
    elif range_diff < 5000:
        increment_steps = 1
    else:
        increment_steps = 5

    # create an array for your price_range using np.arange
    price_range = np.arange(low_value, high_value + increment_steps, increment_steps)

    # create a DataFrame using this array
    pl_table = pd.DataFrame(price_range, columns=['price_range'])

    # calculate cost_contract column

    pl_table['cost_contract'] = pl_table['price_range'] * contract_size

    # add additional columns
    columns = ['difference_leg_1', 'difference_leg_2', 'difference_leg_3', 'difference_leg_4', 'cost', 'profit', 'leg2diagonalForward', 'leg2diagonald1', 'leg2diagonald2', 'leg2diagonalcumulative_norm_d1', 'leg2diagonalcumulative_norm_d2', 'leg2diagonalcall_option', 'leg2diagonalput_option', 'leg2diagonaloption_sell']
    for column in columns:
        pl_table[column] = np.nan

    #define the leg size based on the strategy_mapping 

    leg_size = strategy_mapping[strategy][0]['size']  

    #define a function to fill difference_leg_1

    def calculate_difference_leg1(row):
        price_range = row['price_range']
        if strategy == "Bear Put Spread":
            return max(0, (strike_prices[0] - price_range)) * leg_size * contract_size
        elif strategy == "Long Call":
            return max(0, (price_range - strike_prices[0])) * leg_size * contract_size
        elif strategy == "Long Put":
            return max(0, ((strike_prices[0]*contract_size) - row['cost_contract'])) * leg_size if strike_prices[0]-price_range>=0 else 0
        elif strategy == "Short Call":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Short Put":
            return max(0, ((strike_prices[0]*contract_size) - row['cost_contract'])) * leg_size if strike_prices[0]-price_range>=0 else 0
        elif strategy == "Bear Put Spread":
            return max(0, (strike_prices[0] - price_range)) * leg_size * contract_size
        elif strategy == "Long Call":
            return max(0, (price_range - strike_prices[0])) * leg_size * contract_size
        elif strategy == "Long Put":
            return max(0, ((strike_prices[0]*contract_size) - row['cost_contract'])) * leg_size if strike_prices[0]-price_range>=0 else 0
        elif strategy == "Short Call":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Bull Call Spread":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Short Iron Condor":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Short Iron Butterfly":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Bear Call Spread":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Bull Put Spread":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Short Straddle":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Short Strangle":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Put Broken Wing Butterfly":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Custom Naked Put":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Put Diagonal Spread":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Call Diagonal Spread":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Call Calendar Spread":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Put Calendar Spread":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Custom Naked Call":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Call Broken Wing Butterfly":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Call Backspread":
            return max(0, (price_range - strike_prices[0])) * contract_size * leg_size
        elif strategy == "Put Backspread":
            return max(0, (strike_prices[0] - price_range)) * contract_size * leg_size
        elif strategy == "Synthetic Long":
            return max(0, (price_range - strike_prices[0])) * leg_size * contract_size if strike_prices[0]-price_range<0 else 0
        else:
            return np.nan  

    pl_table['difference_leg_1'] = pl_table.apply(calculate_difference_leg1, axis=1)

    # calculate difference leg 2 

    def calculate_difference_leg2(row):
        price_range = row['price_range']

        # Check if the strategy has a third leg, if not, assign leg_size to 0
        try:
            leg_size = strategy_mapping[strategy][1]['size']  # try to access the second leg
        except IndexError:
            leg_size = 0  # default value if the third leg doesn't exist
            
        if strategy == "Bear Put Spread":
            return max(0, ((strike_prices[1]*contract_size) - row['cost_contract'])) * leg_size if ((strike_prices[1]*contract_size) - row['cost_contract'])>0 else 0
        elif strategy == "Bull Call Spread":
            return max(0, (price_range - strike_prices[1])) * leg_size * contract_size
        elif strategy == "Short Iron Condor":
            return max(0, ((strike_prices[1]*contract_size) - row['cost_contract'])) * leg_size
        elif strategy == "Short Iron Butterfly":
            return max(0, ((strike_prices[1]*contract_size) - row['cost_contract'])) * leg_size
        elif strategy == "Bear Call Spread":
            return max(0, (row['cost_contract'] - (strike_prices[1]*contract_size))) * leg_size
        elif strategy == "Bull Put Spread":
            return max(0, ((strike_prices[1]*contract_size) - row['cost_contract'])) * leg_size
        elif strategy == "Short Straddle":
            return max(0, (price_range - strike_prices[1])) * contract_size * leg_size
        elif strategy == "Short Strangle":
            return max(0, (price_range - strike_prices[1])) * contract_size * leg_size
        elif strategy == "Put Broken Wing Butterfly":
            return max(0, (strike_prices[1] - price_range)) * contract_size * leg_size
        elif strategy == "Custom Naked Put":
            return max(0, (price_range - strike_prices[1])) * contract_size * leg_size
        elif strategy == "Put Diagonal Spread":
            return max(0, (strike_prices[1] - price_range)) * contract_size * leg_size
        elif strategy == "Custom Naked Call":
            return max(0, (strike_prices[1]-price_range)) * contract_size * leg_size
        elif strategy == "Call Broken Wing Butterfly":
            return max(0, (price_range - strike_prices[1])) * contract_size * leg_size
        elif strategy == "Call Backspread":
            return max(0, (price_range - strike_prices[1])) * contract_size * leg_size
        elif strategy == "Put Backspread":
            return max(0, (strike_prices[1] - price_range)) * contract_size * leg_size
        elif strategy == "Synthetic Long":
            return max(0, ((strike_prices[1]*contract_size) - row['cost_contract'])) * leg_size if strike_prices[1] - price_range>0 else 0
        else:
            return np.nan  

    pl_table['difference_leg_2'] = pl_table.apply(calculate_difference_leg2, axis=1)

    # calculate difference leg 3

    def calculate_difference_leg3(row):
        price_range = row['price_range']
        
        # Check if the strategy has a third leg, if not, assign leg_size to 0
        try:
            leg_size = strategy_mapping[strategy][2]['size']  # try to access the third leg
        except IndexError:
            leg_size = 0  # default value if the third leg doesn't exist
        
        if strategy == "Short Iron Condor":
            return max(0, (row['cost_contract'] - strike_prices[2]*contract_size)) * leg_size
        elif strategy == "Short Iron Butterfly":
            return max(0, (row['cost_contract'] - strike_prices[2]* contract_size)) * leg_size
        elif strategy == "Put Broken Wing Butterfly":
            return max(0, (strike_prices[2] - price_range)) * contract_size * leg_size
        elif strategy == "Custom Naked Put":
            return max(0, (price_range - strike_prices[2])) * contract_size * leg_size
        elif strategy == "Custom Naked Call":
            return max(0, (strike_prices[2] - price_range)) * contract_size * leg_size
        elif strategy == "Call Broken Wing Butterfly":
            return max(0, (price_range - strike_prices[2])) * contract_size * leg_size
        else:
            return np.nan
        
    pl_table['difference_leg_3'] = pl_table.apply(calculate_difference_leg3, axis=1)

    # calculate difference leg 4

    def calculate_difference_leg4(row):
        price_range = row['price_range']
        
        # Check if the strategy has a fourth leg, if not, assign leg_size to 0
        try:
            leg_size = strategy_mapping[strategy][3]['size']  # try to access the fourth leg
        except IndexError:
            leg_size = 0  # default value if the fourth leg doesn't exist
        
        if strategy == "Short Iron Condor":
            return max(0, (row['cost_contract'] - strike_prices[3]*contract_size)) * leg_size
        elif strategy == "Short Iron Butterfly":
            return max(0, (row['cost_contract'] - strike_prices[3]*contract_size)) * leg_size
        else:
            return np.nan

    pl_table['difference_leg_4'] = pl_table.apply(calculate_difference_leg4, axis=1)

    # calculate cost 

    def calculate_cost(row):
        total_cost = 0
        for i in range(len(option_prices)):
            direction = strategy_mapping[strategy][i]['direction']
            multiplier = -1 if direction == "Long" else 1
            size = strategy_mapping[strategy][i]['size']
            total_cost += multiplier * option_prices[i] * contract_size * size
        total_cost -= transaction_costs  # Subtract the transaction costs
        return total_cost
        

    pl_table['cost'] = pl_table.apply(calculate_cost, axis=1)

    # define the function for the visualization metrics needed for calendar and diagonal spreads 

    def calculate_visualization_values(days_to_expiry, days_to_expiry2, risk_free_rate, dividend_yield, annualized_volatility, actual_strike_price, trades_data, strategy_mapping, chosen_strategy):
        strategy_legs = strategy_mapping[chosen_strategy]
        visualization_values_list = []

        for i, leg in enumerate(strategy_legs):
            if i == 0 or expiry_date2 is None or not ("diagonal" in chosen_strategy.lower() or "calendar" in chosen_strategy.lower()):
                time_to_maturity = days_to_expiry / 365
            else:
                time_to_maturity = days_to_expiry2 / 365
                if i == 1:
                    visualization_time_to_maturity = (days_to_expiry2 - days_to_expiry) / 365
                    visualization_discount_factor = (1 + (risk_free_rate / 100)) ** (-visualization_time_to_maturity)
                    visualization_dividend_factor = (1 + dividend_yield) ** (-visualization_time_to_maturity)
                    visualization_fwd_spot = visualization_dividend_factor / visualization_discount_factor
                    visualization_forward = visualization_fwd_spot * trades_data['index_price'].iloc[-1]
                    visualization_vol_sqrt = max(annualized_volatility * np.sqrt(visualization_time_to_maturity), 1e-30)
                    visualization_d1 = (np.log(visualization_forward / actual_strike_price[leg['leg']-1]) / visualization_vol_sqrt) + visualization_vol_sqrt * 0.5
                    visualization_d2 = visualization_d1 - visualization_vol_sqrt
                    visualization_cumulative_norm_d1 = norm.cdf(visualization_d1, 0, 1)
                    visualization_cumulative_norm_d2 = norm.cdf(visualization_d2, 0, 1)
                    visualization_call_price = visualization_forward * visualization_discount_factor * visualization_cumulative_norm_d1 - actual_strike_price[leg['leg']-1] * visualization_cumulative_norm_d2 * visualization_discount_factor
                    visualization_put_price = actual_strike_price[leg['leg']-1] * visualization_discount_factor - visualization_forward * visualization_discount_factor + visualization_call_price
                    visualization_values = {
                        'time_to_maturity': visualization_time_to_maturity,
                        'discount_factor': visualization_discount_factor,
                        'dividend_factor': visualization_dividend_factor,
                        'fwd_spot': visualization_fwd_spot,
                        'forward': visualization_forward,
                        'vol_sqrt': visualization_vol_sqrt,
                        'd1': visualization_d1,
                        'd2': visualization_d2,
                        'cumulative_norm_d1': visualization_cumulative_norm_d1,
                        'cumulative_norm_d2': visualization_cumulative_norm_d2,
                        'call_price': visualization_call_price,
                        'put_price': visualization_put_price
                    }

        return visualization_values

    if strategy in ["Put Diagonal Spread", "Call Diagonal Spread", "Call Calendar Spread", "Put Calendar Spread"]:

        # Call the function with the appropriate arguments
        spread_visualisation_metrics = calculate_visualization_values(days_to_expiry, days_to_expiry2, risk_free_rate, dividend_yield, annualized_volatility, strike_prices, trades_data, strategy_mapping, strategy)


    #calculate leg2diagonalForward

    def calculate_leg2diagonalForward(row):
        price_range = row['price_range']
        fwd_spot = spread_visualisation_metrics.get('fwd_spot', 1)  # default to 1 if not present

        if strategy in ["Put Diagonal Spread", "Call Diagonal Spread", "Call Calendar Spread", "Put Calendar Spread"]:
            return price_range * fwd_spot
        else:
            return 0  # default return, adjust as needed

    if "diagonal" in strategy.lower() or "calendar" in strategy.lower():
        pl_table['leg2diagonalForward'] = pl_table.apply(calculate_leg2diagonalForward, axis=1)


    # Calculate leg2diagonald1
    def calculate_leg2diagonald1(row):
        if strategy in ["Put Diagonal Spread", "Call Diagonal Spread", "Call Calendar Spread", "Put Calendar Spread"]:
            vol_sqrt = spread_visualisation_metrics.get('vol_sqrt', 1)
            return np.log(row['leg2diagonalForward'] / strike_prices[1]) / vol_sqrt + vol_sqrt * 0.5
        else:
            return 0
    if "diagonal" in strategy.lower() or "calendar" in strategy.lower():    
        pl_table['leg2diagonald1'] = pl_table.apply(calculate_leg2diagonald1, axis=1)


    # Calculate leg2diagonald2
    def calculate_leg2diagonald2(row):
        return row['leg2diagonald1'] - spread_visualisation_metrics.get('vol_sqrt', 1)

    if "diagonal" in strategy.lower() or "calendar" in strategy.lower():
        pl_table['leg2diagonald2'] = pl_table.apply(calculate_leg2diagonald2, axis=1)


    # Calculate leg2diagonalcumulative_norm_d1
    def calculate_leg2diagonalcumulative_norm_d1(row):
        return norm.cdf(row['leg2diagonald1'], 0, 1)

    if "diagonal" in strategy.lower() or "calendar" in strategy.lower():
        pl_table['leg2diagonalcumulative_norm_d1'] = pl_table.apply(calculate_leg2diagonalcumulative_norm_d1, axis=1)


    # Calculate leg2diagonalcumulative_norm_d2
    def calculate_leg2diagonalcumulative_norm_d2(row):
        return norm.cdf(row['leg2diagonald2'], 0, 1)

    if "diagonal" in strategy.lower() or "calendar" in strategy.lower():
        pl_table['leg2diagonalcumulative_norm_d2'] = pl_table.apply(calculate_leg2diagonalcumulative_norm_d2, axis=1)


    # Calculate leg2diagonalcall_option
    def calculate_leg2diagonalcall_option(row):
        discount_factor = spread_visualisation_metrics.get('discount_factor', 1)
        return row['leg2diagonalForward'] * discount_factor * row['leg2diagonalcumulative_norm_d1'] - strike_prices[1] * row['leg2diagonalcumulative_norm_d2'] * discount_factor

    if "diagonal" in strategy.lower() or "calendar" in strategy.lower():
        pl_table['leg2diagonalcall_option'] = pl_table.apply(calculate_leg2diagonalcall_option, axis=1)


    # Calculate leg2diagonalput_option
    def calculate_leg2diagonalput_option(row):
        discount_factor = spread_visualisation_metrics.get('discount_factor', 1)
        return strike_prices[1] * discount_factor - row['leg2diagonalForward'] * discount_factor + row['leg2diagonalcall_option']
    if "diagonal" in strategy.lower() or "calendar" in strategy.lower():
        pl_table['leg2diagonalput_option'] = pl_table.apply(calculate_leg2diagonalput_option, axis=1)

    # Calculate leg2diagonaloption_sell
    def calculate_leg2diagonaloption_sell(row):
        leg_size = strategy_mapping[strategy][1]['size']  # Fetch the size of the second leg for the specific strategy
        if strategy == "Put Diagonal Spread":
            return row['leg2diagonalput_option'] * contract_size * leg_size
        elif strategy in ["Call Diagonal Spread", "Call Calendar Spread"]:
            return row['leg2diagonalcall_option'] * contract_size * leg_size
        elif strategy == "Put Calendar Spread":
            return row['leg2diagonalput_option'] * contract_size * leg_size
        else:
            return 0
    if "diagonal" in strategy.lower() or "calendar" in strategy.lower():
        pl_table['leg2diagonaloption_sell'] = pl_table.apply(calculate_leg2diagonaloption_sell, axis=1)


    # calculate profit 

    def calculate_profit(row):

        difference_leg_1 = row['difference_leg_1']
        difference_leg_2 = row['difference_leg_2']
        difference_leg_3 = row.get('difference_leg_3', 0)
        difference_leg_4 = row.get('difference_leg_4', 0)
        cost = row['cost']
        leg2diagonaloption_sell = row.get('leg2diagonaloption_sell', 0)

        if strategy == "Bear Put Spread":
            return difference_leg_1 - difference_leg_2 + cost
        elif strategy == "Long Call":
            return difference_leg_1 + cost
        elif strategy == "Long Put":
            return difference_leg_1 + cost
        elif strategy == "Short Call":
            return cost - difference_leg_1
        elif strategy == "Short Put":
            return cost - difference_leg_1
        elif strategy == "Bull Call Spread":
            return difference_leg_1 - difference_leg_2 + cost
        elif strategy == "Short Iron Condor":
            return difference_leg_1 - difference_leg_2 - difference_leg_3 + difference_leg_4 + cost
        elif strategy == "Short Iron Butterfly":
            return difference_leg_1 - difference_leg_2 - difference_leg_3 + difference_leg_4 + cost
        elif strategy == "Bear Call Spread":
            return difference_leg_2 - difference_leg_1 + cost
        elif strategy == "Bull Put Spread":
            return difference_leg_1 - difference_leg_2 + cost
        elif strategy == "Short Straddle":
            return cost - difference_leg_1 - difference_leg_2
        elif strategy == "Short Strangle":
            return cost - difference_leg_1 - difference_leg_2
        elif strategy == "Put Broken Wing Butterfly":
            return cost + difference_leg_1 - difference_leg_2 + difference_leg_3
        elif strategy == "Custom Naked Put":
            return cost - difference_leg_1 - difference_leg_2 + difference_leg_3
        elif strategy == "Put Diagonal Spread":
            return -difference_leg_1 + leg2diagonaloption_sell + cost
        elif strategy == "Call Diagonal Spread":
            return -difference_leg_1 + leg2diagonaloption_sell + cost
        elif strategy == "Call Calendar Spread":
            return -difference_leg_1 + leg2diagonaloption_sell + cost
        elif strategy == "Put Calendar Spread":
            return -difference_leg_1 + leg2diagonaloption_sell + cost
        elif strategy == "Custom Naked Call":
            return -difference_leg_1 - difference_leg_2 + difference_leg_3 + cost
        elif strategy == "Call Broken Wing Butterfly":
            return difference_leg_1 - difference_leg_2 + difference_leg_3 + cost
        elif strategy == "Call Backspread":
            return -difference_leg_1 + difference_leg_2 + cost
        elif strategy == "Put Backspread":
            return -difference_leg_1 + difference_leg_2 + cost
        elif strategy == "Synthetic Long":
            return difference_leg_1 - difference_leg_2
        else:
            return np.nan

    pl_table['profit'] = pl_table.apply(calculate_profit, axis=1)


    # calculate probability of price being below q at expiration

    # Calculate V at time t
    V = annualized_volatility
    t = days_to_expiry / 365
    Vt = V * np.sqrt(t)

    # Current stock price
    p = trades_data['index_price'].iloc[-1]

    # Calculate P(below) for each price in pl_table['price_range']
    pl_table['p_below'] = pl_table['price_range'].apply(lambda q: norm.cdf(np.log(q / p) / Vt))

    # Calculate P(at q) by taking the difference between each value in p_below and its previous value
    pl_table['p_at_q'] = pl_table['p_below'].diff().fillna(pl_table['p_below'].iloc[0])  # Using fillna for the first value

    # Calculate probability_of_profit
    pl_table['probability_of_profit'] = pl_table['p_at_q'] * pl_table['profit']

    #calculate min/max price over the duration of the trade based on volatility

    def calculate_min_max_price_vol_based(strategy, df, annualized_volatility):
        strategy = strategy.lower()
        close_price = df['Close'].iloc[-1]

        if strategy == "long call" or strategy == "short call":
            return close_price * (1 + annualized_volatility)
        elif strategy == "long put" or strategy == "short put":
            return close_price * (1 - annualized_volatility)
        elif strategy == "custom naked put":
            return close_price * annualized_volatility
        else:
            return np.nan

    min_max_price = calculate_min_max_price_vol_based(strategy, df, annualized_volatility)

    #define cost of opening

    cost_of_opening = pl_table['cost'][0]

    #define function to calculate max loss at expiration

    def calculate_max_loss_at_expiration(strategy, strike_prices, cost_of_opening, contract_size, leg_size, min_max_price, pl_table):
        strategy = strategy.lower()
        if strategy == "bear put spread" or strategy == "long call" or strategy == "long put":
            return cost_of_opening
        elif strategy == "short call":
            return ((strike_prices[0] * contract_size) - (min_max_price * contract_size)) * leg_size
        elif strategy == "short put":
            return ((min_max_price * contract_size) - (strike_prices[0] * contract_size)) * leg_size
        elif strategy == "bull call spread":
            return cost_of_opening
        elif strategy == "short iron condor":
            return cost_of_opening - ((strike_prices[1] - strike_prices[0]) * contract_size * leg_size)
        elif strategy == "bear call spread":
            return ((strike_prices[1] - strike_prices[0]) * contract_size * leg_size - cost_of_opening) * -1
        elif strategy == "short iron butterfly":
            return (((strike_prices[3] - strike_prices[2]) * contract_size * leg_size) - cost_of_opening) * (-1)
        elif strategy == "bull put spread":
            return (((strike_prices[1] - strike_prices[0]) * contract_size * leg_size) - cost_of_opening) * -1
        elif strategy == "short straddle" or strategy == "short strangle":
            return (strike_prices[0] * contract_size - cost_of_opening / leg_size) * -1 * leg_size
        elif strategy == "put broken wing butterfly":
            return (((strike_prices[0] - strike_prices[1]) * contract_size * leg_size) - cost_of_opening) * (-1)
        elif strategy == "custom naked put":
            return cost_of_opening - (((strike_prices[0] * contract_size) - (min_max_price * contract_size)) * leg_size)
        elif strategy == "put diagonal spread":
            return (((strike_prices[0] - strike_prices[1]) * contract_size * leg_size - cost_of_opening) if cost_of_opening > 0 else ((strike_prices[0] - strike_prices[1]) * contract_size * leg_size + cost_of_opening)) * -1
        elif strategy == "call diagonal spread":
            return (((strike_prices[1] - strike_prices[0]) * contract_size * leg_size - cost_of_opening) if cost_of_opening > 0 else ((strike_prices[1] - strike_prices[0]) * contract_size * leg_size + cost_of_opening)) * -1
        elif strategy == "call calendar spread" or strategy == "put calendar spread":
            return cost_of_opening
        elif strategy == "custom naked call" or strategy == "call backspread" or strategy == "put backspread":
            return pl_table['profit'].min()
        elif strategy == "call broken wing butterfly":
            return (((strike_prices[1] - strike_prices[0]) * contract_size * leg_size) - cost_of_opening) * (-1)
        else:
            return None
        
    max_loss_at_expiration = calculate_max_loss_at_expiration(strategy, strike_prices, cost_of_opening, contract_size, leg_size, min_max_price, pl_table)


    # define function to calculate max profit at expiration 

    def calculate_max_profit_at_expiration(strategy, strike_prices, cost_of_opening, contract_size, leg_size, min_max_price, pl_table, strategy_mapping):
        strategy = strategy.lower()
        if strategy == "bear put spread":
            return (strike_prices[0] * contract_size * leg_size) - (strike_prices[1] * contract_size * strategy_mapping[strategy][1]['size']) - abs(cost_of_opening)
        elif strategy == "long call":
            return (min_max_price * contract_size * leg_size) - (strike_prices[0] * contract_size * leg_size) - abs(cost_of_opening)
        elif strategy == "long put":
            return (strike_prices[0] * contract_size * leg_size) - (min_max_price * contract_size * leg_size) - abs(cost_of_opening)
        elif strategy in ["short call", "short put", "short iron condor", "short iron butterfly", "bear call spread", "bull put spread", "short straddle", "short strangle", "custom naked call"]:
            return cost_of_opening
        elif strategy in ["bull call spread"]:
            return (strike_prices[1] - strike_prices[0]) * contract_size * leg_size - abs(cost_of_opening)
        elif strategy in ["put broken wing butterfly", "call broken wing butterfly"]:
            return (((strike_prices[0] - strike_prices[1]) * contract_size * leg_size) + cost_of_opening)
        elif strategy == "custom naked put":
            return cost_of_opening
        elif strategy in ["put diagonal spread", "call diagonal spread"]:
            return pl_table['profit'].max() * leg_size
        elif strategy in ["call calendar spread", "put calendar spread", "call backspread", "put backspread"]:
            return pl_table['profit'].max()
        else:
            return None
        
    max_profit_at_expiration = calculate_max_profit_at_expiration(strategy, strike_prices, cost_of_opening, contract_size, leg_size, min_max_price, pl_table, strategy_mapping)

    # define function for break-even price point 1

    def calculate_break_even_pp_1(strategy, strike_prices, cost_of_opening, contract_size, leg_size, max_loss_at_expiration, strategy_mapping):
        strategy = strategy.lower()
        if strategy == "bear put spread":
            return ((strike_prices[0] * contract_size) - (abs(cost_of_opening) / leg_size)) / contract_size
        elif strategy == "short call" or strategy == "bear call spread" or strategy == "custom naked call":
            return ((strike_prices[0] * contract_size) + (cost_of_opening / leg_size)) / contract_size
        elif strategy in ["short put", "long call", "bull call spread", "short iron condor", "short iron butterfly", "bull put spread", "short straddle", "short strangle", "custom naked put"]:
            #return ((strike_prices[0] * contract_size) - (cost_of_opening / leg_size)) / contract_size
            return strike_prices[1] - cost_of_opening/contract_size
        elif strategy == "long put":
            return (strike_prices[0] - (abs(cost_of_opening) / contract_size)) / leg_size
        elif strategy == "put broken wing butterfly":
            return ((np.median([strike_prices[1],strike_prices[2]]) * contract_size) - (cost_of_opening / leg_size)) / contract_size
        elif strategy in ["put diagonal spread", "call diagonal spread", "call calendar spread", "put calendar spread"]:
            return "can't be calculated, use chart for guidance"
        elif strategy == "call broken wing butterfly":
            return np.median(strike_prices[1:2]) * strategy_mapping[strategy][2]['size'] + abs(cost_of_opening)
        elif strategy == "call backspread":
            return strike_prices[1] + abs(max_loss_at_expiration) / contract_size / leg_size
        elif strategy == "put backspread" or strategy == "synthetic long":
            return strike_prices[0]
        else:
            return None

    break_even_price_point_1 = calculate_break_even_pp_1(strategy, strike_prices, cost_of_opening, contract_size, leg_size, max_loss_at_expiration, strategy_mapping)

    # define function for break-even price point 2 where relevant

    def calculate_break_even_pp_2(strategy, strike_prices, cost_of_opening, contract_size, max_loss_at_expiration, strategy_mapping):
        if strategy in ["Short Iron Condor", "Short Iron Butterfly"]:
            #return ((strike_prices[2]*100) + cost_of_opening / strategy_mapping[strategy][2]['size']) / contract_size
            return strike_prices[2] + cost_of_opening/contract_size
        elif strategy in ["Short Straddle", "Short Strangle"]:
            return ((strike_prices[1] * contract_size) + (cost_of_opening / strategy_mapping[strategy][0]['size'])) / contract_size
        elif strategy in ["Put Diagonal Spread", "Call Diagonal Spread", "Call Calendar Spread", "Put Calendar Spread"]:
            return "can't be calculated, use chart for guidance"
        elif strategy == "Call Backspread":
            return strike_prices[0]
        elif strategy == "Put Backspread":
            return strike_prices[1] - abs(max_loss_at_expiration) / contract_size / strategy_mapping[strategy][0]['size']
        else:
            return None

    if strategy in ["Short Iron Condor", "Short Iron Butterfly", "Short Straddle", "Short Strangle","Put Diagonal Spread", 
                    "Call Diagonal Spread", "Call Calendar Spread", "Put Calendar Spread", "Call Backspread", "Put Backspread"]:
        break_even_price_point_2 = calculate_break_even_pp_2(strategy, strike_prices, cost_of_opening, contract_size, max_loss_at_expiration, strategy_mapping)


    # check if break_even_price_point_1 is a string and if it is, skip the code
    if type(break_even_price_point_1) is not str:

        try:

            # calculate expected returns

            pl_table['price_range'] = pl_table['price_range'].astype(float)
            break_even_price_point_1 = float(break_even_price_point_1)

            if strategy in ["Short Iron Condor", "Short Iron Butterfly", "Short Straddle", "Short Strangle","Put Diagonal Spread", 
                            "Call Diagonal Spread", "Call Backspread", "Put Backspread"]:
                break_even_price_point_2 = float(break_even_price_point_2)
                filtered_rows = pl_table[(pl_table['price_range'] >= break_even_price_point_1) & (pl_table['price_range'] <= break_even_price_point_2)]
                expected_return = filtered_rows['probability_of_profit'].sum()

            if strategy in ["Short Call"]:
                filtered_rows = pl_table[(pl_table['price_range'] <= break_even_price_point_1)]
                expected_return = filtered_rows['probability_of_profit'].sum()
        
        except NameError:
            pass


    # check if break_even_price_point_1 is a string and if it is, skip the code
    if type(break_even_price_point_1) is not str:

        # calculate probability of price being above break-even 1

        if strategy in ["Short Iron Condor", "Short Iron Butterfly", "Short Straddle", "Short Strangle","Put Diagonal Spread", 
                        "Call Diagonal Spread", "Call Backspread", "Put Backspread"]:
            # Find the index of the price range value closest to break_even_price_point_1
            closest_index_be1 = (pl_table['price_range'] - break_even_price_point_1).abs().idxmin()

            # Lookup the corresponding p_below value
            p_below_value_be1 = pl_table.loc[closest_index_be1, 'p_below']

            # Compute probability_price_above_breakeven_1
            probability_price_above_breakeven_1 = 1 - p_below_value_be1

            # calculate probability of price being below break-even 2

            # Find the index of the price range value closest to break_even_price_point_1
            closest_index_be2 = (pl_table['price_range'] - break_even_price_point_2).abs().idxmin()

            # Lookup the corresponding p_below value
            p_below_value_be2 = pl_table.loc[closest_index_be2, 'p_below']

            # Compute probability_price_above_breakeven_1
            probability_price_above_breakeven_2 = 1 - p_below_value_be2

            # Compute probability_price_above_breakeven_1
            probability_price_below_breakeven_2 = p_below_value_be2

            # Compute probability of price being in the profit zone 

            probability_profit_zone = probability_price_below_breakeven_2-p_below_value_be1


        if strategy in ["Short Call"]:
            # Find the index of the price range value closest to break_even_price_point_1
            closest_index_be1 = (pl_table['price_range'] - break_even_price_point_1).abs().idxmin()

            # Lookup the corresponding p_below value
            p_below_value_be1 = pl_table.loc[closest_index_be1, 'p_below']

            # Compute probability_price_above_breakeven_1
            probability_price_above_breakeven_1 = 1 - p_below_value_be1

            #define probability of profit zone 

            probability_profit_zone = p_below_value_be1

    #########
    ### DISPLAY THE PLOTLY CHART IN STREAMLIT 
    #########

    # Create a line plot
    fig = go.Figure()

    # Add line plot
    fig.add_trace(go.Scatter(x=pl_table['price_range'], y=pl_table['profit'], mode='lines', showlegend=False))

    # Add horizontal line
    fig.add_trace(go.Scatter(x=pl_table['price_range'], y=np.zeros(len(pl_table['price_range'])), mode='lines', line=dict(color="black"), showlegend=False))

    # Add filled area plot for profit >= 0
    fig.add_trace(go.Scatter(
        x=pl_table['price_range'], 
        y=np.where(pl_table['profit'] >= 0, pl_table['profit'], 0), 
        fill='tozeroy', 
        fillcolor='rgba(0,200,0,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
    ))

    # Add filled area plot for profit < 0
    fig.add_trace(go.Scatter(
        x=pl_table['price_range'], 
        y=np.where(pl_table['profit'] < 0, pl_table['profit'], 0), 
        fill='tozeroy', 
        fillcolor='rgba(200,0,0,0.2)',
        line_color='rgba(255,255,255,0)',
        showlegend=False,
    ))

    # Set title, labels and layout options
    fig.update_layout(
        title={
            'text': f"P/L Chart for {tickerSymbol} {strategy}",
            'y':0.9,
            'x':0.55,
            'xanchor': 'center',
            'yanchor': 'top'},
        title_font_size=20,
        autosize=False,
        width=1000, # Increased width
        height=750, # Increased height
        margin=dict(
            l=1,
            r=1,
            b=100,
            t=100,
            pad=1
        ),
        xaxis_title="Price Range",
        yaxis_title="Profit",
        showlegend=False, # Hide the legend
        font=dict(
            family="Arial",
            size=18,
            color="RebeccaPurple"
        )
    )

    # Join the instrument_names list into a single string
    instrument_names_str = ', '.join(instrument_names)

    # Set a subtitle
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=1.0,
        text=f"Instrument Names: {instrument_names_str}",  # Displayed instrument names
        showarrow=False,
        font=dict(size=14)
    )

    # Format variables
    #max_loss_at_expiration = format(max_loss_at_expiration, '.2f')
    #max_profit_at_expiration = format(max_profit_at_expiration, '.2f')
    #break_even_price_point_1 = format(break_even_price_point_1, '.2f')

    # Annotations
    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=-0.19,
        text=f"Maximum loss at expiration: {format(max_loss_at_expiration, '.2f')}", 
        showarrow=False,
        font=dict(size=14)
    )

    fig.add_annotation(
        xref="paper",
        yref="paper",
        x=0.5,
        y=0.28,
        text=f"Maximum profit at expiration: {format(max_profit_at_expiration, '.2f')}", 
        showarrow=False,
        font=dict(size=14)
    )

    # check if break_even_price_point_1 is a str

    # check if break_even_price_point_1 is a string and if it is, skip the code
    if type(break_even_price_point_1) is not str:

        fig.add_annotation(
            xref="paper",
            yref="paper",
            x=0.5,
            y=0.22,
            text=f"Break-even price point 1: {format(break_even_price_point_1, '.2f')}", 
            showarrow=False,
            font=dict(size=14)
        )

    # check if break_even_price_point_1 is a string and if it is, skip the code
    if type(break_even_price_point_1) is not str:

        # Check if break_even_price_point_2 is defined
        try:
            if isinstance(break_even_price_point_2, str):
                break_even_price_point_2 = float(break_even_price_point_2)
            break_even_price_point_2_str = format(break_even_price_point_2, '.2f')
            
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.19,
                text=f"Break-even price point 2: {break_even_price_point_2_str}", 
                showarrow=False,
                font=dict(size=14)
            )
        except NameError:
            pass

    # check if break_even_price_point_1 is a string and if it is, skip the code
    if type(break_even_price_point_1) is not str:
        
        try:

            # add expected return
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.25,
                text=f"Expected return: {format(expected_return, '.2f')}", 
                showarrow=False,
                font=dict(size=14)
            )
        except NameError:
            pass

    # check if break_even_price_point_1 is a string and if it is, skip the code
    if type(break_even_price_point_1) is not str:

        try:

            # add probability of price being in the profit zone at expiry
            fig.add_annotation(
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.16,
                text=f"Probability of price being in the profit zone at expiry: {format(probability_profit_zone*100, '.1f')}", 
                showarrow=False,
                font=dict(size=14)
            )
        
        except NameError:
            pass
    
    # Create columns for layout
    col1, col2, col3 = st.columns([0.2,12,0.2])

    # Display chart in the center column
    with col2:

        st.plotly_chart(fig,use_container_width=True)

#####
# FOOTER NOTES
#####

footer = """
    <style>
        .footer {
            position: fixed;
            bottom: 5px;
            left: 50%;
            transform: translateX(-50%);
            text-align: center;
        }
    </style>
    <div class="footer">
        <p>Built by <a href="https://twitter.com/orioldc" target="_blank">orioldc</a></p>
    </div>
"""

st.markdown(footer, unsafe_allow_html=True)
