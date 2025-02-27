"""
https://github.com/groq/groq-api-cookbook/blob/main/tutorials/llama3-stock-market-function-calling/llama3-stock-market-function-calling.ipynb
"""

from corecode.Utilities import (get_environment_variable, load_environment_file)
from commonapi.Messages import (
    create_system_message,
    create_user_message,
    create_tool_message
)
from moregroq.Tools import ParseFunctionAsTool, ToolCallProcessor
from moregroq.Wrappers.ChatCompletionConfiguration import Tool

from moregroq.Wrappers import GroqAPIWrapper

from TestUtilities.TestSetup import (
    get_stock_info,
    get_historical_price
)

import json
from pathlib import Path

load_environment_file()

get_stock_info_message = (
    "Return the correct stock info value given the appropriate symbol and key. "
    "Infer valid key from the user prompt; it must be one of the following:"
    """\n\n"""
    "address1, city, state, zip, country, phone, website, industry, "
    "industryKey, industryDisp, sector, sectorKey, sectorDisp, "
    "longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, "
    "boardRisk, compensationRisk, shareHolderRightsRisk, overallRisk, "
    "governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint, "
    "previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, "
    "regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, "
    "dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, "
    "volume, regularMarketVolume, averageVolume, averageVolume10days, "
    "averageDailyVolume10Day, bid, ask, bidSize, askSize, marketCap, "
    "fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, "
    "fiftyDayAverage, twoHundredDayAverage, currency, enterpriseValue, "
    "profitMargins, floatShares, sharesOutstanding, sharesShort, "
    "sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, "
    "sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions, "
    "shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, "
    "priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, "
    "earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, "
    "pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, "
    "SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, "
    "quoteType, symbol, underlyingSymbol, shortName, longName, "
    "firstTradeDateEpochUtc, timeZoneFullName, timeZoneShortName, uuid, "
    "messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice, "
    "targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, "
    "recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, "
    "ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, "
    "debtToEquity, revenuePerShare, returnOnAssets, returnOnEquity, "
    "freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, "
    "grossMargins, ebitdaMargins, operatingMargins, financialCurrency, "
    "trailingPegRatio"
    """\n\n"""
    "If asked generically for 'stock price', use currentPrice"
)
    
get_historical_price_message = (
    "Fetches historical stock prices for a given symbol from 'start_date' to "
    "'end_date'. "
    """\n"""
    "- symbol (str): Stock ticker symbol. "
    """\n"""
    "- end_date (date): Typically today unless a specific end date is "
    "provided. End date MUST be greater than start date "
    """\n"""
    "- start_date (date): Set explicitly, or calculated as 'end_date - date interval' "
    "(for example, if prompted 'over the past 6 months', date interval = 6 "
    "months so start_date would be 6 months earlier than today's date). "
    "Default to '1900-01-01' if vaguely asked for historical price. Start date "
    "must always be before the current date"
)

def test_using_tools():
    function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(get_stock_info)
    assert function_definition.name == 'get_stock_info'
    assert len(function_definition.parameters.properties) == 2
    assert function_definition.parameters.properties[0].name == 'symbol'
    assert function_definition.parameters.properties[0].type == 'str'
    function_definition.parameters.properties[0].description = \
        "The stock ticker symbol"
    assert function_definition.parameters.properties[1].name == 'key'
    assert function_definition.parameters.properties[1].type == 'str'
    function_definition.parameters.properties[1].description = \
        "A valid key to the stock info"

    get_historical_price_function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(
            get_historical_price)
    assert \
        len(get_historical_price_function_definition.parameters.properties) == 3
    get_historical_price_function_definition.parameters.properties[0].description = \
        "The stock ticker symbol"
    get_historical_price_function_definition.parameters.properties[1].description = \
        "The start date"
    get_historical_price_function_definition.parameters.properties[2].description = \
        "The end date"

    tools = [
        Tool(function=function_definition),
        Tool(function=get_historical_price_function_definition)
    ]

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.tools = tools
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tool_choice = "auto"

    query1 = 'What is the market cap of Meta'
    messages = [
        create_system_message("You are a helpful assistant."),
        create_user_message(query1)
    ]

    response = groq_api_wrapper.create_chat_completion(messages)

    available_functions = {
        "get_stock_info": get_stock_info,
        "get_historical_price": get_historical_price
    }

    tool_call_processor = ToolCallProcessor(
        available_functions=available_functions,
        messages=messages
    )

    process_result = tool_call_processor.process_response(
        response.choices[0].message)

    assert process_result == 1

    response = groq_api_wrapper.create_chat_completion(
        tool_call_processor.messages)
    print("response.choices[0].message", response.choices[0].message)
    print(
        "tool_call_processor.current_tool_calls",
        tool_call_processor.current_tool_calls)

    assert "market cap" in response.choices[0].message.content or \
        "Market cap" in response.choices[0].message.content
    assert "Meta" in response.choices[0].message.content or \
        "meta" in response.choices[0].message.content

    query2 = 'How does the volume of Apple compare to that of Microsoft?'

    messages = [
        create_system_message("You are a helpful assistant."),
        create_user_message(query2)
    ]

    response = groq_api_wrapper.create_chat_completion(messages)

    tool_call_processor.messages = messages
    process_result = tool_call_processor.process_response(
        response.choices[0].message)

    assert process_result == 2

    response = groq_api_wrapper.create_chat_completion(
        tool_call_processor.messages)
    print("response.choices[0].message", response.choices[0].message)
    print(
        "tool_call_processor.current_tool_calls",
        tool_call_processor.current_tool_calls)

    assert "volume" in response.choices[0].message.content or \
        "Volume" in response.choices[0].message.content

def test_using_call_with_tool_calls():
    get_stock_info_function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(get_stock_info)
    get_stock_info_function_definition.parameters.properties[0].description = \
        "The stock ticker symbol"
    get_stock_info_function_definition.parameters.properties[1].description = \
        "A valid key to the stock info"

    get_historical_price_function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(
            get_historical_price)
    get_historical_price_function_definition.parameters.properties[0].description = \
        "The stock ticker symbol"
    get_historical_price_function_definition.parameters.properties[1].description = \
        "The start date"
    get_historical_price_function_definition.parameters.properties[2].description = \
        "The end date"

    tools = [
        Tool(function=get_stock_info_function_definition),
        Tool(function=get_historical_price_function_definition)
    ]

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.tools = tools
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tool_choice = "auto"

    available_functions = {
        "get_stock_info": get_stock_info,
        "get_historical_price": get_historical_price
    }

    tool_call_processor = ToolCallProcessor(
        available_functions=available_functions)

    query1 = \
        'Show the historical price of the S&P 500 over the past 3 years? (Today is 4/23/2024)'

    messages = [
        create_system_message("You are a helpful assistant."),
        create_user_message(query1)
    ]

    process_result, response = tool_call_processor.call_with_tool_calls(
        messages, groq_api_wrapper)

    assert process_result == 1

    print("response.choices[0].message", response.choices[0].message)
    print(
        "tool_call_processor.current_tool_calls",
        tool_call_processor.current_tool_calls)

    assert "S&P 500" in response.choices[0].message.content or \
        "sp 500" in response.choices[0].message.content
    assert "Historical price" in response.choices[0].message.content or \
        "historical price" in response.choices[0].message.content

    query2 = 'Compare the price of Google and Amazon throughout 2023'

    messages = [
        create_system_message("You are a helpful assistant."),
        create_user_message(query2)
    ]

    process_result, response = tool_call_processor.call_with_tool_calls(
        messages, groq_api_wrapper)

    assert process_result == 2
    
    print("response.choices[0].message", response.choices[0].message)
    print(
        "tool_call_processor.current_tool_calls",
        tool_call_processor.current_tool_calls)

    assert "Google" in response.choices[0].message.content or \
        "google" in response.choices[0].message.content
    assert "Amazon" in response.choices[0].message.content or \
        "amazon" in response.choices[0].message.content
        
import pandas as pd
import plotly.graph_objects as go

def plot_price_over_time(historical_price_dfs, is_local=False):
    full_df = pd.DataFrame(columns=['Date'])
    for df in historical_price_dfs:
        full_df = full_df.merge(df, on='Date', how='outer')

    # Create a Plotly figure
    fig = go.Figure()

    # Dynamically add a trace for each stock symbol in the DataFrame
    # Skip the first column since it's the date
    for column in full_df.columns[1:]:
        fig.add_trace(go.Scatter(
            x = full_df['Date'],
            y = full_df[column],
            mode = 'lines+markers',
            name = column
        ))

    # Update the layout to add titles and format axis labels
    fig.update_layout(
        title=\
            'Stock Price Over Time: ' + ', '.join(full_df.columns.tolist()[1:]),
        xaxis_title = 'Date',
        yaxis_title = 'Stock Price (USD)',
        yaxis_tickprefix='$',
        yaxis_tickformat=',.2f',
        xaxis=dict(
            tickangle=45,
            nticks=20,
            tickfont=dict(size=10)
        ),
        yaxis=dict(
            # Enable y-axis grid lines
            showgrid=True,
            # Set grid line color
            gridcolor='lightgray',
        ),
        legend_title_text='Stock Symbol',
        # Set plot background to white
        plot_bgcolor='white',
        # Set overall figure background to white
        paper_bgcolor='white',
        legend=dict(
            # Optional: Set legend background to white
            bgcolor='white',
            bordercolor='black'
        )
    )

    # Originally, the code said:
    # Show the figure - this just generates a static .png. If running locally
    # you can use fig.show(renderer='iframe') to output a dynamic plotly plot.
    # Show the plot

    # Save figure to current working directory
    output_path = Path.cwd() / "stock_price_comparison.png"
    fig.write_image(str(output_path))
    
    # Still show the figure if requested
    fig.show(renderer='iframe') if is_local else fig.show('png')

from datetime import date

def define_stock_info_tools():
    get_stock_info_function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(get_stock_info)
    get_stock_info_function_definition.parameters.properties[0].description = \
        "The stock ticker symbol"
    get_stock_info_function_definition.parameters.properties[1].description = \
        "A valid key to the stock info"

    get_historical_price_function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(
            get_historical_price)
    get_historical_price_function_definition.parameters.properties[0].description = \
        "The stock ticker symbol"
    get_historical_price_function_definition.parameters.properties[1].description = \
        "The start date"
    get_historical_price_function_definition.parameters.properties[2].description = \
        "The end date"
    
    return [
        get_stock_info_function_definition,
        get_historical_price_function_definition]

stock_info_tools = [
    Tool(function=define_stock_info_tools()[0]),
    Tool(function=define_stock_info_tools()[1])
]

def test_call_functions_steps_for_putting_it_all_together():
    system_prompt = (
        "You are a helpful finance assistant that analyzes stocks and stock "
        "prices. Today is {today}.".format(today = date.today()))

    user_prompt_1 = "What is the beta for meta stock?"

    messages = [
        create_system_message(system_prompt),
        create_user_message(user_prompt_1)
    ]

    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.tools = stock_info_tools
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tool_choice = "auto"

    response = groq_api_wrapper.create_chat_completion(messages)

    available_functions = {
        "get_stock_info": get_stock_info,
        "get_historical_price": get_historical_price
    }

    tool_calls = getattr(response.choices[0].message, 'tool_calls', None)

    print("len(tool_calls): ", len(tool_calls))

    historical_price_dfs = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions.get(function_name)
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(**function_args)

        if function_name == 'get_historical_price':
            historical_price_dfs.append(function_response)
        else:
            messages.append(create_tool_message(
                name=function_name,
                tool_call_id=tool_call.id,
                content=str(function_response)))

    print("len(historical_price_dfs): ", len(historical_price_dfs))
    print("historical_price_dfs: ", historical_price_dfs)

    assert len(historical_price_dfs) == 0
    assert historical_price_dfs == []

def call_functions(groq_api_wrapper, user_prompt):
    system_prompt = (
        "You are a helpful finance assistant that analyzes stocks and stock "
        "prices. Today is {today}.".format(today = date.today()))

    messages = [
        create_system_message(system_prompt),
        create_user_message(user_prompt)
    ]

    response = groq_api_wrapper.create_chat_completion(messages)
    available_functions = {
        "get_stock_info": get_stock_info,
        "get_historical_price": get_historical_price
    }

    tool_calls = getattr(response.choices[0].message, 'tool_calls', None)

    historical_price_dfs = []
    symbols = []

    for tool_call in tool_calls:
        function_name = tool_call.function.name
        function_to_call = available_functions.get(function_name)
        function_args = json.loads(tool_call.function.arguments)
        function_response = function_to_call(**function_args)

        if function_name == 'get_historical_price':
            historical_price_dfs.append(function_response)
            symbols.append(function_args['symbol'])
        else:
            messages.append(create_tool_message(
                name=function_name,
                tool_call_id=tool_call.id,
                content=str(function_response)))

    if len(historical_price_dfs) > 0:
        plot_price_over_time(historical_price_dfs, is_local=True)
        symbols = ' and '.join(symbols)
        messages.append(create_tool_message(
            content=\
                'Tell the user that a historical stock price chart for {symbols} been generated.'.format(symbols=symbols),
            # Tool call id has to be a string.
            tool_call_id="0",
            # Name was arbitrary but had to be set.
            name="message_to_user"
        ))
    return (
        groq_api_wrapper.create_chat_completion(messages),
        messages,
        historical_price_dfs,
        symbols
    )

def test_call_functions():
    groq_api_wrapper = GroqAPIWrapper(get_environment_variable("GROQ_API_KEY"))
    groq_api_wrapper.configuration.tools = stock_info_tools
    groq_api_wrapper.configuration.model = "llama-3.3-70b-versatile"
    groq_api_wrapper.configuration.tool_choice = "auto"

    user_prompt = \
        "Compare the stock price of Google, Apple and Meta over the past 6 months"
    
    response, messages, historical_price_dfs, symbols = call_functions(
        groq_api_wrapper, user_prompt)
    
    print("response.choices[0].message", response.choices[0].message)
    print("messages", messages)
    print("historical_price_dfs", historical_price_dfs)
    print("symbols", symbols)

