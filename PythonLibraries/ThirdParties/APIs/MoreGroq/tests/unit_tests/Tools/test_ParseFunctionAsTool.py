from moregroq.Tools import ParseFunctionAsTool

from inspect import cleandoc

from TestUtilities.TestSetup import calculate

# From
# https://github.com/groq/groq-api-cookbook/blob/main/tutorials/llama3-stock-market-function-calling/llama3-stock-market-function-calling.ipynb

def get_stock_info(symbol, key):
    '''Return the correct stock info value given the appropriate symbol and key. Infer valid key from the user prompt; it must be one of the following:

    address1, city, state, zip, country, phone, website, industry, industryKey, industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary, fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk, shareHolderRightsRisk, overallRisk, governanceEpochDate, compensationAsOfEpochDate, maxAge, priceHint, previousClose, open, dayLow, dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow, regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta, trailingPE, forwardPE, volume, regularMarketVolume, averageVolume, averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize, marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months, fiftyDayAverage, twoHundredDayAverage, currency, enterpriseValue, profitMargins, floatShares, sharesOutstanding, sharesShort, sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest, sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions, shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue, priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter, earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps, pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange, SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType, symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc, timeZoneFullName, timeZoneShortName, uuid, messageBoardId, gmtOffSetMilliseconds, currentPrice, targetHighPrice, targetLowPrice, targetMeanPrice, targetMedianPrice, recommendationMean, recommendationKey, numberOfAnalystOpinions, totalCash, totalCashPerShare, ebitda, totalDebt, quickRatio, currentRatio, totalRevenue, debtToEquity, revenuePerShare, returnOnAssets, returnOnEquity, freeCashflow, operatingCashflow, earningsGrowth, revenueGrowth, grossMargins, ebitdaMargins, operatingMargins, financialCurrency, trailingPegRatio
    
    If asked generically for 'stock price', use currentPrice
    '''
    data = yf.Ticker(symbol)
    stock_info = data.info
    return stock_info[key]

def get_historical_price(symbol, start_date, end_date):
    """
    Fetches historical stock prices for a given symbol from 'start_date' to 'end_date'.
    - symbol (str): Stock ticker symbol.
    - end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date
    - start_date (date): Set explicitly, or calculated as 'end_date - date interval' (for example, if prompted 'over the past 6 months', date interval = 6 months so start_date would be 6 months earlier than today's date). Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date
    """

    data = yf.Ticker(symbol)
    hist = data.history(start=start_date, end=end_date)
    hist = hist.reset_index()
    hist[symbol] = hist['Close']
    return hist[['Date', symbol]]

def test_parse_for_docstring_arguments_name():
    docstring, arguments, param_info, name = \
        ParseFunctionAsTool.parse_for_docstring_arguments_name(get_stock_info)

    # Uncomment to print the docstring, arguments, and param_info
    #print(docstring)
    #print(arguments)
    #print(param_info)
    assert "Return the correct stock info value given the appropriate symbol and key." \
        in docstring
    assert "address1, city, state, zip, country, phone, website, industry, industryKey," \
        in docstring
    assert "If asked generically for 'stock price', use currentPrice" \
        in docstring

    assert arguments == ('symbol', 'key')
    assert param_info['symbol']['default'] is None
    assert param_info['symbol']['annotation'] == 'Any'
    assert param_info['symbol']['kind'] == 'POSITIONAL_OR_KEYWORD'
    assert param_info['key']['default'] is None
    assert param_info['key']['annotation'] == 'Any'
    assert param_info['key']['kind'] == 'POSITIONAL_OR_KEYWORD'

    assert name == 'get_stock_info'

    docstring, arguments, param_info, name = \
        ParseFunctionAsTool.parse_for_docstring_arguments_name(get_historical_price)

    # Uncomment to print the docstring, arguments, and param_info
    #print(docstring)
    #print(arguments)
    #print(param_info)
    assert "Fetches historical stock prices for a given symbol from 'start_date' to 'end_date'." \
        in docstring
    assert "symbol (str): Stock ticker symbol." \
        in docstring
    assert "end_date (date): Typically today unless a specific end date is provided. End date MUST be greater than start date" \
        in docstring
    assert "start_date (date): Set explicitly, or calculated as 'end_date - date interval' (for example, if prompted 'over the past 6 months', date interval = 6 months so start_date would be 6 months earlier than today's date). Default to '1900-01-01' if vaguely asked for historical price. Start date must always be before the current date" \
        in docstring
    assert arguments == ('symbol', 'start_date', 'end_date')
    assert param_info['symbol']['default'] is None
    assert param_info['symbol']['annotation'] == 'Any'
    assert param_info['symbol']['kind'] == 'POSITIONAL_OR_KEYWORD'
    assert param_info['start_date']['default'] is None
    assert param_info['start_date']['annotation'] == 'Any'
    assert param_info['start_date']['kind'] == 'POSITIONAL_OR_KEYWORD'
    assert param_info['end_date']['default'] is None
    assert param_info['end_date']['annotation'] == 'Any'
    assert param_info['end_date']['kind'] == 'POSITIONAL_OR_KEYWORD'
    assert name == 'get_historical_price'

# Following examples are from
# https://modelcontextprotocol.io/quickstart/server

async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.

    Args:
        state: Two-letter US state code (e.g. CA, NY)
    """
    url = f"{NWS_API_BASE}/alerts/active/area/{state}"
    data = await make_nws_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forecast for a location.

    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    # First get the forecast grid endpoint
    points_url = f"{NWS_API_BASE}/points/{latitude},{longitude}"
    points_data = await make_nws_request(points_url)

    if not points_data:
        return "Unable to fetch forecast data for this location."

    # Get the forecast URL from the points response
    forecast_url = points_data["properties"]["forecast"]
    forecast_data = await make_nws_request(forecast_url)

    if not forecast_data:
        return "Unable to fetch detailed forecast."

    # Format the periods into a readable forecast
    periods = forecast_data["properties"]["periods"]
    forecasts = []
    for period in periods[:5]:  # Only show next 5 periods
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

from inspect import Parameter

def test__get_detailed_type_info():
    type_info, type_annotation = \
        ParseFunctionAsTool._get_detailed_type_info(get_stock_info)
    assert type_info['symbol'] == 'Any'
    assert type_info['key'] == 'Any'

    assert type_annotation['symbol'] == Parameter.empty
    assert type_annotation['key'] == Parameter.empty

    type_info, type_annotation = \
        ParseFunctionAsTool._get_detailed_type_info(get_historical_price)
    assert type_info['symbol'] == 'Any'
    assert type_info['start_date'] == 'Any'
    assert type_info['end_date'] == 'Any'

    assert type_annotation['symbol'] == Parameter.empty
    assert type_annotation['start_date'] == Parameter.empty
    assert type_annotation['end_date'] == Parameter.empty

    type_info, type_annotation = \
        ParseFunctionAsTool._get_detailed_type_info(get_alerts)
    assert type_info['state'] == 'str'

    assert type_annotation['state'] == str

    type_info, type_annotation = \
        ParseFunctionAsTool._get_detailed_type_info(get_forecast)
    assert type_info['latitude'] == 'float'
    assert type_info['longitude'] == 'float'

    assert type_annotation['latitude'] == float
    assert type_annotation['longitude'] == float

from inspect import cleandoc

def test__parse_docstring_sections():
    sections = \
        ParseFunctionAsTool._parse_docstring_sections(get_stock_info)
    assert len(sections) == 1
    assert sections['description'] == cleandoc(get_stock_info.__doc__)

    sections = \
        ParseFunctionAsTool._parse_docstring_sections(get_historical_price)
    assert len(sections) == 1
    assert sections['description'] == cleandoc(get_historical_price.__doc__)

    sections = \
        ParseFunctionAsTool._parse_docstring_sections(get_alerts)
    assert len(sections) == 2
    assert sections['description'] == "Get weather alerts for a US state."
    assert sections['Args'] == "state: Two-letter US state code (e.g. CA, NY)"

    sections = \
        ParseFunctionAsTool._parse_docstring_sections(get_forecast)
    assert len(sections) == 2
    assert sections['description'] == "Get weather forecast for a location."
    assert sections['Args'] == \
        "latitude: Latitude of the location\n    longitude: Longitude of the location"

def test__parse_docstring_sections_on_calculate():
    sections = \
        ParseFunctionAsTool._parse_docstring_sections(calculate)
    assert len(sections) == 3
    assert sections['description'] == "Evaluate a mathematical expression."

    assert sections['Args'] == \
        "expression: The mathematical expression to evaluate."

def test__extract_parameter_descriptions():
    param_descriptions = \
        ParseFunctionAsTool._extract_parameter_descriptions(get_stock_info)
    assert param_descriptions == {'symbol': '', 'key': ''}

    param_descriptions = \
        ParseFunctionAsTool._extract_parameter_descriptions(get_historical_price)
    assert param_descriptions == {'symbol': '', 'start_date': '', 'end_date': ''}

    param_descriptions = \
        ParseFunctionAsTool._extract_parameter_descriptions(get_alerts)
    assert param_descriptions['state'] == "Two-letter US state code (e.g. CA, NY)"

    param_descriptions = \
        ParseFunctionAsTool._extract_parameter_descriptions(get_forecast)
    assert param_descriptions['latitude'] == "Latitude of the location"
    assert param_descriptions['longitude'] == "Longitude of the location"
    

def test_parse_for_function_definition():
    function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(get_stock_info)
    assert function_definition.name == 'get_stock_info'

    # Uncomment to print the function_definition
    #print("function_definition.description: ", function_definition.description)

    assert function_definition.description == cleandoc(get_stock_info.__doc__)

    assert function_definition.parameters.properties[0].name == 'symbol'
    assert function_definition.parameters.properties[0].type == 'Any'
    assert function_definition.parameters.properties[0].description == ""
    assert function_definition.parameters.properties[0].required == True
    assert function_definition.parameters.properties[1].name == 'key'
    assert function_definition.parameters.properties[1].type == 'Any'
    assert function_definition.parameters.properties[1].description == ""
    assert function_definition.parameters.properties[1].required == True
    
    function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(get_historical_price)
    assert function_definition.name == 'get_historical_price'
    assert function_definition.description == cleandoc(get_historical_price.__doc__)
    assert function_definition.parameters.properties[0].name == 'symbol'
    assert function_definition.parameters.properties[0].type == 'Any'
    assert function_definition.parameters.properties[0].description == ""
    assert function_definition.parameters.properties[0].required == True
    assert function_definition.parameters.properties[1].name == 'start_date'
    assert function_definition.parameters.properties[1].type == 'Any'
    assert function_definition.parameters.properties[1].description == ""
    assert function_definition.parameters.properties[1].required == True
    assert function_definition.parameters.properties[2].name == 'end_date'
    assert function_definition.parameters.properties[2].type == 'Any'
    assert function_definition.parameters.properties[2].description == ""
    assert function_definition.parameters.properties[2].required == True

    function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(get_alerts)
    assert function_definition.name == 'get_alerts'
    assert function_definition.description == \
        "Get weather alerts for a US state."
    assert function_definition.parameters.properties[0].name == 'state'
    assert function_definition.parameters.properties[0].type == 'string'
    assert function_definition.parameters.properties[0].description == \
        "Two-letter US state code (e.g. CA, NY)"
    assert function_definition.parameters.properties[0].required == True

    function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(get_forecast)
    assert function_definition.name == 'get_forecast'
    assert function_definition.description == \
        "Get weather forecast for a location."
    assert function_definition.parameters.properties[0].name == 'latitude'
    assert function_definition.parameters.properties[0].type == 'number'
    assert function_definition.parameters.properties[0].description == \
        "Latitude of the location"
    assert function_definition.parameters.properties[0].required == True
    assert function_definition.parameters.properties[1].name == 'longitude'
    assert function_definition.parameters.properties[1].type == 'number'
    assert function_definition.parameters.properties[1].description == \
        "Longitude of the location"
    assert function_definition.parameters.properties[1].required == True

def test_parse_for_function_definition_with_calculate():
    function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(calculate)
    assert function_definition.name == 'calculate'
    assert function_definition.description == \
        "Evaluate a mathematical expression."
    assert function_definition.parameters.properties[0].name == 'expression'
    assert function_definition.parameters.properties[0].type == 'string'
    print("function_definition:", function_definition)