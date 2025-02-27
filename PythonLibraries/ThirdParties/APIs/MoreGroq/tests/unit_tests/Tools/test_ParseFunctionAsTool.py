from moregroq.Tools import ParseFunctionAsTool

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

def test_parse_for_function_definition():
    function_definition = \
        ParseFunctionAsTool.parse_for_function_definition(get_stock_info)
    assert function_definition.name == 'get_stock_info'
    assert function_definition.description == get_stock_info.__doc__
    assert function_definition.parameters.properties[0].name == 'symbol'
    assert function_definition.parameters.properties[0].type == 'string'
    assert function_definition.parameters.properties[0].description == \
        get_stock_info.__doc__
    assert function_definition.parameters.properties[0].required == True
    assert function_definition.parameters.properties[1].name == 'key'
    assert function_definition.parameters.properties[1].type == 'string'
    assert function_definition.parameters.properties[1].description == \
        get_stock_info.__doc__
    assert function_definition.parameters.properties[1].required == True