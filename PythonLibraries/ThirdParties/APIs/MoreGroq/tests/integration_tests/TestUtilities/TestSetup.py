from dataclasses import dataclass
import json

# https://console.groq.com/docs/tool-use
def calculate(expression: str):
    """Evaluate a mathematical expression.
    
    Args:
        expression: The mathematical expression to evaluate.

    Returns:
        str: The result of the mathematical expression.
    """
    try:
        # Attempt to evaluate the math expression
        result = eval(expression)
        return json.dumps({"result": result})
    except Exception as e:
        return json.dumps({"error": f"Invalid expression: {str(e)}"})

def get_bakery_prices(bakery_item: str):
    """
    Define a tool, or function, that the LLM can invoke to fetch pricing for
    bakery items.
    """
    if bakery_item == "croissant":
        return 4.25
    elif bakery_item == "brownie":
        return 2.50
    elif bakery_item == "cappuccino":
        return 4.75
    else:
        return "We're currently sold out!"

import yfinance as yf

def get_stock_info(symbol: str, key: str):
    """Return the correct stock info value given the appropriate symbol and key.
    Infer valid key from the user prompt; it must be one of the following:

    address1, city, state, zip, country, phone, website, industry, industryKey,
    industryDisp, sector, sectorKey, sectorDisp, longBusinessSummary,
    fullTimeEmployees, companyOfficers, auditRisk, boardRisk, compensationRisk,
    shareHolderRightsRisk, overallRisk, governanceEpochDate,
    compensationAsOfEpochDate, maxAge, priceHint, previousClose, open, dayLow,
    dayHigh, regularMarketPreviousClose, regularMarketOpen, regularMarketDayLow,
    regularMarketDayHigh, dividendRate, dividendYield, exDividendDate, beta,
    trailingPE, forwardPE, volume, regularMarketVolume, averageVolume,
    averageVolume10days, averageDailyVolume10Day, bid, ask, bidSize, askSize,
    marketCap, fiftyTwoWeekLow, fiftyTwoWeekHigh, priceToSalesTrailing12Months,
    fiftyDayAverage, twoHundredDayAverage, currency, enterpriseValue,
    profitMargins, floatShares, sharesOutstanding, sharesShort,
    sharesShortPriorMonth, sharesShortPreviousMonthDate, dateShortInterest,
    sharesPercentSharesOut, heldPercentInsiders, heldPercentInstitutions,
    shortRatio, shortPercentOfFloat, impliedSharesOutstanding, bookValue,
    priceToBook, lastFiscalYearEnd, nextFiscalYearEnd, mostRecentQuarter,
    earningsQuarterlyGrowth, netIncomeToCommon, trailingEps, forwardEps,
    pegRatio, enterpriseToRevenue, enterpriseToEbitda, 52WeekChange,
    SandP52WeekChange, lastDividendValue, lastDividendDate, exchange, quoteType,
    symbol, underlyingSymbol, shortName, longName, firstTradeDateEpochUtc,
    
    If asked generically for 'stock price', use currentPrice
    """
    data = yf.Ticker(symbol)
    stock_info = data.info
    return stock_info[key]

def get_historical_price(symbol, start_date, end_date):
    """Fetches historical stock prices for a given symbol from 'start_date' to
    'end_date'.
    - symbol (str): Stock ticker symbol.
    - end_date (date): Typically today unless a specific end date is provided.
    End date MUST be greater than start date
    - start_date (date): Set explicitly, or calculated as 'end_date - date
    interval' (for example, if prompted 'over the past 6 months',
    date interval = 6 months so start_date would be 6 months earlier than
    today's date). Default to '1900-01-01' if vaguely asked for historical
    price. Start date must always be before the current date
    """
    data = yf.Ticker(symbol)
    hist = data.history(start=start_date, end=end_date)
    hist = hist.reset_index()
    hist[symbol] = hist['Close']
    return hist[['Date', symbol]]

# Test functions

def echo(message: str) -> str:
    """Echo a message back to the user.
    
    Args:
        message (str): The message to echo back to the user.

    Returns:
        str: The message echoed back to the user.
    """
    return message

def reverse_string(input_string: str) -> str:
    """Reverse a string.
    
    Args:
        input_string (str): The string to reverse.

    Returns:
        str: The reversed string.
    """
    return input_string[::-1]

def count_words(text: str):
    """Count the number of words in a string.
    
    Args:
        text (str): The string to count the words in.

    Returns:
        int: The number of words in the string.
    """
    return len(text.split())

def get_current_system_time():
    """Get the current system time.
    
    Returns:
        str: The current system time.
    """
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def create_purchasing_order(product_id: str, quantity: int) -> dict:
    """Create a purchasing order.
    
    Args:
        product_id (str): The ID of the product to purchase.
        quantity (int): The quantity of the product to purchase.

    Returns:
        dict: The purchasing order.
    """
    # Initialize counter if it doesn't exist
    if not hasattr(create_purchasing_order, "counter"):
        create_purchasing_order.counter = 0
    
    # Increment counter
    create_purchasing_order.counter += 1
    
    # Create order with unique ID
    order = {
        "order_id": f"PO-{create_purchasing_order.counter:04d}",
        "product_id": product_id,
        "quantity": quantity}
    
    return order

class PizzaCustomerSupportFiniteStateMachine:
    STATES = {
        'start',
        'selecting_type',
        'customizing_toppings',
        'confirming_order',
        'order_placed',
        'order_cancelled'
    }

    ALPHABET = {
        'order_pizza',
        'select_type',
        'add_topping',
        'remove_topping',
        'confirm_order',
        'cancel_order',
        'unknown_input'}

    INITIAL_STATE = 'start'

    FINAL_STATES = {'order_placed', 'order_cancelled'}

    @staticmethod
    def get_transition_function():

        transition_function = {}
        for s in PizzaCustomerSupportFiniteStateMachine.STATES:
            for a in PizzaCustomerSupportFiniteStateMachine.ALPHABET:
                if s == 'start':
                    transition_function[(s, a)] = 'selecting_type' \
                        if a == 'order_pizza' else 'start'
                elif s == 'selecting_type':
                    transition_function[(s, a)] = 'customizing_toppings' \
                        if a == 'select_type' else 'order_cancelled' \
                            if a == 'cancel_order' else 'selecting_type'
                elif s == 'customizing_toppings':
                    transition_function[(s, a)] = 'customizing_toppings' \
                        if a in {'add_topping', 'remove_topping'} \
                        else 'confirming_order' \
                            if a == 'confirm_order' else 'order_cancelled' \
                                if a == 'cancel_order' else 'customizing_toppings'
                elif s == 'confirming_order':
                    transition_function[(s, a)] = 'order_placed' \
                        if a == 'confirm_order' else 'order_cancelled' \
                            if a == 'cancel_order' else 'confirming_order'
                elif s == 'order_placed':
                    transition_function[(s, a)] = 'order_placed'
                elif s == 'order_cancelled':
                    transition_function[(s, a)] = 'order_cancelled'
        return transition_function

    SYSTEM_PROMPT = \
    """
You are a customer support chatbot for a pizza restaurant. Your task is to guide users through the process of ordering a pizza using a state machine. The state machine has the following input alphabet symbols: 'order_pizza', 'select_type', 'add_topping', 'remove_topping', 'confirm_order', 'cancel_order', 'unknown_input'.

Based on the user's message, determine their intent and map it to one of these symbols. Here are examples:
- 'I want to order a pizza' → 'order_pizza'
- 'I’d like a pepperoni pizza' → 'select_type'
- 'Add extra cheese' → 'add_topping'
- 'Remove olives' → 'remove_topping'
- 'Yes, place the order' → 'confirm_order'
- 'No, cancel it' → 'cancel_order'
- 'What’s the weather like?' → 'unknown_input'

If the user’s intent is unclear or doesn’t match these categories, use 'unknown_input' and ask for clarification. Always respond with a JSON object containing the symbol, e.g., {'symbol': 'order_pizza'}. Only use symbols from the defined alphabet.
"""