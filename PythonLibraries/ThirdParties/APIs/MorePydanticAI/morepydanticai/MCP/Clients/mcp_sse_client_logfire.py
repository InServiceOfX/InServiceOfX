import logfire

def instrument_client_code_with_logfire():

    logfire.configure()
    logfire.instrument_pydantic_ai()

if __name__ == "__main__":
    instrument_client_code_with_logfire()
