from textwrap import dedent

class ToolsForTest:
    @staticmethod
    def get_current_temperature(location: str, unit: str):
        """
        Get the current temperature at a location.
        
        Args:
            location: The location to get the temperature for, in the format
            "City, Country"
            unit: The unit to return the temperature in. (choices: ["celsius",
            "fahrenheit"])
        """
        # A real function should probably actually get the temperature!
        return 22.

    @staticmethod
    def get_current_wind_speed(location: str):
        """
        Get the current wind speed in km/h at a given location.
        
        Args:
            location: The location to get the wind speed for, in the format
            "City, Country"
        """
        # A real function should probably actually get the wind speed!
        return 6.

    @staticmethod
    def create_wind_weather_system_message():
        """
        Create an enhanced system message that helps the LLM intelligently route
        tool calls for weather-related queries.
        """
        return dedent(
            """You are an intelligent weather assistant with access to two 
            specialized tools:

    AVAILABLE TOOLS:
    1. get_current_temperature(location, unit) - Gets temperature data
    - Use when users ask about: temperature, heat, cold, warmth, degrees, "how 
    hot/cold"
    - Always specify unit: "celsius" or "fahrenheit" based on location norms
    - Examples: "What's the temperature in Paris?" → use temperature tool

    2. get_current_wind_speed(location) - Gets wind speed data  
    - Use when users ask about: wind, breeze, gusts, "how windy", wind speed
    - Examples: "How windy is Tokyo?" → use wind speed tool

    TOOL SELECTION STRATEGY:
    - For temperature-only queries: Use only get_current_temperature
    - For wind-only queries: Use only get_current_wind_speed  
    - For comprehensive weather queries: Use BOTH tools
    - For general weather questions: Use BOTH tools to provide complete 
    information

    EXAMPLES OF QUERY ROUTING:
    - "What's the temperature in London?" → get_current_temperature only
    - "How windy is New York?" → get_current_wind_speed only
    - "What's the weather like in Tokyo?" → BOTH tools
    - "Tell me about conditions in Paris" → BOTH tools
    - "Is it hot in Miami?" → get_current_temperature only
    - "Is it windy in Chicago?" → get_current_wind_speed only

    RESPONSE GUIDELINES:
    - Always extract the location from the user's query
    - Use appropriate temperature units (Celsius for most countries, Fahrenheit 
    for US)
    - Provide clear, helpful responses combining tool results
    - If using both tools, present temperature and wind information together
    - Be conversational and helpful in your responses

    Remember: You can call multiple tools in sequence to provide comprehensive 
    weather information when appropriate.""")

