from typing import Any
import httpx
from mcp.server.fastmcp import FastMCP

# Initialize FastMCP server
# https://modelcontextprotocol.io/quickstart/server#building-your-server
# The FastMCP class uses Python type hints and docstrings to automatically
# generate tool definitions, making it easy to create and maintain MCP tools.
mcp = FastMCP("weather")

# Constants
NSW_API_BASE = "https://api.weather.gov"
USER_AGENT = "weather-app/1.0"

# Helper functions
# Helper functions for querying and formatting data from National Weather
# Service APIs

async def make_nws_request(url: str) -> dict[str, Any] | None:
    """Make a request to the NWS API with proper error handling."""
    print("make_nws_request called with url: ", url)
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "application/geo+json"
    }
    async with httpx.AsyncClient() as client:
        try:
            response = await client.get(url, headers=headers, timeout=30.0)
            response.raise_for_status()
            return response.json()
        except Exception:
            return None

def format_alert(feature: dict) -> str:
    """Format an alert feature into a readable string."""
    print("format_alert called with feature: ", feature)
    props = feature["properties"]
    return f"""
Event: {props.get('event', 'Unknown')}
Area: {props.get('areaDesc', 'Unknown')}
Severity: {props.get('severity', 'Unknown')}
Description: {props.get('description', 'No description available')}
Instructions: {props.get('instruction', 'No specific instructions provided')}
"""

@mcp.tool()
async def get_alerts(state: str) -> str:
    """Get weather alerts for a US state.
    
    Args:
        state: Two-letter US state code (e.g. CA, NY)        
    """
    print(f"get_alerts called with state: {state}")
    url = f"{NSW_API_BASE}/alerts/active/area/{state}"
    data = await make_nsw_request(url)

    if not data or "features" not in data:
        return "Unable to fetch alerts or no alerts found."

    if not data["features"]:
        return "No active alerts for this state."

    alerts = [format_alert(feature) for feature in data["features"]]
    return "\n---\n".join(alerts)

@mcp.tool()
async def get_forecast(latitude: float, longitude: float) -> str:
    """Get weather forcast for a location.
    
    Args:
        latitude: Latitude of the location
        longitude: Longitude of the location
    """
    print(
        "get_forecast called with latitude: ",
        latitude,
        "longitude: ",
        longitude)
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
    # Only show next 5 periods
    for period in periods[:5]:
        forecast = f"""
{period['name']}:
Temperature: {period['temperature']}°{period['temperatureUnit']}
Wind: {period['windSpeed']} {period['windDirection']}
Forecast: {period['detailedForecast']}
"""
        forecasts.append(forecast)

    return "\n---\n".join(forecasts)

if __name__ == "__main__":
    print("Now running mcp, mcp.run(transport='stdio')")
    # Initialize and run the server
    mcp.run(transport='stdio')