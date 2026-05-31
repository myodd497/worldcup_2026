"""
OpenWeatherMap wrapper — returns current forecast for a given city.
"""
from __future__ import annotations

import os
import httpx

_BASE = "https://api.openweathermap.org/data/2.5/weather"


def _is_weather_enabled() -> bool:
    return os.getenv("ENABLE_WEATHER", "false").lower() == "true"


def get_venue_weather(city: str) -> dict:
    if not _is_weather_enabled():
        return {
            "city": city,
            "temp_c": 22.0,
            "feels_like_c": 22.0,
            "humidity_pct": 50,
            "description": "Weather disabled (test mode)",
            "wind_mps": 0.0,
        }

    params = {
        "q": city,
        "appid": os.environ["OPENWEATHER_API_KEY"],
        "units": "metric",
    }
    with httpx.Client(timeout=10) as client:
        resp = client.get(_BASE, params=params)
        resp.raise_for_status()
        data = resp.json()

    return {
        "city": city,
        "temp_c": data["main"]["temp"],
        "feels_like_c": data["main"]["feels_like"],
        "humidity_pct": data["main"]["humidity"],
        "description": data["weather"][0]["description"].capitalize(),
        "wind_mps": data["wind"]["speed"],
    }


if __name__ == "__main__":
    print(get_venue_weather("New York"))
