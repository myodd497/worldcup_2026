"""
Data schemas — Pydantic models for all data contracts across agents and tools.
"""
from __future__ import annotations

from datetime import datetime
from pydantic import BaseModel


class MatchFixture(BaseModel):
    fixture_id: int
    date: datetime
    venue: str
    venue_city: str
    referee: str
    home_team: str
    away_team: str
    home_lineup: list[str] = []
    away_lineup: list[str] = []


class WeatherInfo(BaseModel):
    city: str
    temp_c: float
    feels_like_c: float
    humidity_pct: int
    description: str
    wind_mps: float


class NewsArticle(BaseModel):
    title: str
    url: str
    source: str
    published_at: str


class SentimentSummary(BaseModel):
    tweet_count: int
    positive_pct: float
    negative_pct: float
    neutral_pct: float
    top_topic: str
    sample_tweets: list[str] = []


class MatchPrediction(BaseModel):
    home_team: str
    away_team: str
    home_win_pct: float
    draw_pct: float
    away_win_pct: float
    model_version: str
