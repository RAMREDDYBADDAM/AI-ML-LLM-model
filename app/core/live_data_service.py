# app/core/live_data_service.py
"""
LIVE DATA SERVICE - Fetches real-time market data from Yahoo Finance.

FEATURES:
- Real-time stock prices with timestamps
- Price change calculations (day, 5-day, month)
- Trend detection (Bullish/Bearish/Sideways)
- Data freshness enforcement (15-min threshold)
- Confidence scoring based on data quality

USAGE:
    from app.core.live_data_service import (
        get_live_stock_data,
        get_market_overview,
        get_live_response,
    )
"""
import os
import re
from datetime import datetime, timedelta
from typing import Dict, Any, List, Optional, Literal
import logging

logger = logging.getLogger(__name__)

# Data freshness threshold (15 minutes)
DATA_FRESHNESS_THRESHOLD_MINUTES = 15


class LiveDataService:
    """
    Service for fetching and formatting live market data.
    Always includes timestamps and freshness indicators.
    """
    
    def __init__(self):
        self._yahoo_service = None
        self._cache_time = None
    
    @property
    def yahoo(self):
        """Lazy load Yahoo service."""
        if self._yahoo_service is None:
            try:
                from app.core.yahoo_service import yahoo_service
                self._yahoo_service = yahoo_service
            except ImportError:
                logger.error("Yahoo Finance service not available")
                self._yahoo_service = None
        return self._yahoo_service
    
    def is_available(self) -> bool:
        """Check if live data service is available."""
        if self.yahoo is None:
            return False
        return self.yahoo.is_available()
    
    def _calculate_trend(self, current: float, prev_close: float, 
                         week_ago: Optional[float] = None) -> Literal["Bullish", "Bearish", "Sideways"]:
        """
        Determine trend based on price movements.
        
        Bullish: >1% up from previous close
        Bearish: >1% down from previous close
        Sideways: Within +/-1%
        """
        if prev_close is None or prev_close == 0:
            return "Sideways"
        
        day_change_pct = ((current - prev_close) / prev_close) * 100
        
        if day_change_pct > 1.0:
            return "Bullish"
        elif day_change_pct < -1.0:
            return "Bearish"
        else:
            return "Sideways"
    
    def _calculate_confidence(self, data: Dict) -> Literal["Low", "Medium", "High"]:
        """
        Calculate confidence based on data completeness.
        
        High: All fields present, data fresh
        Medium: Most fields present OR data slightly delayed
        Low: Missing critical fields OR data stale
        """
        required_fields = ['current_price', 'market_cap', 'pe_ratio']
        present = sum(1 for f in required_fields if data.get(f) is not None)
        
        if present == len(required_fields):
            return "High"
        elif present >= 1:
            return "Medium"
        else:
            return "Low"
    
    def _check_data_freshness(self, timestamp: datetime) -> tuple[bool, str]:
        """
        Check if data is within freshness threshold.
        
        Returns:
            (is_fresh: bool, message: str)
        """
        now = datetime.now()
        age_minutes = (now - timestamp).total_seconds() / 60
        
        if age_minutes <= DATA_FRESHNESS_THRESHOLD_MINUTES:
            return True, ""
        else:
            return False, f"Live data may be delayed ({int(age_minutes)} minutes old)"
    
    def get_live_stock_data(self, ticker: str) -> Dict[str, Any]:
        """
        Fetch live stock data for a single ticker.
        
        Returns standardized response with:
        - live_price, price_change, trend
        - timestamp, confidence
        - reasoning array
        """
        timestamp = datetime.now()
        
        if not self.is_available():
            return self._error_response("Yahoo Finance service unavailable", timestamp)
        
        try:
            from app.core.yahoo_service import get_company_info, get_stock_history
            
            # Get current company info
            info = get_company_info(ticker.upper())
            if not info or not info.get('current_price'):
                return self._error_response(f"No data found for {ticker}", timestamp)
            
            current_price = info.get('current_price')
            prev_close = info.get('52_week_low')  # Fallback if no prev close
            
            # Try to get recent history for better trend analysis
            history = get_stock_history(ticker, period="5d", interval="1d")
            
            # Calculate changes
            day_change = 0
            day_change_pct = 0
            week_change_pct = 0
            
            if history is not None and len(history) >= 2:
                prev_close = history.iloc[-2]['close'] if 'close' in history.columns else current_price
                week_open = history.iloc[0]['open'] if 'open' in history.columns else current_price
                
                day_change = current_price - prev_close
                day_change_pct = (day_change / prev_close * 100) if prev_close else 0
                week_change_pct = ((current_price - week_open) / week_open * 100) if week_open else 0
            
            # Determine trend
            trend = self._calculate_trend(current_price, prev_close)
            
            # Build reasoning
            reasoning = []
            if day_change_pct > 0:
                reasoning.append(f"Price up {day_change_pct:.2f}% today")
            elif day_change_pct < 0:
                reasoning.append(f"Price down {abs(day_change_pct):.2f}% today")
            else:
                reasoning.append("Price unchanged today")
            
            if week_change_pct != 0:
                direction = "up" if week_change_pct > 0 else "down"
                reasoning.append(f"5-day trend: {direction} {abs(week_change_pct):.2f}%")
            
            if info.get('pe_ratio'):
                reasoning.append(f"P/E ratio: {info['pe_ratio']:.2f}")
            
            # Check freshness
            is_fresh, freshness_msg = self._check_data_freshness(timestamp)
            if not is_fresh:
                reasoning.insert(0, freshness_msg)
            
            return {
                "summary": f"{info.get('name', ticker)} ({ticker}) trading at ${current_price:.2f}",
                "live_price": current_price,
                "price_change": f"{'+' if day_change >= 0 else ''}{day_change_pct:.2f}%",
                "trend": trend,
                "reasoning": reasoning,
                "timestamp": timestamp.isoformat(),
                "confidence": self._calculate_confidence(info),
                "data": {
                    "ticker": ticker.upper(),
                    "name": info.get('name'),
                    "market_cap": info.get('market_cap'),
                    "pe_ratio": info.get('pe_ratio'),
                    "eps": info.get('eps'),
                    "52w_high": info.get('52_week_high'),
                    "52w_low": info.get('52_week_low'),
                    "sector": info.get('sector'),
                    "industry": info.get('industry')
                },
                "source": "yahoo_finance",
                "is_fresh": is_fresh
            }
            
        except Exception as e:
            logger.error(f"Error fetching live data for {ticker}: {e}")
            return self._error_response(str(e), timestamp)
    
    def get_market_overview(self, top_n: int = 10) -> Dict[str, Any]:
        """
        Get live market overview with top N companies by market cap.
        """
        timestamp = datetime.now()
        
        if not self.is_available():
            return self._error_response("Yahoo Finance service unavailable", timestamp)
        
        try:
            companies = self.yahoo.get_top_companies_data()
            if not companies:
                return self._error_response("No market data available", timestamp)
            
            # Sort by market cap
            sorted_companies = sorted(
                [c for c in companies if c.get('market_cap')],
                key=lambda x: x.get('market_cap', 0),
                reverse=True
            )[:top_n]
            
            # Calculate overall market trend
            gainers = sum(1 for c in sorted_companies if c.get('current_price', 0) > 0)
            
            top_list = []
            for i, c in enumerate(sorted_companies, 1):
                price = c.get('current_price')
                cap = c.get('market_cap', 0)
                top_list.append({
                    "rank": i,
                    "ticker": c.get('ticker'),
                    "name": c.get('name', '')[:30],
                    "live_price": price,
                    "market_cap": self._format_market_cap(cap),
                    "sector": c.get('sector', 'N/A')[:20]
                })
            
            return {
                "summary": f"Top {len(top_list)} companies by market cap",
                "trend": "Bullish" if gainers > len(sorted_companies) / 2 else "Mixed",
                "reasoning": [
                    f"Showing top {len(top_list)} by market capitalization",
                    "Rankings based on current market values",
                    "Prices from Yahoo Finance"
                ],
                "timestamp": timestamp.isoformat(),
                "confidence": "High" if len(top_list) >= 5 else "Medium",
                "top_companies": top_list,
                "metric": "market_cap",
                "source": "yahoo_finance",
                "is_fresh": True
            }
            
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return self._error_response(str(e), timestamp)
    
    def _format_market_cap(self, value: float) -> str:
        """Format market cap in human readable form."""
        if value is None:
            return "N/A"
        if value >= 1e12:
            return f"${value/1e12:.2f}T"
        if value >= 1e9:
            return f"${value/1e9:.2f}B"
        if value >= 1e6:
            return f"${value/1e6:.2f}M"
        return f"${value:,.0f}"
    
    def _error_response(self, error: str, timestamp: datetime) -> Dict[str, Any]:
        """Generate standardized error response."""
        return {
            "summary": "Unable to fetch live data",
            "live_price": None,
            "price_change": None,
            "trend": "Unknown",
            "reasoning": [error, "Falling back to cached data if available"],
            "timestamp": timestamp.isoformat(),
            "confidence": "Low",
            "source": "error",
            "is_fresh": False
        }


# ============================================================
# SINGLETON & CONVENIENCE FUNCTIONS
# ============================================================

live_data_service = LiveDataService()


def get_live_stock_data(ticker: str) -> Dict[str, Any]:
    """Get live stock data for a ticker."""
    return live_data_service.get_live_stock_data(ticker)


def get_market_overview(top_n: int = 10) -> Dict[str, Any]:
    """Get live market overview."""
    return live_data_service.get_market_overview(top_n)


def is_live_data_available() -> bool:
    """Check if live data service is available."""
    return live_data_service.is_available()


def get_live_response(question: str) -> Dict[str, Any]:
    """
    Main entry point for live data queries.
    Parses question and routes to appropriate handler.
    """
    q = question.lower()
    
    # Extract ticker if mentioned
    ticker_pattern = r'\b([A-Z]{1,5})\b'
    
    # Known company name to ticker mapping
    company_map = {
        'apple': 'AAPL', 'microsoft': 'MSFT', 'google': 'GOOGL',
        'alphabet': 'GOOGL', 'amazon': 'AMZN', 'nvidia': 'NVDA',
        'tesla': 'TSLA', 'meta': 'META', 'facebook': 'META',
        'netflix': 'NFLX', 'amd': 'AMD', 'intel': 'INTC',
        'walmart': 'WMT', 'jpmorgan': 'JPM', 'berkshire': 'BRK-B'
    }
    
    # Check for company name
    for name, ticker in company_map.items():
        if name in q:
            return get_live_stock_data(ticker)
    
    # Check for ticker symbols in original question
    tickers = re.findall(r'\b([A-Z]{1,5})\b', question)
    common_tickers = {'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META', 'NFLX', 'AMD', 'JPM'}
    for t in tickers:
        if t in common_tickers:
            return get_live_stock_data(t)
    
    # Check for "top N" pattern
    top_match = re.search(r'top\s*(\d+)', q)
    if top_match or any(kw in q for kw in ['top', 'largest', 'biggest', 'best']):
        n = int(top_match.group(1)) if top_match else 10
        return get_market_overview(top_n=n)
    
    # Default to market overview
    return get_market_overview(top_n=5)
