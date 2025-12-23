"""
Yahoo Finance Data Service
===========================
Integrates Yahoo Finance as a data provider for historical stock data.

INTEGRATION POINTS:
- Used by sp500_analytics.py for price data
- Used by sp500_companies.py for company fundamentals
- Can be disabled via USE_YAHOO_DATA=false in .env

USAGE:
    from app.core.yahoo_service import (
        get_stock_history,
        get_company_info,
        get_multiple_stocks,
        yahoo_service,  # singleton instance
    )
    
    # Get historical data
    df = get_stock_history("AAPL", period="1y")
    
    # Get company info
    info = get_company_info("AAPL")

Author: Financial AI Assistant
Version: 1.0.0
"""

import os
import logging
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
from functools import lru_cache
import pandas as pd

# Try to import yfinance - graceful fallback if not installed
try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False
    yf = None

from dotenv import load_dotenv

load_dotenv()

# ============================================================
# CONFIGURATION
# ============================================================

# Configuration flags - can be overridden via environment variables
USE_YAHOO_DATA = os.getenv("USE_YAHOO_DATA", "true").lower() == "true"
YAHOO_CACHE_TTL = int(os.getenv("YAHOO_CACHE_TTL", "3600"))  # 1 hour default

# Default symbols to fetch (configurable via .env)
DEFAULT_SYMBOLS = os.getenv(
    "YAHOO_SYMBOLS",
    "AAPL,MSFT,GOOGL,AMZN,NVDA,TSLA,META,BRK-B,JNJ,V,WMT,JPM,PG,HD,MA"
).split(",")

# Logging setup
logger = logging.getLogger(__name__)

# ============================================================
# IN-MEMORY CACHE
# ============================================================

class YahooCache:
    """
    Lightweight in-memory cache for Yahoo Finance data.
    Prevents repeated API calls within TTL window.
    """
    
    def __init__(self, ttl_seconds: int = 3600):
        self._cache: Dict[str, Tuple[Any, datetime]] = {}
        self._ttl = timedelta(seconds=ttl_seconds)
    
    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if datetime.now() - timestamp < self._ttl:
                logger.debug(f"Cache HIT: {key}")
                return value
            else:
                # Expired - remove from cache
                del self._cache[key]
                logger.debug(f"Cache EXPIRED: {key}")
        return None
    
    def set(self, key: str, value: Any) -> None:
        """Store value in cache with current timestamp."""
        self._cache[key] = (value, datetime.now())
        logger.debug(f"Cache SET: {key}")
    
    def clear(self) -> None:
        """Clear all cached data."""
        self._cache.clear()
        logger.info("Yahoo cache cleared")
    
    def stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "entries": len(self._cache),
            "ttl_seconds": self._ttl.total_seconds(),
        }


# Global cache instance
_yahoo_cache = YahooCache(ttl_seconds=YAHOO_CACHE_TTL)

# ============================================================
# YAHOO FINANCE SERVICE CLASS
# ============================================================

class YahooFinanceService:
    """
    Service class for Yahoo Finance data operations.
    
    Features:
    - Historical OHLCV data
    - Company fundamentals
    - Caching to avoid repeated API calls
    - Graceful error handling for invalid symbols
    """
    
    def __init__(self):
        self.enabled = USE_YAHOO_DATA and YFINANCE_AVAILABLE
        self.cache = _yahoo_cache
        
        if not YFINANCE_AVAILABLE:
            logger.warning(
                "yfinance not installed. Install with: pip install yfinance"
            )
        elif not USE_YAHOO_DATA:
            logger.info("Yahoo Finance disabled via USE_YAHOO_DATA=false")
        else:
            logger.info("Yahoo Finance service initialized")
    
    def is_available(self) -> bool:
        """Check if Yahoo Finance service is available and enabled."""
        return self.enabled
    
    def get_stock_history(
        self,
        symbol: str,
        period: str = "1y",
        interval: str = "1d",
        start: Optional[str] = None,
        end: Optional[str] = None,
    ) -> Optional[pd.DataFrame]:
        """
        Fetch historical OHLCV data for a stock symbol.
        
        Args:
            symbol: Stock ticker symbol (e.g., "AAPL")
            period: Data period - 1d, 5d, 1mo, 3mo, 6mo, 1y, 2y, 5y, 10y, ytd, max
            interval: Data interval - 1m, 2m, 5m, 15m, 30m, 60m, 90m, 1h, 1d, 5d, 1wk, 1mo
            start: Start date (YYYY-MM-DD) - overrides period if provided
            end: End date (YYYY-MM-DD) - overrides period if provided
            
        Returns:
            DataFrame with columns: Open, High, Low, Close, Volume, Adj Close
            Returns None if fetch fails or symbol is invalid
        """
        if not self.enabled:
            logger.debug(f"Yahoo disabled, skipping fetch for {symbol}")
            return None
        
        # Normalize symbol
        symbol = symbol.upper().strip()
        
        # Check cache first
        cache_key = f"history_{symbol}_{period}_{interval}_{start}_{end}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            logger.info(f"Fetching Yahoo data for {symbol} (period={period})")
            
            ticker = yf.Ticker(symbol)
            
            if start and end:
                df = ticker.history(start=start, end=end, interval=interval)
            else:
                df = ticker.history(period=period, interval=interval)
            
            if df is None or df.empty:
                logger.warning(f"No data returned for {symbol}")
                return None
            
            # Standardize column names to lowercase
            df.columns = df.columns.str.lower()
            
            # Reset index to make date a column
            df = df.reset_index()
            df = df.rename(columns={"Date": "date", "index": "date"})
            
            # Ensure date column exists and is datetime
            if "date" in df.columns:
                df["date"] = pd.to_datetime(df["date"])
            
            # Add symbol column for reference
            df["symbol"] = symbol
            
            # Cache the result
            self.cache.set(cache_key, df)
            
            logger.info(f"Fetched {len(df)} rows for {symbol}")
            return df
            
        except Exception as e:
            logger.error(f"Error fetching {symbol}: {e}")
            return None
    
    def get_company_info(self, symbol: str) -> Optional[Dict[str, Any]]:
        """
        Fetch company fundamental information.
        
        Args:
            symbol: Stock ticker symbol
            
        Returns:
            Dictionary with company info including:
            - shortName, longName
            - sector, industry
            - marketCap, revenue, profit
            - website, description
            Returns None if fetch fails
        """
        if not self.enabled:
            return None
        
        symbol = symbol.upper().strip()
        
        # Check cache
        cache_key = f"info_{symbol}"
        cached = self.cache.get(cache_key)
        if cached is not None:
            return cached
        
        try:
            logger.info(f"Fetching company info for {symbol}")
            
            ticker = yf.Ticker(symbol)
            info = ticker.info
            
            if not info or "symbol" not in info:
                logger.warning(f"No info returned for {symbol}")
                return None
            
            # Extract relevant fields with safe defaults
            company_info = {
                "ticker": symbol,
                "name": info.get("shortName") or info.get("longName") or symbol,
                "long_name": info.get("longName"),
                "sector": info.get("sector", "Unknown"),
                "industry": info.get("industry", "Unknown"),
                "market_cap": info.get("marketCap"),
                "revenue": info.get("totalRevenue"),
                "profit": info.get("netIncomeToCommon"),
                "eps": info.get("trailingEps"),
                "pe_ratio": info.get("trailingPE"),
                "dividend_yield": info.get("dividendYield"),
                "52_week_high": info.get("fiftyTwoWeekHigh"),
                "52_week_low": info.get("fiftyTwoWeekLow"),
                "current_price": info.get("currentPrice") or info.get("regularMarketPrice"),
                "website": info.get("website"),
                "description": info.get("longBusinessSummary"),
                "employees": info.get("fullTimeEmployees"),
                "country": info.get("country"),
                "currency": info.get("currency", "USD"),
            }
            
            # Cache result
            self.cache.set(cache_key, company_info)
            
            return company_info
            
        except Exception as e:
            logger.error(f"Error fetching info for {symbol}: {e}")
            return None
    
    def get_multiple_stocks(
        self,
        symbols: List[str],
        period: str = "1y",
        interval: str = "1d",
    ) -> pd.DataFrame:
        """
        Fetch historical data for multiple symbols.
        
        Args:
            symbols: List of ticker symbols
            period: Data period
            interval: Data interval
            
        Returns:
            Combined DataFrame with all symbols' data
        """
        if not self.enabled:
            return pd.DataFrame()
        
        all_data = []
        
        for symbol in symbols:
            df = self.get_stock_history(symbol, period=period, interval=interval)
            if df is not None and not df.empty:
                all_data.append(df)
        
        if not all_data:
            return pd.DataFrame()
        
        return pd.concat(all_data, ignore_index=True)
    
    def get_sp500_index(
        self,
        period: str = "max",
        interval: str = "1mo",
    ) -> Optional[pd.DataFrame]:
        """
        Fetch S&P 500 index historical data.
        Uses ^GSPC (S&P 500) or SPY (ETF) as proxy.
        
        Args:
            period: Data period (default: max for all available history)
            interval: Data interval (default: 1mo for monthly)
            
        Returns:
            DataFrame with S&P 500 historical data
        """
        # Try S&P 500 index first, fall back to SPY ETF
        for symbol in ["^GSPC", "SPY"]:
            df = self.get_stock_history(symbol, period=period, interval=interval)
            if df is not None and not df.empty:
                # Rename 'close' to 'sp500' for compatibility
                if "close" in df.columns:
                    df = df.rename(columns={"close": "sp500"})
                return df
        
        return None
    
    def get_top_companies_data(
        self,
        symbols: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Fetch company info for top S&P 500 companies.
        
        Args:
            symbols: List of symbols (uses DEFAULT_SYMBOLS if None)
            
        Returns:
            List of company info dictionaries
        """
        if not self.enabled:
            return []
        
        symbols = symbols or DEFAULT_SYMBOLS
        companies = []
        
        for symbol in symbols:
            info = self.get_company_info(symbol)
            if info:
                companies.append(info)
        
        # Sort by market cap descending
        companies.sort(
            key=lambda x: x.get("market_cap") or 0,
            reverse=True
        )
        
        return companies
    
    def clear_cache(self) -> None:
        """Clear all cached Yahoo Finance data."""
        self.cache.clear()
    
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        return {
            "enabled": self.enabled,
            "yfinance_available": YFINANCE_AVAILABLE,
            **self.cache.stats()
        }


# ============================================================
# SINGLETON INSTANCE & CONVENIENCE FUNCTIONS
# ============================================================

# Singleton service instance
yahoo_service = YahooFinanceService()


def get_stock_history(
    symbol: str,
    period: str = "1y",
    interval: str = "1d",
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Optional[pd.DataFrame]:
    """Convenience function - see YahooFinanceService.get_stock_history"""
    return yahoo_service.get_stock_history(symbol, period, interval, start, end)


def get_company_info(symbol: str) -> Optional[Dict[str, Any]]:
    """Convenience function - see YahooFinanceService.get_company_info"""
    return yahoo_service.get_company_info(symbol)


def get_multiple_stocks(
    symbols: List[str],
    period: str = "1y",
    interval: str = "1d",
) -> pd.DataFrame:
    """Convenience function - see YahooFinanceService.get_multiple_stocks"""
    return yahoo_service.get_multiple_stocks(symbols, period, interval)


def get_sp500_index(period: str = "max", interval: str = "1mo") -> Optional[pd.DataFrame]:
    """Convenience function - see YahooFinanceService.get_sp500_index"""
    return yahoo_service.get_sp500_index(period, interval)


def is_yahoo_available() -> bool:
    """Check if Yahoo Finance is available and enabled."""
    return yahoo_service.is_available()


# ============================================================
# DATA MERGING UTILITIES
# ============================================================

def merge_with_existing(
    existing_df: pd.DataFrame,
    yahoo_df: pd.DataFrame,
    on_column: str = "date",
    prefer_yahoo: bool = True,
) -> pd.DataFrame:
    """
    Merge Yahoo Finance data with existing dataset.
    
    Args:
        existing_df: Current dataset
        yahoo_df: Yahoo Finance data
        on_column: Column to merge on
        prefer_yahoo: If True, Yahoo data takes precedence for overlapping dates
        
    Returns:
        Merged DataFrame
    """
    if existing_df is None or existing_df.empty:
        return yahoo_df
    
    if yahoo_df is None or yahoo_df.empty:
        return existing_df
    
    # Ensure both have the merge column
    if on_column not in existing_df.columns or on_column not in yahoo_df.columns:
        logger.warning(f"Merge column '{on_column}' not found, returning Yahoo data")
        return yahoo_df if prefer_yahoo else existing_df
    
    # Standardize date formats
    existing_df = existing_df.copy()
    yahoo_df = yahoo_df.copy()
    
    existing_df[on_column] = pd.to_datetime(existing_df[on_column])
    yahoo_df[on_column] = pd.to_datetime(yahoo_df[on_column])
    
    if prefer_yahoo:
        # Keep Yahoo data, fill gaps with existing
        merged = pd.concat([yahoo_df, existing_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=[on_column], keep="first")
    else:
        # Keep existing data, fill gaps with Yahoo
        merged = pd.concat([existing_df, yahoo_df], ignore_index=True)
        merged = merged.drop_duplicates(subset=[on_column], keep="first")
    
    merged = merged.sort_values(on_column).reset_index(drop=True)
    
    logger.info(f"Merged data: {len(merged)} total rows")
    return merged


# ============================================================
# INITIALIZATION CHECK
# ============================================================

if __name__ == "__main__":
    # Quick test when run directly
    print("=" * 50)
    print("Yahoo Finance Service Test")
    print("=" * 50)
    print(f"yfinance installed: {YFINANCE_AVAILABLE}")
    print(f"USE_YAHOO_DATA: {USE_YAHOO_DATA}")
    print(f"Service enabled: {yahoo_service.is_available()}")
    print(f"Default symbols: {DEFAULT_SYMBOLS}")
    
    if yahoo_service.is_available():
        print("\nFetching AAPL data...")
        df = get_stock_history("AAPL", period="5d")
        if df is not None:
            print(f"Got {len(df)} rows")
            print(df.head())
        
        print("\nFetching AAPL info...")
        info = get_company_info("AAPL")
        if info:
            print(f"Name: {info.get('name')}")
            print(f"Sector: {info.get('sector')}")
            print(f"Market Cap: ${info.get('market_cap', 0):,.0f}")
