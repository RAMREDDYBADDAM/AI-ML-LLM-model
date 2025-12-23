"""
S&P 500 Analytics Module - Advanced time-series analysis and visualization

Provides comprehensive analytics for S&P 500 historical data including:
- Time-series analysis
- Statistical metrics
- Correlation analysis
- Trend detection
- Comparative analysis
"""

from typing import Dict, Any, List, Optional
import pandas as pd
from datetime import datetime, timedelta
from app.config import settings
import os

from app.core.response_utils import (
    analytics_response,
    merge_date_range,
    safe_int,
    safe_number,
)

# In-memory cache for S&P 500 data
_sp500_cache = None


def clear_sp500_cache():
    """Clear the S&P 500 data cache to force reload from source."""
    global _sp500_cache
    _sp500_cache = None
    print("S&P 500 cache cleared")


def _ensure_datetime(df: pd.DataFrame) -> pd.DataFrame:
    if 'date' in df.columns and df['date'].dtype == 'object':
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
    return df


def _meta_for_df(df: pd.DataFrame, aggregation: str) -> Dict[str, Any]:
    df = _ensure_datetime(df)
    start = df['date'].min() if 'date' in df.columns and not df.empty else None
    end = df['date'].max() if 'date' in df.columns and not df.empty else None
    return {
        "company": "SP500 Index",
        "date_range": merge_date_range(start, end),
        "aggregation": aggregation,
    }


# ============================================================
# YAHOO FINANCE INTEGRATION
# ============================================================
# Set USE_YAHOO_DATA=true in .env to enable Yahoo Finance
# Set USE_YAHOO_DATA=false to use only local CSV data

def _load_sp500_from_yahoo() -> Optional[pd.DataFrame]:
    """
    YAHOO FINANCE INTEGRATION
    Try to load S&P 500 data from Yahoo Finance.
    Returns None if Yahoo is disabled or unavailable.
    """
    try:
        from app.core.yahoo_service import is_yahoo_available, get_sp500_index
        
        if not is_yahoo_available():
            return None
        
        print("Fetching S&P 500 data from Yahoo Finance...")
        df = get_sp500_index(period="max", interval="1mo")
        
        if df is not None and not df.empty:
            print(f"Yahoo Finance: Loaded {len(df)} S&P 500 data points")
            return df
            
    except ImportError:
        print("Yahoo service not available (yfinance not installed)")
    except Exception as e:
        print(f"Error loading from Yahoo Finance: {e}")
    
    return None


def _load_sp500_from_csv() -> Optional[pd.DataFrame]:
    """Try to load S&P 500 data from local CSV file."""
    try:
        csv_path = os.path.join(os.path.dirname(__file__), "..", "..", "data", "sample_financial_data.csv")
        if os.path.exists(csv_path):
            df = pd.read_csv(csv_path)
            df = df.rename(columns={
                'Date': 'date',
                'SP500': 'sp500',
                'Dividend': 'dividend',
                'Earnings': 'earnings',
                'Consumer Price Index': 'consumer_price_index',
                'Long Interest Rate': 'long_interest_rate',
                'Real Price': 'real_price',
                'Real Dividend': 'real_dividend',
                'Real Earnings': 'real_earnings',
                'PE10': 'pe10'
            })
            df.columns = df.columns.str.lower()
            # Handle mixed date formats (with/without time component)
            df['date'] = pd.to_datetime(df['date'], format='mixed')
            return df
    except Exception as e:
        print(f"Error loading CSV: {e}")
    return None


def get_sp500_data() -> pd.DataFrame:
    """
    Get S&P 500 data from cache, Yahoo Finance, or CSV.
    
    DATA SOURCE PRIORITY:
    1. In-memory cache (if available)
    2. Yahoo Finance (if USE_YAHOO_DATA=true and yfinance installed)
    3. Local CSV file (fallback)
    
    To disable Yahoo Finance: set USE_YAHOO_DATA=false in .env
    """
    global _sp500_cache
    
    if _sp500_cache is not None and not _sp500_cache.empty:
        return _sp500_cache
    
    # YAHOO FINANCE: Try Yahoo first if enabled
    yahoo_df = _load_sp500_from_yahoo()
    
    # CSV FALLBACK: Load local CSV
    csv_df = _load_sp500_from_csv()
    
    # MERGE STRATEGY: Combine data sources intelligently
    if yahoo_df is not None and not yahoo_df.empty:
        if csv_df is not None and not csv_df.empty:
            # Smart merge: Use CSV for fundamental data (PE10, earnings, dividend)
            # and Yahoo for latest price data
            try:
                # Ensure date columns are datetime
                csv_df = csv_df.copy()
                yahoo_df = yahoo_df.copy()
                csv_df['date'] = pd.to_datetime(csv_df['date'])
                yahoo_df['date'] = pd.to_datetime(yahoo_df['date'])
                
                # Get fundamental columns from CSV that Yahoo doesn't have
                fundamental_cols = ['dividend', 'earnings', 'consumer_price_index', 
                                   'long_interest_rate', 'real_price', 'real_dividend', 
                                   'real_earnings', 'pe10']
                csv_has_fundamentals = any(col in csv_df.columns for col in fundamental_cols)
                
                if csv_has_fundamentals:
                    # Use CSV as base (has fundamental data), merge Yahoo price updates
                    # First, update CSV with any newer Yahoo data
                    csv_max_date = csv_df['date'].max()
                    yahoo_newer = yahoo_df[yahoo_df['date'] > csv_max_date]
                    
                    if len(yahoo_newer) > 0:
                        # Add newer Yahoo rows to CSV
                        df = pd.concat([csv_df, yahoo_newer], ignore_index=True)
                        df = df.sort_values('date').reset_index(drop=True)
                        print(f"Merged data: CSV ({len(csv_df)}) + Yahoo newer ({len(yahoo_newer)}) = {len(df)} rows")
                    else:
                        df = csv_df
                        print(f"Using CSV data: {len(df)} rows (has fundamental metrics)")
                else:
                    # CSV doesn't have fundamentals, use Yahoo
                    df = yahoo_df
                    print(f"Using Yahoo data: {len(df)} rows")
            except Exception as e:
                print(f"Merge error: {e}, falling back to CSV")
                df = csv_df if csv_df is not None else yahoo_df
        else:
            df = yahoo_df
    elif csv_df is not None and not csv_df.empty:
        df = csv_df
    else:
        return pd.DataFrame()
    
    _sp500_cache = df
    return df


def ingest_sp500_data(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Ingest S&P 500 CSV data into cache (and optionally PostgreSQL).
    """
    global _sp500_cache
    
    try:
        # Standardize column names
        df.columns = df.columns.str.lower()
        _sp500_cache = df.copy()
        
        return {
            "success": True,
            "rows_inserted": len(df),
            "message": f"Loaded {len(df)} S&P 500 records into cache"
        }
        
    except Exception as e:
        return {
            "success": False,
            "error": f"Ingestion failed: {str(e)}"
        }


def get_sp500_summary() -> Dict[str, Any]:
    """Get S&P 500 data summary."""
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "daily"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available. Please upload the CSV file first."],
        )
    
    try:
        df = _ensure_datetime(df)
        latest_idx = -1 if len(df) > 0 else None
        latest_values = {}
        for metric in ("sp500", "dividend", "pe10"):
            if metric in df.columns and latest_idx is not None:
                latest_values[metric] = safe_number(df[metric].iloc[latest_idx])
            else:
                latest_values[metric] = None

        numeric_cols = [
            "sp500",
            "dividend",
            "earnings",
            "consumer_price_index",
            "long_interest_rate",
            "real_price",
            "real_dividend",
            "real_earnings",
            "pe10",
        ]
        stats_rows: List[Dict[str, Any]] = []
        for col in numeric_cols:
            if col not in df.columns:
                continue
            col_series = pd.to_numeric(df[col], errors='coerce')
            stats_rows.append(
                {
                    "metric": col,
                    "avg": safe_number(col_series.mean()),
                    "min": safe_number(col_series.min()),
                    "max": safe_number(col_series.max()),
                    "stddev": safe_number(col_series.std()),
                }
            )

        summary = {
            "latest_values": latest_values,
            "record_count": safe_int(len(df), 0),
            "status": "ok",
        }

        return analytics_response(meta=_meta_for_df(df, "daily"), data=stats_rows, summary=summary)
    except Exception as e:
        return analytics_response(
            meta=_meta_for_df(df, "daily"),
            data=[],
            summary={"status": "error"},
            errors=[f"Failed to calculate summary: {e}"],
        )


def get_time_series_data(
    start_date: Optional[str] = None,
    end_date: Optional[str] = None,
    metrics: Optional[List[str]] = None,
    limit: int = 1000
) -> Dict[str, Any]:
    """Get time-series data with optional filtering."""
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "daily"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )
    
    try:
        df = _ensure_datetime(df.copy())
        
        if start_date:
            df = df[df['date'] >= pd.to_datetime(start_date)]
        if end_date:
            df = df[df['date'] <= pd.to_datetime(end_date)]
        
        metrics = metrics or ['date', 'sp500', 'dividend', 'earnings', 'pe10']
        if 'date' not in metrics:
            metrics = ['date'] + metrics
        available_metrics = [m for m in metrics if m in df.columns]
        df_filtered = df[available_metrics].tail(limit).reset_index(drop=True)
        
        records: List[Dict[str, Any]] = []
        for _, row in df_filtered.iterrows():
            record: Dict[str, Any] = {}
            for col in available_metrics:
                val = row[col]
                if isinstance(val, pd.Timestamp):
                    record[col] = val.strftime("%Y-%m-%d")
                elif pd.isna(val):
                    record[col] = None
                elif isinstance(val, (int, float)):
                    record[col] = safe_number(val)
                else:
                    try:
                        record[col] = safe_number(float(val))
                    except Exception:
                        record[col] = str(val)
            records.append(record)

        summary = {
            "record_count": safe_int(len(records), 0),
            "metrics": available_metrics,
            "status": "ok",
        }

        return analytics_response(meta=_meta_for_df(df_filtered, "daily"), data=records, summary=summary)
    except Exception as e:
        return analytics_response(
            meta=_meta_for_df(df, "daily"),
            data=[],
            summary={"status": "error"},
            errors=[f"Failed to fetch time series: {e}"],
        )


def get_year_over_year_growth() -> Dict[str, Any]:
    """Calculate year-over-year growth rates for S&P 500."""
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "yearly"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )
    
    try:
        df = _ensure_datetime(df.copy())
        df['year'] = df['date'].dt.year
        yearly = df.groupby('year').agg({'sp500': 'mean'}).reset_index()
        yearly['yoy_growth'] = yearly['sp500'].pct_change() * 100
        yearly = yearly[yearly['year'] >= 1950].tail(50)

        growth_rows: List[Dict[str, Any]] = []
        for _, row in yearly.iterrows():
            growth_rows.append(
                {
                    "year": int(row['year']),
                    "sp500": safe_number(row['sp500']),
                    "yoy_growth": safe_number(row['yoy_growth']),
                }
            )

        summary = {
            "record_count": safe_int(len(growth_rows), 0),
            "status": "ok",
        }

        # meta date range uses min/max year boundaries
        start_year = growth_rows[0]['year'] if growth_rows else None
        end_year = growth_rows[-1]['year'] if growth_rows else None
        meta = {
            "company": "SP500 Index",
            "date_range": merge_date_range(f"{start_year}-01-01" if start_year else None, f"{end_year}-12-31" if end_year else None),
            "aggregation": "yearly",
        }

        return analytics_response(meta=meta, data=growth_rows, summary=summary)
    except Exception as e:
        return analytics_response(
            meta=_meta_for_df(df, "yearly"),
            data=[],
            summary={"status": "error"},
            errors=[f"Failed to calculate YoY growth: {e}"],
        )


def get_sp500_growth_analysis() -> Dict[str, Any]:
    """Compute annual growth percentages over the last 25 years."""
    df = get_sp500_data()

    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "yearly"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )

    try:
        df = _ensure_datetime(df.copy())
        df['year'] = df['date'].dt.year
        yearly = (
            df.groupby('year')['sp500']
            .mean()
            .dropna()
            .reset_index()
            .sort_values('year')
        )
        yearly['return_pct'] = yearly['sp500'].pct_change() * 100

        rows: List[Dict[str, Any]] = []
        for _, row in yearly.tail(50).iterrows():
            rows.append(
                {
                    "period": int(row['year']),
                    "return_pct": safe_number(row['return_pct']) or 0.0,
                }
            )

        summary = {
            "record_count": safe_int(len(rows), 0),
            "status": "ok",
        }
        meta = {
            "company": "SP500 Index",
            "date_range": merge_date_range(
                f"{rows[0]['period']}-01-01" if rows else None,
                f"{rows[-1]['period']}-12-31" if rows else None,
            ),
            "aggregation": "yearly",
        }

        return analytics_response(meta=meta, data=rows, summary=summary)
    except Exception as e:
        return analytics_response(
            meta=_meta_for_df(df, "yearly"),
            data=[],
            summary={"status": "error"},
            errors=[f"Failed to calculate growth analysis: {e}"],
        )


def get_correlation_matrix() -> Dict[str, Any]:
    """Calculate correlation matrix for key metrics."""
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "correlation"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )
    
    try:
        df_numeric = df.copy()
        metrics = ['sp500', 'dividend', 'earnings', 'consumer_price_index', 'long_interest_rate', 'pe10']
        df_numeric = df_numeric[[m for m in metrics if m in df_numeric.columns]].apply(pd.to_numeric, errors='coerce')

        rows: List[Dict[str, Any]] = []
        for col in df_numeric.columns:
            if col == 'sp500':
                continue
            corr = df_numeric['sp500'].corr(df_numeric[col])
            if pd.isna(corr):
                continue
            rows.append({"metric": col, "correlation": safe_number(corr)})

        summary = {
            "record_count": safe_int(len(rows), 0),
            "status": "ok",
        }

        return analytics_response(meta=_meta_for_df(df_numeric, "correlation"), data=rows, summary=summary)
    except Exception as e:
        return analytics_response(
            meta=_meta_for_df(df, "correlation"),
            data=[],
            summary={"status": "error"},
            errors=[f"Failed to calculate correlations: {e}"],
        )


def get_decade_performance() -> Dict[str, Any]:
    """Get performance summary by decade."""
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "decade"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )
    
    try:
        df = _ensure_datetime(df.copy())
        df['decade'] = (df['date'].dt.year // 10) * 10
        decade_stats = df.groupby('decade').agg({
            'sp500': ['count', 'mean', 'min', 'max'],
            'pe10': 'mean',
            'dividend': 'mean',
            'long_interest_rate': 'mean'
        }).reset_index()
        
        rows: List[Dict[str, Any]] = []
        for _, row in decade_stats.iterrows():
            # Use .iloc[0] to properly extract scalar values from Series
            decade_val = row['decade']
            if hasattr(decade_val, 'iloc'):
                decade_val = decade_val.iloc[0]
            rows.append(
                {
                    "decade": int(decade_val),
                    "data_points": safe_int(row[('sp500', 'count')], 0),
                    "avg_sp500": safe_number(row[('sp500', 'mean')]),
                    "min_sp500": safe_number(row[('sp500', 'min')]),
                    "max_sp500": safe_number(row[('sp500', 'max')]),
                    "avg_pe10": safe_number(row[('pe10', 'mean')]),
                    "avg_dividend": safe_number(row[('dividend', 'mean')]),
                }
            )

        rows = list(reversed(rows))
        summary = {"record_count": safe_int(len(rows), 0), "status": "ok"}
        meta = _meta_for_df(df, "decade")
        return analytics_response(meta=meta, data=rows, summary=summary)
    except Exception as e:
        return analytics_response(
            meta=_meta_for_df(df, "decade"),
            data=[],
            summary={"status": "error"},
            errors=[f"Failed to fetch decade performance: {e}"],
        )


def get_volatility_analysis(period_days: int = 365) -> Dict[str, Any]:
    """Calculate volatility metrics for specified period."""
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "volatility"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )
    
    try:
        df = _ensure_datetime(df.copy())
        start_date = df['date'].max() - timedelta(days=period_days)
        df_period = df[df['date'] >= start_date].copy()
        df_period = df_period.sort_values('date')
        df_period['sp500_numeric'] = pd.to_numeric(df_period['sp500'], errors='coerce')
        df_period['daily_return'] = df_period['sp500_numeric'].pct_change() * 100

        volatility = safe_number(df_period['daily_return'].std())
        avg_return = safe_number(df_period['daily_return'].mean())
        min_return = safe_number(df_period['daily_return'].min())
        max_return = safe_number(df_period['daily_return'].max())

        rows: List[Dict[str, Any]] = []
        for _, row in df_period.tail(120).iterrows():
            rows.append(
                {
                    "date": row['date'].strftime('%Y-%m-%d') if isinstance(row['date'], pd.Timestamp) else str(row['date']),
                    "daily_return": safe_number(row['daily_return']),
                }
            )

        summary = {
            "period_days": period_days,
            "volatility": volatility,
            "avg_daily_return": avg_return,
            "min_return": min_return,
            "max_return": max_return,
            "status": "ok",
        }

        meta = {
            "company": "SP500 Index",
            "date_range": merge_date_range(df_period['date'].min(), df_period['date'].max()),
            "aggregation": "daily",
        }

        return analytics_response(meta=meta, data=rows, summary=summary)
    except Exception as e:
        return analytics_response(
            meta=_meta_for_df(df, "volatility"),
            data=[],
            summary={"status": "error"},
            errors=[f"Failed to calculate volatility: {e}"],
        )


# ============================================================
# ADVANCED FINANCIAL ANALYTICS
# ============================================================

def get_market_insights() -> Dict[str, Any]:
    """
    Generate comprehensive market insights based on computed metrics.
    
    Computes:
    - Rolling returns (1Y, 5Y, 10Y CAGR)
    - Rolling volatility (30D, 1Y)
    - Maximum drawdown
    - Market regime (Bull/Bear/Sideways using 200-day MA)
    - Valuation status (using P/E percentiles)
    - Trend strength score (0-100)
    - Risk warnings and opportunity signals
    """
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "insights"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )
    
    try:
        df = _ensure_datetime(df.copy())
        df = df.sort_values('date')
        df['sp500_numeric'] = pd.to_numeric(df['sp500'], errors='coerce')
        df['pe10_numeric'] = pd.to_numeric(df.get('pe10', pd.Series()), errors='coerce')
        
        latest = df.iloc[-1]
        latest_price = float(latest['sp500_numeric']) if pd.notna(latest['sp500_numeric']) else 0
        latest_date = latest['date'].strftime('%Y-%m-%d') if hasattr(latest['date'], 'strftime') else str(latest['date'])
        
        # ============ ROLLING RETURNS (CAGR) ============
        def calc_cagr(start_val, end_val, years):
            if start_val <= 0 or end_val <= 0 or years <= 0:
                return None
            return ((end_val / start_val) ** (1 / years) - 1) * 100
        
        returns = {}
        for years, label in [(1, "1Y"), (5, "5Y"), (10, "10Y")]:
            lookback = df[df['date'] >= (df['date'].max() - timedelta(days=years*365))]
            if len(lookback) >= 2:
                start_price = float(lookback['sp500_numeric'].iloc[0])
                end_price = float(lookback['sp500_numeric'].iloc[-1])
                cagr = calc_cagr(start_price, end_price, years)
                returns[label] = round(cagr, 2) if cagr else None
            else:
                returns[label] = None
        
        # ============ VOLATILITY ============
        # Detect data frequency (monthly vs daily)
        date_diff = df['date'].diff().dt.days.median()
        is_monthly = date_diff > 20  # Monthly data has ~30 days between records
        annualization_factor = 12 if is_monthly else 252  # 12 months or 252 trading days
        
        df['period_return'] = df['sp500_numeric'].pct_change() * 100
        
        # Recent volatility (annualized)
        lookback_periods = 12 if is_monthly else 30  # 1 year for monthly, 30 days for daily
        vol_recent = df['period_return'].tail(lookback_periods).std() * (annualization_factor ** 0.5) if len(df) >= lookback_periods else None
        
        # 1-year volatility (annualized)
        periods_in_year = 12 if is_monthly else 252
        df_1y = df.tail(periods_in_year) if len(df) >= periods_in_year else df
        vol_1y = df_1y['period_return'].std() * (annualization_factor ** 0.5) if len(df_1y) >= (6 if is_monthly else 30) else None
        
        # Calculate volatility percentile using rolling window approach
        vol_percentile = None
        if vol_1y is not None:
            # Calculate rolling volatilities for percentile
            window_size = periods_in_year
            min_periods = window_size // 2
            df['rolling_vol'] = df['period_return'].rolling(window=window_size, min_periods=min_periods).std() * (annualization_factor ** 0.5)
            rolling_vols = df['rolling_vol'].dropna()
            if len(rolling_vols) > 10:
                vol_percentile = (rolling_vols < vol_1y).mean() * 100
        
        # ============ MAXIMUM DRAWDOWN ============
        df['cummax'] = df['sp500_numeric'].cummax()
        df['drawdown'] = (df['sp500_numeric'] - df['cummax']) / df['cummax'] * 100
        max_drawdown = float(df['drawdown'].min()) if not df['drawdown'].isna().all() else None
        
        # Current drawdown
        current_drawdown = float(df['drawdown'].iloc[-1]) if not pd.isna(df['drawdown'].iloc[-1]) else 0
        
        # ============ MARKET REGIME (200-day MA equivalent) ============
        # For monthly data: 200 days ≈ 10 months
        # For daily data: 200 days
        ma_window = 10 if is_monthly else 200
        ma_min_periods = 5 if is_monthly else 50
        df['ma_200'] = df['sp500_numeric'].rolling(window=ma_window, min_periods=ma_min_periods).mean()
        latest_ma200 = float(df['ma_200'].iloc[-1]) if pd.notna(df['ma_200'].iloc[-1]) else None
        
        if latest_ma200 and latest_price:
            ma_diff_pct = ((latest_price - latest_ma200) / latest_ma200) * 100
            if ma_diff_pct > 5:
                regime = "Bull"
                regime_desc = "Above 200-day MA"
            elif ma_diff_pct > 0:
                regime = "Bullish"
                regime_desc = "Above 200-day MA"
            elif ma_diff_pct > -5:
                regime = "Sideways"
                regime_desc = "Near 200-day MA"
            else:
                regime = "Bear"
                regime_desc = "Below 200-day MA"
        else:
            regime = "Unknown"
            regime_desc = "Insufficient data"
            ma_diff_pct = None
        
        # ============ VALUATION STATUS (P/E Percentile) ============
        pe_current = float(latest['pe10_numeric']) if pd.notna(latest.get('pe10_numeric')) else None
        pe_percentile = None
        valuation_status = "Unknown"
        
        if pe_current and not df['pe10_numeric'].isna().all():
            pe_series = df['pe10_numeric'].dropna()
            pe_percentile = (pe_series < pe_current).mean() * 100
            
            if pe_percentile >= 80:
                valuation_status = "Overvalued"
            elif pe_percentile >= 60:
                valuation_status = "Elevated"
            elif pe_percentile >= 40:
                valuation_status = "Fair"
            elif pe_percentile >= 20:
                valuation_status = "Attractive"
            else:
                valuation_status = "Undervalued"
        
        pe_avg = float(df['pe10_numeric'].mean()) if not df['pe10_numeric'].isna().all() else None
        
        # ============ TREND STRENGTH SCORE (0-100) ============
        trend_score = 50  # Base score
        
        # Price vs MA200 contribution (+/- 20 points)
        if ma_diff_pct is not None:
            trend_score += min(20, max(-20, ma_diff_pct * 2))
        
        # Recent momentum (3-month return) contribution (+/- 15 points)
        df_3m = df[df['date'] >= (df['date'].max() - timedelta(days=90))]
        if len(df_3m) >= 2:
            momentum_3m = ((df_3m['sp500_numeric'].iloc[-1] / df_3m['sp500_numeric'].iloc[0]) - 1) * 100
            trend_score += min(15, max(-15, momentum_3m))
        
        # Volatility contribution (-15 for high vol, +10 for low vol)
        # Compare current volatility to historical median
        if vol_1y is not None:
            hist_vol_median = df['rolling_vol'].median() if 'rolling_vol' in df.columns and not df['rolling_vol'].isna().all() else None
            if hist_vol_median:
                if vol_1y > hist_vol_median * 1.2:
                    trend_score -= 15
                elif vol_1y < hist_vol_median * 0.8:
                    trend_score += 10
        
        trend_score = min(100, max(0, round(trend_score)))
        
        # ============ GENERATE INSIGHTS ============
        insights = []
        warnings = []
        opportunities = []
        
        # Market summary - use clearer language for MA comparison
        if regime in ["Bull", "Bullish"]:
            if abs(ma_diff_pct) > 10:
                insights.append(f"Market is in a strong bull phase, significantly above the 200-day moving average (+{abs(ma_diff_pct):.1f}%).")
            else:
                insights.append(f"Market is in a bull phase, trading {abs(ma_diff_pct):.1f}% above the 200-day moving average.")
        elif regime == "Bear":
            if abs(ma_diff_pct) > 10:
                insights.append(f"Market is in a bearish phase, significantly below the 200-day moving average ({ma_diff_pct:.1f}%).")
            else:
                insights.append(f"Market is in a bearish phase, trading {abs(ma_diff_pct):.1f}% below the 200-day moving average.")
        else:
            insights.append(f"Market is consolidating near the 200-day moving average ({ma_diff_pct:+.1f}%).")
        
        # Volatility insight
        if vol_1y:
            if vol_percentile and vol_percentile > 70:
                warnings.append(f"Volatility is elevated ({vol_1y:.1f}% annualized), indicating higher risk.")
            elif vol_percentile and vol_percentile < 30:
                insights.append(f"Volatility is below average ({vol_1y:.1f}% annualized), indicating calmer markets.")
        
        # Valuation insight
        if pe_percentile is not None:
            if pe_percentile >= 80:
                warnings.append(f"Valuation is elevated (PE10: {pe_current:.1f}, {pe_percentile:.0f}th percentile historically).")
            elif pe_percentile <= 30:
                opportunities.append(f"Valuation is attractive (PE10: {pe_current:.1f}, {pe_percentile:.0f}th percentile historically).")
        
        # Drawdown opportunity
        if current_drawdown and current_drawdown < -15:
            opportunities.append(f"Current drawdown of {current_drawdown:.1f}% may present buying opportunity if trend is improving.")
        
        # Build summary text
        summary_text = " ".join(insights)
        if warnings:
            summary_text += " ⚠️ " + " ".join(warnings)
        if opportunities:
            summary_text += " 💡 " + " ".join(opportunities)
        
        # ============ RESPONSE ============
        result = {
            "latest_price": round(latest_price, 2),
            "latest_date": latest_date,
            "returns": {
                "1y": round(returns.get("1Y", 0) / 100, 4) if returns.get("1Y") else None,
                "5y_cagr": round(returns.get("5Y", 0) / 100, 4) if returns.get("5Y") else None,
                "10y_cagr": round(returns.get("10Y", 0) / 100, 4) if returns.get("10Y") else None,
            },
            "volatility": {
                "30d": round(vol_recent / 100, 4) if vol_recent else None,
                "1y": round(vol_1y / 100, 4) if vol_1y else None,
                "percentile": round(vol_percentile, 0) if vol_percentile else None,
            },
            "max_drawdown": {
                "value": round(max_drawdown / 100, 4) if max_drawdown else None,
                "current": round(current_drawdown / 100, 4) if current_drawdown else 0,
            },
            "market_regime": {
                "regime": regime,
                "description": regime_desc,
                "ma_200": round(latest_ma200, 2) if latest_ma200 else None,
                "price_vs_ma_pct": round(ma_diff_pct, 2) if ma_diff_pct else None,
            },
            "valuation": {
                "current_pe10": round(pe_current, 2) if pe_current else None,
                "avg_pe10": round(pe_avg, 2) if pe_avg else None,
                "percentile": round(pe_percentile, 0) if pe_percentile else None,
                "status": valuation_status,
            },
            "trend_score": {
                "score": trend_score,
                "label": "Strong" if trend_score >= 70 else "Moderate" if trend_score >= 40 else "Weak",
            },
            "summary_text": summary_text,
            "warnings": warnings,
            "opportunities": opportunities,
            "status": "ok",
        }
        
        # Return flat result for easier frontend consumption
        return result
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return {
            "status": "error",
            "error": f"Failed to generate insights: {e}",
        }



def get_enhanced_decade_performance() -> Dict[str, Any]:
    """
    Get enhanced decade performance with CAGR, max drawdown, and volatility.
    """
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "decade"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )
    
    try:
        df = _ensure_datetime(df.copy())
        df = df.sort_values('date')
        df['sp500_numeric'] = pd.to_numeric(df['sp500'], errors='coerce')
        df['decade'] = (df['date'].dt.year // 10) * 10
        df['daily_return'] = df['sp500_numeric'].pct_change()
        
        rows = []
        for decade, group in df.groupby('decade'):
            if len(group) < 10:
                continue
            
            # Start and end values for CAGR
            start_val = group['sp500_numeric'].iloc[0]
            end_val = group['sp500_numeric'].iloc[-1]
            years = (group['date'].max() - group['date'].min()).days / 365.25
            
            # CAGR
            cagr = None
            if start_val > 0 and end_val > 0 and years > 0:
                cagr = ((end_val / start_val) ** (1 / years) - 1) * 100
            
            # Volatility (annualized)
            volatility = group['daily_return'].std() * (252 ** 0.5) * 100 if len(group) > 30 else None
            
            # Max Drawdown
            group = group.copy()
            group['cummax'] = group['sp500_numeric'].cummax()
            group['drawdown'] = (group['sp500_numeric'] - group['cummax']) / group['cummax'] * 100
            max_dd = group['drawdown'].min()
            
            # Total return
            total_return = ((end_val / start_val) - 1) * 100 if start_val > 0 else None
            
            rows.append({
                "decade": int(decade),
                "label": f"{int(decade)}s",
                "start_price": round(start_val, 2) if pd.notna(start_val) else None,
                "end_price": round(end_val, 2) if pd.notna(end_val) else None,
                "cagr": round(cagr / 100, 4) if cagr else None,
                "total_return": round(total_return / 100, 4) if total_return else None,
                "volatility": round(volatility / 100, 4) if volatility else None,
                "max_drawdown": round(max_dd / 100, 4) if pd.notna(max_dd) else None,
                "data_points": len(group),
            })
        
        # Sort by decade ascending for chart display
        rows = sorted(rows, key=lambda x: x['decade'])
        
        return {
            "status": "ok",
            "decades": rows,
            "count": len(rows),
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to calculate decade performance: {e}",
        }


def get_full_correlation_matrix() -> Dict[str, Any]:
    """
    Get full correlation matrix between all metrics.
    """
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "correlation"),
            data=[],
            summary={"status": "no-data"},
            errors=["No S&P 500 data available"],
        )
    
    try:
        metrics = ['sp500', 'dividend', 'earnings', 'pe10']
        labels = {
            'sp500': 'S&P 500',
            'dividend': 'Dividend',
            'earnings': 'Earnings',
            'pe10': 'PE10'
        }
        
        available = [m for m in metrics if m in df.columns]
        df_numeric = df[available].apply(pd.to_numeric, errors='coerce')
        
        corr_matrix = df_numeric.corr()
        
        # Build matrix as nested structure
        matrix_data = []
        for row_metric in available:
            row_data = {
                "metric": row_metric,
                "label": labels.get(row_metric, row_metric.upper()),
            }
            for col_metric in available:
                val = corr_matrix.loc[row_metric, col_metric]
                row_data[col_metric] = round(val, 3) if pd.notna(val) else None
            matrix_data.append(row_data)
        
        # Also provide flat correlation list for S&P 500
        correlations = {}
        for col in available:
            if col != 'sp500':
                val = corr_matrix.loc['sp500', col]
                correlations[col] = {
                    "value": round(val, 3) if pd.notna(val) else None,
                    "strength": _correlation_strength(val) if pd.notna(val) else "N/A",
                }
        
        return {
            "status": "ok",
            "correlations": correlations,
            "metrics": available,
        }
        
    except Exception as e:
        return {
            "status": "error",
            "error": f"Failed to calculate correlations: {e}",
        }


def _correlation_strength(corr: float) -> str:
    """Classify correlation strength."""
    abs_corr = abs(corr)
    if abs_corr >= 0.8:
        return "Very Strong"
    elif abs_corr >= 0.6:
        return "Strong"
    elif abs_corr >= 0.4:
        return "Moderate"
    elif abs_corr >= 0.2:
        return "Weak"
    else:
        return "Very Weak"


def get_sp500_timeseries(metric: str = "sp500", start_year: Optional[int] = None) -> Dict[str, Any]:
    """Get time series data for a specific metric."""
    df = get_sp500_data()
    
    if df is None or df.empty:
        return analytics_response(
            meta=_meta_for_df(pd.DataFrame(columns=['date']), "daily"),
            data=[],
            summary={"status": "no-data"},
            errors=["No data available"],
        )
    
    try:
        df = _ensure_datetime(df.copy())
        if metric not in df.columns:
            return analytics_response(
                meta=_meta_for_df(df, "daily"),
                data=[],
                summary={"status": "error"},
                errors=[f"Metric '{metric}' not found in data"],
            )
        if start_year and 'date' in df.columns:
            df = df[df['date'].dt.year >= start_year]
        
        rows: List[Dict[str, Any]] = []
        for _, row in df.iterrows():
            rows.append(
                {
                    "date": row['date'].strftime('%Y-%m-%d') if hasattr(row['date'], 'strftime') else str(row['date']),
                    "value": safe_number(row[metric]),
                }
            )
        
        summary = {
            "metric": metric,
            "record_count": safe_int(len(rows), 0),
            "status": "ok",
        }
        return analytics_response(meta=_meta_for_df(df, "daily"), data=rows, summary=summary)
    except Exception as e:
        return analytics_response(
            meta=_meta_for_df(df, "daily"),
            data=[],
            summary={"status": "error"},
            errors=[f"Failed to get time series: {e}"],
        )
