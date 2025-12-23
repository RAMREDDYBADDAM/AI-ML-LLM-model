#!/usr/bin/env python3
"""
CSV to PostgreSQL Ingestion Pipeline
=====================================
Production-quality data ingestion script for Financial AI Assistant.

This script:
- Loads CSV files from data/raw_csv/
- Validates and cleans the data
- Inserts into PostgreSQL financial_metrics table
- Uses transactions for data integrity
- Provides comprehensive logging

Author: Financial AI Assistant Team
Version: 1.0.0
"""

import os
import sys
import glob
import logging
from pathlib import Path
from datetime import datetime
from typing import Optional, List, Tuple

import pandas as pd
from sqlalchemy import create_engine, text, inspect
from sqlalchemy.engine import Engine
from sqlalchemy.exc import SQLAlchemyError
from dotenv import load_dotenv

# ============================================================
# CONFIGURATION
# ============================================================

# Load environment variables from .env file
load_dotenv()

# Project paths
PROJECT_ROOT = Path(__file__).parent.parent
RAW_CSV_DIR = PROJECT_ROOT / "data" / "raw_csv"
LOG_DIR = PROJECT_ROOT / "logs"

# Database configuration from environment
DB_HOST = os.getenv("DB_HOST", "localhost")
DB_PORT = os.getenv("DB_PORT", "5432")
DB_NAME = os.getenv("DB_NAME", "financial_db")
DB_USER = os.getenv("DB_USER", "postgres")
DB_PASSWORD = os.getenv("DB_PASSWORD", "")

# Table configuration
TARGET_TABLE = "financial_metrics"

# Expected columns and their types
EXPECTED_COLUMNS = {
    "company": str,
    "year": int,
    "revenue": "Int64",  # Nullable integer
    "profit": "Int64",
    "debt": "Int64",
    "eps": float,
    "sector": str,
}

# ============================================================
# LOGGING SETUP
# ============================================================

def setup_logging() -> logging.Logger:
    """
    Configure logging with both file and console handlers.
    
    Returns:
        logging.Logger: Configured logger instance
    """
    # Create logs directory if it doesn't exist
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    # Create logger
    logger = logging.getLogger("csv_to_sql")
    logger.setLevel(logging.DEBUG)
    
    # Clear existing handlers
    logger.handlers.clear()
    
    # File handler - detailed logs
    log_filename = LOG_DIR / f"ingestion_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    file_handler = logging.FileHandler(log_filename, encoding='utf-8')
    file_handler.setLevel(logging.DEBUG)
    file_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(funcName)s:%(lineno)d | %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    file_handler.setFormatter(file_format)
    
    # Console handler - info and above
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_format = logging.Formatter(
        '%(asctime)s | %(levelname)-8s | %(message)s',
        datefmt='%H:%M:%S'
    )
    console_handler.setFormatter(console_format)
    
    # Add handlers
    logger.addHandler(file_handler)
    logger.addHandler(console_handler)
    
    return logger


# Initialize logger
logger = setup_logging()

# ============================================================
# DATABASE FUNCTIONS
# ============================================================

def get_db_engine() -> Optional[Engine]:
    """
    Create and return a SQLAlchemy engine for PostgreSQL connection.
    
    Uses environment variables for configuration:
    - DB_HOST: Database host (default: localhost)
    - DB_PORT: Database port (default: 5432)
    - DB_NAME: Database name (default: financial_db)
    - DB_USER: Database user (default: postgres)
    - DB_PASSWORD: Database password
    
    Returns:
        Engine: SQLAlchemy engine instance, or None if connection fails
    """
    try:
        # Build connection URL
        connection_url = f"postgresql://{DB_USER}:{DB_PASSWORD}@{DB_HOST}:{DB_PORT}/{DB_NAME}"
        
        logger.info(f"Connecting to PostgreSQL at {DB_HOST}:{DB_PORT}/{DB_NAME}")
        
        # Create engine with connection pooling
        engine = create_engine(
            connection_url,
            pool_size=5,
            max_overflow=10,
            pool_timeout=30,
            pool_recycle=1800,  # Recycle connections after 30 minutes
            echo=False,  # Set to True for SQL debugging
        )
        
        # Test connection
        with engine.connect() as conn:
            conn.execute(text("SELECT 1"))
            logger.info("✓ Database connection successful")
        
        return engine
        
    except SQLAlchemyError as e:
        logger.error(f"✗ Database connection failed: {e}")
        return None
    except Exception as e:
        logger.error(f"✗ Unexpected error creating engine: {e}")
        return None


def create_table_if_not_exists(engine: Engine) -> bool:
    """
    Create the financial_metrics table if it doesn't exist.
    
    Args:
        engine: SQLAlchemy engine instance
        
    Returns:
        bool: True if table exists or was created successfully
    """
    create_table_sql = """
    CREATE TABLE IF NOT EXISTS financial_metrics (
        id SERIAL PRIMARY KEY,
        company VARCHAR(100) NOT NULL,
        year INTEGER NOT NULL,
        revenue BIGINT,
        profit BIGINT,
        debt BIGINT,
        eps DOUBLE PRECISION,
        sector VARCHAR(100),
        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
        UNIQUE(company, year)
    );
    
    -- Create indexes for common queries
    CREATE INDEX IF NOT EXISTS idx_financial_metrics_company 
        ON financial_metrics(company);
    CREATE INDEX IF NOT EXISTS idx_financial_metrics_year 
        ON financial_metrics(year);
    CREATE INDEX IF NOT EXISTS idx_financial_metrics_sector 
        ON financial_metrics(sector);
    """
    
    try:
        with engine.begin() as conn:
            conn.execute(text(create_table_sql))
        logger.info(f"✓ Table '{TARGET_TABLE}' is ready")
        return True
    except SQLAlchemyError as e:
        logger.error(f"✗ Failed to create table: {e}")
        return False


def table_exists(engine: Engine, table_name: str) -> bool:
    """
    Check if a table exists in the database.
    
    Args:
        engine: SQLAlchemy engine instance
        table_name: Name of the table to check
        
    Returns:
        bool: True if table exists
    """
    inspector = inspect(engine)
    return table_name in inspector.get_table_names()


# ============================================================
# CSV LOADING FUNCTIONS
# ============================================================

def load_csv_files(csv_directory: Path = RAW_CSV_DIR) -> Tuple[Optional[pd.DataFrame], List[str]]:
    """
    Load and concatenate all CSV files from the specified directory.
    
    Args:
        csv_directory: Path to directory containing CSV files
        
    Returns:
        Tuple containing:
        - DataFrame with all CSV data combined, or None if no files found
        - List of successfully loaded file paths
    """
    # Ensure directory exists
    if not csv_directory.exists():
        logger.warning(f"CSV directory does not exist: {csv_directory}")
        csv_directory.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created directory: {csv_directory}")
        return None, []
    
    # Find all CSV files
    csv_files = list(csv_directory.glob("*.csv"))
    
    if not csv_files:
        logger.warning(f"No CSV files found in {csv_directory}")
        return None, []
    
    logger.info(f"Found {len(csv_files)} CSV file(s) to process")
    
    dataframes = []
    loaded_files = []
    
    for csv_file in csv_files:
        try:
            logger.info(f"Loading: {csv_file.name}")
            
            # Read CSV with flexible parsing
            df = pd.read_csv(
                csv_file,
                encoding='utf-8',
                na_values=['', 'NA', 'N/A', 'null', 'NULL', 'None', '-'],
                skipinitialspace=True,
            )
            
            # Normalize column names (lowercase, strip whitespace)
            df.columns = df.columns.str.lower().str.strip().str.replace(' ', '_')
            
            # Add source file tracking
            df['_source_file'] = csv_file.name
            
            dataframes.append(df)
            loaded_files.append(str(csv_file))
            
            logger.info(f"  → Loaded {len(df):,} rows from {csv_file.name}")
            
        except pd.errors.EmptyDataError:
            logger.warning(f"  → Skipping empty file: {csv_file.name}")
        except pd.errors.ParserError as e:
            logger.error(f"  → Parse error in {csv_file.name}: {e}")
        except Exception as e:
            logger.error(f"  → Failed to load {csv_file.name}: {e}")
    
    if not dataframes:
        logger.error("No data could be loaded from CSV files")
        return None, []
    
    # Concatenate all dataframes
    combined_df = pd.concat(dataframes, ignore_index=True)
    logger.info(f"✓ Combined total: {len(combined_df):,} rows from {len(loaded_files)} file(s)")
    
    return combined_df, loaded_files


# ============================================================
# DATA CLEANING FUNCTIONS
# ============================================================

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Validate and clean the financial data.
    
    Cleaning steps:
    1. Standardize column names
    2. Remove duplicate rows
    3. Handle missing values
    4. Convert data types
    5. Validate data ranges
    6. Remove invalid records
    
    Args:
        df: Raw DataFrame from CSV files
        
    Returns:
        pd.DataFrame: Cleaned and validated DataFrame
    """
    logger.info("Starting data cleaning process...")
    initial_count = len(df)
    
    # Track cleaning statistics
    stats = {
        "initial_rows": initial_count,
        "duplicates_removed": 0,
        "nulls_handled": 0,
        "invalid_removed": 0,
    }
    
    # ----------------------------------------------------------
    # Step 1: Ensure required columns exist
    # ----------------------------------------------------------
    required_cols = ['company', 'year']
    missing_cols = [col for col in required_cols if col not in df.columns]
    
    if missing_cols:
        logger.error(f"Missing required columns: {missing_cols}")
        logger.info(f"Available columns: {list(df.columns)}")
        raise ValueError(f"Missing required columns: {missing_cols}")
    
    # ----------------------------------------------------------
    # Step 2: Remove exact duplicate rows
    # ----------------------------------------------------------
    df_deduped = df.drop_duplicates()
    stats["duplicates_removed"] = initial_count - len(df_deduped)
    
    if stats["duplicates_removed"] > 0:
        logger.info(f"  → Removed {stats['duplicates_removed']:,} duplicate rows")
    
    df = df_deduped.copy()
    
    # ----------------------------------------------------------
    # Step 3: Clean string columns
    # ----------------------------------------------------------
    string_cols = ['company', 'sector']
    for col in string_cols:
        if col in df.columns:
            # Strip whitespace and standardize case
            df[col] = df[col].astype(str).str.strip().str.title()
            # Replace 'Nan' strings with actual NaN
            df[col] = df[col].replace(['Nan', 'None', ''], pd.NA)
    
    # ----------------------------------------------------------
    # Step 4: Convert numeric columns
    # ----------------------------------------------------------
    numeric_conversions = {
        'year': 'Int64',
        'revenue': 'Int64',
        'profit': 'Int64',
        'debt': 'Int64',
        'eps': 'float64',
    }
    
    for col, dtype in numeric_conversions.items():
        if col in df.columns:
            try:
                # Remove any currency symbols or commas
                if df[col].dtype == object:
                    df[col] = df[col].astype(str).str.replace(r'[$,]', '', regex=True)
                
                df[col] = pd.to_numeric(df[col], errors='coerce')
                
                if dtype == 'Int64':
                    df[col] = df[col].astype('Int64')
                    
            except Exception as e:
                logger.warning(f"  → Could not convert {col} to {dtype}: {e}")
    
    # ----------------------------------------------------------
    # Step 5: Handle missing values in required columns
    # ----------------------------------------------------------
    # Company and year are required - remove rows with missing values
    before_null_removal = len(df)
    df = df.dropna(subset=['company', 'year'])
    stats["nulls_handled"] = before_null_removal - len(df)
    
    if stats["nulls_handled"] > 0:
        logger.info(f"  → Removed {stats['nulls_handled']:,} rows with missing company/year")
    
    # ----------------------------------------------------------
    # Step 6: Validate data ranges
    # ----------------------------------------------------------
    before_validation = len(df)
    
    # Year should be reasonable (1900-2030)
    df = df[(df['year'] >= 1900) & (df['year'] <= 2030)]
    
    # Revenue, profit, debt should not be astronomically large (sanity check)
    # This catches data entry errors like extra zeros
    MAX_VALUE = 10**15  # 1 quadrillion
    
    for col in ['revenue', 'profit', 'debt']:
        if col in df.columns:
            df = df[(df[col].isna()) | (df[col].abs() < MAX_VALUE)]
    
    stats["invalid_removed"] = before_validation - len(df)
    
    if stats["invalid_removed"] > 0:
        logger.info(f"  → Removed {stats['invalid_removed']:,} rows with invalid values")
    
    # ----------------------------------------------------------
    # Step 7: Remove business duplicates (same company + year)
    # ----------------------------------------------------------
    # Keep the last occurrence (most recent data)
    before_biz_dedup = len(df)
    df = df.drop_duplicates(subset=['company', 'year'], keep='last')
    biz_dupes = before_biz_dedup - len(df)
    
    if biz_dupes > 0:
        logger.info(f"  → Removed {biz_dupes:,} duplicate company/year combinations")
    
    # ----------------------------------------------------------
    # Step 8: Select final columns for database
    # ----------------------------------------------------------
    final_columns = ['company', 'year', 'revenue', 'profit', 'debt', 'eps', 'sector']
    available_columns = [col for col in final_columns if col in df.columns]
    
    df = df[available_columns].copy()
    
    # ----------------------------------------------------------
    # Summary
    # ----------------------------------------------------------
    final_count = len(df)
    removed_total = initial_count - final_count
    
    logger.info(f"✓ Data cleaning complete:")
    logger.info(f"  → Initial rows:  {initial_count:,}")
    logger.info(f"  → Final rows:    {final_count:,}")
    logger.info(f"  → Rows removed:  {removed_total:,} ({removed_total/initial_count*100:.1f}%)")
    
    return df


# ============================================================
# DATABASE INSERTION FUNCTIONS
# ============================================================

def insert_to_db(
    df: pd.DataFrame,
    engine: Engine,
    table_name: str = TARGET_TABLE,
    if_exists: str = 'append',
    chunk_size: int = 1000,
) -> Tuple[bool, int]:
    """
    Insert cleaned DataFrame into PostgreSQL database.
    
    Uses chunked insertion with transaction handling for reliability.
    
    Args:
        df: Cleaned DataFrame to insert
        engine: SQLAlchemy engine instance
        table_name: Target table name
        if_exists: How to handle existing data ('append', 'replace', 'fail')
        chunk_size: Number of rows per batch insert
        
    Returns:
        Tuple of (success: bool, rows_inserted: int)
    """
    if df is None or df.empty:
        logger.warning("No data to insert")
        return False, 0
    
    total_rows = len(df)
    logger.info(f"Inserting {total_rows:,} rows into '{table_name}'...")
    
    try:
        # Use a transaction for the entire insertion
        with engine.begin() as conn:
            # If replacing, truncate the table first
            if if_exists == 'replace':
                logger.info(f"  → Truncating existing data in '{table_name}'")
                conn.execute(text(f"TRUNCATE TABLE {table_name} RESTART IDENTITY CASCADE"))
            
            # Insert in chunks for better memory management
            rows_inserted = 0
            
            for start_idx in range(0, total_rows, chunk_size):
                end_idx = min(start_idx + chunk_size, total_rows)
                chunk = df.iloc[start_idx:end_idx]
                
                # Insert chunk
                chunk.to_sql(
                    table_name,
                    conn,
                    if_exists='append',
                    index=False,
                    method='multi',  # Use multi-row INSERT for speed
                )
                
                rows_inserted += len(chunk)
                
                # Progress logging for large datasets
                if total_rows > chunk_size:
                    progress = (rows_inserted / total_rows) * 100
                    logger.info(f"  → Progress: {rows_inserted:,}/{total_rows:,} ({progress:.1f}%)")
        
        logger.info(f"✓ Successfully inserted {rows_inserted:,} rows into '{table_name}'")
        return True, rows_inserted
        
    except SQLAlchemyError as e:
        logger.error(f"✗ Database insertion failed: {e}")
        return False, 0
    except Exception as e:
        logger.error(f"✗ Unexpected error during insertion: {e}")
        return False, 0


def upsert_to_db(
    df: pd.DataFrame,
    engine: Engine,
    table_name: str = TARGET_TABLE,
) -> Tuple[bool, int, int]:
    """
    Insert or update records using PostgreSQL's ON CONFLICT (upsert).
    
    This is useful for incremental updates where some records may already exist.
    
    Args:
        df: Cleaned DataFrame to upsert
        engine: SQLAlchemy engine instance
        table_name: Target table name
        
    Returns:
        Tuple of (success: bool, inserted: int, updated: int)
    """
    if df is None or df.empty:
        logger.warning("No data to upsert")
        return False, 0, 0
    
    logger.info(f"Upserting {len(df):,} rows into '{table_name}'...")
    
    # Build upsert query
    columns = list(df.columns)
    col_list = ', '.join(columns)
    placeholders = ', '.join([f':{col}' for col in columns])
    
    update_cols = [col for col in columns if col not in ['company', 'year']]
    update_set = ', '.join([f'{col} = EXCLUDED.{col}' for col in update_cols])
    
    upsert_sql = f"""
    INSERT INTO {table_name} ({col_list})
    VALUES ({placeholders})
    ON CONFLICT (company, year) 
    DO UPDATE SET {update_set}, updated_at = CURRENT_TIMESTAMP
    """
    
    try:
        with engine.begin() as conn:
            records = df.to_dict('records')
            
            for record in records:
                conn.execute(text(upsert_sql), record)
        
        logger.info(f"✓ Successfully upserted {len(df):,} rows")
        return True, len(df), 0  # Can't easily distinguish inserts vs updates
        
    except SQLAlchemyError as e:
        logger.error(f"✗ Upsert failed: {e}")
        return False, 0, 0


# ============================================================
# MAIN PIPELINE
# ============================================================

def run_ingestion_pipeline(
    csv_directory: Path = RAW_CSV_DIR,
    mode: str = 'append',  # 'append', 'replace', or 'upsert'
) -> dict:
    """
    Execute the full CSV to PostgreSQL ingestion pipeline.
    
    Args:
        csv_directory: Path to directory containing CSV files
        mode: Insertion mode ('append', 'replace', 'upsert')
        
    Returns:
        dict: Pipeline execution summary
    """
    logger.info("=" * 60)
    logger.info("STARTING CSV TO SQL INGESTION PIPELINE")
    logger.info("=" * 60)
    
    start_time = datetime.now()
    
    result = {
        "status": "failed",
        "files_loaded": [],
        "rows_processed": 0,
        "rows_inserted": 0,
        "duration_seconds": 0,
        "errors": [],
    }
    
    try:
        # ----------------------------------------------------------
        # Step 1: Connect to database
        # ----------------------------------------------------------
        logger.info("\n[Step 1/5] Connecting to database...")
        engine = get_db_engine()
        
        if engine is None:
            result["errors"].append("Failed to connect to database")
            return result
        
        # ----------------------------------------------------------
        # Step 2: Create table if needed
        # ----------------------------------------------------------
        logger.info("\n[Step 2/5] Ensuring table exists...")
        if not create_table_if_not_exists(engine):
            result["errors"].append("Failed to create table")
            return result
        
        # ----------------------------------------------------------
        # Step 3: Load CSV files
        # ----------------------------------------------------------
        logger.info(f"\n[Step 3/5] Loading CSV files from {csv_directory}...")
        df, loaded_files = load_csv_files(csv_directory)
        
        if df is None or df.empty:
            logger.warning("No data to process")
            result["status"] = "no_data"
            result["files_loaded"] = loaded_files
            return result
        
        result["files_loaded"] = loaded_files
        result["rows_processed"] = len(df)
        
        # ----------------------------------------------------------
        # Step 4: Clean data
        # ----------------------------------------------------------
        logger.info("\n[Step 4/5] Cleaning and validating data...")
        df_clean = clean_data(df)
        
        if df_clean.empty:
            logger.warning("All data was filtered out during cleaning")
            result["status"] = "no_valid_data"
            return result
        
        # ----------------------------------------------------------
        # Step 5: Insert into database
        # ----------------------------------------------------------
        logger.info(f"\n[Step 5/5] Inserting data (mode: {mode})...")
        
        if mode == 'upsert':
            success, inserted, _ = upsert_to_db(df_clean, engine)
        else:
            success, inserted = insert_to_db(
                df_clean, 
                engine, 
                if_exists='replace' if mode == 'replace' else 'append'
            )
        
        if success:
            result["status"] = "success"
            result["rows_inserted"] = inserted
        else:
            result["errors"].append("Database insertion failed")
        
    except Exception as e:
        logger.error(f"Pipeline failed with error: {e}")
        result["errors"].append(str(e))
    
    finally:
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds()
        result["duration_seconds"] = round(duration, 2)
        
        # Print summary
        logger.info("\n" + "=" * 60)
        logger.info("PIPELINE SUMMARY")
        logger.info("=" * 60)
        logger.info(f"Status:         {result['status'].upper()}")
        logger.info(f"Files loaded:   {len(result['files_loaded'])}")
        logger.info(f"Rows processed: {result['rows_processed']:,}")
        logger.info(f"Rows inserted:  {result['rows_inserted']:,}")
        logger.info(f"Duration:       {result['duration_seconds']:.2f} seconds")
        
        if result["errors"]:
            logger.error(f"Errors:         {result['errors']}")
        
        logger.info("=" * 60)
    
    return result


# ============================================================
# CLI ENTRY POINT
# ============================================================

def main():
    """
    Command-line entry point for the ingestion pipeline.
    
    Usage:
        python csv_to_sql.py [--mode append|replace|upsert] [--csv-dir PATH]
    """
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Ingest CSV financial data into PostgreSQL",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python csv_to_sql.py                    # Append data from default directory
  python csv_to_sql.py --mode replace     # Replace all existing data
  python csv_to_sql.py --mode upsert      # Update existing, insert new
  python csv_to_sql.py --csv-dir ./data   # Use custom CSV directory
        """
    )
    
    parser.add_argument(
        '--mode', 
        choices=['append', 'replace', 'upsert'],
        default='append',
        help='Data insertion mode (default: append)'
    )
    
    parser.add_argument(
        '--csv-dir',
        type=Path,
        default=RAW_CSV_DIR,
        help=f'Directory containing CSV files (default: {RAW_CSV_DIR})'
    )
    
    args = parser.parse_args()
    
    # Run pipeline
    result = run_ingestion_pipeline(
        csv_directory=args.csv_dir,
        mode=args.mode,
    )
    
    # Exit with appropriate code
    sys.exit(0 if result["status"] == "success" else 1)


if __name__ == "__main__":
    main()
