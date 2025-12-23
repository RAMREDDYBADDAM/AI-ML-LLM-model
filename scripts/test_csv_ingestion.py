#!/usr/bin/env python3
"""
Test script for CSV to SQL ingestion pipeline.
Run this to verify the ingestion works correctly.
"""

import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.csv_to_sql import (
    setup_logging,
    load_csv_files,
    clean_data,
    RAW_CSV_DIR,
)


def test_csv_loading():
    """Test CSV loading without database."""
    print("\n" + "=" * 50)
    print("TESTING CSV LOADING AND CLEANING")
    print("=" * 50)
    
    # Load CSVs
    df, files = load_csv_files(RAW_CSV_DIR)
    
    if df is None:
        print("❌ No CSV files found!")
        print(f"   Please add CSV files to: {RAW_CSV_DIR}")
        return False
    
    print(f"\n✓ Loaded {len(df)} rows from {len(files)} file(s)")
    print(f"\nColumns found: {list(df.columns)}")
    print(f"\nSample data (first 5 rows):")
    print(df.head().to_string())
    
    # Clean data
    print("\n" + "-" * 50)
    print("CLEANING DATA...")
    print("-" * 50)
    
    df_clean = clean_data(df)
    
    print(f"\n✓ Cleaned data: {len(df_clean)} rows")
    print(f"\nData types:")
    print(df_clean.dtypes)
    
    print(f"\nSample cleaned data (first 5 rows):")
    print(df_clean.head().to_string())
    
    print(f"\nUnique companies: {df_clean['company'].nunique()}")
    print(f"Year range: {df_clean['year'].min()} - {df_clean['year'].max()}")
    
    if 'sector' in df_clean.columns:
        print(f"Sectors: {df_clean['sector'].unique().tolist()}")
    
    return True


if __name__ == "__main__":
    # Run test
    success = test_csv_loading()
    
    print("\n" + "=" * 50)
    if success:
        print("✓ TEST PASSED - CSV loading and cleaning works!")
        print("\nTo run full ingestion with PostgreSQL:")
        print("  python scripts/csv_to_sql.py --mode replace")
    else:
        print("❌ TEST FAILED")
    print("=" * 50)
