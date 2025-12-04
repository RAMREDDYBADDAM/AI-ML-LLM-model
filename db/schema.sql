-- Financial RAG Project: Sample Database Schema
-- PostgreSQL DDL for financial data (companies, metrics, quarterly reports, etc.)
--
-- Usage:
--   psql -U user -d database -f db/schema.sql
--
-- Then insert sample data as shown at the bottom.

-- ============================================================================
-- TABLE: companies
-- ============================================================================
CREATE TABLE IF NOT EXISTS companies (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    sector VARCHAR(100),
    country VARCHAR(100),
    founded_year INT,
    website VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE: financial_metrics
-- ============================================================================
-- Stores key financial metrics (revenue, net income, assets, etc.) by quarter/year
CREATE TABLE IF NOT EXISTS financial_metrics (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    period VARCHAR(20) NOT NULL,  -- e.g., "2024-Q1", "2023-FY"
    revenue NUMERIC(15, 2),  -- in millions
    net_income NUMERIC(15, 2),
    operating_income NUMERIC(15, 2),
    total_assets NUMERIC(15, 2),
    total_liabilities NUMERIC(15, 2),
    equity NUMERIC(15, 2),
    eps NUMERIC(10, 2),  -- earnings per share
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, period)
);

-- ============================================================================
-- TABLE: quarterly_reports
-- ============================================================================
-- Links to quarterly report summaries and documents
CREATE TABLE IF NOT EXISTS quarterly_reports (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    quarter VARCHAR(10) NOT NULL,  -- e.g., "Q1", "Q2", "Q3", "Q4"
    year INT NOT NULL,
    report_date DATE,
    summary TEXT,
    highlights TEXT,
    document_path VARCHAR(255),  -- path to ingested PDF or document
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, year, quarter)
);

-- ============================================================================
-- TABLE: products
-- ============================================================================
-- Main products/services offered by each company
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    launch_date DATE,
    revenue_contribution NUMERIC(5, 2),  -- percentage of total revenue
    status VARCHAR(50),  -- "active", "discontinued", "in_development"
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE: market_trends
-- ============================================================================
-- Industry trends, market data, analyst opinions
CREATE TABLE IF NOT EXISTS market_trends (
    id SERIAL PRIMARY KEY,
    sector VARCHAR(100) NOT NULL,
    trend_date DATE,
    description TEXT,
    impact_score INT,  -- 1-10 scale
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE: analyst_ratings
-- ============================================================================
-- Analyst recommendations for each company
CREATE TABLE IF NOT EXISTS analyst_ratings (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    analyst_name VARCHAR(255),
    firm VARCHAR(255),
    rating VARCHAR(50),  -- "Buy", "Hold", "Sell", "Outperform", etc.
    price_target NUMERIC(10, 2),
    rating_date DATE,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEX: Optimize common queries
-- ============================================================================
CREATE INDEX idx_financial_metrics_company ON financial_metrics(company_id);
CREATE INDEX idx_financial_metrics_period ON financial_metrics(period);
CREATE INDEX idx_quarterly_reports_company ON quarterly_reports(company_id);
CREATE INDEX idx_quarterly_reports_year_quarter ON quarterly_reports(year, quarter);
CREATE INDEX idx_products_company ON products(company_id);
CREATE INDEX idx_market_trends_sector ON market_trends(sector);
CREATE INDEX idx_market_trends_date ON market_trends(trend_date);
CREATE INDEX idx_analyst_ratings_company ON analyst_ratings(company_id);

-- ============================================================================
-- SAMPLE DATA (for testing)
-- ============================================================================

-- Insert sample companies
INSERT INTO companies (ticker, name, sector, country, founded_year, website)
VALUES 
  ('AAPL', 'Apple Inc.', 'Technology', 'USA', 1976, 'https://www.apple.com'),
  ('MSFT', 'Microsoft Corporation', 'Technology', 'USA', 1975, 'https://www.microsoft.com'),
  ('GOOGL', 'Alphabet Inc.', 'Technology', 'USA', 1998, 'https://www.google.com'),
  ('TSLA', 'Tesla Inc.', 'Automotive', 'USA', 2003, 'https://www.tesla.com'),
  ('AMZN', 'Amazon.com Inc.', 'E-commerce/Cloud', 'USA', 1994, 'https://www.amazon.com')
ON CONFLICT (ticker) DO NOTHING;

-- Insert sample financial metrics for Apple
INSERT INTO financial_metrics (company_id, period, revenue, net_income, operating_income, total_assets, total_liabilities, equity, eps)
SELECT id, '2024-Q1', 123456.00, 34567.00, 45678.00, 987654.00, 123456.00, 864198.00, 2.25
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT (company_id, period) DO NOTHING;

INSERT INTO financial_metrics (company_id, period, revenue, net_income, operating_income, total_assets, total_liabilities, equity, eps)
SELECT id, '2024-Q2', 134567.00, 38901.00, 51234.00, 1001234.00, 145678.00, 855556.00, 2.52
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT (company_id, period) DO NOTHING;

-- Insert sample products for Apple
INSERT INTO products (company_id, name, description, launch_date, revenue_contribution, status)
SELECT id, 'iPhone', 'Flagship smartphone device', '2007-06-29', 52.00, 'active'
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT DO NOTHING;

INSERT INTO products (company_id, name, description, launch_date, revenue_contribution, status)
SELECT id, 'iPad', 'Tablet device', '2010-04-03', 8.50, 'active'
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT DO NOTHING;

INSERT INTO products (company_id, name, description, launch_date, revenue_contribution, status)
SELECT id, 'MacBook', 'Personal computers', '2006-01-10', 15.30, 'active'
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT DO NOTHING;

-- Insert sample quarterly reports
INSERT INTO quarterly_reports (company_id, quarter, year, report_date, summary, highlights)
SELECT id, 'Q1', 2024, '2024-02-01', 
  'Strong Q1 performance with iPhone sales leading growth.',
  'iPhone revenue +5% YoY, Services segment record high, China sales stabilized'
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT (company_id, year, quarter) DO NOTHING;

-- Insert sample analyst ratings
INSERT INTO analyst_ratings (company_id, analyst_name, firm, rating, price_target, rating_date, rationale)
SELECT id, 'Jane Smith', 'Goldman Sachs', 'Buy', 225.50, '2024-03-15',
  'Strong services growth and new AI features justify premium valuation'
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT DO NOTHING;

-- Insert sample market trends
INSERT INTO market_trends (sector, trend_date, description, impact_score, source)
VALUES 
  ('Technology', '2024-01-15', 'AI adoption accelerating across enterprise sector', 9, 'Market Research'),
  ('Technology', '2024-02-20', 'Semiconductor supply chain stabilized', 7, 'Industry Report'),
  ('Automotive', '2024-01-10', 'EV market growing faster than expected', 8, 'Analyst Report')
ON CONFLICT DO NOTHING;
