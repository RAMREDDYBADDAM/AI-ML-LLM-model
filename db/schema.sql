-- ============================================================================
-- FINANCIAL RAG PROJECT – PostgreSQL Database Schema
-- ============================================================================

-- Usage:
--   psql -U postgres -d finance_db -f db/schema.sql
--

-- ============================================================================
-- EXAMPLE QUERIES YOU CAN ASK THE SYSTEM
-- ============================================================================
--
-- 1️⃣  SQL / ANALYTICS QUESTIONS (Numeric Data & Aggregations)
-- These questions trigger SQL agent to query the database directly.
--
--  📊 Financial Metrics
--    "What was Apple's revenue in 2024 Q1?"
--    "Compare Apple's revenue between Q1 and Q2 of 2024."
--    "What is Apple's total assets and liabilities for 2024-Q2?"
--    "What is the EPS for Apple for 2024 Q2?"
--    "Which company has the highest revenue in Q1 2024?"
--    "Show the revenue growth rate for Apple year-over-year."
--
--  🏭 Products
--    "List Apple's top revenue-contributing products."
--    "What percentage of Apple's revenue comes from iPhone?"
--    "When was the MacBook launched?"
--    "How many products does Apple have in active status?"
--
--  📈 Quarterly Reports
--    "Summarize Apple's Q1 2024 quarterly performance."
--    "What were the highlights of Apple's Q1 2024 report?"
--    "Show all quarterly reports for Apple in 2024."
--
--  📝 Analyst Ratings
--    "What did Goldman Sachs rate Apple?"
--    "What is the latest analyst price target for Apple?"
--    "List all analyst ratings for Apple."
--    "What firms have rated Apple and what are their ratings?"
--
--  🌍 Market Trends
--    "What are the top market trends in the Technology sector?"
--    "Show all market trends with impact score above 7."
--    "How is AI adoption impacting technology companies?"
--
-- ============================================================================
--
-- 2️⃣  DOCUMENT / RAG QUESTIONS (Context-based from Vectorstore)
-- These questions are answered using document embeddings and retrieval.
--
--  🎓 Explanations & Context
--    "Explain Apple's recent financial performance in simple terms."
--    "What are the major trends affecting the Technology sector?"
--    "Give a summary of Apple's quarterly highlights."
--    "What are the risks and opportunities for Apple in 2024?"
--    "Summarize Apple's business segments and revenue drivers."
--    "What is Apple's competitive advantage?"
--    "Explain Apple's supply chain strategy."
--
-- ============================================================================
--
-- 3️⃣  HYBRID QUESTIONS (SQL + RAG Combined)
-- These questions require both numeric data AND narrative explanation.
--
--  🔥 Combined Analysis
--    "Is Apple growing or declining? Use numbers and explanation."
--    "Compare Apple's revenue growth and summarize key reasons behind it."
--    "How do Apple's financial metrics align with analyst expectations?"
--    "What is Apple's financial health and what trends influence it?"
--    "Analyze Apple's profitability trend and explain the business factors."
--
--  📌 Strategic Questions
--    "Should investors consider buying Apple stock? Provide metrics and analyst reasoning."
--    "How is the semiconductor market trend affecting Apple's performance?"
--    "What are Apple's growth opportunities based on financial data and market trends?"
--    "Compare Apple vs Microsoft on revenue and profitability."
--
-- ============================================================================
--
-- 4️⃣  NATURAL LANGUAGE → SQL (Auto-Generated Queries)
-- The AI system can also translate natural language to SQL automatically:
--
--    "Show all Apple's quarterly revenue sorted by period."
--    "Which Apple product has the highest revenue contribution?"
--    "List all market trends with impact score above 8 and sort by date."
--    "Find companies in the Technology sector with highest revenue in 2024-Q1."
--    "Show analyst price targets for Apple from different firms."
--
-- ============================================================================
--
-- 5️⃣  GENERAL FINANCIAL KNOWLEDGE QUESTIONS
-- These are answered by the configured LLM (Ollama or OpenAI):
--
--    "What drives Apple's profitability?"
--    "How does Apple compare with other tech companies?"
--    "What are the major factors influencing the EV market?" (Tesla-related)
--    "Explain EPS in simple terms."
--    "What is market capitalization?"
--    "How do quarterly earnings reports impact stock prices?"
--
-- ============================================================================

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
CREATE TABLE IF NOT EXISTS financial_metrics (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    period VARCHAR(20) NOT NULL,   -- e.g., '2024-Q1'
    revenue NUMERIC(15, 2),
    net_income NUMERIC(15, 2),
    operating_income NUMERIC(15, 2),
    total_assets NUMERIC(15, 2),
    total_liabilities NUMERIC(15, 2),
    equity NUMERIC(15, 2),
    eps NUMERIC(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, period)
);

-- ============================================================================
-- TABLE: quarterly_reports
-- ============================================================================
CREATE TABLE IF NOT EXISTS quarterly_reports (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    quarter VARCHAR(10) NOT NULL,
    year INT NOT NULL,
    report_date DATE,
    summary TEXT,
    highlights TEXT,
    document_path VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(company_id, year, quarter)
);

-- ============================================================================
-- TABLE: products
-- ============================================================================
CREATE TABLE IF NOT EXISTS products (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    launch_date DATE,
    revenue_contribution NUMERIC(5, 2),  -- percent of total revenue
    status VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE: market_trends
-- ============================================================================
CREATE TABLE IF NOT EXISTS market_trends (
    id SERIAL PRIMARY KEY,
    sector VARCHAR(100) NOT NULL,
    trend_date DATE NOT NULL,
    description TEXT,
    impact_score INT,
    source VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- TABLE: analyst_ratings
-- ============================================================================
CREATE TABLE IF NOT EXISTS analyst_ratings (
    id SERIAL PRIMARY KEY,
    company_id INT NOT NULL REFERENCES companies(id) ON DELETE CASCADE,
    analyst_name VARCHAR(255),
    firm VARCHAR(255),
    rating VARCHAR(50),
    price_target NUMERIC(10, 2),
    rating_date DATE,
    rationale TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================================================
-- INDEXES
-- ============================================================================
CREATE INDEX IF NOT EXISTS idx_financial_metrics_company ON financial_metrics(company_id);
CREATE INDEX IF NOT EXISTS idx_financial_metrics_period ON financial_metrics(period);
CREATE INDEX IF NOT EXISTS idx_quarterly_reports_company ON quarterly_reports(company_id);
CREATE INDEX IF NOT EXISTS idx_quarterly_reports_year_quarter ON quarterly_reports(year, quarter);
CREATE INDEX IF NOT EXISTS idx_products_company ON products(company_id);
CREATE INDEX IF NOT EXISTS idx_market_trends_sector ON market_trends(sector);
CREATE INDEX IF NOT EXISTS idx_market_trends_date ON market_trends(trend_date);
CREATE INDEX IF NOT EXISTS idx_analyst_ratings_company ON analyst_ratings(company_id);

-- ============================================================================
-- SAMPLE DATA
-- ============================================================================

-- Companies
INSERT INTO companies (ticker, name, sector, country, founded_year, website)
VALUES 
    ('AAPL', 'Apple Inc.', 'Technology', 'USA', 1976, 'https://www.apple.com'),
    ('MSFT', 'Microsoft Corporation', 'Technology', 'USA', 1975, 'https://www.microsoft.com'),
    ('GOOGL', 'Alphabet Inc.', 'Technology', 'USA', 1998, 'https://www.google.com'),
    ('TSLA', 'Tesla Inc.', 'Automotive', 'USA', 2003, 'https://www.tesla.com'),
    ('AMZN', 'Amazon.com Inc.', 'E-commerce/Cloud', 'USA', 1994, 'https://www.amazon.com')
ON CONFLICT (ticker) DO NOTHING;

-- Apple Financial Metrics
INSERT INTO financial_metrics (company_id, period, revenue, net_income, operating_income, total_assets, total_liabilities, equity, eps)
SELECT id, '2024-Q1', 123456.00, 34567.00, 45678.00, 987654.00, 123456.00, 864198.00, 2.25
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT (company_id, period) DO NOTHING;

INSERT INTO financial_metrics (company_id, period, revenue, net_income, operating_income, total_assets, total_liabilities, equity, eps)
SELECT id, '2024-Q2', 134567.00, 38901.00, 51234.00, 1001234.00, 145678.00, 855556.00, 2.52
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT (company_id, period) DO NOTHING;

-- Apple Products
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

-- Apple Quarterly Report
INSERT INTO quarterly_reports (company_id, quarter, year, report_date, summary, highlights)
SELECT id, 'Q1', 2024, '2024-02-01',
       'Strong Q1 performance with iPhone sales leading growth.',
       'iPhone revenue +5% YoY, Services at a record high'
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT (company_id, year, quarter) DO NOTHING;

-- Analyst Rating
INSERT INTO analyst_ratings (company_id, analyst_name, firm, rating, price_target, rating_date, rationale)
SELECT id, 'Jane Smith', 'Goldman Sachs', 'Buy', 225.50, '2024-03-15',
       'Strong services growth and new AI features justify premium valuation'
FROM companies WHERE ticker = 'AAPL'
ON CONFLICT DO NOTHING;

-- Market Trends
INSERT INTO market_trends (sector, trend_date, description, impact_score, source)
VALUES
    ('Technology', '2024-01-15', 'AI adoption accelerating across enterprise sector', 9, 'Market Research'),
    ('Technology', '2024-02-20', 'Semiconductor supply chain stabilized', 7, 'Industry Report'),
    ('Automotive', '2024-01-10', 'EV market growing faster than expected', 8, 'Analyst Report')
ON CONFLICT DO NOTHING;
