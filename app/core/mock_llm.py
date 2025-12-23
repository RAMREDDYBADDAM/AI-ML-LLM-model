"""
Fallback LLM implementation for testing without OpenAI API key.
Provides structured financial analyst responses.
"""
from typing import Dict, Any
import re


class MockLLM:
    """Enhanced mock LLM that returns structured financial analyst responses."""
    
    def __init__(self):
        self.company_info = {
            "apple": {
                "name": "Apple Inc.",
                "ticker": "AAPL",
                "sector": "Technology",
                "products": "iPhone, Mac, iPad, Apple Watch, services (Apple Music, iCloud, App Store)",
                "revenue": "$383 billion (FY2023)",
                "market_cap": "$3+ trillion",
            },
            "microsoft": {
                "name": "Microsoft Corporation",
                "ticker": "MSFT",
                "sector": "Technology",
                "products": "Windows, Office 365, Azure cloud, Xbox, LinkedIn",
                "revenue": "$211 billion (FY2023)",
                "market_cap": "$2.8+ trillion",
            },
            "google": {
                "name": "Alphabet Inc. (Google)",
                "ticker": "GOOGL",
                "sector": "Technology",
                "products": "Google Search, YouTube, Google Cloud, Android",
                "revenue": "$307 billion (FY2023)",
                "market_cap": "$1.9+ trillion",
            },
            "amazon": {
                "name": "Amazon.com Inc.",
                "ticker": "AMZN",
                "sector": "Consumer Discretionary",
                "products": "E-commerce, AWS cloud, Prime, Alexa",
                "revenue": "$574 billion (FY2023)",
                "market_cap": "$1.8+ trillion",
            },
            "tesla": {
                "name": "Tesla Inc.",
                "ticker": "TSLA",
                "sector": "Consumer Discretionary",
                "products": "Electric vehicles, energy storage, solar panels",
                "revenue": "$96 billion (FY2023)",
                "market_cap": "$800+ billion",
            },
            "nvidia": {
                "name": "NVIDIA Corporation",
                "ticker": "NVDA",
                "sector": "Technology",
                "products": "GPUs, AI chips, data center processors, gaming",
                "revenue": "$60 billion (FY2024)",
                "market_cap": "$3+ trillion",
            },
        }
    
    def _find_company(self, text: str) -> Dict[str, Any]:
        """Find company mentioned in the text."""
        text_lower = text.lower()
        for key, info in self.company_info.items():
            if key in text_lower or info["ticker"].lower() in text_lower:
                return info
        return None
    
    def _generate_structured_response(self, question: str, context: str = "") -> str:
        """Generate a structured financial analyst response."""
        question_lower = question.lower()
        
        # Check for company-specific questions
        company = self._find_company(question)
        
        # Check for S&P 500 questions
        if any(term in question_lower for term in ['s&p', 'sp500', 's&p 500', 'market', 'index']):
            return self._sp500_response(question)
        
        # Check for revenue/financial questions
        if any(term in question_lower for term in ['revenue', 'profit', 'earnings', 'income', 'financial']):
            return self._financial_response(question, company)
        
        # Check for comparison questions
        if any(term in question_lower for term in ['compare', 'vs', 'versus', 'better', 'difference']):
            return self._comparison_response(question)
        
        # Check for trend/growth questions
        if any(term in question_lower for term in ['trend', 'growth', 'performance', 'outlook']):
            return self._trend_response(question, company)
        
        # Company-specific response
        if company:
            return self._company_response(company)
        
        # Default structured response
        return self._default_response(question, context)
    
    def _sp500_response(self, question: str) -> str:
        return """## Summary
The S&P 500 index is a market-capitalization-weighted index of 500 leading publicly traded companies in the U.S. Based on the available data, the index provides a comprehensive view of U.S. equity market performance.

## Context & Data Scope
- **Index Coverage**: 500 large-cap U.S. companies
- **Weighting**: Market-cap weighted
- **Data Available**: Historical price data spanning multiple decades
- **Sectors Represented**: All 11 GICS sectors

## Key Insights

### Performance Characteristics
- Historically returns ~10% annually on average (including dividends)
- Serves as a benchmark for the overall U.S. stock market
- Represents approximately 80% of U.S. equity market capitalization

### Pros ✅
1. **Diversification**: Exposure to 500 companies across all sectors
2. **Liquidity**: Highly liquid and easy to trade
3. **Track Record**: Long history of positive returns over extended periods
4. **Transparency**: Composition and methodology are publicly available

### Cons ⚠️
1. **Concentration Risk**: Top 10 companies represent ~30% of index weight
2. **U.S. Only**: No international exposure
3. **Large-Cap Bias**: Excludes small and mid-cap opportunities
4. **Passive Exposure**: No protection during market downturns

## Data-Grounded Suggestions
1. Review the S&P 500 Analytics tab for detailed historical performance
2. Check decade-by-decade returns for long-term perspective
3. Consider sector breakdowns to understand composition changes
4. Use the date filters to analyze specific time periods of interest"""

    def _financial_response(self, question: str, company: Dict = None) -> str:
        if company:
            return f"""## Summary
Analysis of {company['name']} ({company['ticker']}) financial performance based on available data.

## Context & Data Scope
- **Company**: {company['name']}
- **Ticker**: {company['ticker']}
- **Sector**: {company['sector']}
- **Revenue**: {company['revenue']}
- **Market Cap**: {company['market_cap']}

## Key Insights

### Financial Highlights
- {company['name']} is a leading company in the {company['sector']} sector
- Primary products/services: {company['products']}
- Market position: Among the largest publicly traded companies globally

### Pros ✅
1. **Strong Revenue**: Demonstrated consistent revenue generation
2. **Market Leader**: Dominant position in core business segments
3. **Brand Value**: High brand recognition and customer loyalty
4. **Innovation**: Continued investment in R&D and new products

### Cons ⚠️
1. **Competition**: Faces strong competition in all business segments
2. **Valuation**: Premium valuation may limit upside potential
3. **Regulatory Risk**: Subject to increasing regulatory scrutiny
4. **Market Dependency**: Performance tied to broader economic conditions

## Data-Grounded Suggestions
1. Review quarterly earnings reports for most recent performance
2. Compare key metrics with sector peers
3. Monitor industry trends that may impact future performance
4. Consider macroeconomic factors affecting the sector"""
        else:
            return """## Summary
Financial analysis request received. Here is a general overview of key financial metrics and considerations.

## Context & Data Scope
- **Analysis Type**: General financial metrics
- **Data Sources**: Sample S&P 500 company data
- **Time Period**: Based on most recent available data

## Key Insights

### Important Financial Metrics
- **Revenue**: Total income from business operations
- **Net Income**: Profit after all expenses and taxes
- **Profit Margin**: Percentage of revenue retained as profit
- **EPS (Earnings Per Share)**: Net income divided by shares outstanding

### Pros ✅ of Financial Analysis
1. **Informed Decisions**: Data-driven investment choices
2. **Risk Assessment**: Understanding of financial health
3. **Trend Identification**: Recognition of growth patterns
4. **Comparative Analysis**: Benchmarking against peers

### Cons ⚠️ to Consider
1. **Historical Data**: Past performance doesn't guarantee future results
2. **Market Factors**: External conditions affect all companies
3. **Accounting Methods**: Different companies use different methods
4. **Limited Scope**: Financial data alone doesn't tell the full story

## Data-Grounded Suggestions
1. Ask about specific companies for detailed analysis
2. Use the Insights tab to view revenue leaders
3. Check the S&P 500 Analytics for market context
4. Compare multiple companies within the same sector"""

    def _comparison_response(self, question: str) -> str:
        return """## Summary
Comparative analysis request received. Effective comparison requires examining multiple financial and operational metrics.

## Context & Data Scope
- **Analysis Type**: Comparative Assessment
- **Methodology**: Multi-factor comparison framework
- **Available Data**: S&P 500 company fundamentals

## Key Comparison Framework

### Financial Metrics to Compare
| Metric | Purpose | Best For |
|--------|---------|----------|
| Revenue | Size comparison | Understanding scale |
| Profit Margin | Efficiency | Operational quality |
| Growth Rate | Momentum | Future potential |
| P/E Ratio | Valuation | Investment timing |

### Pros ✅ of Comparative Analysis
1. **Relative Value**: Identifies undervalued opportunities
2. **Sector Context**: Performance relative to peers
3. **Trend Comparison**: Growth trajectory differences
4. **Risk Assessment**: Relative stability and volatility

### Cons ⚠️ to Consider
1. **Different Business Models**: Not all companies are directly comparable
2. **Accounting Differences**: Metrics may be calculated differently
3. **Market Conditions**: External factors affect companies differently
4. **Time Period Sensitivity**: Results vary by timeframe analyzed

## Data-Grounded Suggestions
1. Specify the companies you want to compare
2. Use the Revenue Leaders section in Insights tab
3. Check the Sector Comparison table for peer analysis
4. Consider both quantitative metrics and qualitative factors"""

    def _trend_response(self, question: str, company: Dict = None) -> str:
        context = f"for {company['name']}" if company else "in the market"
        return f"""## Summary
Trend analysis {context} based on available historical data and market indicators.

## Context & Data Scope
- **Analysis Period**: Multi-year historical data
- **Focus**: Growth trends and performance patterns
- **Data Source**: S&P 500 historical records

## Key Trend Insights

### Growth Indicators
- **Year-over-Year Growth**: Annual performance changes
- **Decade Performance**: Long-term secular trends
- **Volatility Patterns**: Market stability indicators

### Pros ✅ of Trend Analysis
1. **Pattern Recognition**: Identifies recurring market cycles
2. **Long-term Perspective**: Filters out short-term noise
3. **Historical Context**: Past performance provides framework
4. **Risk Management**: Helps identify potential downturns

### Cons ⚠️ to Consider
1. **Past ≠ Future**: Historical trends may not continue
2. **Structural Changes**: Economy and markets evolve
3. **Black Swan Events**: Unexpected events disrupt patterns
4. **Sample Bias**: Limited data in certain periods

## Data-Grounded Suggestions
1. View the Year-over-Year Growth chart in S&P 500 Analytics
2. Check Decade Performance for long-term trends
3. Use date filters to analyze specific periods
4. Compare current trends with historical averages"""

    def _company_response(self, company: Dict) -> str:
        return f"""## Summary
Overview of {company['name']} ({company['ticker']}), a leading company in the {company['sector']} sector.

## Context & Data Scope
- **Company**: {company['name']}
- **Ticker Symbol**: {company['ticker']}
- **Sector**: {company['sector']}
- **Annual Revenue**: {company['revenue']}
- **Market Capitalization**: {company['market_cap']}

## Key Insights

### Business Overview
{company['name']} is known for: {company['products']}

### Investment Considerations

#### Pros ✅
1. **Market Position**: Leading position in {company['sector']} sector
2. **Revenue Scale**: Strong revenue generation ({company['revenue']})
3. **Market Cap**: Significant market presence ({company['market_cap']})
4. **Product Portfolio**: Diversified product/service offerings
5. **Brand Recognition**: Strong global brand presence

#### Cons ⚠️
1. **Competitive Pressure**: Intense competition in core markets
2. **Market Saturation**: Mature markets may limit growth
3. **Regulatory Scrutiny**: Subject to regulatory oversight
4. **Valuation Concerns**: Premium pricing may limit upside
5. **Macro Sensitivity**: Performance tied to economic cycles

## Data-Grounded Suggestions
1. Review recent quarterly earnings for current performance
2. Compare with sector peers using the Insights tab
3. Monitor news and regulatory developments
4. Consider portfolio diversification across sectors"""

    def _default_response(self, question: str, context: str = "") -> str:
        return f"""## Summary
Analysis request received. Based on the available data and context, here is a comprehensive response.

## Context & Data Scope
- **Query**: {question[:100]}{'...' if len(question) > 100 else ''}
- **Data Available**: S&P 500 historical data, company fundamentals
- **Analysis Framework**: Financial analyst perspective

## Key Insights

### Available Analysis Options
The Financial RAG system can help with:
1. **S&P 500 Analysis**: Historical performance, trends, volatility
2. **Company Research**: Fundamentals, sector comparison
3. **Financial Metrics**: Revenue, profit margins, growth rates
4. **Market Context**: Decade performance, correlations

### Pros ✅ of Using This System
1. **Data-Driven**: Responses based on actual market data
2. **Structured Analysis**: Professional financial analyst format
3. **Multiple Perspectives**: Quantitative and qualitative insights
4. **Actionable Suggestions**: Clear next steps provided

### Cons ⚠️ / Limitations
1. **Historical Data**: Based on past performance only
2. **Sample Data**: May not include all companies
3. **No Real-Time**: Data refreshes periodically
4. **Not Financial Advice**: For informational purposes only

## Data-Grounded Suggestions
1. Try asking about specific companies (Apple, Microsoft, etc.)
2. Explore S&P 500 historical performance
3. Use the dashboard tabs for visual analysis
4. Ask about specific financial metrics or comparisons

*Note: For the best experience, ensure Ollama LLM is running for enhanced AI responses.*"""

    def invoke(self, prompt: str) -> Dict[str, Any]:
        """Generate structured response based on prompt content."""
        # Extract the question from the prompt
        question = prompt
        context = ""
        
        # Try to extract question and context from formatted prompts
        if "USER QUESTION:" in prompt:
            parts = prompt.split("USER QUESTION:")
            if len(parts) > 1:
                question = parts[1].split("\n")[0].strip()
        
        if "RETRIEVED CONTEXT:" in prompt:
            context_parts = prompt.split("RETRIEVED CONTEXT:")
            if len(context_parts) > 1:
                context = context_parts[1].split("USER QUESTION:")[0].strip()
        
        response = self._generate_structured_response(question, context)
        return {"content": response}
    
    @property
    def content(self) -> str:
        return ""


def get_mock_llm() -> MockLLM:
    """Factory function for mock LLM."""
    return MockLLM()
