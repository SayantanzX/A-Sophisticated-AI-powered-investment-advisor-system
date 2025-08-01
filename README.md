# 💰 AI Investment Advisor & Portfolio Manager

A sophisticated ML-powered investment advisor system with multimodal analysis capabilities, real-time market data integration, and voice-based queries.

## 🌟 Features

### Core Investment Features
- **Real-time Stock Analysis**: Live market data with technical indicators
- **Portfolio Optimization**: Modern Portfolio Theory + ML algorithms
- **Risk Assessment**: Comprehensive questionnaire + behavioral analysis
- **Trading Signals**: Combined technical analysis and sentiment signals
- **Market Sentiment Analysis**: News + social media sentiment tracking

### Advanced AI Capabilities
- **Voice Investment Queries**: Ask questions like "Should I buy Apple stock?"
- **Multimodal Analysis**: Text + voice + market data integration
- **Intelligent Recommendations**: Context-aware stock suggestions
- **Real-time News Analysis**: Financial news sentiment processing
- **Behavioral Tracking**: User activity and preference learning

### Technical Features
- **Modern Portfolio Theory**: Optimal asset allocation algorithms
- **Technical Analysis**: RSI, MACD, Moving Averages, Bollinger Bands
- **Machine Learning Models**: Predictive analytics for stock recommendations
- **Real-time Data**: Live market feeds and news integration
- **Interactive Dashboard**: Professional Streamlit interface

## 🚀 Quick Start

### 1. Clone and Setup
```bash
git clone <your-repo>
cd ai-investment-advisor
pip install -r requirements.txt
```

### 2. Get API Keys
You'll need these free API keys:

#### News API (newsapi.org)
1. Visit [https://newsapi.org/register](https://newsapi.org/register)
2. Sign up for free account (500 requests/day)
3. Copy your API key

#### Alpha Vantage API (alphavantage.co)
1. Visit [https://www.alphavantage.co/support/#api-key](https://www.alphavantage.co/support/#api-key)
2. Get free API key (500 requests/day)
3. Copy your API key

### 3. Configure API Keys
Open `investment_advisor.py` and update these lines:
```python
NEWS_API_KEY = "your_news_api_key_here"
ALPHA_VANTAGE_API_KEY = "your_alpha_vantage_key_here"
```

### 4. Run the Application
```bash
streamlit run streamlit_investment_app.py
```

## 📁 Project Structure

```
ai-investment-advisor/
├── investment_advisor.py          # Core ML system
├── streamlit_investment_app.py    # Streamlit interface
├── requirements.txt               # Dependencies
├── README.md                     # This file
```

## 🧠 System Architecture

### Core Components

#### 1. EnhancedUserPortfolioTracker
```python
- Track user investment behavior
- Monitor trading patterns
- Portfolio performance tracking
- Device and system metrics
```

#### 2. MarketDataAnalyzer
```python
- Real-time stock data (Yahoo Finance)
- Technical indicators calculation
- Market overview and indices
- Historical data processing
```

#### 3. NewsAndSentimentAnalyzer
```python
- Financial news aggregation
- VADER + FinBERT sentiment analysis
- Market sentiment scoring
- News impact assessment
```

#### 4. RiskAssessmentEngine
```python
- Investment risk questionnaire
- Behavioral risk analysis
- Risk-based asset allocation
- Dynamic risk profiling
```

#### 5. PortfolioOptimizer
```python
- Modern Portfolio Theory implementation
- Efficient frontier calculation
- Risk-return optimization
- Diversification analysis
```

#### 6. TradingSignalGenerator
```python
- Technical analysis signals
- Sentiment-based signals
- ML-driven predictions
- Risk-adjusted recommendations
```

## 🎯 Usage Examples

### Basic Stock Analysis
```python
# Initialize the system
advisor = EnhancedInvestmentAdvisorSystem(NEWS_API_KEY, ALPHA_VANTAGE_API_KEY)

# Get recommendations for a user
user_id = "investor_123"
recommendations = advisor.get_enhanced_investment_recommendations(user_id)

# View top recommendations
for stock in recommendations['recommended_stocks'][:5]:
    print(f"{stock['symbol']}: {stock['recommendation_score']:.2f}")
```

### Voice Investment Queries
```python
# Process voice input
voice_query = "Should I invest in Tesla stock right now?"
results = advisor.process_voice_investment_query(voice_query, user_id)

if results['success']:
    print(f"Investment advice: {results['recommendations']}")
```

### Risk Assessment
```python
# Calculate user risk profile
questionnaire = {
    'time_horizon': 4,  # 5-10 years
    'risk_reaction': 3,  # Hold and wait
    'income_percentage': 3,  # 10-20%
    'investment_goal': 4,  # Capital appreciation
    'experience_level': 3   # Moderate experience
}

risk_profile = advisor.risk_engine.calculate_risk_score(
    user_id, questionnaire
)
print(f"Risk Level: {risk_profile['risk_level']}")
```

### Portfolio Optimization
```python
# Optimize portfolio for selected stocks
symbols = ['AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA']
market_data = {symbol: advisor.market_analyzer.get_stock_data(symbol) 
               for symbol in symbols}

optimization = advisor.portfolio_optimizer.optimize_portfolio(
    symbols, risk_profile, market_data
)

print("Optimal allocation:")
for symbol, weight in optimization['optimal_weights'].items():
    print(f"{symbol}: {weight}%")
```

## 🎙️ Voice Commands

The system understands natural language investment queries:

### Stock Analysis
- "What do you think about Apple stock?"
- "Should I buy Microsoft shares?"
- "Tell me about Tesla's performance"
- "Is Amazon a good investment right now?"

### Market Research
- "What are the best tech stocks to buy?"
- "Show me some safe dividend stocks"
- "Which sectors are performing well?"
- "What's the market sentiment today?"

### Portfolio Questions
- "How should I diversify my portfolio?"
- "What's the optimal allocation for aggressive growth?"
- "Should I sell my losing positions?"
- "When is a good time to buy the dip?"

## 📊 Available Features

### Investment Analysis
- ✅ Real-time stock prices and charts
- ✅ Technical indicators (RSI, MACD, Moving Averages)
- ✅ Fundamental analysis (P/E, Market Cap, Beta)
- ✅ Trading signal generation
- ✅ Risk-adjusted recommendations

### Portfolio Management
- ✅ Risk assessment questionnaire
- ✅ Modern Portfolio Theory optimization
- ✅ Asset allocation recommendations
- ✅ Diversification analysis
- ✅ Performance tracking

### Market Intelligence
- ✅ Real-time market overview
- ✅ Sector analysis and trends
- ✅ Financial news sentiment analysis
- ✅ Market sentiment scoring
- ✅ Economic indicator tracking

### User Experience
- ✅ Voice-based queries
- ✅ Interactive Streamlit dashboard
- ✅ Real-time data visualization
- ✅ Personalized recommendations
- ✅ Activity tracking and learning

## 🔧 Advanced Configuration

### Custom Stock Universe
```python
# Add your preferred stocks to track
advisor.popular_stocks = [
    'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA',
    'NVDA', 'META', 'NFLX', 'AMD', 'CRM'
    # Add more symbols here
]
```

### Risk Profile Customization
```python
# Modify risk assessment questions
custom_questions = [
    {
        'question': 'Your custom risk question?',
        'options': {
            'Conservative answer': 1,
            'Moderate answer': 3,
            'Aggressive answer': 5
        }
    }
]
advisor.risk_engine.risk_questionnaire = custom_questions
```

### Technical Indicator Settings
```python
# Customize technical analysis parameters
def custom_technical_analysis(stock_data):
    hist = stock_data['historical_data']
    
    # Custom SMA periods
    hist['SMA_10'] = ta.trend.sma_indicator(hist['Close'], window=10)
    hist['SMA_30'] = ta.trend.sma_indicator(hist['Close'], window=30)
    
    # Custom RSI period
    hist['RSI_14'] = ta.momentum.rsi(hist['Close'], window=14)
    
    return hist
```

## 🚀 Deployment Options

### Local Development
```bash
streamlit run streamlit_investment_app.py --server.port 8501
```


## 📈 Performance Optimization

### Caching Strategy
- Market data cached for 5 minutes
- News data cached for 30 minutes
- Technical indicators computed once per session
- User profiles persisted across sessions

### Memory Management
- Automatic cleanup of old cache entries
- Efficient pandas operations
- Streamlined API calls
- Optimized data structures

## 🔒 Security Best Practices

### API Key Management
```python
# Use environment variables
import os
NEWS_API_KEY = os.getenv('NEWS_API_KEY')
ALPHA_VANTAGE_API_KEY = os.getenv('ALPHA_VANTAGE_API_KEY')
```

### Data Protection
- No personal financial data stored
- Session-based user tracking
- Secure API communications
- Input validation and sanitization

## 🐛 Troubleshooting

### Common Issues

#### API Rate Limits
```python
# Handle rate limit errors gracefully
try:
    stock_data = get_stock_data(symbol)
except RateLimitError:
    st.warning("API rate limit reached. Please try again later.")
    return cached_data
```

#### Missing Dependencies
```bash
# Install missing packages individually
pip install yfinance ta nltk transformers
```

#### Voice Recognition Issues
- Ensure microphone permissions are granted
- Use Chrome/Edge browsers for best compatibility
- Check internet connection for cloud speech processing

### Performance Issues
- Clear browser cache and cookies
- Restart the Streamlit app
- Check API key validity
- Monitor system resources

## 📊 Example Output

### Sample Risk Assessment Result
```json
{
    "risk_score": 3.8,
    "risk_level": "Aggressive",
    "risk_category": "high",
    "recommended_allocation": {
        "stocks": 80,
        "bonds": 15,
        "cash": 5
    }
}
```

### Sample Stock Recommendation
```json
{
    "symbol": "AAPL",
    "current_price": 185.64,
    "price_change_pct": 2.3,
    "recommendation_score": 0.87,
    "trading_signals": {
        "overall_signal": "BUY",
        "confidence": 0.78,
        "technical_signals": {
            "moving_average": "BUY",
            "rsi": "HOLD",
            "macd": "BUY"
        }
    }
}
```

### Sample Portfolio Optimization
```json
{
    "optimal_weights": {
        "AAPL": 25.3,
        "MSFT": 22.1,
        "GOOGL": 18.7,
        "AMZN": 15.2,
        "TSLA": 12.4,
        "NVDA": 6.3
    },
    "expected_return": 12.8,
    "expected_volatility": 18.4,
    "sharpe_ratio": 0.69
}
```

## 🎯 Future Enhancements

### Planned Features
- [ ] Options trading analysis
- [ ] Cryptocurrency integration
- [ ] Advanced charting tools
- [ ] Backtesting capabilities
- [ ] Paper trading simulation
- [ ] Social trading features
- [ ] Mobile app development
- [ ] Advanced ML models (LSTM, GRU)

### Integration Opportunities
- [ ] Broker API integration (Alpaca, Interactive Brokers)
- [ ] Portfolio tracking with real accounts
- [ ] Tax optimization features
- [ ] Retirement planning tools
- [ ] ESG (Environmental, Social, Governance) scoring
