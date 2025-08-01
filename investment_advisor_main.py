from typing import Any, Dict, List
import pandas as pd
import numpy as np
import requests
import json
from datetime import datetime, timedelta
import time
import schedule
import nltk
from nltk.sentiment import SentimentIntensityAnalyzer
import psutil
import platform
from collections import defaultdict
import logging
import warnings
import os
import random
from PIL import Image
from transformers import pipeline
import yfinance as yf
import ta
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import plotly.graph_objects as go
import plotly.express as px

os.environ["STREAMLIT_BROWSER_GATHERUSAGESTATS"] = "false"

warnings.filterwarnings('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# API Keys (you'll need to get these)
NEWS_API_KEY = "32deb9c6a12e4d60b4656bc9f7b31fa7"  # Get from newsapi.org
ALPHA_VANTAGE_API_KEY = "8WJOTH902615G9B3"  # Get from alphavantage.co

class EnhancedUserPortfolioTracker:
    def __init__(self):
        self.user_sessions = {}
        self.device_data = {}
        self.trading_history = defaultdict(list)
        self.system_metrics = {}
        self.portfolio_performance = defaultdict(dict)
        
    def track_system_metrics(self, user_id):
        """Track system-level metrics for trading environment"""
        try:
            battery_info = psutil.sensors_battery()
            battery_percent = battery_info.percent if battery_info else None
        except (FileNotFoundError, AttributeError):
            battery_percent = None

        metrics = {
            'cpu_usage': psutil.cpu_percent(),
            'memory_usage': psutil.virtual_memory().percent,
            'battery': battery_percent,
            'system': platform.system(),
            'trading_session_active': True,
            'last_market_check': datetime.now().isoformat()
        }
        self.system_metrics[user_id] = metrics
        return metrics

    def track_investment_behavior(self, user_id, action_data):
        """Track detailed investment behavior"""
        current_time = datetime.now()
        investment_data = {
            'timestamp': current_time,
            'action_type': action_data.get('action_type'),  # 'view', 'buy', 'sell', 'research'
            'symbol': action_data.get('symbol'),
            'amount': action_data.get('amount', 0),
            'price': action_data.get('price', 0),
            'risk_level': action_data.get('risk_level', 'medium'),
            'sentiment_score': action_data.get('sentiment_score', 0)
        }
        self.trading_history[user_id].append(investment_data)

    def track_activity(self, user_id, activity_type, details=None):
        """Enhanced activity tracking for investment activities"""
        current_time = datetime.now()
        
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'session_start': current_time,
                'last_activity': current_time,
                'activity_count': 0,
                'activities': defaultdict(list),
                'stock_views': defaultdict(int),
                'watchlist': [],
                'research_history': [],
                'risk_tolerance': 'medium',
                'investment_goals': [],
                'device_info': self.track_system_metrics(user_id),
                'portfolio_value': 100000,  # Starting virtual portfolio
                'available_cash': 100000
            }
        
        session = self.user_sessions[user_id]
        session['last_activity'] = current_time
        session['activity_count'] += 1
        
        # Track specific activity with details
        activity_data = {
            'timestamp': current_time,
            'details': details
        }
        session['activities'][activity_type].append(activity_data)
        
        # Update specific metrics based on activity type
        if activity_type == 'stock_view' and details:
            session['stock_views'][details['symbol']] += 1
        elif activity_type == 'research' and details:
            session['research_history'].append(details['query'])
        elif activity_type == 'watchlist_add' and details:
            if details['symbol'] not in session['watchlist']:
                session['watchlist'].append(details['symbol'])
        elif activity_type == 'trading' and details:
            self.track_investment_behavior(user_id, details)
            
    def get_user_activity(self, user_id):
        """Get comprehensive user investment activity data"""
        if user_id in self.user_sessions:
            session = self.user_sessions[user_id]
            current_time = datetime.now()
            duration = (current_time - session['session_start']).total_seconds() / 60
            
            # Get recent trading history
            recent_trades = self.trading_history[user_id][-10:] if user_id in self.trading_history else []
            
            return {
                'session_duration': duration,
                'activity_count': session['activity_count'],
                'stock_views': dict(session['stock_views']),
                'watchlist_size': len(session['watchlist']),
                'watchlist': session['watchlist'],
                'research_history': session['research_history'][-5:],
                'risk_tolerance': session['risk_tolerance'],
                'recent_trades': [
                    {**trade_data, 'timestamp': trade_data['timestamp'].isoformat()}
                    for trade_data in recent_trades
                ],
                'portfolio_value': session['portfolio_value'],
                'available_cash': session['available_cash'],
                'device_info': self.track_system_metrics(user_id),
                'last_activity': session['last_activity'].isoformat() 
            }
        return None

class MarketDataAnalyzer:
    def __init__(self, alpha_vantage_key):
        self.api_key = alpha_vantage_key
        self.market_cache = {}
        self.cache_duration = timedelta(minutes=5)  # Shorter cache for market data
        
    def get_stock_data(self, symbol, period='1y'):
        """Get detailed stock data with caching"""
        current_time = datetime.now()
        cache_key = f"{symbol}_{period}"
        
        # Check cache
        if cache_key in self.market_cache:
            cache_time, cached_data = self.market_cache[cache_key]
            if current_time - cache_time < self.cache_duration:
                return cached_data
        
        try:
            # Using yfinance for reliable stock data
            stock = yf.Ticker(symbol)
            hist = stock.history(period=period)
            info = stock.info
            
            if not hist.empty:
                # Calculate technical indicators
                hist['SMA_20'] = ta.trend.sma_indicator(hist['Close'], window=20)
                hist['SMA_50'] = ta.trend.sma_indicator(hist['Close'], window=50)
                hist['RSI'] = ta.momentum.rsi(hist['Close'])
                hist['MACD'] = ta.trend.macd_diff(hist['Close'])
                hist['BB_upper'], hist['BB_middle'], hist['BB_lower'] = ta.volatility.bollinger_hband(hist['Close']), ta.volatility.bollinger_mavg(hist['Close']), ta.volatility.bollinger_lband(hist['Close'])
                
                # Current market data
                current_price = hist['Close'].iloc[-1]
                price_change = current_price - hist['Close'].iloc[-2]
                price_change_pct = (price_change / hist['Close'].iloc[-2]) * 100
                
                market_data = {
                    'symbol': symbol,
                    'current_price': current_price,
                    'price_change': price_change,
                    'price_change_pct': price_change_pct,
                    'volume': hist['Volume'].iloc[-1],
                    'market_cap': info.get('marketCap', 0),
                    'pe_ratio': info.get('trailingPE', 0),
                    'dividend_yield': info.get('dividendYield', 0),
                    'beta': info.get('beta', 1.0),
                    'sector': info.get('sector', 'Unknown'),
                    'industry': info.get('industry', 'Unknown'),
                    'historical_data': hist,
                    'technical_indicators': {
                        'sma_20': hist['SMA_20'].iloc[-1],
                        'sma_50': hist['SMA_50'].iloc[-1],
                        'rsi': hist['RSI'].iloc[-1],
                        'macd': hist['MACD'].iloc[-1]
                    },
                    'volatility': hist['Close'].pct_change().std() * np.sqrt(252),  # Annualized volatility
                    'avg_volume': hist['Volume'].mean()
                }
                
                # Cache the data
                self.market_cache[cache_key] = (current_time, market_data)
                return market_data
                
        except Exception as e:
            logger.error(f"Error fetching stock data for {symbol}: {e}")
            return None

    def get_market_overview(self):
        """Get overall market indicators"""
        try:
            # Major market indices
            indices = ['^GSPC', '^DJI', '^IXIC', '^VIX']  # S&P 500, Dow, NASDAQ, VIX
            market_overview = {}
            
            for index in indices:
                data = self.get_stock_data(index, '5d')
                if data:
                    market_overview[index] = {
                        'current_price': data['current_price'],
                        'price_change_pct': data['price_change_pct'],
                        'symbol': index
                    }
            
            return market_overview
        except Exception as e:
            logger.error(f"Error fetching market overview: {e}")
            return {}

class NewsAndSentimentAnalyzer:
    def __init__(self, news_api_key):
        self.news_api_key = news_api_key
        self.sentiment_cache = {}
        self.news_cache = {}
        self.cache_duration = timedelta(minutes=30)
        
        # Initialize sentiment analyzer
        try:
            nltk.data.find('sentiment/vader_lexicon')
        except LookupError:
            nltk.download('vader_lexicon')
        
        self.sentiment_analyzer = SentimentIntensityAnalyzer()
        
        # Initialize transformers pipeline for financial sentiment
        try:
            self.financial_sentiment = pipeline(
                "sentiment-analysis",
                model="ProsusAI/finbert"
            )
        except:
            logger.warning("FinBERT model not available, using VADER only")
            self.financial_sentiment = None
    
    def get_financial_news(self, query="stock market", sources="bloomberg,reuters,cnbc"):
        """Get financial news with caching"""
        current_time = datetime.now()
        cache_key = f"news_{query}_{sources}"
        
        # Check cache
        if cache_key in self.news_cache:
            cache_time, cached_data = self.news_cache[cache_key]
            if current_time - cache_time < self.cache_duration:
                return cached_data
        
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                'q': query,
                'sources': sources,
                'sortBy': 'publishedAt',
                'pageSize': 50,
                'apiKey': self.news_api_key,
                'language': 'en',
                'from': (current_time - timedelta(days=1)).isoformat()
            }
            
            response = requests.get(url, params=params, timeout=10)
            
            if response.status_code == 200:
                news_data = response.json()
                articles = news_data.get('articles', [])
                
                # Process articles with sentiment
                processed_articles = []
                for article in articles[:20]:  # Process top 20 articles
                    sentiment = self.analyze_text_sentiment(article.get('title', '') + ' ' + article.get('description', ''))
                    
                    processed_article = {
                        'title': article.get('title'),
                        'description': article.get('description'),
                        'url': article.get('url'),
                        'source': article.get('source', {}).get('name'),
                        'published_at': article.get('publishedAt'),
                        'sentiment': sentiment
                    }
                    processed_articles.append(processed_article)
                
                # Cache the processed data
                self.news_cache[cache_key] = (current_time, processed_articles)
                return processed_articles
                
        except Exception as e:
            logger.error(f"Error fetching news: {e}")
            return []
    
    def analyze_text_sentiment(self, text):
        """Analyze sentiment of financial text"""
        if not text:
            return {'compound': 0, 'pos': 0, 'neu': 0, 'neg': 0, 'label': 'neutral'}
        
        # Check cache
        if text in self.sentiment_cache:
            return self.sentiment_cache[text]
        
        # VADER sentiment
        vader_scores = self.sentiment_analyzer.polarity_scores(text)
        
        result = vader_scores.copy()
        
        # Add FinBERT sentiment if available
        if self.financial_sentiment:
            try:
                finbert_result = self.financial_sentiment(text)[0]
                result['finbert_label'] = finbert_result['label']
                result['finbert_score'] = finbert_result['score']
            except:
                pass
        
        # Determine overall label
        if result['compound'] >= 0.05:
            result['label'] = 'positive'
        elif result['compound'] <= -0.05:
            result['label'] = 'negative'
        else:
            result['label'] = 'neutral'
        
        # Cache result
        self.sentiment_cache[text] = result
        return result
    
    def get_market_sentiment_score(self, symbol=None):
        """Get overall market sentiment score"""
        query = f"{symbol} stock" if symbol else "stock market"
        news_articles = self.get_financial_news(query)
        
        if not news_articles:
            return {'overall_sentiment': 0, 'sentiment_distribution': {}, 'article_count': 0}
        
        sentiments = [article['sentiment']['compound'] for article in news_articles]
        sentiment_labels = [article['sentiment']['label'] for article in news_articles]
        
        # Calculate sentiment distribution
        sentiment_dist = {
            'positive': sentiment_labels.count('positive'),
            'neutral': sentiment_labels.count('neutral'),
            'negative': sentiment_labels.count('negative')
        }
        
        overall_sentiment = np.mean(sentiments)
        
        return {
            'overall_sentiment': overall_sentiment,
            'sentiment_distribution': sentiment_dist,
            'article_count': len(news_articles),
            'recent_articles': news_articles[:5]  # Top 5 recent articles
        }

class RiskAssessmentEngine:
    def __init__(self):
        self.risk_questionnaire = self._initialize_risk_questions()
        self.user_risk_profiles = {}
        
    def _initialize_risk_questions(self):
        """Initialize risk assessment questionnaire"""
        return [
            {
                'question': 'What is your investment time horizon?',
                'options': {
                    'Less than 1 year': 1,
                    '1-3 years': 2,
                    '3-5 years': 3,
                    '5-10 years': 4,
                    'More than 10 years': 5
                }
            },
            {
                'question': 'How would you react to a 20% drop in your portfolio value?',
                'options': {
                    'Sell everything immediately': 1,
                    'Sell some investments': 2,
                    'Hold and wait': 3,
                    'Buy more at lower prices': 4,
                    'Excited about the buying opportunity': 5
                }
            },
            {
                'question': 'What percentage of your total income do you plan to invest?',
                'options': {
                    'Less than 5%': 1,
                    '5-10%': 2,
                    '10-20%': 3,
                    '20-30%': 4,
                    'More than 30%': 5
                }
            },
            {
                'question': 'What is your primary investment goal?',
                'options': {
                    'Capital preservation': 1,
                    'Income generation': 2,
                    'Balanced growth': 3,
                    'Capital appreciation': 4,
                    'Aggressive growth': 5
                }
            },
            {
                'question': 'How familiar are you with investing?',
                'options': {
                    'Complete beginner': 1,
                    'Some knowledge': 2,
                    'Moderate experience': 3,
                    'Experienced investor': 4,
                    'Professional/Expert': 5
                }
            }
        ]
    
    def calculate_risk_score(self, user_id, questionnaire_responses, behavioral_data=None):
        """Calculate comprehensive risk score"""
        # Base score from questionnaire
        base_score = sum(questionnaire_responses.values()) / len(questionnaire_responses)
        
        # Adjust based on behavioral data if available
        behavioral_adjustment = 0
        if behavioral_data:
            # Analyze trading frequency (high frequency = higher risk tolerance)
            if behavioral_data.get('trading_frequency', 0) > 10:
                behavioral_adjustment += 0.5
            
            # Analyze portfolio concentration
            stock_views = behavioral_data.get('stock_views', {})
            if stock_views:
                concentration = max(stock_views.values()) / sum(stock_views.values())
                if concentration > 0.5:  # Concentrated on few stocks
                    behavioral_adjustment += 0.3
        
        final_score = min(5, max(1, base_score + behavioral_adjustment))
        
        # Categorize risk level
        if final_score <= 2:
            risk_level = 'Conservative'
            risk_category = 'low'
        elif final_score <= 3:
            risk_level = 'Moderate'
            risk_category = 'medium'
        elif final_score <= 4:
            risk_level = 'Aggressive'
            risk_category = 'high'
        else:
            risk_level = 'Very Aggressive'
            risk_category = 'very_high'
        
        risk_profile = {
            'user_id': user_id,
            'risk_score': final_score,
            'risk_level': risk_level,
            'risk_category': risk_category,
            'questionnaire_responses': questionnaire_responses,
            'behavioral_adjustment': behavioral_adjustment,
            'timestamp': datetime.now().isoformat(),
            'recommended_allocation': self._get_recommended_allocation(risk_category)
        }
        
        # Store the profile
        self.user_risk_profiles[user_id] = risk_profile
        return risk_profile
    
    def _get_recommended_allocation(self, risk_category):
        """Get recommended asset allocation based on risk category"""
        allocations = {
            'low': {'stocks': 30, 'bonds': 60, 'cash': 10},
            'medium': {'stocks': 60, 'bonds': 30, 'cash': 10},
            'high': {'stocks': 80, 'bonds': 15, 'cash': 5},
            'very_high': {'stocks': 90, 'bonds': 5, 'cash': 5}
        }
        return allocations.get(risk_category, allocations['medium'])

class PortfolioOptimizer:
    def __init__(self):
        self.optimization_cache = {}
        self.asset_correlations = {}
        
    def optimize_portfolio(self, symbols, risk_profile, market_data, historical_period='1y'):
        """Optimize portfolio using Modern Portfolio Theory + ML"""
        try:
            # Get historical data for all symbols
            returns_data = []
            for symbol in symbols:
                stock_data = market_data.get(symbol)
                if stock_data and 'historical_data' in stock_data:
                    hist = stock_data['historical_data']
                    returns = hist['Close'].pct_change().dropna()
                    returns_data.append(returns)
            
            if len(returns_data) < 2:
                return None
            
            # Create returns matrix
            returns_df = pd.concat(returns_data, axis=1)
            returns_df.columns = symbols
            returns_df = returns_df.dropna()
            
            # Calculate expected returns and covariance matrix
            expected_returns = returns_df.mean() * 252  # Annualized
            cov_matrix = returns_df.cov() * 252  # Annualized
            
            # Risk-based optimization
            risk_tolerance = risk_profile.get('risk_score', 3) / 5  # Normalize to 0-1
            
            # Generate efficient frontier points
            num_portfolios = 10000
            results = np.zeros((3, num_portfolios))
            weights_array = np.zeros((num_portfolios, len(symbols)))
            
            # Random portfolio generation
            for i in range(num_portfolios):
                weights = np.random.random(len(symbols))
                weights /= np.sum(weights)  # Normalize to sum to 1
                weights_array[i] = weights
                
                # Portfolio return and risk
                portfolio_return = np.sum(weights * expected_returns)
                portfolio_volatility = np.sqrt(np.dot(weights.T, np.dot(cov_matrix, weights)))
                
                results[0, i] = portfolio_return
                results[1, i] = portfolio_volatility
                results[2, i] = portfolio_return / portfolio_volatility  # Sharpe ratio
            
            # Find optimal portfolio based on risk tolerance
            # Higher risk tolerance -> optimize for return
            # Lower risk tolerance -> optimize for Sharpe ratio
            if risk_tolerance > 0.7:  # High risk tolerance
                optimal_idx = np.argmax(results[0])  # Maximize return
            elif risk_tolerance < 0.3:  # Low risk tolerance
                optimal_idx = np.argmin(results[1])  # Minimize risk
            else:  # Moderate risk tolerance
                optimal_idx = np.argmax(results[2])  # Maximize Sharpe ratio
            
            optimal_weights = weights_array[optimal_idx]
            
            # Create portfolio recommendation
            portfolio_weights = {}
            for i, symbol in enumerate(symbols):
                if optimal_weights[i] > 0.01:  # Only include weights > 1%
                    portfolio_weights[symbol] = round(optimal_weights[i] * 100, 2)
            
            # Normalize weights to sum to 100%
            total_weight = sum(portfolio_weights.values())
            if total_weight > 0:
                portfolio_weights = {k: round(v/total_weight * 100, 2) for k, v in portfolio_weights.items()}
            
            optimization_result = {
                'symbols': symbols,
                'optimal_weights': portfolio_weights,
                'expected_return': results[0, optimal_idx] * 100,  # Convert to percentage
                'expected_volatility': results[1, optimal_idx] * 100,
                'sharpe_ratio': results[2, optimal_idx],
                'risk_profile_used': risk_profile['risk_level'],
                'optimization_date': datetime.now().isoformat(),
                'diversification_score': len([w for w in optimal_weights if w > 0.05])  # Assets with >5% allocation
            }
            
            return optimization_result
            
        except Exception as e:
            logger.error(f"Error in portfolio optimization: {e}")
            return None

class TradingSignalGenerator:
    def __init__(self):
        self.signal_cache = {}
        self.ml_models = {}
        self.technical_indicators = {}
        
    def generate_trading_signals(self, symbol, market_data, sentiment_data, risk_profile):
        """Generate comprehensive trading signals"""
        try:
            signals = {
                'symbol': symbol,
                'timestamp': datetime.now().isoformat(),
                'technical_signals': {},
                'sentiment_signals': {},
                'ml_signals': {},
                'overall_signal': 'HOLD',
                'confidence': 0.5,
                'risk_adjusted_signal': 'HOLD'
            }
            
            if not market_data:
                return signals
            
            # Technical Analysis Signals
            tech_indicators = market_data.get('technical_indicators', {})
            current_price = market_data.get('current_price', 0)
            
            # Moving Average Signals
            sma_20 = tech_indicators.get('sma_20', current_price)
            sma_50 = tech_indicators.get('sma_50', current_price)
            
            if current_price > sma_20 > sma_50:
                signals['technical_signals']['moving_average'] = 'BUY'
            elif current_price < sma_20 < sma_50:
                signals['technical_signals']['moving_average'] = 'SELL'
            else:
                signals['technical_signals']['moving_average'] = 'HOLD'
            
            # RSI Signals
            rsi = tech_indicators.get('rsi', 50)
            if rsi < 30:
                signals['technical_signals']['rsi'] = 'BUY'  # Oversold
            elif rsi > 70:
                signals['technical_signals']['rsi'] = 'SELL'  # Overbought
            else:
                signals['technical_signals']['rsi'] = 'HOLD'
            
            # MACD Signals
            macd = tech_indicators.get('macd', 0)
            if macd > 0:
                signals['technical_signals']['macd'] = 'BUY'
            elif macd < 0:
                signals['technical_signals']['macd'] = 'SELL'
            else:
                signals['technical_signals']['macd'] = 'HOLD'
            
            # Sentiment Analysis Signals
            if sentiment_data:
                overall_sentiment = sentiment_data.get('overall_sentiment', 0)
                if overall_sentiment > 0.1:
                    signals['sentiment_signals']['news_sentiment'] = 'BUY'
                elif overall_sentiment < -0.1:
                    signals['sentiment_signals']['news_sentiment'] = 'SELL'
                else:
                    signals['sentiment_signals']['news_sentiment'] = 'HOLD'
            
            # Combine signals
            buy_signals = 0
            sell_signals = 0
            total_signals = 0
            
            for signal_type in ['technical_signals', 'sentiment_signals']:
                for signal_name, signal_value in signals[signal_type].items():
                    total_signals += 1
                    if signal_value == 'BUY':
                        buy_signals += 1
                    elif signal_value == 'SELL':
                        sell_signals += 1
            
            if total_signals > 0:
                buy_ratio = buy_signals / total_signals
                sell_ratio = sell_signals / total_signals
                
                if buy_ratio > 0.6:
                    signals['overall_signal'] = 'BUY'
                    signals['confidence'] = buy_ratio
                elif sell_ratio > 0.6:
                    signals['overall_signal'] = 'SELL'
                    signals['confidence'] = sell_ratio
                else:
                    signals['overall_signal'] = 'HOLD'
                    signals['confidence'] = max(buy_ratio, sell_ratio)
            
            # Risk-adjusted signal
            risk_category = risk_profile.get('risk_category', 'medium')
            if risk_category in ['low', 'very_low'] and signals['overall_signal'] == 'SELL':
                signals['risk_adjusted_signal'] = 'SELL'  # Conservative investors should sell on negative signals
            elif risk_category in ['high', 'very_high'] and signals['overall_signal'] == 'BUY':
                signals['risk_adjusted_signal'] = 'BUY'  # Aggressive investors can buy on positive signals
            else:
                signals['risk_adjusted_signal'] = 'HOLD'  # Default to hold for moderate risk
            
            return signals
            
        except Exception as e:
            logger.error(f"Error generating trading signals for {symbol}: {e}")
            return signals

class EnhancedInvestmentAdvisorSystem:
    def __init__(self, news_api_key, alpha_vantage_key):
        self.user_tracker = EnhancedUserPortfolioTracker()
        self.market_analyzer = MarketDataAnalyzer(alpha_vantage_key)
        self.news_analyzer = NewsAndSentimentAnalyzer(news_api_key)
        self.risk_engine = RiskAssessmentEngine()
        self.portfolio_optimizer = PortfolioOptimizer()
        self.signal_generator = TradingSignalGenerator()
        
        # Popular stock symbols for recommendations
        self.popular_stocks = [
            'AAPL', 'MSFT', 'GOOGL', 'AMZN', 'TSLA', 'META', 'NVDA', 'NFLX',
            'JPM', 'V', 'WMT', 'PG', 'HD', 'MA', 'BAC', 'DIS', 'ADBE', 'CRM'
        ]
        
        # Sector ETFs for diversification
        self.sector_etfs = {
            'Technology': 'XLK',
            'Healthcare': 'XLV',
            'Financial': 'XLF',
            'Energy': 'XLE',
            'Consumer Discretionary': 'XLY',
            'Utilities': 'XLU',
            'Real Estate': 'XLRE',
            'Materials': 'XLB'
        }

    def process_voice_investment_query(self, voice_text: str, user_id: str) -> Dict[str, Any]:
        """Process voice input for investment queries"""
        if not voice_text or not voice_text.strip():
            return {"success": False, "error": "No voice text provided"}
        
        try:
            # Preprocess voice text for investment understanding
            processed_query = self.preprocess_investment_voice(voice_text)
            
            # Analyze sentiment from voice text
            sentiment = self.news_analyzer.analyze_text_sentiment(voice_text)
            
            # Track this voice interaction
            self.user_tracker.track_activity(user_id, 'voice_research', {
                'query': voice_text,
                'processed_query': processed_query,
                'timestamp': datetime.now().isoformat()
            })
            
            # Extract investment intent and symbols
            investment_intent = self.extract_investment_intent(processed_query)
            
            # Generate investment recommendations
            recommendations = self.get_enhanced_investment_recommendations(
                user_id=user_id,
                investment_intent=investment_intent,
                user_query=processed_query
            )
            
            return {
                "success": True,
                "recommendations": recommendations,
                "processed_query": processed_query,
                "original_voice_text": voice_text,
                "investment_intent": investment_intent,
                "sentiment_analysis": sentiment,
                "total_recommendations": len(recommendations.get('recommended_stocks', []))
            }
            
        except Exception as e:
            logger.error(f"Error in voice investment processing: {e}")
            return {"success": False, "error": str(e)}

    def preprocess_investment_voice(self, voice_text: str) -> str:
        """Preprocess voice text for investment understanding"""
        if not voice_text:
            return ""
        
        # Clean and normalize
        processed = voice_text.lower().strip()
        
        # Investment-specific corrections
        corrections = {
            'apple stock': 'AAPL',
            'microsoft': 'MSFT',
            'google': 'GOOGL',
            'amazon': 'AMZN',
            'tesla': 'TSLA',
            'facebook': 'META',
            'meta': 'META',
            'nvidia': 'NVDA',
            'netflix': 'NFLX',
            'jp morgan': 'JPM',
            'jpmorgan': 'JPM',
            'visa': 'V',
            'walmart': 'WMT',
            'procter and gamble': 'PG',
            'home depot': 'HD',
            'mastercard': 'MA',
            'bank of america': 'BAC',
            'disney': 'DIS',
            'adobe': 'ADBE',
            'salesforce': 'CRM'
        }
        
        for phrase, symbol in corrections.items():
            processed = processed.replace(phrase, symbol)
        
        return processed

    def extract_investment_intent(self, query: str) -> Dict[str, Any]:
        """Extract investment intent from user query"""
        intent = {
            'action': 'research',  # research, buy, sell, analyze
            'symbols': [],
            'sectors': [],
            'risk_level': 'medium',
            'time_horizon': 'medium',
            'amount': None
        }
        
        # Extract action intent
        if any(word in query for word in ['buy', 'purchase', 'invest in']):
            intent['action'] = 'buy'
        elif any(word in query for word in ['sell', 'exit', 'dump']):
            intent['action'] = 'sell'
        elif any(word in query for word in ['analyze', 'research', 'tell me about']):
            intent['action'] = 'analyze'
        
        # Extract symbols (simple regex for stock symbols)
        import re
        symbols = re.findall(r'\b[A-Z]{1,5}\b', query.upper())
        intent['symbols'] = symbols
        
        # Extract risk preferences
        if any(word in query for word in ['safe', 'conservative', 'low risk']):
            intent['risk_level'] = 'low'
        elif any(word in query for word in ['aggressive', 'high risk', 'growth']):
            intent['risk_level'] = 'high'
        
        # Extract time horizon
        if any(word in query for word in ['short term', 'quick', 'day trading']):
            intent['time_horizon'] = 'short'
        elif any(word in query for word in ['long term', 'retirement', 'years']):
            intent['time_horizon'] = 'long'
        
        return intent

    def get_basic_recommendations(self, user_id):
        """Generate basic investment recommendations"""
        # Get user activity data
        activity_data = self.user_tracker.get_user_activity(user_id)
        
        # Get market overview
        market_overview = self.market_analyzer.get_market_overview()
        
        # Get overall market sentiment
        market_sentiment = self.news_analyzer.get_market_sentiment_score()
        
        # Select stocks based on user activity and market conditions
        recommended_stocks = []
        
        # If user has viewed specific stocks, prioritize similar ones
        if activity_data and activity_data['stock_views']:
            viewed_stocks = list(activity_data['stock_views'].keys())
            # Add viewed stocks and similar ones
            for stock in viewed_stocks[:5]:  # Top 5 viewed stocks
                stock_data = self.market_analyzer.get_stock_data(stock)
                if stock_data:
                    recommended_stocks.append(stock_data)
        
        # Add popular stocks if we need more recommendations
        remaining_slots = max(0, 10 - len(recommended_stocks))
        for stock_symbol in self.popular_stocks[:remaining_slots]:
            if stock_symbol not in [s['symbol'] for s in recommended_stocks]:
                stock_data = self.market_analyzer.get_stock_data(stock_symbol)
                if stock_data:
                    recommended_stocks.append(stock_data)
        
        # Add basic scoring
        for stock in recommended_stocks:
            stock['recommendation_score'] = self.calculate_basic_score(stock, market_sentiment)
        
        # Sort by recommendation score
        recommended_stocks.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        
        return {
            'recommended_stocks': recommended_stocks[:10],
            'market_overview': market_overview,
            'market_sentiment': market_sentiment,
            'user_activity': activity_data
        }

    def calculate_basic_score(self, stock_data, market_sentiment):
        """Calculate basic recommendation score for a stock"""
        score = 0.5  # Base score
        
        # Price momentum (positive if price increased recently)
        if stock_data.get('price_change_pct', 0) > 0:
            score += 0.1
        elif stock_data.get('price_change_pct', 0) < -5:  # Significant drop might be buying opportunity
            score += 0.05
        
        # Technical indicators
        tech_indicators = stock_data.get('technical_indicators', {})
        rsi = tech_indicators.get('rsi', 50)
        
        # RSI scoring (prefer slightly oversold to neutral)
        if 30 <= rsi <= 45:  # Slightly oversold
            score += 0.2
        elif 45 < rsi <= 55:  # Neutral
            score += 0.1
        elif rsi > 70:  # Overbought
            score -= 0.1
        
        # Market sentiment influence
        sentiment_score = market_sentiment.get('overall_sentiment', 0)
        score += sentiment_score * 0.1  # Small influence from market sentiment
        
        # Volatility preference (penalize very high volatility)
        volatility = stock_data.get('volatility', 0.2)
        if volatility > 0.4:  # High volatility
            score -= 0.1
        elif 0.1 <= volatility <= 0.3:  # Moderate volatility
            score += 0.05
        
        return max(0, min(1, score))  # Clamp between 0 and 1

    def get_enhanced_investment_recommendations(self, user_id, investment_intent=None, user_query=None):
        """Generate enhanced investment recommendations with full analysis"""
        # Get basic recommendations first
        basic_results = self.get_basic_recommendations(user_id)
        
        # Get user risk profile if available
        risk_profile = self.risk_engine.user_risk_profiles.get(user_id)
        if not risk_profile:
            # Create default moderate risk profile
            risk_profile = {
                'risk_score': 3,
                'risk_level': 'Moderate',
                'risk_category': 'medium',
                'recommended_allocation': {'stocks': 60, 'bonds': 30, 'cash': 10}
            }
        
        enhanced_results = basic_results.copy()
        enhanced_results['risk_profile'] = risk_profile
        
        # Generate trading signals for each recommended stock
        recommended_stocks = enhanced_results['recommended_stocks']
        for stock in recommended_stocks:
            symbol = stock['symbol']
            
            # Get sentiment data for this stock
            stock_sentiment = self.news_analyzer.get_market_sentiment_score(symbol)
            
            # Generate trading signals
            trading_signals = self.signal_generator.generate_trading_signals(
                symbol, stock, stock_sentiment, risk_profile
            )
            
            stock['trading_signals'] = trading_signals
            stock['stock_sentiment'] = stock_sentiment
            
            # Update recommendation score with signals
            signal_adjustment = 0
            if trading_signals['overall_signal'] == 'BUY':
                signal_adjustment = 0.2 * trading_signals['confidence']
            elif trading_signals['overall_signal'] == 'SELL':
                signal_adjustment = -0.2 * trading_signals['confidence']
            
            stock['recommendation_score'] = min(1, max(0, 
                stock.get('recommendation_score', 0.5) + signal_adjustment
            ))
        
        # Re-sort by updated recommendation score
        recommended_stocks.sort(key=lambda x: x.get('recommendation_score', 0), reverse=True)
        enhanced_results['recommended_stocks'] = recommended_stocks
        
        # Portfolio optimization if user has multiple stocks
        if len(recommended_stocks) >= 3:
            top_symbols = [stock['symbol'] for stock in recommended_stocks[:8]]
            market_data_dict = {stock['symbol']: stock for stock in recommended_stocks[:8]}
            
            portfolio_optimization = self.portfolio_optimizer.optimize_portfolio(
                top_symbols, risk_profile, market_data_dict
            )
            enhanced_results['portfolio_optimization'] = portfolio_optimization
        
        # Sector analysis
        sector_analysis = self.analyze_sector_opportunities()
        enhanced_results['sector_analysis'] = sector_analysis
        
        # Investment intent specific recommendations
        if investment_intent:
            enhanced_results['intent_specific_advice'] = self.generate_intent_specific_advice(
                investment_intent, enhanced_results
            )
        
        return enhanced_results

    def analyze_sector_opportunities(self):
        """Analyze sector-specific opportunities"""
        sector_data = {}
        
        for sector, etf_symbol in self.sector_etfs.items():
            etf_data = self.market_analyzer.get_stock_data(etf_symbol)
            if etf_data:
                sector_sentiment = self.news_analyzer.get_market_sentiment_score(f"{sector} stocks")
                
                sector_data[sector] = {
                    'etf_symbol': etf_symbol,
                    'performance': etf_data.get('price_change_pct', 0),
                    'sentiment': sector_sentiment.get('overall_sentiment', 0),
                    'recommendation': 'HOLD'
                }
                
                # Simple sector recommendation logic
                if (etf_data.get('price_change_pct', 0) > 2 and 
                    sector_sentiment.get('overall_sentiment', 0) > 0.1):
                    sector_data[sector]['recommendation'] = 'BUY'
                elif (etf_data.get('price_change_pct', 0) < -5 or 
                      sector_sentiment.get('overall_sentiment', 0) < -0.1):
                    sector_data[sector]['recommendation'] = 'AVOID'
        
        return sector_data

    def generate_intent_specific_advice(self, investment_intent, results):
        """Generate advice specific to user intent"""
        advice = []
        
        action = investment_intent.get('action', 'research')
        risk_level = investment_intent.get('risk_level', 'medium')
        time_horizon = investment_intent.get('time_horizon', 'medium')
        
        if action == 'buy':
            if risk_level == 'low':
                advice.append("Consider starting with blue-chip stocks or ETFs for stability.")
                advice.append("Focus on dividend-paying stocks for steady income.")
            elif risk_level == 'high':
                advice.append("Growth stocks and tech companies align with your risk tolerance.")
                advice.append("Consider smaller positions in high-volatility stocks.")
            
            if time_horizon == 'long':
                advice.append("Dollar-cost averaging can help reduce timing risk.")
                advice.append("Focus on companies with strong fundamentals and growth potential.")
            elif time_horizon == 'short':
                advice.append("Be cautious with short-term trading - consider technical analysis.")
                advice.append("Set stop-loss orders to manage downside risk.")
        
        elif action == 'sell':
            advice.append("Review your original investment thesis - has it changed?")
            advice.append("Consider tax implications of selling (capital gains).")
            advice.append("Evaluate if rebalancing might be better than selling entirely.")
        
        # Specific symbol advice
        symbols = investment_intent.get('symbols', [])
        for symbol in symbols:
            stock_info = next((s for s in results['recommended_stocks'] if s['symbol'] == symbol), None)
            if stock_info:
                signals = stock_info.get('trading_signals', {})
                overall_signal = signals.get('overall_signal', 'HOLD')
                confidence = signals.get('confidence', 0.5)
                
                advice.append(f"{symbol}: Current signal is {overall_signal} with {confidence:.1%} confidence.")
        
        return advice

# Initialize the system
def create_investment_advisor():
    """Factory function to create the investment advisor system"""
    return EnhancedInvestmentAdvisorSystem(NEWS_API_KEY, ALPHA_VANTAGE_API_KEY)

# Test function
def test_investment_advisor():
    """Test the investment advisor system"""
    advisor = create_investment_advisor()
    
    # Test user
    user_id = 'test_investor'
    
    # Simulate user activities
    advisor.user_tracker.track_activity(user_id, 'stock_view', 
        {'symbol': 'AAPL', 'time_spent': 120})
    
    advisor.user_tracker.track_activity(user_id, 'research', 
        {'query': 'best tech stocks 2024'})
    
    advisor.user_tracker.track_activity(user_id, 'watchlist_add', 
        {'symbol': 'MSFT'})
    
    # Test risk assessment
    questionnaire_responses = {
        'time_horizon': 4,
        'risk_reaction': 3,
        'income_percentage': 3,
        'investment_goal': 3,
        'experience_level': 3
    }
    
    activity_data = advisor.user_tracker.get_user_activity(user_id)
    risk_profile = advisor.risk_engine.calculate_risk_score(
        user_id, questionnaire_responses, activity_data
    )
    
    # Get recommendations
    results = advisor.get_enhanced_investment_recommendations(user_id)
    
    print("\n=== Investment Advisor Test Results ===")
    print(f"\nRisk Profile: {risk_profile['risk_level']} (Score: {risk_profile['risk_score']:.2f})")
    print(f"Recommended Allocation: {risk_profile['recommended_allocation']}")
    
    print(f"\nMarket Overview:")
    market_overview = results.get('market_overview', {})
    for index, data in market_overview.items():
        print(f"  {index}: {data.get('current_price', 0):.2f} ({data.get('price_change_pct', 0):+.2f}%)")
    
    print(f"\nTop 5 Stock Recommendations:")
    for i, stock in enumerate(results['recommended_stocks'][:5], 1):
        signals = stock.get('trading_signals', {})
        print(f"{i}. {stock['symbol']} - ${stock['current_price']:.2f} "
              f"({stock['price_change_pct']:+.2f}%) "
              f"Signal: {signals.get('overall_signal', 'N/A')} "
              f"Score: {stock.get('recommendation_score', 0):.2f}")
    
    if 'portfolio_optimization' in results:
        opt = results['portfolio_optimization']
        print(f"\nOptimal Portfolio Allocation:")
        for symbol, weight in opt['optimal_weights'].items():
            print(f"  {symbol}: {weight}%")
        print(f"Expected Annual Return: {opt['expected_return']:.2f}%")
        print(f"Expected Volatility: {opt['expected_volatility']:.2f}%")
    
    print(f"\nSector Analysis:")
    sector_analysis = results.get('sector_analysis', {})
    for sector, data in sector_analysis.items():
        print(f"  {sector}: {data['recommendation']} "
              f"({data['performance']:+.2f}% performance)")

if __name__ == "__main__":
    # Run test
    test_investment_advisor()