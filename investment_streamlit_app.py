import os
import re
import streamlit as st
import pandas as pd
import json
import time
import requests
from datetime import datetime, timedelta
import plotly.express as px
import plotly.graph_objects as go
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import speech_recognition as sr
from PIL import Image
import io
import tempfile
import streamlit.components.v1 as components
import numpy as np
from typing import Any, Dict, List
import logging

# Import your investment advisor system
try:
    from investment_advisor_main import EnhancedInvestmentAdvisorSystem, NEWS_API_KEY, ALPHA_VANTAGE_API_KEY
    ML_SYSTEM_AVAILABLE = True
except ImportError:
    ML_SYSTEM_AVAILABLE = False
    st.error("Investment Advisor System not found. Please ensure 'investment_advisor_main.py' is in the same directory.")

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Configure Streamlit page
st.set_page_config(
    page_title="AI Investment Advisor",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for investment-specific styling
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        color: #1e3a8a;
        text-align: center;
        margin-bottom: 2rem;
    }
    .portfolio-card {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 1rem 0;
    }
    .stock-card {
        background: #f8fafc;
        padding: 1rem;
        border-radius: 8px;
        border-left: 4px solid #3b82f6;
        margin: 0.5rem 0;
    }
    .risk-low {
        background: linear-gradient(135deg, #10b981 0%, #059669 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-medium {
        background: linear-gradient(135deg, #f59e0b 0%, #d97706 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .risk-high {
        background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%);
        color: white;
        padding: 0.5rem;
        border-radius: 5px;
        text-align: center;
    }
    .signal-buy {
        background: #dcfce7;
        color: #166534;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .signal-sell {
        background: #fef2f2;
        color: #991b1b;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .signal-hold {
        background: #f3f4f6;
        color: #374151;
        padding: 0.3rem 0.6rem;
        border-radius: 15px;
        font-weight: bold;
        font-size: 0.8rem;
    }
    .market-positive {
        color: #059669;
        font-weight: bold;
    }
    .market-negative {
        color: #dc2626;
        font-weight: bold;
    }
    .voice-investment-section {
        background: linear-gradient(135deg, #6366f1 0%, #4f46e5 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 2rem 0;
        color: white;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
def initialize_session_state():
    if 'user_id' not in st.session_state:
        st.session_state.user_id = f"investor_{int(time.time())}"
    
    if 'investment_recommendations' not in st.session_state:
        st.session_state.investment_recommendations = None
    
    if 'risk_profile' not in st.session_state:
        st.session_state.risk_profile = None
    
    if 'portfolio_value' not in st.session_state:
        st.session_state.portfolio_value = 100000  # Virtual starting amount
    
    if 'voice_transcription' not in st.session_state:
        st.session_state.voice_transcription = ""
    
    if 'investment_advisor' not in st.session_state and ML_SYSTEM_AVAILABLE:
        st.session_state.investment_advisor = EnhancedInvestmentAdvisorSystem(
            NEWS_API_KEY, ALPHA_VANTAGE_API_KEY
        )

initialize_session_state()

# Voice Recording Component for Investment Queries
def create_investment_voice_component():
    """Create voice input component specifically for investment queries"""
    
    component_html = """
    <div style="text-align: center; padding: 15px; border: 2px solid #6366f1; border-radius: 10px; margin: 10px 0; background: linear-gradient(135deg, #f8fafc, #e2e8f0);">
        <h4 style="color: #4f46e5; margin-bottom: 10px;">🎙️ Voice Investment Assistant</h4>
        
        <div style="margin: 10px 0;">
            <button id="startBtn" onclick="startRecording()" style="
                background: linear-gradient(135deg, #10b981, #059669);
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 15px;
                cursor: pointer;
                font-size: 14px;
                margin: 3px;
                font-weight: bold;
                box-shadow: 0 2px 4px rgba(0, 0, 0, 0.1);
            ">
                🎤 Ask Investment Question
            </button>
            
            <button id="stopBtn" onclick="stopRecording()" disabled style="
                background: #6b7280;
                color: white;
                padding: 10px 20px;
                border: none;
                border-radius: 15px;
                cursor: not-allowed;
                font-size: 14px;
                margin: 3px;
                font-weight: bold;
            ">
                ⏹️ Stop Recording
            </button>
        </div>
        
        <div id="status" style="
            margin: 10px 0; 
            font-weight: bold; 
            color: #059669;
            padding: 8px;
            border-radius: 5px;
            background: #ecfdf5;
        ">Ready to record your investment question</div>
        
        <div id="transcription" style="
            margin: 10px 0; 
            padding: 12px; 
            background: #f0f9ff; 
            border: 1px solid #6366f1;
            border-radius: 6px; 
            min-height: 60px; 
            text-align: left;
            font-size: 14px;
        ">Try asking: "What are good tech stocks to buy?" or "Should I sell my Apple stock?"</div>
        
        <input type="hidden" id="voice_result" value="">
    </div>

    <script>
        let recognition;
        let isRecording = false;
        let finalTranscript = '';
        
        function updateStatus(message, color = '#059669', bgColor = '#ecfdf5') {
            const statusDiv = document.getElementById('status');
            statusDiv.style.color = color;
            statusDiv.style.backgroundColor = bgColor;
            statusDiv.innerHTML = message;
        }

        function sendToStreamlit(text) {
            if (text && text.trim().length > 0) {
                const cleanText = text.trim();
                
                // Store in sessionStorage
                sessionStorage.setItem('investment_voice_query', cleanText);
                sessionStorage.setItem('voice_timestamp', Date.now().toString());
                
                // Send via multiple methods
                const data = {
                    transcription: cleanText,
                    timestamp: Date.now(),
                    type: 'investment_query'
                };
                
                if (window.Streamlit) {
                    window.Streamlit.setComponentValue(data);
                }
                
                window.parent.postMessage({
                    type: 'streamlit:setComponentValue',
                    value: data
                }, '*');
                
                window.parent.postMessage({
                    type: 'investment_voice_ready',
                    query: cleanText,
                    timestamp: Date.now()
                }, '*');
            }
        }

        function initSpeechRecognition() {
            if (!('webkitSpeechRecognition' in window) && !('SpeechRecognition' in window)) {
                updateStatus('❌ Speech recognition not supported', '#dc2626', '#fef2f2');
                document.getElementById('startBtn').disabled = true;
                return false;
            }

            const SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
            recognition = new SpeechRecognition();
            
            recognition.continuous = true;
            recognition.interimResults = true;
            recognition.lang = 'en-US';
            
            recognition.onstart = function() {
                isRecording = true;
                finalTranscript = '';
                updateStatus('🔴 Listening... Ask your investment question!', '#dc2626', '#fecaca');
                
                document.getElementById('startBtn').disabled = true;
                document.getElementById('stopBtn').disabled = false;
                document.getElementById('stopBtn').style.background = '#ef4444';
                document.getElementById('stopBtn').style.cursor = 'pointer';
                document.getElementById('transcription').innerHTML = '🎤 Listening for your investment question...';
            };
            
            recognition.onresult = function(event) {
                let interimTranscript = '';
                finalTranscript = '';
                
                for (let i = 0; i < event.results.length; i++) {
                    if (event.results[i].isFinal) {
                        finalTranscript += event.results[i][0].transcript + ' ';
                    } else {
                        interimTranscript += event.results[i][0].transcript;
                    }
                }
                
                const display = finalTranscript + 
                    (interimTranscript ? '<span style="color: #6b7280; font-style: italic;">' + interimTranscript + '</span>' : '');
                
                document.getElementById('transcription').innerHTML = display || '🎤 Listening...';
            };
            
            recognition.onerror = function(event) {
                updateStatus('❌ Error: ' + event.error, '#dc2626', '#fef2f2');
                resetButtons();
            };
            
            recognition.onend = function() {
                isRecording = false;
                
                if (finalTranscript.trim()) {
                    const cleanQuery = finalTranscript.trim();
                    updateStatus('✅ Investment question captured! Processing...', '#059669', '#dcfce7');
                    document.getElementById('transcription').innerHTML = cleanQuery;
                    
                    sendToStreamlit(cleanQuery);
                    
                    setTimeout(() => {
                        updateStatus('✅ Ready for investment analysis!', '#059669', '#dcfce7');
                    }, 1000);
                } else {
                    updateStatus('⚠️ No question detected. Try again.', '#ca8a04', '#fefce8');
                    document.getElementById('transcription').innerHTML = 'No speech detected. Please try again.';
                }
                
                resetButtons();
            };
            
            return true;
        }

        function startRecording() {
            if (!recognition && !initSpeechRecognition()) {
                return;
            }
            
            try {
                recognition.start();
            } catch (error) {
                updateStatus('❌ Failed to start recording', '#dc2626', '#fef2f2');
                resetButtons();
            }
        }

        function stopRecording() {
            if (recognition && isRecording) {
                recognition.stop();
                updateStatus('⏳ Processing your investment question...', '#ca8a04', '#fefce8');
            }
        }

        function resetButtons() {
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
            document.getElementById('stopBtn').style.background = '#6b7280';
            document.getElementById('stopBtn').style.cursor = 'not-allowed';
        }

        // Initialize on load
        document.addEventListener('DOMContentLoaded', function() {
            initSpeechRecognition();
        });
    </script>
    """
    
    return st.components.v1.html(component_html, height=300)

# Risk Assessment Questionnaire
def render_risk_assessment():
    """Render risk assessment questionnaire"""
    st.subheader("📊 Investment Risk Assessment")
    
    if st.session_state.risk_profile:
        # Show existing risk profile
        risk_profile = st.session_state.risk_profile
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Risk Score", f"{risk_profile['risk_score']:.1f}/5")
        with col2:
            risk_level = risk_profile['risk_level']
            if 'Conservative' in risk_level:
                st.markdown(f'<div class="risk-low">{risk_level}</div>', unsafe_allow_html=True)
            elif 'Aggressive' in risk_level:
                st.markdown(f'<div class="risk-high">{risk_level}</div>', unsafe_allow_html=True)
            else:
                st.markdown(f'<div class="risk-medium">{risk_level}</div>', unsafe_allow_html=True)
        with col3:
            allocation = risk_profile['recommended_allocation']
            st.write(f"**Stocks:** {allocation['stocks']}%")
            st.write(f"**Bonds:** {allocation['bonds']}%")
            st.write(f"**Cash:** {allocation['cash']}%")
        
        if st.button("🔄 Retake Risk Assessment"):
            st.session_state.risk_profile = None
            st.rerun()
    
    else:
        # Risk assessment questionnaire
        with st.form("risk_assessment"):
            st.write("Please answer the following questions to determine your investment risk profile:")
            
            q1 = st.selectbox(
                "1. What is your investment time horizon?",
                ["Less than 1 year", "1-3 years", "3-5 years", "5-10 years", "More than 10 years"]
            )
            
            q2 = st.selectbox(
                "2. How would you react to a 20% drop in your portfolio value?",
                ["Sell everything immediately", "Sell some investments", "Hold and wait", 
                 "Buy more at lower prices", "Excited about the buying opportunity"]
            )
            
            q3 = st.selectbox(
                "3. What percentage of your total income do you plan to invest?",
                ["Less than 5%", "5-10%", "10-20%", "20-30%", "More than 30%"]
            )
            
            q4 = st.selectbox(
                "4. What is your primary investment goal?",
                ["Capital preservation", "Income generation", "Balanced growth", 
                 "Capital appreciation", "Aggressive growth"]
            )
            
            q5 = st.selectbox(
                "5. How familiar are you with investing?",
                ["Complete beginner", "Some knowledge", "Moderate experience", 
                 "Experienced investor", "Professional/Expert"]
            )
            
            if st.form_submit_button("Calculate Risk Profile", type="primary"):
                # Map responses to scores
                response_map = {
                    q1: {"Less than 1 year": 1, "1-3 years": 2, "3-5 years": 3, "5-10 years": 4, "More than 10 years": 5},
                    q2: {"Sell everything immediately": 1, "Sell some investments": 2, "Hold and wait": 3, 
                         "Buy more at lower prices": 4, "Excited about the buying opportunity": 5},
                    q3: {"Less than 5%": 1, "5-10%": 2, "10-20%": 3, "20-30%": 4, "More than 30%": 5},
                    q4: {"Capital preservation": 1, "Income generation": 2, "Balanced growth": 3, 
                         "Capital appreciation": 4, "Aggressive growth": 5},
                    q5: {"Complete beginner": 1, "Some knowledge": 2, "Moderate experience": 3, 
                         "Experienced investor": 4, "Professional/Expert": 5}
                }
                
                questionnaire_responses = {
                    'time_horizon': response_map[q1][q1],
                    'risk_reaction': response_map[q2][q2],
                    'income_percentage': response_map[q3][q3],
                    'investment_goal': response_map[q4][q4],
                    'experience_level': response_map[q5][q5]
                }
                
                if ML_SYSTEM_AVAILABLE:
                    advisor = st.session_state.investment_advisor
                    activity_data = advisor.user_tracker.get_user_activity(st.session_state.user_id)
                    
                    risk_profile = advisor.risk_engine.calculate_risk_score(
                        st.session_state.user_id, questionnaire_responses, activity_data
                    )
                    
                    st.session_state.risk_profile = risk_profile
                    st.success("✅ Risk profile calculated successfully!")
                    st.rerun()

def render_portfolio_overview():
    """Render portfolio overview and performance"""
    st.subheader("💼 Portfolio Overview")
    
    # Portfolio metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Portfolio Value", f"${st.session_state.portfolio_value:,.2f}", "2.3%")
    
    with col2:
        st.metric("Available Cash", f"${st.session_state.portfolio_value * 0.1:,.2f}")
    
    with col3:
        st.metric("Today's P&L", "$1,234.56", "0.8%")
    
    with col4:
        st.metric("Total Return", "15.7%", "2.1%")
    
    # Sample portfolio allocation chart
    if st.session_state.risk_profile:
        allocation = st.session_state.risk_profile['recommended_allocation']
        
        fig_pie = go.Figure(data=[go.Pie(
            labels=['Stocks', 'Bonds', 'Cash'],
            values=[allocation['stocks'], allocation['bonds'], allocation['cash']],
            hole=0.3,
            marker_colors=['#3b82f6', '#10b981', '#f59e0b']
        )])
        
        fig_pie.update_traces(textposition='inside', textinfo='percent+label')
        fig_pie.update_layout(
            title="Recommended Asset Allocation",
            showlegend=True,
            height=300
        )
        
        st.plotly_chart(fig_pie, use_container_width=True)

def render_market_overview():
    """Render market overview with major indices"""
    st.subheader("📈 Market Overview")
    
    if ML_SYSTEM_AVAILABLE and st.session_state.investment_advisor:
        advisor = st.session_state.investment_advisor
        
        with st.spinner("Loading market data..."):
            market_overview = advisor.market_analyzer.get_market_overview()
            
            if market_overview:
                cols = st.columns(len(market_overview))
                
                for i, (index, data) in enumerate(market_overview.items()):
                    with cols[i]:
                        price = data.get('current_price', 0)
                        change_pct = data.get('price_change_pct', 0)
                        
                        # Format index name
                        index_names = {
                            '^GSPC': 'S&P 500',
                            '^DJI': 'Dow Jones',
                            '^IXIC': 'NASDAQ',
                            '^VIX': 'VIX'
                        }
                        
                        display_name = index_names.get(index, index)
                        
                        st.metric(
                            display_name,
                            f"{price:.2f}",
                            f"{change_pct:+.2f}%"
                        )
            
            # Market sentiment
            market_sentiment = advisor.news_analyzer.get_market_sentiment_score()
            
            if market_sentiment:
                sentiment_score = market_sentiment.get('overall_sentiment', 0)
                sentiment_dist = market_sentiment.get('sentiment_distribution', {})
                
                col1, col2 = st.columns(2)
                
                with col1:
                    if sentiment_score > 0.1:
                        st.success(f"📈 Market Sentiment: Positive ({sentiment_score:.2f})")
                    elif sentiment_score < -0.1:
                        st.error(f"📉 Market Sentiment: Negative ({sentiment_score:.2f})")
                    else:
                        st.info(f"➡️ Market Sentiment: Neutral ({sentiment_score:.2f})")
                
                with col2:
                    if sentiment_dist:
                        total_articles = sum(sentiment_dist.values())
                        if total_articles > 0:
                            st.write(f"**News Analysis:** {total_articles} articles")
                            st.write(f"Positive: {sentiment_dist.get('positive', 0)}")
                            st.write(f"Neutral: {sentiment_dist.get('neutral', 0)}")
                            st.write(f"Negative: {sentiment_dist.get('negative', 0)}")

def render_investment_recommendations():
    """Render investment recommendations"""
    st.subheader("🎯 Investment Recommendations")
    
    if st.session_state.investment_recommendations:
        results = st.session_state.investment_recommendations
        
        # Check if this is voice-based recommendation
        if results.get('has_voice_input'):
            st.info(f"🎤 Recommendations based on your query: '{results.get('voice_text_used', '')}'")
        
        recommended_stocks = results.get('recommended_stocks', [])
        
        if recommended_stocks:
            st.write(f"**Top {len(recommended_stocks)} Stock Recommendations:**")
            
            for i, stock in enumerate(recommended_stocks[:10], 1):
                with st.container():
                    col1, col2, col3, col4 = st.columns([1, 3, 2, 1])
                    
                    with col1:
                        st.markdown(f"### #{i}")
                        score = stock.get('recommendation_score', 0)
                        st.metric("Score", f"{score:.2f}")
                    
                    with col2:
                        st.markdown(f"**{stock['symbol']}**")
                        st.write(f"Sector: {stock.get('sector', 'N/A')}")
                        st.write(f"Industry: {stock.get('industry', 'N/A')}")
                        
                        # Display price info
                        current_price = stock.get('current_price', 0)
                        price_change = stock.get('price_change_pct', 0)
                        
                        if price_change > 0:
                            st.markdown(f"Price: **${current_price:.2f}** <span class='market-positive'>+{price_change:.2f}%</span>", unsafe_allow_html=True)
                        else:
                            st.markdown(f"Price: **${current_price:.2f}** <span class='market-negative'>{price_change:.2f}%</span>", unsafe_allow_html=True)
                    
                    with col3:
                        # Trading signals
                        signals = stock.get('trading_signals', {})
                        overall_signal = signals.get('overall_signal', 'HOLD')
                        confidence = signals.get('confidence', 0.5)
                        
                        if overall_signal == 'BUY':
                            st.markdown(f'<div class="signal-buy">BUY ({confidence:.1%})</div>', unsafe_allow_html=True)
                        elif overall_signal == 'SELL':
                            st.markdown(f'<div class="signal-sell">SELL ({confidence:.1%})</div>', unsafe_allow_html=True)
                        else:
                            st.markdown(f'<div class="signal-hold">HOLD ({confidence:.1%})</div>', unsafe_allow_html=True)
                        
                        # Technical indicators
                        tech_indicators = stock.get('technical_indicators', {})
                        if tech_indicators:
                            rsi = tech_indicators.get('rsi', 50)
                            st.write(f"RSI: {rsi:.1f}")
                    
                    with col4:
                        if st.button(f"📊 Analyze", key=f"analyze_{stock['symbol']}_{i}"):
                            st.session_state[f"show_analysis_{stock['symbol']}"] = True
                        
                        if st.button(f"➕ Add to Watchlist", key=f"watchlist_{stock['symbol']}_{i}"):
                            if ML_SYSTEM_AVAILABLE:
                                advisor = st.session_state.investment_advisor
                                advisor.user_tracker.track_activity(
                                    st.session_state.user_id, 
                                    'watchlist_add', 
                                    {'symbol': stock['symbol']}
                                )
                                st.success(f"Added {stock['symbol']} to watchlist!")
                    
                    # Show detailed analysis if requested
                    if st.session_state.get(f"show_analysis_{stock['symbol']}", False):
                        with st.expander(f"📊 Detailed Analysis - {stock['symbol']}", expanded=True):
                            
                            analysis_col1, analysis_col2 = st.columns(2)
                            
                            with analysis_col1:
                                st.write("**Financial Metrics:**")
                                st.write(f"Market Cap: ${stock.get('market_cap', 0):,.0f}")
                                st.write(f"P/E Ratio: {stock.get('pe_ratio', 0):.2f}")
                                st.write(f"Beta: {stock.get('beta', 1.0):.2f}")
                                st.write(f"Dividend Yield: {stock.get('dividend_yield', 0)*100:.2f}%")
                                st.write(f"Volatility: {stock.get('volatility', 0)*100:.1f}%")
                            
                            with analysis_col2:
                                st.write("**Technical Analysis:**")
                                tech_signals = signals.get('technical_signals', {})
                                for signal_name, signal_value in tech_signals.items():
                                    st.write(f"{signal_name.replace('_', ' ').title()}: {signal_value}")
                                
                                # Stock sentiment
                                stock_sentiment = stock.get('stock_sentiment')
                                if stock_sentiment:
                                    sentiment_score = stock_sentiment.get('overall_sentiment', 0)
                                    st.write(f"**News Sentiment:** {sentiment_score:.2f}")
                            
                            # Price chart (mock data for demonstration)
                            if 'historical_data' in stock:
                                hist_data = stock['historical_data']
                                if not hist_data.empty:
                                    fig_price = go.Figure()
                                    fig_price.add_trace(go.Scatter(
                                        x=hist_data.index,
                                        y=hist_data['Close'],
                                        mode='lines',
                                        name='Price',
                                        line=dict(color='#3b82f6', width=2)
                                    ))
                                    
                                    fig_price.update_layout(
                                        title=f"{stock['symbol']} Price Chart",
                                        xaxis_title="Date",
                                        yaxis_title="Price ($)",
                                        height=300
                                    )
                                    
                                    st.plotly_chart(fig_price, use_container_width=True)
                    
                    st.divider()
            
            # Portfolio optimization results
            if 'portfolio_optimization' in results:
                st.subheader("🎯 Optimal Portfolio Allocation")
                
                optimization = results['portfolio_optimization']
                
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write("**Recommended Allocation:**")
                    optimal_weights = optimization['optimal_weights']
                    
                    for symbol, weight in optimal_weights.items():
                        st.write(f"**{symbol}:** {weight}%")
                
                with col2:
                    st.metric("Expected Annual Return", f"{optimization['expected_return']:.2f}%")
                    st.metric("Expected Volatility", f"{optimization['expected_volatility']:.2f}%")
                    st.metric("Sharpe Ratio", f"{optimization['sharpe_ratio']:.2f}")
                    st.metric("Diversification Score", optimization['diversification_score'])
                
                # Allocation pie chart
                if optimal_weights:
                    fig_allocation = go.Figure(data=[go.Pie(
                        labels=list(optimal_weights.keys()),
                        values=list(optimal_weights.values()),
                        hole=0.3
                    )])
                    
                    fig_allocation.update_layout(
                        title="Optimal Portfolio Allocation",
                        height=400
                    )
                    
                    st.plotly_chart(fig_allocation, use_container_width=True)
            
            # Sector analysis
            if 'sector_analysis' in results:
                st.subheader("🏭 Sector Analysis")
                
                sector_analysis = results['sector_analysis']
                
                sector_data = []
                for sector, data in sector_analysis.items():
                    sector_data.append({
                        'Sector': sector,
                        'ETF': data['etf_symbol'],
                        'Performance': data['performance'],
                        'Sentiment': data['sentiment'],
                        'Recommendation': data['recommendation']
                    })
                
                if sector_data:
                    sector_df = pd.DataFrame(sector_data)
                    
                    # Color code the recommendations
                    def highlight_recommendation(val):
                        if val == 'BUY':
                            return 'background-color: #dcfce7; color: #166534'
                        elif val == 'AVOID':
                            return 'background-color: #fef2f2; color: #991b1b'
                        else:
                            return 'background-color: #f3f4f6; color: #374151'
                    
                    styled_df = sector_df.style.applymap(
                        highlight_recommendation, 
                        subset=['Recommendation']
                    )
                    
                    st.dataframe(styled_df, use_container_width=True)
        
        else:
            st.warning("No stock recommendations available. Try generating recommendations first.")

def render_voice_investment_section():
    """Render the voice investment query section"""
    st.markdown("## 🎙️ Voice Investment Assistant")
    
    # Simple text input as fallback
    st.info("💡 **Tip:** Type your investment question below, or use the voice feature if available.")
    
    # Text input for investment queries
    investment_query = st.text_area(
        "Ask your investment question:",
        placeholder="e.g., What are good tech stocks to buy? Should I sell my Apple stock?",
        height=100,
        help="Type your investment question here for AI analysis"
    )
    
    # Voice component (simplified)
    st.markdown("### 🎤 Voice Input (Optional)")
    voice_component = create_investment_voice_component()
    
    # Check for voice input from session state
    current_voice_query = st.session_state.get('voice_transcription', '').strip()
    
    # Use voice input if available, otherwise use text input
    final_query = current_voice_query if current_voice_query else investment_query
    
    if final_query and len(final_query.strip()) > 3:
        st.success(f"✅ Query ready: {final_query}")
        
        col1, col2 = st.columns([3, 1])
        
        with col1:
            if st.button("🎯 Get Investment Analysis", type="primary", use_container_width=True):
                if ML_SYSTEM_AVAILABLE:
                    advisor = st.session_state.investment_advisor
                    
                    with st.spinner("🔍 Analyzing your investment query..."):
                        try:
                            # Process investment query
                            if current_voice_query:
                                voice_results = advisor.process_voice_investment_query(
                                    current_voice_query, 
                                    st.session_state.user_id
                                )
                            else:
                                # Process text query
                                voice_results = advisor.process_voice_investment_query(
                                    final_query, 
                                    st.session_state.user_id
                                )
                            
                            if voice_results['success']:
                                st.session_state.investment_recommendations = voice_results['recommendations']
                                st.session_state.investment_recommendations['has_voice_input'] = bool(current_voice_query)
                                st.session_state.investment_recommendations['voice_text_used'] = final_query
                                
                                st.success("✅ Investment analysis complete!")
                                st.rerun()
                            else:
                                st.error(f"❌ Error: {voice_results['error']}")
                        except Exception as e:
                            st.error(f"❌ Analysis failed: {str(e)}")
                else:
                    st.error("Investment advisor system not available")
        
        with col2:
            if st.button("🗑️ Clear", use_container_width=True):
                st.session_state.voice_transcription = ""
                st.rerun()
    
    # Voice troubleshooting info
    with st.expander("🔧 Voice Feature Troubleshooting"):
        st.markdown("""
        **If voice feature isn't working:**
        
        1. **Browser**: Use Chrome or Edge
        2. **Permissions**: Allow microphone access
        3. **HTTPS**: Voice works on localhost or HTTPS
        4. **Alternative**: Use the text input above
        
        **Common issues:**
        - Microphone not detected
        - Browser doesn't support Web Speech API
        - Security restrictions
        
        **Solution**: Type your question in the text area above for the same analysis!
        """)

# Main application
def main():
    # Header
    st.markdown('<h1 class="main-header">💰 AI Investment Advisor</h1>', unsafe_allow_html=True)
    
    # Sidebar
    with st.sidebar:
        st.header("🎯 Investment Dashboard")
        
        # User info
        st.subheader("👤 Investor Profile")
        st.write(f"**User ID:** {st.session_state.user_id}")
        
        # Risk assessment
        render_risk_assessment()
        
        st.divider()
        
        # Quick actions
        st.subheader("⚡ Quick Actions")
        
        if st.button("📈 Get Market Recommendations", use_container_width=True):
            if ML_SYSTEM_AVAILABLE:
                advisor = st.session_state.investment_advisor
                
                with st.spinner("Analyzing market conditions..."):
                    # Simulate some user activity
                    advisor.user_tracker.track_activity(
                        st.session_state.user_id, 
                        'research', 
                        {'query': 'market analysis'}
                    )
                    
                    # Get enhanced recommendations
                    results = advisor.get_enhanced_investment_recommendations(
                        st.session_state.user_id
                    )
                    
                    st.session_state.investment_recommendations = results
                    st.success("✅ Market analysis complete!")
                    st.rerun()
        
        if st.button("🔄 Refresh Data", use_container_width=True):
            # Clear cache and refresh
            if ML_SYSTEM_AVAILABLE:
                advisor = st.session_state.investment_advisor
                advisor.market_analyzer.market_cache.clear()
                advisor.news_analyzer.news_cache.clear()
                st.success("Data refreshed!")
        
        if st.button("📊 Portfolio Analysis"):
            st.info("Portfolio analysis feature coming soon!")
        
        # System status
        st.subheader("⚙️ System Status")
        if ML_SYSTEM_AVAILABLE:
            st.success("✅ Investment Advisor Active")
            st.success("✅ Market Data Connected")
            st.success("✅ News Analysis Ready")
            st.success("✅ Voice Recognition Ready")
        else:
            st.error("❌ System Unavailable")
    
    # Main content
    col1, col2 = st.columns([2, 1])
    
    with col1:
        # Portfolio Overview (moved to main area)
        render_portfolio_overview()
        
        st.divider()
        
        # Voice Investment Assistant (moved to main area)
        render_voice_investment_section()
        
        st.divider()
        
        # Investment recommendations
        render_investment_recommendations()
        
        # Sample news section
        if st.session_state.investment_recommendations:
            results = st.session_state.investment_recommendations
            market_sentiment = results.get('market_sentiment')
            
            if market_sentiment and market_sentiment.get('recent_articles'):
                st.subheader("📰 Recent Financial News")
                
                articles = market_sentiment['recent_articles'][:3]  # Show top 3 articles
                
                for article in articles:
                    with st.container():
                        st.markdown(f"**{article['title']}**")
                        st.write(article['description'])
                        
                        sentiment = article['sentiment']
                        sentiment_label = sentiment['label']
                        sentiment_score = sentiment['compound']
                        
                        col_news1, col_news2, col_news3 = st.columns(3)
                        with col_news1:
                            st.write(f"Source: {article['source']}")
                        with col_news2:
                            if sentiment_label == 'positive':
                                st.success(f"Sentiment: Positive ({sentiment_score:.2f})")
                            elif sentiment_label == 'negative':
                                st.error(f"Sentiment: Negative ({sentiment_score:.2f})")
                            else:
                                st.info(f"Sentiment: Neutral ({sentiment_score:.2f})")
                        with col_news3:
                            st.link_button("Read More", article['url'])
                        
                        st.divider()
    
    with col2:
        # Market overview
        render_market_overview()
        
        st.divider()
        
        # Market overview
        render_market_overview()
        
        st.divider()
        
        # Watchlist (sample)
        st.subheader("👁️ Watchlist")
        
        if ML_SYSTEM_AVAILABLE:
            advisor = st.session_state.investment_advisor
            activity_data = advisor.user_tracker.get_user_activity(st.session_state.user_id)
            
            if activity_data and activity_data.get('watchlist'):
                watchlist = activity_data['watchlist']
                
                for symbol in watchlist[:5]:  # Show top 5 watchlist items
                    stock_data = advisor.market_analyzer.get_stock_data(symbol)
                    if stock_data:
                        price = stock_data.get('current_price', 0)
                        change_pct = stock_data.get('price_change_pct', 0)
                        
                        col_w1, col_w2 = st.columns(2)
                        with col_w1:
                            st.write(f"**{symbol}**")
                        with col_w2:
                            if change_pct > 0:
                                st.markdown(f'<span class="market-positive">${price:.2f} (+{change_pct:.1f}%)</span>', unsafe_allow_html=True)
                            else:
                                st.markdown(f'<span class="market-negative">${price:.2f} ({change_pct:.1f}%)</span>', unsafe_allow_html=True)
            else:
                st.info("No stocks in watchlist yet. Add some stocks to track them here!")
        
        st.divider()
        
        # Quick market stats
        st.subheader("📊 Quick Stats")
        st.metric("Fear & Greed Index", "65", "Greed")
        st.metric("VIX (Volatility)", "18.5", "-2.1")
        st.metric("10Y Treasury", "4.2%", "+0.1%")

if __name__ == "__main__":
    main()