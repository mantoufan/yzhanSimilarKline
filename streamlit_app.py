import streamlit as st
import adata
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from plotly.subplots import make_subplots
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean

def search_securities(query):
    """
    Search for securities (indices, stocks, funds, bonds) by code or name
    """
    results = []
    
    # Search indices
    try:
        indices_df = adata.stock.info.all_index_code()
        indices = indices_df[
            (indices_df['index_code'].str.contains(query, case=False)) |
            (indices_df['name'].str.contains(query, case=False))
        ]
        for _, row in indices.iterrows():
            results.append({
                'code': row['index_code'],
                'name': row['name'],
                'type': 'index',
                'exchange': ''
            })
    except:
        pass

    # Search stocks
    try:
        stocks_df = adata.stock.info.all_code()
        stocks = stocks_df[
            (stocks_df['stock_code'].str.contains(query, case=False)) |
            (stocks_df['short_name'].str.contains(query, case=False))
        ]
        for _, row in stocks.iterrows():
            results.append({
                'code': row['stock_code'],
                'name': row['short_name'],
                'type': 'stock',
                'exchange': row['exchange']
            })
    except:
        pass

    # Search ETFs
    try:
        etfs_df = adata.fund.info.all_etf_exchange_traded_info()
        etfs = etfs_df[
            (etfs_df['fund_code'].str.contains(query, case=False)) |
            (etfs_df['short_name'].str.contains(query, case=False))
        ]
        for _, row in etfs.iterrows():
            results.append({
                'code': row['fund_code'],
                'name': row['short_name'],
                'type': 'etf',
                'exchange': ''
            })
    except:
        pass

    # Search bonds
    try:
        bonds_df = adata.bond.info.all_convert_code()
        bonds = bonds_df[
            (bonds_df['bond_code'].str.contains(query, case=False)) |
            (bonds_df['bond_name'].str.contains(query, case=False))
        ]
        for _, row in bonds.iterrows():
            results.append({
                'code': row['bond_code'],
                'name': row['bond_name'],
                'type': 'bond',
                'exchange': ''
            })
    except:
        pass

    return results

def get_market_data(code, security_type, days=365*3):
    """
    Get market data based on security type
    """
    end_date = datetime.now() + timedelta(days=7)
    start_date = datetime.now() - timedelta(days=days)
    
    df = None
    
    if security_type == 'stock':
        df = adata.stock.market.get_market(
            stock_code=code,
            start_date=start_date.strftime('%Y-%m-%d'),
            k_type=1,
            adjust_type=1
        )
    elif security_type == 'etf':
        df = adata.fund.market.get_market_etf(
            fund_code=code,
            k_type=1
        )
    elif security_type == 'index':
        df = adata.stock.market.get_market_index(
            index_code=code,
            start_date=start_date.strftime('%Y-%m-%d'),
            k_type=1
        )
    
    if df is not None and not df.empty:
        # Convert date column to datetime
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # Convert price columns to numeric
        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna(subset=numeric_columns)
        
        return df
    
    return None

def normalize_window(window):
    """Normalize price series to percentage changes from first point"""
    # Convert to numeric type, coerce errors to NaN
    numeric_window = pd.to_numeric(window, errors='coerce')
    if numeric_window.isna().any():
        return None
    return (numeric_window - numeric_window.iloc[0]) / numeric_window.iloc[0] * 100

def calculate_similarity(window1, window2):
    """
    Calculate similarity between two windows using correlation and euclidean distance
    """
    if len(window1) != len(window2):
        return 0
    
    # Normalize both windows
    norm1 = normalize_window(window1)
    norm2 = normalize_window(window2)
    
    # Check if normalization was successful
    if norm1 is None or norm2 is None:
        return 0
    
    # Calculate correlation
    try:
        corr, _ = pearsonr(norm1, norm2)
        
        # Calculate normalized euclidean distance
        dist = euclidean(norm1, norm2)
        normalized_dist = 1 / (1 + dist/len(window1))
        
        # Combine both metrics
        similarity = (corr + 1)/2 * 0.7 + normalized_dist * 0.3
        
        return similarity
    except:
        return 0

def find_similar_patterns(df, window_size=20):
    """
    Find similar historical patterns
    """
    if df is None or len(df) < window_size * 2:
        return []
    
    # Get the most recent window
    recent_window = df.tail(window_size)['close']
    
    similar_patterns = []
    
    # Slide through historical data
    # Stop at 30 days before the most recent data to avoid overlap
    max_i = len(df) - window_size * 2 - 30
    
    for i in range(max_i):
        historical_window = df.iloc[i:i+window_size]['close']
        
        # Calculate similarity
        similarity = calculate_similarity(recent_window, historical_window)
        
        similar_patterns.append({
            'start_date': df.iloc[i]['trade_date'],
            'end_date': df.iloc[i+window_size-1]['trade_date'],
            'data': df.iloc[i:i+window_size],
            'similarity': similarity
        })
    
    # Sort by similarity and get only the most similar pattern
    similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_patterns[:1]

def get_market_data(code, security_type, days=365*3):
    """
    Get market data based on security type
    """
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)
    
    df = None
    
    if security_type == 'stock':
        df = adata.stock.market.get_market(
            stock_code=code,
            start_date=start_date.strftime('%Y-%m-%d'),
            k_type=1,
            adjust_type=1
        )
    elif security_type == 'etf':
        df = adata.fund.market.get_market_etf(
            fund_code=code,
            k_type=1
        )
    elif security_type == 'index':
        df = adata.stock.market.get_market_index(
            index_code=code,
            start_date=start_date.strftime('%Y-%m-%d'),
            k_type=1
        )
    
    if df is not None and not df.empty:
        df['trade_date'] = pd.to_datetime(df['trade_date'])
        
        # Convert price columns to numeric
        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        
        # Remove any rows with NaN values
        df = df.dropna(subset=numeric_columns)
        
        return df
    
    return None

def find_similar_patterns(df, window_size=30):
    """
    Find similar historical patterns
    """
    if df is None or len(df) < window_size * 2:
        return []
    
    # Get the most recent window (30 days)
    recent_window = df.tail(window_size)['close']
    
    similar_patterns = []
    
    # Slide through historical data
    # Stop at window_size + 7 days before the recent window to ensure we have future data
    max_i = len(df) - (window_size * 2 + 7)
    
    for i in range(max_i):
        historical_window = df.iloc[i:i+window_size]['close']
        
        # Calculate similarity
        similarity = calculate_similarity(recent_window, historical_window)
        
        if similarity > 0:  # Only include if there's some similarity
            similar_patterns.append({
                'start_date': df.iloc[i]['trade_date'],
                'end_date': df.iloc[i+window_size-1]['trade_date'],
                'pattern_data': df.iloc[i:i+window_size],  # 30 days pattern
                'future_data': df.iloc[i+window_size:i+window_size+7],  # next 7 days
                'similarity': similarity
            })
    
    # Sort by similarity and get the most similar pattern
    similar_patterns.sort(key=lambda x: x['similarity'], reverse=True)
    return similar_patterns[:1]

def get_current_window(df):
    """
    Get current window data with correct future dates
    Returns:
        - current data (30 days)
        - future dates (7 days)
    """
    if df is None or df.empty:
        return None, None
    
    # Calculate dates
    now = datetime.now()
    future_end = now + timedelta(days=7)
    past_start = now - timedelta(days=30)
    
    # Get current window data
    current_data = df[df['trade_date'] >= past_start].tail(30)
    
    # Create future dates
    future_dates = pd.date_range(
        start=current_data['trade_date'].max() + pd.Timedelta(days=1),
        end=future_end
    )
    
    return current_data, future_dates

def plot_kline(df, title, future_df=None, future_dates=None, show_future_data=False):
    """
    Create single K-line chart with separator line and future area
    """
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    
    # Add main candlestick trace
    fig.add_trace(go.Candlestick(
        x=df['trade_date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线'
    ))
    
    # Get the last date and price range
    last_date = df['trade_date'].max()
    y_range = [df['low'].min(), df['high'].max()]
    
    # Adjust y_range if we have future data
    if future_df is not None and not future_df.empty:
        y_range = [
            min(y_range[0], future_df['low'].min()),
            max(y_range[1], future_df['high'].max())
        ]
    
    # Add vertical separator line
    fig.add_vline(
        x=last_date,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        opacity=0.7
    )
    
    if future_df is not None and not future_df.empty and show_future_data:
        # Add actual future data for historical pattern
        fig.add_trace(go.Candlestick(
            x=future_df['trade_date'],
            open=future_df['open'],
            high=future_df['high'],
            low=future_df['low'],
            close=future_df['close'],
            name='后续走势'
        ))
    elif future_dates is not None:
        # Add blank space for future using provided dates
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=[None] * len(future_dates),
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        height=350,
        title={
            'text': title,
            'y': 0.98,
            'x': 0.5,
            'xanchor': 'center',
            'yanchor': 'top'
        },
        showlegend=False,
        template='plotly_white',
        margin=dict(t=50, l=50, r=30, b=50),
        yaxis_title='价格',
        xaxis_title='日期',
        yaxis_range=y_range
    )
    
    # Remove rangeslider
    fig.update_xaxes(rangeslider=dict(visible=False))
    
    return fig

def display_market_analysis(current_df, similar_patterns):
    """
    Display current and similar K-line charts in separate divs
    """
    if not similar_patterns:
        return
    
    most_similar = similar_patterns[0]
    
    # Get current window data
    current_data, future_dates = get_current_window(current_df)
    
    # Current K-line chart
    with st.container():
        st.markdown("### 当前K线图")
        if current_data is not None and not current_data.empty:
            # For current chart, show empty future space
            fig_current = plot_kline(current_data, "", future_dates=future_dates)
            st.plotly_chart(fig_current, use_container_width=True)
        else:
            st.write("无法生成K线图")
    
    # Similar pattern K-line chart
    with st.container():
        st.markdown(f"### 历史相似的K线图")
        st.markdown(f"相似度: {most_similar['similarity']:.2%}")
        st.markdown(f"历史时间段: {most_similar['start_date'].strftime('%Y-%m-%d')} - {most_similar['end_date'].strftime('%Y-%m-%d')}")
        # For historical chart, show actual future data
        fig_similar = plot_kline(
            most_similar['pattern_data'], 
            "", 
            future_df=most_similar['future_data'],
            show_future_data=True
        )
        st.plotly_chart(fig_similar, use_container_width=True)

def main():
    st.title('金融数据查询系统')
    
    query = st.text_input('输入指数/股票/基金/债券的代码或名称进行搜索', '')
    
    if query:
        results = search_securities(query)
        
        if results:
            st.write(f'找到 {len(results)} 个结果:')
            
            for result in results:
                with st.expander(f"{result['name']} ({result['code']}) - {result['type'].upper()}"):
                    st.write(f"代码: {result['code']}")
                    st.write(f"名称: {result['name']}")
                    st.write(f"类型: {result['type'].upper()}")
                    if result['exchange']:
                        st.write(f"交易所: {result['exchange']}")
                    
                    # Get extended market data for pattern matching
                    market_data = get_market_data(result['code'], result['type'])
                    
                    if market_data is not None:
                        # Get recent month data for current display
                        current_month_data = market_data.tail(30).copy()
                        
                        # Find similar patterns
                        similar_patterns = find_similar_patterns(market_data)
                        
                        # Display charts in separate divs
                        display_market_analysis(current_month_data, similar_patterns)
                    else:
                        st.write("暂无K线数据")
        else:
            st.warning('未找到匹配的结果')

if __name__ == "__main__":
    main()