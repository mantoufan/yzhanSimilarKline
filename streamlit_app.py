import streamlit as st
import requests
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from datetime import datetime, timedelta
from scipy.stats import pearsonr
from scipy.spatial.distance import euclidean
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import jieba
import re
import adata
import os
from dotenv import load_dotenv
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor
from typing import List, Dict, Optional
from functools import wraps
import json

# 加载 .env 文件
load_dotenv()

# API 配置 - 优先从 .env 读取，如果没有则从 streamlit 环境变量读取
def get_config(key, default_value=""):
    """安全地获取配置值，优先从环境变量获取，然后尝试从 secrets 获取"""
    try:
        # 首先尝试从环境变量获取
        value = os.getenv(key)
        if value is not None:
            return value
            
        # 如果环境变量不存在，尝试从 secrets 获取
        try:
            return st.secrets.get(key, default_value)
        except FileNotFoundError:
            return default_value
    except Exception:
        return default_value

# 使用新的配置获取函数
API_KEY = get_config('API_KEY')
API_BASE = get_config('API_BASE', "https://api.openai.com")
MODEL = get_config('MODEL', "gpt-4o-mini")
PROXY_URL = get_config('PROXY_URL')

def file_cache(cache_dir="./data_cache", expire_days=1):
    """
    文件缓存装饰器，将数据存储到本地文件系统
    
    参数：
        cache_dir: 缓存目录路径
        expire_days: 缓存过期天数，默认1天
    """
    def decorator(func):
        def wrapper(*args, **kwargs):
            # 创建缓存目录
            os.makedirs(cache_dir, exist_ok=True)
            
            # 构建缓存文件路径，使用函数名和参数作为缓存键
            cache_key = f"{func.__name__}_{str(args)}_{str(kwargs)}"
            cache_file = os.path.join(cache_dir, f"{cache_key}.json")
            meta_file = os.path.join(cache_dir, f"{cache_key}_meta.json")
            
            # 检查缓存是否存在且未过期
            if os.path.exists(cache_file) and os.path.exists(meta_file):
                with open(meta_file, 'r') as f:
                    meta = json.load(f)
                cache_time = datetime.strptime(meta['timestamp'], 
                                             '%Y-%m-%d %H:%M:%S')
                
                # 如果缓存未过期，直接从文件加载数据
                if datetime.now() - cache_time < timedelta(days=expire_days):
                    try:
                        with open(cache_file, 'r') as f:
                            return json.load(f)
                    except Exception as e:
                        print(f"读取缓存文件出错: {str(e)}")
            
            # 如果缓存不存在或已过期，重新获取数据
            results = func(*args, **kwargs)
            
            # 保存数据到缓存文件
            try:
                # 保存数据
                with open(cache_file, 'w') as f:
                    json.dump(results, f, ensure_ascii=False, indent=2)
                
                # 保存元数据
                meta = {
                    'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
                    'function': func.__name__,
                    'args': str(args),
                    'kwargs': str(kwargs)
                }
                with open(meta_file, 'w') as f:
                    json.dump(meta, f, ensure_ascii=False, indent=2)
                    
            except Exception as e:
                print(f"写入缓存文件出错: {str(e)}")
            
            return results
        return wrapper
    return decorator

@file_cache(cache_dir="./securities_cache", expire_days=30)
def load_security_data(security_type: str) -> pd.DataFrame:
    """
    加载证券数据，支持本地文件缓存
    
    参数：
        security_type: 证券类型 ('index', 'stock', 'etf')
    
    返回：
        pd.DataFrame: 包含证券信息的数据框
    """
    try:
        if security_type == 'index':
            return adata.stock.info.all_index_code()
        elif security_type == 'stock':
            return adata.stock.info.all_code()
        elif security_type == 'etf':
            return adata.fund.info.all_etf_exchange_traded_info()
        else:
            return pd.DataFrame()
    except Exception as e:
        print(f"加载{security_type}数据时出错: {str(e)}")
        return pd.DataFrame()

def preprocess_query(query: str) -> str:
    """预处理搜索关键词"""
    query = re.sub(r'[^\w\s]', '', query)
    query = query.lower()
    query = ' '.join(query.split())
    return query

@file_cache(cache_dir="./search_cache", expire_days=30)
def search_single_type(query: str, security_type: str) -> List[Dict]:
    """
    在单个证券类型中搜索
    
    Args:
        query: 搜索关键词
        security_type: 证券类型
    
    Returns:
        List[Dict]: 搜索结果列表
    """
    results = []
    df = load_security_data(security_type)
    
    if df.empty:
        return results
        
    try:
        # 根据证券类型确定代码和名称列
        code_col = {
            'index': 'index_code',
            'stock': 'stock_code',
            'etf': 'fund_code'
        }.get(security_type)
        
        name_col = 'name' if security_type == 'index' else 'short_name'
        
        # 创建搜索条件
        code_match = df[code_col].str.contains(query, case=False, na=False)
        name_match = df[name_col].str.contains(query, case=False, na=False)
        
        # 应用搜索条件
        matched_df = df[code_match | name_match]
        
        # 提取结果
        for _, row in matched_df.iterrows():
            results.append({
                'code': row[code_col],
                'name': row[name_col],
                'type': security_type,
                'exchange': row.get('exchange', '')
            })
            
    except Exception as e:
        print(f"搜索{security_type}时出错: {str(e)}")
    
    return results

@lru_cache(maxsize=2056)
def search_securities(query: str) -> List[Dict]:
    """
    搜索证券(指数、股票)
    
    技术特点:
    1. 使用 LRU 缓存优化数据加载
    2. 多线程并行搜索提升性能
    3. 关键词预处理提高匹配准确性
    4. 异常处理确保功能稳定性
    5. 类型注解增强代码可读性
    
    Args:
        query: 搜索关键词(代码或名称)
    
    Returns:
        List[Dict]: 搜索结果列表，每个结果包含:
            - code: 证券代码
            - name: 证券名称
            - type: 证券类型
            - exchange: 交易所
    """
    if not query or len(query.strip()) == 0:
        return []
        
    # 预处理查询关键词
    query = preprocess_query(query)
    
    # 使用线程池并行搜索不同类型的证券
    security_types = ['index', 'stock', 'etf']
    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = [
            executor.submit(search_single_type, query, security_type)
            for security_type in security_types
        ]
        
        # 收集所有结果
        all_results = []
        for future in futures:
            try:
                results = future.result()
                all_results.extend(results)
            except Exception as e:
                print(f"获取搜索结果时出错: {str(e)}")
                
    # 按相关度排序结果
    all_results.sort(key=lambda x: (
        # 完全匹配代码的优先级最高
        -int(x['code'].lower() == query),
        # 其次是包含代码的
        -int(query in x['code'].lower()),
        # 再次是包含名称的
        -int(query in x['name'].lower()),
        # 最后按代码长度排序
        len(x['code'])
    ))
    
    return all_results

def get_market_data(code, security_type, days=365*3):
    """获取市场数据"""
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
        numeric_columns = ['open', 'high', 'low', 'close']
        for col in numeric_columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
        df = df.dropna(subset=numeric_columns)
        return df
    
    return None

def normalize_window(window):
    """标准化价格序列"""
    numeric_window = pd.to_numeric(window, errors='coerce')
    if numeric_window.isna().any():
        return None
    return (numeric_window - numeric_window.iloc[0]) / numeric_window.iloc[0] * 100

def calculate_similarity(window1, window2):
    """计算两个窗口之间的相似度"""
    if len(window1) != len(window2):
        return 0
    
    norm1 = normalize_window(window1)
    norm2 = normalize_window(window2)
    
    if norm1 is None or norm2 is None:
        return 0
    
    try:
        corr, _ = pearsonr(norm1, norm2)
        dist = euclidean(norm1, norm2)
        normalized_dist = 1 / (1 + dist/len(window1))
        similarity = (corr + 1)/2 * 0.7 + normalized_dist * 0.3
        return similarity
    except:
        return 0

def find_similar_patterns(df, window_size=30, top_n=3):
    """
    查找最相似的历史模式
    
    技术点：
    - K线模式识别：通过滑动窗口方法识别历史K线形态
    - 欧几里得距离：计算K线序列间的距离相似度
    - 皮尔逊相关系数：计算走势的相关性
    - 时间序列标准化：将不同时期的价格序列标准化以便比较
    """
    if df is None or len(df) < window_size * 2:
        return []
    
    recent_window = df.tail(window_size)['close'].values
    recent_window_norm = (recent_window - recent_window[0]) / recent_window[0] * 100
    
    similar_patterns = []
    max_i = len(df) - (window_size * 2 + 7)
    
    # 批量计算所有窗口的标准化数据
    all_windows = np.array([
        df['close'].values[i:i+window_size]
        for i in range(max_i)
    ])
    all_windows_norm = np.array([
        (window - window[0]) / window[0] * 100
        for window in all_windows
    ])
    
    # 批量计算相似度
    correlations = np.array([
        pearsonr(recent_window_norm, window_norm)[0]
        for window_norm in all_windows_norm
    ])
    
    # 批量计算欧氏距离
    distances = np.array([
        euclidean(recent_window_norm, window_norm)
        for window_norm in all_windows_norm
    ])
    normalized_distances = 1 / (1 + distances/window_size)
    
    # 计算综合相似度
    similarities = correlations * 0.7 + normalized_distances * 0.3
    
    # 获取最相似的模式
    top_indices = np.argsort(similarities)[-top_n:][::-1]
    
    for i in top_indices:
        similar_patterns.append({
            'start_date': df.iloc[i]['trade_date'],
            'end_date': df.iloc[i+window_size-1]['trade_date'],
            'pattern_data': df.iloc[i:i+window_size].copy(),
            'future_data': df.iloc[i+window_size:i+window_size+7].copy(),
            'similarity': similarities[i]
        })
    
    return similar_patterns

def analyze_future_trends(similar_patterns):
    """
    分析历史K线后续走势
    
    技术点：
    - 历史模式匹配：基于相似K线模式的历史表现
    - 统计概率分析：计算涨跌概率和幅度分布
    - 趋势预测建模：构建未来可能的走势预测
    """
    if not similar_patterns:
        return None
        
    stats = {
        'up': {str(i): {'count': 0, 'max': 0, 'min': float('inf'), 'mean': 0, 'values': []} 
               for i in range(1, 8)},
        'down': {str(i): {'count': 0, 'max': 0, 'min': float('inf'), 'mean': 0, 'values': []} 
                 for i in range(1, 8)}
    }
    
    for pattern in similar_patterns:
        future_data = pattern['future_data']
        
        for i in range(len(future_data)):
            day = str(i + 1)
            current_price = future_data.iloc[i]['close']
            prev_price = pattern['pattern_data'].iloc[-1]['close'] if i == 0 else future_data.iloc[i-1]['close']
            
            change_rate = ((current_price - prev_price) / prev_price) * 100
            
            category = 'up' if change_rate >= 0 else 'down'
            change_rate = abs(change_rate)
            
            stats[category][day]['count'] += 1
            stats[category][day]['values'].append(change_rate)
            stats[category][day]['max'] = max(stats[category][day]['max'], change_rate)
            stats[category][day]['min'] = min(stats[category][day]['min'], change_rate)
    
    total_patterns = len(similar_patterns)
    for category in ['up', 'down']:
        for day in stats[category]:
            day_stats = stats[category][day]
            if day_stats['count'] > 0:
                day_stats['probability'] = day_stats['count'] / total_patterns
                day_stats['mean'] = sum(day_stats['values']) / day_stats['count']
                rounded_values = [round(x, 2) for x in day_stats['values']]
                value_counts = {}
                for v in rounded_values:
                    value_counts[v] = value_counts.get(v, 0) + 1
                day_stats['mode'] = max(value_counts.items(), key=lambda x: x[1])[0]
                
                if day_stats['min'] == float('inf'):
                    day_stats['min'] = 0
                
            del day_stats['values']
    
    return stats

def analyze_holding_returns(similar_patterns):
    """
    分析不同持有期的收益情况
    
    技术点：
    - 持仓期收益率计算：计算不同持有期的收益表现
    - 风险度量(标准差/波动率)：评估投资风险水平
    - 胜率统计：分析不同持有期的盈利概率
    """
    if not similar_patterns:
        return None
        
    stats = {str(i): {
        'returns': [],
        'max_prices': [],
        'min_prices': [],
        'win_count': 0,
        'loss_count': 0,
    } for i in range(1, 8)}
    
    for pattern in similar_patterns:
        entry_price = pattern['pattern_data'].iloc[-1]['close']
        future_data = pattern['future_data']
        
        for days in range(1, 8):
            day_key = str(days)
            if days <= len(future_data):
                holding_period_data = future_data.iloc[:days]
                exit_price = holding_period_data.iloc[-1]['close']
                returns = (exit_price - entry_price) / entry_price * 100
                
                stats[day_key]['returns'].append(returns)
                
                if returns > 0:
                    stats[day_key]['win_count'] += 1
                else:
                    stats[day_key]['loss_count'] += 1
                
                stats[day_key]['max_prices'].append(holding_period_data['high'].max())
                stats[day_key]['min_prices'].append(holding_period_data['low'].min())
    
    analysis_results = {}
    for days, day_stats in stats.items():
        returns_array = np.array(day_stats['returns'])
        total_trades = len(returns_array)
        
        if total_trades > 0:
            analysis_results[days] = {
                'avg_return': np.mean(returns_array),
                'max_return': np.max(returns_array),
                'min_return': np.min(returns_array),
                'std_return': np.std(returns_array),
                'win_rate': day_stats['win_count'] / total_trades,
                'trade_count': total_trades,
                'max_price_change': (np.max(day_stats['max_prices']) - entry_price) / entry_price * 100,
                'min_price_change': (np.min(day_stats['min_prices']) - entry_price) / entry_price * 100
            }
    
    return analysis_results

class ChineseTextVectorizer:
    """中文文本向量化器"""
    
    def __init__(self, vector_size=100):
        self.vector_size = vector_size
        self.tfidf = TfidfVectorizer(
            tokenizer=self._parallel_tokenize,  # 使用并行分词
            token_pattern=None,
            max_features=2000  # 减少特征数量
        )
        self.svd = TruncatedSVD(
            n_components=vector_size,
            random_state=42,
            algorithm='randomized'  # 使用随机算法加速
        )
        self.is_fitted = False

         # 预加载结巴词典
        jieba.initialize()
    
    @lru_cache(maxsize=1000)  # 缓存分词结果
    def _tokenize(self, text):
        text = re.sub(r'[^\w\s]', '', text)
        words = jieba.lcut(text)
        return [w for w in words if w.strip()]
    
    def fit(self, texts):
        tfidf_matrix = self.tfidf.fit_transform(texts)
        self.svd.fit(tfidf_matrix)
        self.is_fitted = True
    
    def transform(self, text):
        tfidf_vector = self.tfidf.transform([text])
        vector = self.svd.transform(tfidf_vector)
        norm = np.linalg.norm(vector)
        if norm > 0:
            vector = vector / norm
        return vector.flatten()
    
    def _parallel_tokenize(self, text):
        # 对较长的文本进行分块
        if len(text) < 1000:  # 短文本直接处理
            return self._tokenize(text)
            
        # 长文本并行处理
        chunks = self._split_text(text)
        with ThreadPoolExecutor(max_workers=4) as executor:
            results = list(executor.map(self._tokenize, chunks))
        return [token for chunk_result in results for token in chunk_result]

def call_llm_api(prompt):
    """调用 LLM API"""
    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    
    data = {
        "model": MODEL,
        "messages": [
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.7
    }
    
    try:
        response = requests.post(
            f"{API_BASE}/v1/chat/completions",
            headers=headers,
            json=data,
            timeout=30
        )
        response.raise_for_status()
        result = response.json()
        return result["choices"][0]["message"]["content"]
    except Exception as e:
        st.error(f"API 调用出错: {str(e)}")
        return None

def compute_embedding(text, vectorizer=None):
    """计算文本的向量表示"""
    if vectorizer is None:
        vectorizer = ChineseTextVectorizer()
        
    try:
        if not text.strip():
            return np.zeros(vectorizer.vector_size)
            
        if not vectorizer.is_fitted:
            vectorizer.fit([text])
            
        embedding = vectorizer.transform(text)
        return embedding
        
    except Exception as e:
        print(f"计算文本向量时发生错误: {str(e)}")
        return np.zeros(vectorizer.vector_size)

def compute_similarity(query_embedding, chunk_embedding):
    """计算两个向量之间的余弦相似度"""
    if query_embedding.shape != chunk_embedding.shape:
        raise ValueError("向量维度不匹配")
        
    similarity = np.dot(query_embedding, chunk_embedding)
    return similarity

def retrieve_relevant_chunks(query, chunks, top_k=3):
    """检索与查询最相关的文本块"""
    vectorizer = ChineseTextVectorizer(vector_size=50)  # 减少向量维度
    
    # 预处理所有文本
    all_texts = [query] + [chunk_text for _, chunk_text in chunks]
    vectorizer.fit(all_texts)
    
    # 批量计算向量
    query_embedding = compute_embedding(query, vectorizer)
    chunk_embeddings = np.vstack([
        compute_embedding(chunk_text, vectorizer)
        for _, chunk_text in chunks
    ])
    
    # 批量计算相似度
    similarities = np.dot(chunk_embeddings, query_embedding)
    
    # 获取top_k个最相似的chunk
    top_indices = np.argsort(similarities)[-top_k:][::-1]
    return [chunks[i] for i in top_indices]

def create_text_chunks(security, current_df, similar_patterns, holding_stats):
    """
    将市场数据分成多个独立的文本块，便于后续检索

    技术点：
    - RAG检索增强生成：构建结构化的文本块供检索
    - 向量化(TF-IDF + SVD)：将文本转化为向量表示
    - 余弦相似度匹配：计算查询与文本块的相似度
    
    这个函数将各类市场数据转换成结构化的文本块，包括：
    1. 证券基本信息（代码、名称等）
    2. 最新行情数据
    3. 近期价格走势
    4. 历史表现分析
    5. 相似K线分析
    6. 持仓收益分析
    """
    chunks = []
    
    # 构建基本信息文本块
    basic_info = f"""
            证券基本信息：
            名称：{security['name']}
            代码：{security['code']}
            类型：{security['type']}
            交易所：{security['exchange'] if security['exchange'] else '未知'}
    """
    chunks.append(("basic_info", basic_info))
    
    if current_df is not None and not current_df.empty:
        # 添加最新行情信息
        latest_data = current_df.iloc[-1]
        latest_market = f"""
            最新市场行情（{latest_data['trade_date'].strftime('%Y-%m-%d')}）：
            收盘价：{latest_data['close']:.2f}
            开盘价：{latest_data['open']:.2f}
            最高价：{latest_data['high']:.2f}
            最低价：{latest_data['low']:.2f}
            成交量：{latest_data.get('volume', '未知')}
        """
        chunks.append(("latest_market", latest_market))
        
        # 添加近期走势分析（如果有足够数据）
        if len(current_df) >= 21:
            recent_trend = f"""
            近期价格走势：
            5日涨跌幅：{((latest_data['close'] - current_df.iloc[-6]['close'])/current_df.iloc[-6]['close']*100):.2f}%
            10日涨跌幅：{((latest_data['close'] - current_df.iloc[-11]['close'])/current_df.iloc[-11]['close']*100):.2f}%
            20日涨跌幅：{((latest_data['close'] - current_df.iloc[-21]['close'])/current_df.iloc[-21]['close']*100):.2f}%
            """
            chunks.append(("recent_trend", recent_trend))
        
        # 添加历史表现分析（最近3年）
        three_year_data = current_df.tail(252 * 3).copy()
        numeric_columns = ['open', 'high', 'low', 'close', 'volume']
        for col in numeric_columns:
            if col in three_year_data.columns:
                three_year_data[col] = pd.to_numeric(three_year_data[col], errors='coerce')
        
        if not three_year_data.empty:
            # 按年度统计市场表现
            yearly_data = {}
            for year in three_year_data['trade_date'].dt.year.unique():
                year_df = three_year_data[three_year_data['trade_date'].dt.year == year]
                if not year_df.empty:
                    yearly_data[year] = {
                        'start_price': year_df.iloc[0]['close'],
                        'end_price': year_df.iloc[-1]['close'],
                        'highest': year_df['high'].max(),
                        'lowest': year_df['low'].min(),
                        'vol_mean': pd.to_numeric(year_df.get('volume', pd.Series()), errors='coerce').mean()
                    }
            
            # 构建年度分析文本
            historical_trend = """
            近年市场表现："""
            for year, data in sorted(yearly_data.items(), reverse=True):
                yearly_return = ((data['end_price'] - data['start_price']) / data['start_price'] * 100)
                yearly_volatility = ((data['highest'] - data['lowest']) / data['lowest'] * 100)
                
                historical_trend += f"""
                {year}年度表现：
                年度涨跌幅：{yearly_return:.2f}%
                最高价：{data['highest']:.2f}
                最低价：{data['lowest']:.2f}
                波动率：{yearly_volatility:.2f}%"""
                if pd.notnull(data['vol_mean']):
                    historical_trend += f"""平均成交量：{data['vol_mean']:.0f}"""
            
            # 添加3年整体统计
            total_return = ((latest_data['close'] - three_year_data.iloc[0]['close']) 
                          / three_year_data.iloc[0]['close'] * 100)
            max_price = three_year_data['high'].max()
            min_price = three_year_data['low'].min()
            total_volatility = ((max_price - min_price) / min_price * 100)
            
            historical_trend += f"""
            近年整体统计：
            累计涨跌幅：{total_return:.2f}%
            历史最高价：{max_price:.2f}
            历史最低价：{min_price:.2f}
            价格波动率：{total_volatility:.2f}%
            """
            
            chunks.append(("historical_trend", historical_trend))
    
    # 添加相似K线分析结果
    if similar_patterns:
        for i, pattern in enumerate(similar_patterns[:3], 1):
            future_trend = ""
            for j, row in pattern['future_data'].iterrows():
                future_trend += f"""
            第{j+1}日:{row["close"]:.2f}"""
            
            pattern_info = f"""
            历史相似K线分析（TOP {i}）：
            相似度：{pattern['similarity']:.2%}
            历史时间段：{pattern['start_date'].strftime('%Y-%m-%d')} 至 {pattern['end_date'].strftime('%Y-%m-%d')}
            后续7日实际走势：{future_trend}
                """
            chunks.append((f"similar_pattern_{i}", pattern_info))
    
    # 添加持仓分析数据
    if holding_stats:
        for days, stats in holding_stats.items():
            holding_info = f"""            持仓{days}天分析：
            平均收益：{stats['avg_return']:.2f}%
            胜率：{stats['win_rate']*100:.1f}%
            最大上涨：{stats['max_return']:.2f}%
            最大下跌：{stats['min_return']:.2f}%
            波动率：{stats['std_return']:.2f}%"""
            chunks.append((f"holding_days_{days}", holding_info))
    
    return chunks

def get_analysis_prompt(query, relevant_chunks, chat_history=None):
    """
    构建带有检索上下文和对话历史的分析提示
    
    Args:
        query: 当前用户问题
        relevant_chunks: 相关的市场数据块
        chat_history: 之前的对话历史记录，格式为 [(user_msg, assistant_msg), ...]
    """
    context = "\n".join([chunk for _, chunk in relevant_chunks])
    
    # 构建提示的基础部分
    prompt = f"""作为一位专业的金融分析师，请基于以下相关市场数据回答用户问题。

要求：
1. 只使用提供的数据进行分析，不要添加其他市场信息
2. 明确区分数据支持的结论和不确定的推测
3. 如果数据不足以回答问题，请明确指出
4. 适当提醒投资风险

相关市场数据：
{context}
"""

    # 如果存在对话历史，添加到提示中
    if chat_history and len(chat_history) > 0:
        prompt += "\n对话历史：\n"
        for i, (user_msg, assistant_msg) in enumerate(chat_history, 1):
            prompt += f"第{i}轮问答：\n"
            prompt += f"用户：{user_msg}\n"
            prompt += f"助手：{assistant_msg}\n"
    
    # 添加当前问题
    prompt += f"\n当前用户问题：{query}"
    
    # 添加角色指示
    prompt += """

请基于以上信息和对话历史，遵循以下原则回答：
1. 用专业且通俗的语言回答问题，确保分析逻辑清晰
2. 如果涉及到之前的对话内容，请保持分析的连贯性
3. 在回答中适当提供一些思考的切入点，引导用户进行更深入的提问
4. 如果用户追问某个观点，请进一步展开解释背后的原理和依据
5. 如果某个分析涉及到多个方面，可以明确指出，方便用户选择感兴趣的方向继续探讨
"""
    
    return prompt

def display_market_analysis(current_df, similar_patterns, future_dates=None):
    """
    市场分析主函数，集成K线展示、技术分析和智能问答功能
    """
    if current_df is None or current_df.empty:
        st.warning("无法获取市场数据")
        return None
        
    # 显示当前K线图
    with st.container():
        st.markdown("### 当前K线图")
        if not current_df.empty:
            fig = plot_kline(current_df, "", future_dates=future_dates)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.write("无法生成K线图")
    
    # 分析并显示相似K线
    if similar_patterns: 
        st.markdown("### 历史相似K线图（展示相似度最高的前3条）")
        
        # 显示每个相似模式
        for i, pattern in enumerate(similar_patterns[:3], 1):
            with st.container():
                st.markdown(f"#### TOP {i} 相似度: {pattern['similarity']:.2%}")
                st.markdown(f"历史时间段: {pattern['start_date'].strftime('%Y-%m-%d')} - {pattern['end_date'].strftime('%Y-%m-%d')}")
                
                fig_similar = plot_kline(
                    pattern['pattern_data'],
                    "",
                    future_df=pattern['future_data'],
                    show_future_data=True
                )
                st.plotly_chart(fig_similar, use_container_width=True)
        
        # 显示趋势分析
        display_trend_analysis(similar_patterns)

        # 计算和显示持仓收益分析
        holding_stats = analyze_holding_returns(similar_patterns)
        display_holding_period_analysis(holding_stats)
        
        return holding_stats
    
    return None

def display_rag_qa(security, current_df, similar_patterns, holding_stats):
    """
    显示支持多轮对话的智能问答界面
    """
    st.markdown("""
### 智能问答助手
<div style='background-color: #eef6ff; padding: 12px; border-radius: 8px; margin-bottom: 20px; font-size: 0.9em;'>
    📝 <strong>使用指南</strong>
    <ul style='margin-top: 8px; margin-bottom: 8px;'>
        <li>这是一个支持多轮对话的智能助手，您可以围绕一个话题深入交流</li>
        <li>助手会记住对话内容，您可以基于之前的回答继续提问</li>
        <li>随时可以要求助手解释某个观点，或者提供更详细的分析</li>
        <li>如果分析不够清晰，请告诉助手“具体说明一下吗？”/li>
    </ul>
</div>
""", unsafe_allow_html=True)

    # 创建文本块用于检索
    chunks = create_text_chunks(security, current_df, similar_patterns, holding_stats)
    base_key = f"{security['code']}_{security['type']}"
    
    # 初始化会话状态
    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = []
    
    # 显示历史对话
    for i, (user_msg, assistant_msg) in enumerate(st.session_state.chat_history):
        with st.container():
            # 用户消息
            st.markdown(f"""
            <div style='background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                <span style='color: #666;'>👤 您：</span><br>
                {user_msg}
            </div>
            """, unsafe_allow_html=True)
            
            # 助手回复
            st.markdown(f"""
            <div style='background-color: #e8f4f9; padding: 10px; border-radius: 8px; margin-bottom: 20px;'>
                <span style='color: #666;'>🤖 助手：</span><br>
                {assistant_msg}
            </div>
            """, unsafe_allow_html=True)
    
    # 用户输入新问题
    user_question = st.text_input(
        f"请输入您关于 {security['name']}（{security['code']}）的问题：", 
        key=f'rag_question_{base_key}',
        help="您可以：\n1. 询问基本信息、最新行情、历史表现、相似K线分析等内容\n2. 基于助手的回答继续追问，比如'为什么会这样预测？'\n3. 寻求更详细的解释，如'能具体解释一下这个原因吗？'\n4. 要求展开某个观点，如'刚才说到xxx，能详细分析一下吗？'"
    )
    
    if user_question:
        with st.spinner('正在根据相关数据块思考答案...'):
            # 检索相关文本块
            relevant_chunks = retrieve_relevant_chunks(user_question, chunks, top_k=10)
            
            # 显示检索到的相关文本块
            chunks_content = ""
            for _, chunk_text in relevant_chunks:
                chunks_content += f'<p>{chunk_text}</p>'

            st.markdown(f'<details class="details" style="margin-top: -6px;">'
                    f'<summary class="details-summary">🔍 相关数据块（基于 RAG 结果）</summary>'
                    f'<div class="details-body details-body-related-chunks">'
                    f'{chunks_content}'
                    f'</div>'
                    f'</details>', unsafe_allow_html=True)
            
            # 构建prompt并调用API，加入对话历史
            prompt = get_analysis_prompt(
                user_question, 
                relevant_chunks, 
                st.session_state.chat_history
            )
            response = call_llm_api(prompt)
            
            if response:
                # 保存新的对话记录
                st.session_state.chat_history.append((user_question, response))
                
                # 显示最新的回复
                st.markdown("""
                <div style='background-color: #f0f2f6; padding: 10px; border-radius: 8px; margin-bottom: 10px;'>
                    <span style='color: #666;'>👤 您：</span><br>
                    {user_question}
                </div>
                """.format(user_question=user_question), unsafe_allow_html=True)
                
                st.markdown("""
                <div style='background-color: #e8f4f9; padding: 10px; border-radius: 8px; margin-bottom: 20px;'>
                    <span style='color: #666;'>🤖 助手：</span><br>
                    {response}
                </div>
                """.format(response=response), unsafe_allow_html=True)
                
                # 添加免责声明
                st.markdown("""
                <div class="statement">
                ⚠️ <strong>免责声明：</strong>以上分析仅供参考，不构成投资建议。投资有风险，入市需谨慎。
                </div>
                """, unsafe_allow_html=True)
                
                # 清空输入框
                st.text_input(
                    f"请输入您关于 {security['name']}（{security['code']}）的问题或继续提问：",
                    value="",
                    key=f'rag_question_{base_key}_new',
                    help="您可以询问关于该证券的基本信息、最新行情、历史表现、相似K线分析等问题。您也可以基于之前的对话继续提问。"
                )
            else:
                st.error("抱歉，获取回答失败，请稍后重试。")

def main():
    """
    主函数：实现金融数据分析系统的整体功能流程
    """

    if PROXY_URL: adata.proxy(is_proxy=True, proxy_url=PROXY_URL)

    st.title('A 股数据智能分析系统')

    # 显示标题和作者信息
    st.markdown("""
    <div style='font-size: 0.9em; margin-bottom: 2em;'>
        <h3 style='color: #1f449c; font-size: 1.2em;'>
            📚 课程：商业智能技术 
            <small style='color: #666;'>/ 指导教师：阮光册教授</small>
        </h3>
        <p style='color: #666; font-size: 0.9em; margin-top: 0.5em;'>
            👨‍🎓 学生：吴小宇 
            <span style='margin-left: 1em;'>🔢 学号：71265700016</span>
            <span style='margin-left: 1em;'>⭐️ <a href='https://github.com/mantoufan/yzhanSimilarKline' target='_blank'>项目源码和说明文档</a></span>
        </p>
    </div>
    """, unsafe_allow_html=True)

    # 显示系统说明
    st.markdown("""
    <div style='background-color: #f8f9fa; padding: 1.2em; border-radius: 8px; 
        border-left: 4px solid #4c7ef3; margin-bottom: 2em; font-size: 0.9em;'>
        <h4 style='color: #1f449c; font-size: 1.1em; margin-bottom: 0.8em;'>
            💡 系统功能介绍
        </h4>
        <p style='color: #444; line-height: 1.6; margin-bottom: 0.8em;'>
            本系统结合了技术分析和智能问答功能：
        </p>
        <ul style='list-style-type: none; padding-left: 0.5em; margin: 0;'>
            <li style='margin-bottom: 0.5em;'>
                📊 自动识别相似K线形态 (K线模式识别 + 欧几里得距离 + 皮尔逊相关系数 + 时间序列标准化)
            </li>
            <li style='margin-bottom: 0.5em;'>
                📈 预测可能的价格走势 (历史模式匹配 + 统计概率分析 + 趋势预测建模)
            </li>
            <li style='margin-bottom: 0.5em;'>
                ⚖️ 分析不同持仓期的风险收益 (持仓期收益率计算 + 风险度量 + 胜率统计)
            </li>
            <li style='margin-bottom: 0.5em;'>
                🤖 提供智能问答服务 (RAG检索增强生成 + 向量化 + LLM问答 + 余弦相似度匹配)
            </li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    # 搜索证券
    query = st.text_input('输入指数/股票/基金 ETF 的代码或名称进行搜索', '', key='security_search')
    
    if query:
        results = search_securities(query)
        
        if results:
            st.write(f'找到 {len(results)} 个结果:')
            
            # 显示每个搜索结果
            for result in results:
                with st.expander(f"{result['name']} ({result['code']}) - {result['type'].upper()}", expanded=False):
                    # 显示证券基本信息
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.write(f"📇 代码: {result['code']}")
                    with col2:
                        st.write(f"📝 名称: {result['name']}")
                    with col3:
                        st.write(f"🏢 类型: {result['type'].upper()}")
                    
                    if result['exchange']:
                        st.write(f"🏛️ 交易所: {result['exchange']}")
                    
                    # 获取市场数据
                    market_data = get_market_data(result['code'], result['type'])
                    
                    if market_data is not None:
                        # 获取最近一个月数据用于分析
                        current_month_data = market_data.tail(30).copy()

                        # 生成未来7个交易日的日期
                        last_date = current_month_data['trade_date'].iloc[-1]
                        future_dates = pd.date_range(
                            start=last_date + pd.Timedelta(days=1),
                            periods=7,
                            freq='B'  # 'B' 表示工作日
                        )
                        
                        # 寻找相似K线形态
                        similar_patterns = find_similar_patterns(market_data, window_size=30, top_n=10)
                        
                        # 显示市场分析
                        holding_stats = display_market_analysis(current_month_data, similar_patterns, future_dates)
  
                        # 显示智能问答界面
                        if holding_stats is not None:
                            display_rag_qa(result, market_data.tail(252 * 10).copy(), similar_patterns, holding_stats)
                    else:
                        st.warning("暂无K线数据")
        else:
            st.warning('未找到匹配的结果')

def plot_kline(df, title, future_df=None, future_dates=None, show_future_data=False):
    """
    创建K线图，支持显示历史走势和未来预测
    
    参数说明：
    - df: 当前K线数据
    - title: 图表标题
    - future_df: 未来实际数据（用于历史相似图）
    - future_dates: 未来日期（用于当前K线图）
    - show_future_data: 是否显示未来数据
    """
    if df is None or df.empty:
        return None
    
    fig = go.Figure()
    
    # 添加主K线
    fig.add_trace(go.Candlestick(
        x=df['trade_date'],
        open=df['open'],
        high=df['high'],
        low=df['low'],
        close=df['close'],
        name='K线'
    ))
    
    # 获取价格范围
    y_range = [df['low'].min(), df['high'].max()]
    
    if future_df is not None and not future_df.empty:
        y_range = [
            min(y_range[0], future_df['low'].min()),
            max(y_range[1], future_df['high'].max())
        ]
    
    # 添加分隔线
    last_date = df['trade_date'].max()
    fig.add_vline(
        x=last_date,
        line_width=1,
        line_dash="dash",
        line_color="gray",
        opacity=0.7
    )
    
    # 处理未来数据显示
    if future_df is not None and not future_df.empty and show_future_data:
        fig.add_trace(go.Candlestick(
            x=future_df['trade_date'],
            open=future_df['open'],
            high=future_df['high'],
            low=future_df['low'],
            close=future_df['close'],
            name='后续走势'
        ))
    elif future_dates is not None:
        fig.add_trace(go.Scatter(
            x=future_dates,
            y=[None] * len(future_dates),
            mode='lines',
            line=dict(color='rgba(0,0,0,0)'),
            showlegend=False
        ))
    
    # 更新布局
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
        xaxis_title='交易日期',
        yaxis_range=y_range
    )
    
    # 配置x轴
    fig.update_xaxes(
        rangeslider=dict(visible=False),
        type='date',
        rangebreaks=[dict(bounds=["sat", "mon"])]
    )
    
    return fig

def display_trend_analysis(similar_patterns):
    """
    显示趋势分析结果
    """
    stats = analyze_future_trends(similar_patterns)
    
    if not stats:
        st.write("未找到历史K线数据")
        return
    
    st.markdown(f"### 未来交易日涨跌预测（基于最相似 {len(similar_patterns)} 条历史 K 线）")
    
    # 创建网格展示数据
    html = create_trend_analysis_grid(stats)
    
    # 显示网格布局
    st.markdown(html, unsafe_allow_html=True)
    
    # 添加分析说明
    st.markdown("""
    <details class="details">
        <summary class="details-summary">
        📊 <strong>指标计算方法说明</strong></summary>
        <div class="details-body">
            <p><strong>涨跌概率：</strong>当天上涨/下跌的历史K线数量 ÷ 总K线数量</p>
            <p><strong>涨跌幅指标（相对于前一天收盘价）：</strong></p>
            <ul>
                <li>最高/最低涨跌幅：极值涨跌幅</li>
                <li>平均涨跌幅：所有样本的平均值</li>
                <li>最可能涨跌幅：出现频率最高的涨跌幅</li>
            </ul>
            <p><strong>注意：</strong>所有计算均基于历史相似K线的统计结果，仅供参考。</p>
        </div>
    </details>
    """, unsafe_allow_html=True)

def create_trend_analysis_grid(stats):
    """
    创建趋势分析网格的HTML布局
    """
    metrics_data = {
        'up': [
            ('涨概率', 'probability', '%.1f%%'),
            ('最高涨幅', 'max', '%.2f%%'),
            ('平均涨幅', 'mean', '%.2f%%'),
            ('最可能涨幅', 'mode', '%.2f%%'),
            ('最低涨幅', 'min', '%.2f%%')
        ],
        'down': [
            ('跌概率', 'probability', '%.1f%%'),
            ('最高跌幅', 'max', '%.2f%%'),
            ('平均跌幅', 'mean', '%.2f%%'),
            ('最可能跌幅', 'mode', '%.2f%%'),
            ('最低跌幅', 'min', '%.2f%%')
        ]
    }
    
    # 创建HTML布局
    html = """
    <style>
        .grid-container {
            display: grid;
            grid-template-columns: 100px repeat(7, 1fr);
            gap: 1px;
            background-color: #ddd;
            padding: 1px;
            font-size: 12px;
            font-family: Arial, sans-serif;
        }
        .grid-header {
            background-color: #f4f4f4;
            padding: 8px;
            text-align: center;
            font-weight: bold;
        }
        .grid-label {
            background-color: #f4f4f4;
            padding: 8px;
            text-align: right;
        }
        .grid-cell {
            background-color: white;
            padding: 8px;
            text-align: center;
        }
        .up-value { color: #ff4444; }
        .down-value { color: #00cc00; }
        .details {
            background-color: rgb(240, 242, 246);
            padding: 15px;
            border-radius: 5px;
            margin-top: 16px;
            margin-bottom: 16px;
            transition: background-color 0.2s;
        }
        .details:hover {
            background-color: rgb(230, 232, 236);
        }
        .details-summary {
            margin: 0;
            font-size: 0.9em;
            color: #666;
            cursor: pointer;
        }
        .details-body {
            background-color: #f9f9f9;
            padding: 10px;
            margin: 5px 0;
            border-radius: 4px;
            font-family: monospace;
        }

        .details-body-related-chunks {
            font-size: 12px;
            white-space: pre-wrap;
        }

        .details-body-related-chunks p {
            background: #eee;
            padding: 20px;
        }

        .statement {
            background-color: #fff3cd;
            padding: 10px;
            border-radius: 5px; 
            font-size: 0.8em;
            margin-top: 10px;
            margin-bottom: 10px;
        }
    </style>
    <div class="grid-container">
        <div class="grid-header"></div>
    """
    
    # 添加列标题
    for i in range(1, 8):
        html += f'<div class="grid-header">第{i}日</div>'
    
    # 处理数据行
    for direction in ['up', 'down']:
        for label, metric_key, format_str in metrics_data[direction]:
            html += f'<div class="grid-label">{label}</div>'
            
            for day in range(1, 8):
                day_stats = stats[direction][str(day)]
                if day_stats['count'] > 0:
                    value = day_stats[metric_key]
                    if metric_key == 'probability':
                        value = value * 100
                    cell_content = format_str % value
                    style_class = 'up-value' if direction == 'up' else 'down-value'
                    html += f'<div class="grid-cell"><span class="{style_class}">{cell_content}</span></div>'
                else:
                    html += '<div class="grid-cell">无数据</div>'
    
    html += "</div>"
    return html

def display_holding_period_analysis(stats, initial_money=10000):
    """
    显示不同持有期的收益分析结果
    """
    if not stats:
        st.warning("没有足够的历史数据进行分析")
        return
        
    sample_count = list(stats.values())[0]['trade_count']
    
    st.markdown(f"### 立即买入收益预测（基于最相似的 {sample_count} 条历史 K 线）")
    
    # 创建表格数据
    data = []
    for days, day_stats in stats.items():
        data.append({
            "持有期": f"{days}个交易日",
            "平均收益率": f"{day_stats['avg_return']:.2f}%",
            "胜率": f"{day_stats['win_rate']*100:.1f}%",
            "最大上涨": f"{day_stats['max_return']:.2f}%",
            "最大下跌": f"{day_stats['min_return']:.2f}%",
            "波动率": f"{day_stats['std_return']:.2f}%"
        })
    
    # 显示表格
    st.dataframe(
        data,
        hide_index=True,
        use_container_width=True,
        column_config={
            "持有天数": st.column_config.TextColumn("持有天数", help="从当前交易日收盘价买入的持有时间"),
            "平均收益率": st.column_config.TextColumn("平均收益率", help="历史相似情况下的平均收益率"),
            "胜率": st.column_config.TextColumn("胜率", help="盈利次数/总次数"),
            "最大上涨": st.column_config.TextColumn("最大上涨", help="历史相似情况下的最大上涨幅度"),
            "最大下跌": st.column_config.TextColumn("最大下跌", help="历史相似情况下的最大下跌幅度"),
            "波动率": st.column_config.TextColumn("波动率", help="收益率的标准差,反映风险大小")
        }
    )
    
    # 添加指标计算方法说明
    st.markdown("""
    <details class="details" style="margin-top: -12px;">
        <summary class="details-summary">
        📊 <strong>指标计算方法说明</strong></summary >
        <div class="details-body">
            <p><strong>收益率和风险指标计算公式：</strong></p>
            <ul>
                <li><strong>平均收益率</strong> = (各样本的收益率之和) ÷ 样本数量<br>
                    其中，单个样本收益率 = (卖出价格 - 买入价格) ÷ 买入价格 × 100%</li>
                <li><strong>胜率</strong> = 盈利次数 ÷ 总交易次数 × 100%<br>
                    当收益率 > 0 时，计为盈利；当收益率 ≤ 0 时，计为亏损</li>
                <li><strong>最大上涨</strong> = MAX((卖出价格 - 买入价格) ÷ 买入价格 × 100%)<br>
                    在所有样本中，选取收益率最高的记录</li>
                <li><strong>最大下跌</strong> = MIN((卖出价格 - 买入价格) ÷ 买入价格 × 100%)<br>
                    在所有样本中，选取收益率最低的记录</li>
                <li><strong>波动率</strong> = STDEV(所有样本收益率) = √(∑(收益率 - 平均收益率)² ÷ (样本数量-1))<br>
                    反映收益率的离散程度，波动率越大代表风险越高</li>
            </ul>
            <p><strong>重要提示：</strong></p>
            <ul>
                <li>买入价格为当前K线的收盘价</li>
                <li>卖出价格为持有期结束时的收盘价</li>
                <li>所有统计均基于历史相似K线数据，仅供参考</li>
                <li>过往表现不代表未来收益，投资需谨慎</li>
            </ul>
        </div>
    </details>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()