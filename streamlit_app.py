import streamlit as st
import adata
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

def create_kline_chart(data):
    """
    创建K线图的函数
    
    参数:
        data: pandas DataFrame, 包含OHLC数据的DataFrame
        
    返回:
        plotly图表对象或None（如果创建失败）
    """
    try:
        # 检查数据是否为空
        if data.empty:
            st.error("没有获取到行情数据")
            return None
            
        # 检查必要的列是否存在
        required_columns = ['trade_date', 'open', 'close', 'high', 'low']
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            st.error(f"数据中缺少必要的列: {', '.join(missing_columns)}")
            return None
            
        # 创建数据的副本以避免修改原始数据
        plot_data = data.copy()
        
        # 确保日期列是datetime类型
        plot_data['trade_date'] = pd.to_datetime(plot_data['trade_date'])
        
        # 创建K线图
        fig = go.Figure(data=[go.Candlestick(
            x=plot_data['trade_date'],
            open=plot_data['open'],
            high=plot_data['high'],
            low=plot_data['low'],
            close=plot_data['close']
        )])
        
        # 更新布局
        fig.update_layout(
            title='K线图',
            yaxis_title='价格',
            xaxis_title='日期',
            template='plotly_dark',
            height=600,
            xaxis_rangeslider_visible=False  # 隐藏底部的范围滑块
        )
        
        return fig
        
    except Exception as e:
        st.error(f"创建K线图时出错: {str(e)}")
        return None

def create_constituent_cards(constituents):
    """
    创建成分股卡片视图的函数
    
    参数:
        constituents: pandas DataFrame, 包含成分股信息
    """
    # 将成分股列表分成多列显示
    cols = st.columns(4)
    for idx, constituent in enumerate(constituents.itertuples()):
        with cols[idx % 4]:
            st.container(border=True).markdown(
                f"""
                **{constituent.股票简称}**  
                {constituent.股票代码}
                """
            )

def load_stock_data():
    """加载股票数据"""
    return adata.stock.info.all_code()

def load_index_data():
    """加载指数数据"""
    return adata.stock.info.all_index_code()

def load_etf_data():
    """加载ETF数据"""
    return adata.fund.info.all_etf_exchange_traded_info()

def load_bond_data():
    """加载债券数据"""
    return adata.bond.info.all_convert_code()

def load_all_data():
    """
    加载所有类型的数据并返回字典
    返回一个字典，包含所有数据类型的DataFrame
    """
    try:
        data_dict = {
            "股票": load_stock_data(),
            "指数": load_index_data(),
            "ETF": load_etf_data(),
            "可转债": load_bond_data()
        }
        return data_dict
    except Exception as e:
        st.error(f"加载数据失败: {str(e)}")
        return None

def show_detail_page(code, type_name):
    """
    显示详情页面
    
    参数:
        code: str, 证券代码
        type_name: str, 证券类型（股票/指数/ETF/可转债）
    """
    # 使用三列布局，将返回按钮放在右侧
    col1, col2, col3 = st.columns([8, 1, 1])
    
    with col3:  # 使用最右侧的列放置返回按钮
        if st.button("← 返回", key="return_button", use_container_width=True):
            st.session_state.page = "main"
            st.rerun()
    
    with col1:
        st.title(f"{type_name}详情: {code}")
    
    # 根据不同的证券类型显示不同的详情内容
    if type_name == "股票":
        try:
            # 创建标签页
            tab1, tab2, tab3 = st.tabs(["行情数据", "基本信息", "概念板块"])
            
            with tab1:
                # K线图
                market_data = adata.stock.market.get_market(code)
                if not market_data.empty:
                    # 显示最新行情
                    latest_data = market_data.iloc[-1]
                    cols = st.columns(4)
                    cols[0].metric("最新价", f"¥{latest_data['close']:.2f}", 
                                 f"{(latest_data['close']/latest_data['pre_close']-1)*100:.2f}%")
                    cols[1].metric("成交量", f"{latest_data['volume']/10000:.0f}万手")
                    cols[2].metric("成交额", f"{latest_data['amount']/100000000:.2f}亿")
                    cols[3].metric("换手率", f"{latest_data['turnover_ratio']:.2f}%")
                    
                    # 显示K线图
                    fig = create_kline_chart(market_data)
                    if fig is not None:
                        st.plotly_chart(fig, use_container_width=True)
            
            with tab2:
                # 股本信息
                shares_info = adata.stock.info.get_stock_shares(code)
                if not shares_info.empty:
                    st.subheader("股本结构")
                    cols = st.columns(3)
                    for i, (key, value) in enumerate(shares_info.items()):
                        cols[i % 3].metric(key, f"{value:,.0f}")
                
                # 行业信息
                industry_info = adata.stock.info.get_industry_sw(code)
                if not industry_info.empty:
                    st.subheader("所属行业")
                    st.info(f"申万一级行业: {industry_info.get('申万一级', 'N/A')}  \n"
                           f"申万二级行业: {industry_info.get('申万二级', 'N/A')}")
            
            with tab3:
                # 概念信息
                concept_info = adata.stock.info.get_concept_ths(code)
                if not concept_info.empty:
                    st.subheader("概念板块")
                    for concept in concept_info['概念名称'].unique():
                        st.chip(concept)
                        
        except Exception as e:
            st.error(f"获取数据失败: {str(e)}")
            
    elif type_name == "指数":
        try:
            tab1, tab2 = st.tabs(["行情走势", "成分股"])
            
            with tab1:
                # 指数行情
                index_market = adata.stock.market.get_market_index(code)
                if not index_market.empty:
                    fig = create_kline_chart(index_market)
                    st.plotly_chart(fig, use_container_width=True)
                    
                    # 显示最新行情
                    latest_data = index_market.iloc[-1]
                    cols = st.columns(4)
                    cols[0].metric("最新点位", f"{latest_data['收盘']:.2f}", 
                                 f"{(latest_data['收盘']/latest_data['开盘']-1)*100:.2f}%")
                    cols[1].metric("成交量", f"{latest_data['成交量']/10000:.0f}万手")
                    cols[2].metric("成交额", f"{latest_data['成交额']/100000000:.2f}亿")
            
            with tab2:
                # 成分股列表
                constituent_info = adata.stock.info.index_constituent(code)
                if not constituent_info.empty:
                    st.subheader("成分股列表")
                    create_constituent_cards(constituent_info)
                    
        except Exception as e:
            st.error(f"获取数据失败: {str(e)}")
            
    elif type_name == "ETF":
        try:
            # ETF行情
            etf_market = adata.fund.market.get_market_etf(code)
            if not etf_market.empty:
                fig = create_kline_chart(etf_market)
                st.plotly_chart(fig, use_container_width=True)
                
                # 显示最新行情
                latest_data = etf_market.iloc[-1]
                cols = st.columns(4)
                cols[0].metric("最新价", f"¥{latest_data['收盘']:.3f}", 
                             f"{(latest_data['收盘']/latest_data['开盘']-1)*100:.2f}%")
                cols[1].metric("成交量", f"{latest_data['成交量']/10000:.0f}万手")
                cols[2].metric("成交额", f"{latest_data['成交额']/100000000:.2f}亿")
                
        except Exception as e:
            st.error(f"获取数据失败: {str(e)}")

def filter_dataframe(df, search_term, data_type):
    """
    根据搜索词筛选数据框
    
    参数:
        df: pandas DataFrame, 要搜索的数据
        search_term: str, 搜索关键词
        data_type: str, 数据类型
    返回:
        filtered_df: pandas DataFrame, 筛选后的数据
    """
    if not search_term:
        return df.head(50)  # 默认显示前50条
    
    if data_type == "指数":
        # 处理指数名称中的 *** 
        search_term_clean = search_term.replace('*', '')
        mask = df.apply(lambda x: x.astype(str).str.replace('*', '').str.contains(search_term_clean, case=False, na=False)).any(axis=1)
    else:
        mask = df.apply(lambda x: x.astype(str).str.contains(search_term, case=False, na=False)).any(axis=1)
    
    return df[mask]

def create_search_results(df, search_term, data_type):
    """创建搜索结果展示"""
    # 使用已过滤的数据框直接创建结果
    for idx, row in df.iterrows():
        code = row.iloc[0]  # 假设第一列是代码
        name = row.iloc[1] if len(row) > 1 else ""  # 假设第二列是名称
        
        # 创建可点击的容器
        container = st.container(border=True)
        container.markdown(f"**{name}** ({code})")
        
        # 如果点击了容器
        if container.button("查看详情", key=f"btn_{code}"):
            st.session_state.selected_code = code
            st.session_state.page = "detail"
            st.rerun()

def filter_dataframe(df, search_term, data_type):
    """
    根据搜索词筛选数据框
    
    参数:
        df: pandas DataFrame, 要搜索的数据
        search_term: str, 搜索关键词
        data_type: str, 数据类型
    返回:
        filtered_df: pandas DataFrame, 筛选后的数据
    """
    if not search_term:
        return df.head(50)  # 默认显示前50条
    
    # 将DataFrame的所有列转换为字符串类型进行搜索
    df_str = df.astype(str)
    
    if data_type == "指数":
        # 处理指数名称中的 *** 
        search_term_clean = search_term.replace('*', '')
        # 使用str.contains()进行搜索
        mask = df_str.apply(lambda x: x.str.replace('*', '').str.contains(search_term_clean, case=False, na=False)).any(axis=1)
    else:
        # 使用str.contains()进行搜索
        mask = df_str.apply(lambda x: x.str.contains(search_term, case=False, na=False)).any(axis=1)
    
    return df[mask]

def main():
    """主函数：设置页面配置并控制整体流程"""
    # 设置页面配置
    st.set_page_config(page_title="金融数据查询", layout="wide")
    
    # 初始化会话状态变量
    if "page" not in st.session_state:
        st.session_state.page = "main"
    
    if "selected_code" not in st.session_state:
        st.session_state.selected_code = None
        
    if "selected_type" not in st.session_state:
        st.session_state.selected_type = "指数"  # 默认选择指数
        
    if "last_search_time" not in st.session_state:
        st.session_state.last_search_time = None
        
    if "last_search_term" not in st.session_state:
        st.session_state.last_search_term = None
        
    if "search_delay" not in st.session_state:
        st.session_state.search_delay = 1.5  # 设置搜索延迟为1.5秒
    
    # 根据页面状态显示不同内容
    if st.session_state.page == "detail":
        show_detail_page(st.session_state.selected_code, st.session_state.selected_type)
    else:
        st.title("金融数据查询系统")
        
        # 创建两列布局
        col1, col2 = st.columns([1, 3])
        
        with col1:
            # 数据类型选择
            data_type = st.selectbox(
                "选择数据类型",
                ["股票", "指数", "ETF", "可转债"],
                index=1  # 默认选择指数（索引1）
            )
            st.session_state.selected_type = data_type
            
            # 搜索框
            search_term = st.text_input("搜索代码或名称", 
                                      placeholder="输入关键词搜索...",
                                      key="search_input")
            
            # 处理搜索逻辑
            current_time = datetime.now()
            
            # 检查是否有新的搜索词
            if search_term != st.session_state.last_search_term:
                st.session_state.last_search_time = current_time
                st.session_state.last_search_term = search_term
            
            # 如果距离上次输入超过1.5秒，执行搜索
            perform_search = (
                st.session_state.last_search_time is not None and
                (current_time - st.session_state.last_search_time).total_seconds() >= st.session_state.search_delay
            )
            
        with col2:
            try:
                # 根据选择加载相应的数据
                if data_type == "股票":
                    df = load_stock_data()
                elif data_type == "指数":
                    df = load_index_data()
                elif data_type == "ETF":
                    df = load_etf_data()
                else:  # 可转债
                    df = load_bond_data()
                
                # 根据搜索条件过滤数据
                if perform_search or not search_term:
                    filtered_df = filter_dataframe(df, search_term, data_type)
                    create_search_results(filtered_df, search_term, data_type)
                    
            except Exception as e:
                st.error(f"加载数据失败: {str(e)}")

if __name__ == "__main__":
    main()