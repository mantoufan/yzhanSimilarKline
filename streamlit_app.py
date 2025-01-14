import streamlit as st

# 设置页面标题
st.title('Hello World')

# 显示文本
st.write('欢迎来到我的第一个 Streamlit 应用!')

# 添加一个简单的按钮
if st.button('点击我'):
    st.balloons()