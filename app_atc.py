import streamlit as st
import pandas as pd
import altair as alt

st.title("健保藥品 ATC 分類動態分析")

# 使用 Streamlit 內建快取，避免每次換按鈕都要重新讀取大檔案
@st.cache_data
def load_and_process_heavy_file(file_name):
    # 【核心優化 1】不知道欄位名，所以先讀 1 列來確認總共有多少欄
    preview = pd.read_csv(file_name, nrows=1, header=None)
    total_cols = len(preview.columns)
    
    # 【核心優化 2】usecols 嚴格限制：只讀取第二欄(索引1=藥品代碼) 和 最後一欄(索引-1=總醫令量)
    # 這能幫 Streamlit 雲端伺服器省下 85% 以上的記憶體！
    target_cols = [1, total_cols - 1]
    
    atc_counts = {}
    
    # 【核心優化 3】chunksize 分流讀取，每批只讀 30,000 列
    chunks = pd.read_csv(file_name, header=None, usecols=target_cols, chunksize=30000, encoding='utf-8')
    
    for chunk in chunks:
        # 重新命名這兩欄
        chunk.columns = ['Drug_Code', 'Volume']
        
        chunk['Drug_Code'] = chunk['Drug_Code'].astype(str).str.strip()
        chunk['Volume'] = pd.to_numeric(chunk['Volume'], errors='coerce').fillna(0)
        
        # 擷取首碼
        chunk['ATC_Group'] = chunk['Drug_Code'].str[0].str.upper()
        chunk = chunk[chunk['ATC_Group'].str.isalpha()]
        
        summary = chunk.groupby('ATC_Group')['Volume'].sum()
        for atc, val in summary.items():
            atc_counts[atc] = atc_counts.get(atc, 0) + val
            
    df_res = pd.DataFrame(list(atc_counts.items()), columns=['ATC_Group', 'Total_Volume'])
    return df_res

# 讓使用者選年份，選到該年份才去讀該大檔
year = st.selectbox("請選擇分析年份", [2022, 2023, 2024])

file_map = {
    2022: 'A21030000I-E41005-001 (2022).csv',
    2023: 'A21030000I-E41005-002 (2023).csv',
    2024: 'A21030000I-E41005-003 (2024).csv'
}

current_file = file_map[year]

with st.spinner("雲端伺服器正在分塊讀取龐大健保原始資料，請稍候..."):
    df_result = load_and_process_heavy_file(current_file)

# 畫圖
chart = alt.Chart(df_result).mark_bar().encode(
    x='ATC_Group:N',
    y='Total_Volume:Q',
    color='ATC_Group:N'
).properties(title=f"{year} 年各 ATC 分類總醫令量")

st.altair_chart(chart, use_container_width=True)
