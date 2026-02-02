import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 網頁設定 ---
st.set_page_config(page_title="健保藥品價量標註系統", layout="wide")

# --- 民國日期轉換函數 ---
def parse_roc_date(s):
    try:
        s = str(int(float(s)))
    except:
        return None
    if len(s) == 7:
        year, month, day = int(s[:3]) + 1911, int(s[3:5]), int(s[5:7])
    elif len(s) == 6:
        year, month, day = int(s[:2]) + 1911, int(s[2:4]), int(s[4:6])
    else:
        return None
    try:
        return datetime(year, month, day)
    except:
        return None

# --- 高效資料讀取工具 ---
def local_csv_loader(filename):
    """讀取同目錄下的 CSV 檔案"""
    if not os.path.exists(filename):
        return None
    for enc in ['utf-8-sig', 'utf-8', 'big5', 'cp950']:
        try:
            df = pd.read_csv(filename, encoding=enc)
            df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
            return df
        except:
            continue
    return None

# --- 建立藥價字典 ---
def prepare_price_dict(price_df):
    price_map = {2022: {}, 2023: {}, 2024: {}}
    if price_df is None: return price_map
    pdf = price_df.copy()
    pdf['起'] = pdf['有效起日'].apply(parse_roc_date)
    pdf['迄'] = pdf['有效迄日'].apply(parse_roc_date)
    pdf['支付價'] = pd.to_numeric(pdf['支付價'], errors='coerce').fillna(0.0)
    for year in [2022, 2023, 2024]:
        start_dt, end_dt = datetime(year, 1, 1), datetime(year, 12, 31)
        mask = (pdf['起'] <= end_dt) & (pdf['迄'] >= start_dt)
        temp_df = pdf[mask].copy()
        if not temp_df.empty:
            temp_df['區間起'] = temp_df['起'].apply(lambda d: max(d, start_dt))
            temp_df['區間迄'] = temp_df['迄'].apply(lambda d: min(d, end_dt))
            temp_df['天數'] = (temp_df['區間迄'] - temp_df['區間起']).dt.days + 1
            idx = temp_df.groupby('藥品代號')['天數'].idxmax()
            price_map[year] = dict(zip(temp_df.loc[idx, '藥品代號'], temp_df.loc[idx, '支付價']))
    return price_map

# --- 建立用量字典 ---
def prepare_usage_dict(u22, u23, u24):
    usage_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                usage_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 主介面 ---
st.title("💊 健保藥品分類量價自動標註系統")
st.markdown("請上傳 **分類項目表**，程式將自動抓取同目錄下的藥價與用量檔進行標註。")

# 1. 手動上傳分類表
uploaded_items = st.file_uploader("📤 請上傳分類項目表 (例如: items.csv)", type="csv")

# 2. 自動偵測同目錄下的背景檔案
with st.sidebar:
    st.header("📂 同目錄背景檔案檢查")
    p_df_local = pd.concat([local_csv_loader('Price_ATC1.csv'), local_csv_loader('Price_ATC2.csv')], ignore_index=True)
    u22 = local_csv_loader('A21030000I-E41005-001 (2022).csv')
    u23 = local_csv_loader('A21030000I-E41005-002 (2023).csv')
    u24 = local_csv_loader('A21030000I-E41005-003 (2024).csv')
    
    st.write(f"藥價表: {'✅' if len(p_df_local)>0 else '❌'}")
    st.write(f"2022用量: {'✅' if u22 is not None else '❌'}")
    st.write(f"2023用量: {'✅' if u23 is not None else '❌'}")
    st.write(f"2024用量: {'✅' if u24 is not None else '❌'}")

if uploaded_items:
    # 讀取上傳的檔案
    m_df = pd.read_csv(uploaded_items, encoding='utf-8-sig')
    m_df.columns = [str(c).replace('\n', ' ').strip() for c in m_df.columns]
    
    st.success(f"已讀取上傳表單：共 {len(m_df)} 筆項目。")

    if st.button("🚀 執行全項標註 (附加 2022-2024 價量)", type="primary"):
        if len(p_df_local) == 0:
            st.error("目錄下找不到 Price_ATC1.csv 或 Price_ATC2.csv，無法計算單價！")
        else:
            with st.status("正在匹配目錄下的量價大數據...", expanded=True):
                p_map = prepare_price_dict(p_df_local)
                u_map = prepare_usage_dict(u22, u23, u24)
                
                stats = []
                for _, row in m_df.iterrows():
                    code = str(row['藥品代碼']).strip()
                    res = {'藥品代碼': code}
                    for yr in [2022, 2023, 2024]:
                        p = p_map[yr].get(code, 0.0)
                        q = u_map[yr].get(code, 0.0)
                        res[f'{yr} 銷售量'] = q
                        res[f'{yr} 當年健保單價'] = p
                        res[f'{yr} 當年健保總價'] = round(p * q, 1)
                    stats.append(res)
                
                final_df = pd.merge(m_df, pd.DataFrame(stats), on='藥品代碼', how='left')
            
            st.subheader("分析預覽")
            st.dataframe(final_df.head(100), use_container_width=True)
            
            csv_buf = io.StringIO()
            final_df.to_csv(csv_buf, index=False, encoding='utf-8-sig')
            st.download_button("📥 下載完整標註報表", csv_buf.getvalue(), 
                               file_name=f"健保標註_{datetime.now().strftime('%m%d')}.csv", 
                               mime='text/csv')
else:
    st.info("請先上傳您的分類項目 CSV 檔案以開始運算。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
