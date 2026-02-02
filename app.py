import pandas as pd
import streamlit as st
from datetime import datetime
import io

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品價量標註系統", layout="wide")

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

# --- 核心邏輯：建立高效查找字典 ---
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
            final_prices = temp_df.loc[idx, ['藥品代號', '支付價']]
            price_map[year] = dict(zip(final_prices['藥品代號'], final_prices['支付價']))
    return price_map

def prepare_usage_dict(u22, u23, u24):
    usage_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                usage_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- UI 介面 ---
st.title("💊 健保藥品分類量價自動標註系統 (上傳版)")
st.info("請依序上傳分類表、藥價表及各年度用量表，系統將自動完成 13,950 筆資料標註。")

# --- 側邊欄：檔案上傳區 ---
with st.sidebar:
    st.header("📤 上傳資料檔案")
    # 分類主表
    uploaded_master = st.file_uploader("1. 上傳項目分類表 (CSV)", type="csv")
    # 藥價表
    uploaded_p1 = st.file_uploader("2. 上傳 Price_ATC1 (CSV)", type="csv")
    uploaded_p2 = st.file_uploader("3. 上傳 Price_ATC2 (CSV)", type="csv")
    # 用量表
    uploaded_u22 = st.file_uploader("4. 上傳 2022 用量檔 (CSV)", type="csv")
    uploaded_u23 = st.file_uploader("5. 上傳 2023 用量檔 (CSV)", type="csv")
    uploaded_u24 = st.file_uploader("6. 上傳 2024 用量檔 (CSV)", type="csv")

# --- 讀取與處理 ---
if uploaded_master and uploaded_p1 and uploaded_p2:
    # 讀取函數
    def read_df(file):
        df = pd.read_csv(file, encoding='utf-8-sig')
        df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
        return df

    m_df = read_df(uploaded_master)
    p_df = pd.concat([read_df(uploaded_p1), read_df(uploaded_p2)], ignore_index=True)
    
    u22 = read_df(uploaded_u22) if uploaded_u22 else None
    u23 = read_df(uploaded_u23) if uploaded_u23 else None
    u24 = read_df(uploaded_u24) if uploaded_u24 else None

    st.success(f"✅ 分類表已載入：{len(m_df)} 筆資料")

    if st.button("🚀 開始執行全項標註 (附加 2022-2024 價量)", type="primary"):
        with st.status("運算中，請稍候...", expanded=True):
            p_map = prepare_price_dict(p_df)
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
            
            # 合併數據並保留原始所有欄位 (包含 Unnamed 欄位)
            final_df = pd.merge(m_df, pd.DataFrame(stats), on='藥品代碼', how='left')
            
        st.subheader("標註結果預覽")
        st.dataframe(final_df.head(100), use_container_width=True)
        
        # 下載區
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載標註完成之完整報表",
            data=csv_buffer.getvalue(),
            file_name=f"健保標註結果_{datetime.now().strftime('%m%d')}.csv",
            mime='text/csv'
        )
else:
    st.warning("👈 請先從側邊欄上傳必要的 CSV 檔案。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
