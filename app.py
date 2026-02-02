import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品動態價量系統", layout="wide")

def parse_roc_date(s):
    """民國日期字串轉為西元 datetime"""
    try:
        s = str(int(float(s)))
    except: return None
    if len(s) == 7:
        y, m, d = int(s[:3]) + 1911, int(s[3:5]), int(s[5:7])
    elif len(s) == 6:
        y, m, d = int(s[:2]) + 1911, int(s[2:4]), int(s[4:6])
    else: return None
    try: return datetime(y, m, d)
    except: return None

def local_loader(fn):
    if not os.path.exists(fn): return None
    for enc in ['utf-8-sig', 'utf-8', 'big5', 'cp950']:
        try:
            df = pd.read_csv(fn, encoding=enc)
            df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
            return df
        except: continue
    return None

def prepare_dynamic_price_map(p_df):
    """計算每一年度生效天數最長的單價"""
    price_map = {2022: {}, 2023: {}, 2024: {}}
    if p_df is None: return price_map
    
    # 預處理藥價表
    p_df['起'] = p_df['有效起日'].apply(parse_roc_date)
    p_df['迄'] = p_df['有效迄日'].apply(parse_roc_date)
    p_df['支付價'] = pd.to_numeric(p_df['支付價'], errors='coerce').fillna(0.0)
    
    # 過濾掉日期解析失敗的資料
    p_df = p_df.dropna(subset=['起', '迄'])

    for yr in [2022, 2023, 2024]:
        s_dt, e_dt = datetime(yr, 1, 1), datetime(yr, 12, 31)
        # 篩選該年份有重疊的價格區間
        mask = (p_df['起'] <= e_dt) & (p_df['迄'] >= s_dt)
        year_p = p_df[mask].copy()
        
        if not year_p.empty:
            # 計算在該年度內的生效天數
            year_p['區間起'] = year_p['起'].apply(lambda x: max(x, s_dt))
            year_p['區間迄'] = year_p['迄'].apply(lambda x: min(x, e_dt))
            year_p['生效天數'] = (year_p['區間迄'] - year_p['區間起']).dt.days + 1
            
            # 針對每個藥品，取生效天數最長的那筆價格
            idx = year_p.groupby('藥品代號')['生效天數'].idxmax()
            price_map[yr] = dict(zip(year_p.loc[idx, '藥品代號'], year_p.loc[idx, '支付價']))
            
    return price_map

def prepare_usage_map(u22, u23, u24):
    u_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                u_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return u_map

# --- UI ---
st.title("💊 健保藥品動態價量標註系統")

with st.sidebar:
    st.header("📂 背景數據檢查")
    p_local = pd.concat([local_loader('Price_ATC1.csv'), local_loader('Price_ATC2.csv')], ignore_index=True)
    u22, u23, u24 = local_loader('A21030000I-E41005-001 (2022).csv'), local_loader('A21030000I-E41005-002 (2023).csv'), local_loader('A21030000I-E41005-003 (2024).csv')
    st.write(f"藥價數據: {'✅' if p_local is not None else '❌'}")
    st.write(f"用量數據: {'✅' if u24 is not None else '❌'}")

up_file = st.file_uploader("📤 上傳原始分類表 (items.csv)", type="csv")

if up_file:
    m_df = pd.read_csv(up_file, encoding='utf-8-sig')
    st.success(f"已讀取 {len(m_df)} 筆項目")

    if st.button("🚀 執行精準年度標註", type="primary"):
        with st.status("正在計算每年動態單價...", expanded=True):
            p_map = prepare_dynamic_price_map(p_local)
            u_map = prepare_usage_map(u22, u23, u24)
            
            results = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                res = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    p = p_map[yr].get(code, 0.0)
                    q = u_map[yr].get(code, 0.0)
                    res[f'{yr} 銷售量'] = q
                    res[f'{yr} 當年健保單價'] = p
                    res[f'{yr} 當年健保總價'] = round(p * q, 1)
                results.append(res)
            
            final_df = pd.merge(m_df, pd.DataFrame(results), on='藥品代碼', how='left')
            
        st.subheader("結果預覽 (請檢查不同年份單價是否已有變化)")
        st.dataframe(final_df.head(100))
        
        csv_out = final_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button("📥 下載動態標註報表", csv_out, file_name="健保年度價量分析.csv", mime='text/csv')
