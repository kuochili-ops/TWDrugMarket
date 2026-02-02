import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品價量標註系統", layout="wide")

def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    if not os.path.exists(file):
        return None
    for enc in encodings:
        try:
            df = pd.read_csv(file, encoding=enc)
            # 僅針對標題做清理，內容不動，保留原始結構
            df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
            return df
        except:
            continue
    return None

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

# --- 核心運算：字典映射法 ---
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
            # 找到代碼欄位 (通常是第二欄或標題為藥品代碼)
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                usage_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 載入區 ---
@st.cache_data
def load_data():
    # 改名為 items.csv
    m_df = try_read_csv('items.csv')
    p1, p2 = try_read_csv('Price_ATC1.csv'), try_read_csv('Price_ATC2.csv')
    p_df = pd.concat([p1, p2], ignore_index=True) if (p1 is not None and p2 is not None) else None
    u22 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    u23 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    u24 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    return m_df, p_df, u22, u23, u24

# --- UI 介面 ---
st.title("💊 健保藥品分類量價標註工具")

m_df, p_df, u22, u23, u24 = load_data()

# 檔案檢查
with st.sidebar:
    st.header("📂 檔案狀態檢查")
    st.write(f"items.csv: {'✅' if m_df is not None else '❌'}")
    st.write(f"藥價表: {'✅' if p_df is not None else '❌'}")
    st.write(f"2022-24 用量表: {'✅' if u22 is not None and u24 is not None else '❌'}")

if m_df is not None and p_df is not None:
    st.info(f"成功讀取 `items.csv`，共 {len(m_df)} 筆藥品。")
    
    if st.button("🚀 執行全項標註 (附加 2022-2024 價量)", type="primary"):
        with st.status("運算中...", expanded=True):
            p_map = prepare_price_dict(p_df)
            u_map = prepare_usage_dict(u22, u23, u24)
            
            # 建立附加資料清單
            stats_list = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                res = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    p = p_map[yr].get(code, 0.0)
                    q = u_map[yr].get(code, 0.0)
                    res[f'{yr} 銷售量'] = q
                    res[f'{yr} 當年健保單價'] = p
                    res[f'{yr} 當年健保總價'] = round(p * q, 1)
                stats_list.append(res)
            
            # 精準合併：確保 items.csv 的原始欄位（包括空欄位）完全不動
            merged_df = pd.merge(m_df, pd.DataFrame(stats_list), on='藥品代碼', how='left')
            
            st.write("✅ 標註完成！")
            st.dataframe(merged_df.head(100), use_container_width=True)
            
            # 下載
            csv_data = merged_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="📥 下載標註完成報表",
                data=csv_data,
                file_name=f"標註結果_{datetime.now().strftime('%m%d')}.csv",
                mime='text/csv'
            )
else:
    st.error("請確認目錄下是否有 `items.csv` 以及正確的藥價與用量檔案。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
