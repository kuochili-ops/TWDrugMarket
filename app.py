import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品分類量價標註系統", layout="wide")

def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    if not os.path.exists(file):
        return None
    for enc in encodings:
        try:
            # 讀取時不變動原始內容，僅清理標題空格
            df = pd.read_csv(file, encoding=enc)
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
            # 抓取該年度生效天數最長的一筆作為年度單價
            idx = temp_df.groupby('藥品代號')['天數'].idxmax()
            final_prices = temp_df.loc[idx, ['藥品代號', '支付價']]
            price_map[year] = dict(zip(final_prices['藥品代號'], final_prices['支付價']))
    return price_map

def prepare_usage_dict(u22, u23, u24):
    usage_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            # 自動偵測藥品代碼欄位 (通常是第二欄)
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                usage_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 載入區 ---
@st.cache_data
def load_all_data():
    # 改名後的分類表
    m_df = try_read_csv('items.csv')
    # 藥價表 (ATC1 & ATC2 合併)
    p1 = try_read_csv('Price_ATC1.csv')
    p2 = try_read_csv('Price_ATC2.csv')
    p_df = pd.concat([p1, p2], ignore_index=True) if (p1 is not None and p2 is not None) else None
    # 2022-2024 用量表
    u22 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    u23 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    u24 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    return m_df, p_df, u22, u23, u24

# --- UI 介面 ---
st.title("💊 健保藥品價量自動標註系統")

m_df, p_df, u22, u23, u24 = load_all_data()

# 側邊欄狀態監測
with st.sidebar:
    st.header("📂 檔案準備狀態")
    st.write(f"items.csv: {'✅' if m_df is not None else '❌'}")
    st.write(f"藥價表: {'✅' if p_df is not None else '❌'}")
    st.write(f"用量表 (22-24): {'✅' if u22 is not None and u24 is not None else '❌'}")
    if m_df is None:
        st.warning("請將分類表改名為 `items.csv` 並放置於目錄下。")

if m_df is not None and p_df is not None:
    st.success(f"成功識別 `items.csv`，共 {len(m_df)} 筆項目待標註。")
    
    if st.button("🚀 開始執行全項標註 (附加 2022-2024 價量資訊)", type="primary"):
        with st.status("正在進行高速價量對比...", expanded=True) as status:
            p_map = prepare_price_dict(p_df)
            u_map = prepare_usage_dict(u22, u23, u24)
            
            # 建立標註資料列表
            results = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                item_stats = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    pr = p_map[yr].get(code, 0.0)
                    qt = u_map[yr].get(code, 0.0)
                    item_stats[f'{yr} 銷售量'] = qt
                    item_stats[f'{yr} 當年健保單價'] = pr
                    item_stats[f'{yr} 當年健保總價'] = round(pr * qt, 1)
                results.append(item_stats)
            
            # 以 left join 合併，保留 items.csv 的原始所有欄位結構
            final_df = pd.merge(m_df, pd.DataFrame(results), on='藥品代碼', how='left')
            
            status.update(label="✅ 標註運算完成！", state="complete", expanded=False)
            
        st.subheader("結果預覽 (前 100 筆)")
        st.dataframe(final_df.head(100), use_container_width=True)
        
        # 匯出 CSV 檔案
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載標註完成之 CSV 報表",
            data=csv_buffer.getvalue(),
            file_name=f"健保重新分類標註_{datetime.now().strftime('%m%d')}.csv",
            mime='text/csv'
        )
else:
    st.info("💡 啟動說明：請確保您的分類表檔名為 `items.csv`，且目錄中包含 Price_ATC1/2.csv 與年度用量檔。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
