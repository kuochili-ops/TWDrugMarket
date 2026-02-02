import pandas as pd
import streamlit as st
from datetime import datetime
import io

# --- 網頁設定 ---
st.set_page_config(page_title="健保藥品價量標註系統", layout="wide")

# --- 日期轉換函數 (民國轉西元) ---
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

# --- 核心邏輯：建立年度單價字典 ---
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
            # 抓取年度內生效最久者為代表價
            idx = temp_df.groupby('藥品代號')['天數'].idxmax()
            final_prices = temp_df.loc[idx, ['藥品代號', '支付價']]
            price_map[year] = dict(zip(final_prices['藥品代號'], final_prices['支付價']))
    return price_map

# --- 核心邏輯：建立年度用量字典 ---
def prepare_usage_dict(u22, u23, u24):
    usage_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            # 自動偵測藥品代碼欄位 (通常是標題或第二欄)
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                usage_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 主畫面介面 ---
st.title("💊 健保藥品分類量價自動標註系統")
st.markdown("請在左側選單上傳 **items.csv** 與相關健保資料，系統將自動附加 2022-2024 的價量數據。")

# --- 側邊欄：檔案上傳區 ---
with st.sidebar:
    st.header("📤 上傳 CSV 檔案")
    up_items = st.file_uploader("1. 上傳項目分類表 (items.csv)", type="csv")
    st.divider()
    up_p1 = st.file_uploader("2. 上傳 Price_ATC1.csv", type="csv")
    up_p2 = st.file_uploader("3. 上傳 Price_ATC2.csv", type="csv")
    st.divider()
    up_u22 = st.file_uploader("4. 上傳 2022 用量檔", type="csv")
    up_u23 = st.file_uploader("5. 上傳 2023 用量檔", type="csv")
    up_u24 = st.file_uploader("6. 上傳 2024 用量檔", type="csv")

# --- 執行處理 ---
if up_items and up_p1 and up_p2:
    # 讀取函數 (清理標題)
    def load_csv(file):
        df = pd.read_csv(file, encoding='utf-8-sig')
        df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
        return df

    m_df = load_csv(up_items)
    p_df = pd.concat([load_csv(up_p1), load_csv(up_p2)], ignore_index=True)
    
    u22 = load_csv(up_u22) if up_u22 else None
    u23 = load_csv(up_u23) if up_u23 else None
    u24 = load_csv(up_u24) if up_u24 else None

    st.success(f"✅ 分類表 `{up_items.name}` 讀取成功 (共 {len(m_df)} 筆)")

    if st.button("🚀 開始執行全項價量標註", type="primary"):
        with st.status("正在計算大數據，請稍候...", expanded=True):
            p_map = prepare_price_dict(p_df)
            u_map = prepare_usage_dict(u22, u23, u24)
            
            # 準備標註資料
            stats_list = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                item_data = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    p = p_map[yr].get(code, 0.0)
                    q = u_map[yr].get(code, 0.0)
                    item_data[f'{yr} 銷售量'] = q
                    item_data[f'{yr} 當年健保單價'] = p
                    item_data[f'{yr} 當年健保總價'] = round(p * q, 1)
                stats_list.append(item_data)
            
            # 使用 left join 確保原始 items.csv 的空欄位與結構完全不動
            final_df = pd.merge(m_df, pd.DataFrame(stats_list), on='藥品代碼', how='left')
            
        st.subheader("分析結果預覽 (向右捲動查看新欄位)")
        st.dataframe(final_df.head(100), use_container_width=True)
        
        # 匯出 CSV
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載標註完成之完整分析表",
            data=csv_buffer.getvalue(),
            file_name=f"健保價量標註結果_{datetime.now().strftime('%m%d')}.csv",
            mime='text/csv'
        )
else:
    st.info("👈 請先在側邊欄上傳分類表與藥價表開始操作。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
