import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品分類全項量價附加系統", layout="wide")

def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    if not file or not os.path.exists(file):
        return None
    for enc in encodings:
        try:
            # 讀取 CSV 並不進行欄位過濾，保留原始結構
            df = pd.read_csv(file, encoding=enc)
            # 清理標題文字中的換行與多餘空格，避免欄位對應失敗
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

# --- 字典建立優化 (Mapping) ---
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
            # 取年度內生效天數最長者作為代表藥價
            idx = temp_df.groupby('藥品代號')['天數'].idxmax()
            final_prices = temp_df.loc[idx, ['藥品代號', '支付價']]
            price_map[year] = dict(zip(final_prices['藥品代號'], final_prices['支付價']))
    return price_map

def prepare_usage_dict(u22, u23, u24):
    usage_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            # 偵測正確的藥品代碼與量欄位
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[0]
            c_qty = '含包裹支付的醫令量_合計' if '含包裹支付的醫令量_合計' in df.columns else None
            if c_qty:
                usage_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 讀取所有必要資料 ---
@st.cache_data
def load_all_essential_data():
    master_name = "現行健保收載藥品重新分類項目表_1141229_1140673262.csv"
    m_df = try_read_csv(master_name)
    
    # 藥價表
    p1, p2 = try_read_csv('Price_ATC1.csv'), try_read_csv('Price_ATC2.csv')
    p_df = pd.concat([p1, p2], ignore_index=True) if (p1 is not None and p2 is not None) else None
    
    # 用量表
    u22 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    u23 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    u24 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    
    return m_df, p_df, u22, u23, u24, master_name

# --- 主畫面 ---
st.title("💊 健保藥品分類全項量價附加系統")

m_df, p_df, u22, u23, u24, master_name = load_all_essential_data()

if m_df is not None and p_df is not None:
    st.success(f"✅ 分類項目表已就緒 (共 {len(m_df)} 筆)")
    
    if st.button("🚀 執行全項標註 (包含銷售量、單價、總價)", type="primary"):
        with st.status("正在匹配 2022-2024 價量數據...", expanded=True):
            p_map = prepare_price_dict(p_df)
            u_map = prepare_usage_dict(u22, u23, u24)
            
            stats = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                row_stats = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    price = p_map[yr].get(code, 0.0)
                    qty = u_map[yr].get(code, 0.0)
                    total = round(price * qty, 1)
                    
                    row_stats[f'{yr} 銷售量'] = qty
                    row_stats[f'{yr} 當年健保單價'] = price
                    row_stats[f'{yr} 當年健保總價'] = total
                stats.append(row_stats)
            
            # 使用 left join 合併，確保分類表原始項目完全不變，僅向右增加欄位
            result_df = pd.merge(m_df, pd.DataFrame(stats), on='藥品代碼', how='left')
            
        st.subheader("分析結果預覽 (橫向滾動查看新標註項目)")
        st.dataframe(result_df.head(50), use_container_width=True)
        
        # 匯出 CSV
        csv_out = result_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載完成標註之 CSV 檔",
            data=csv_out,
            file_name=f"健保重新分類標註_{datetime.now().strftime('%m%d')}.csv",
            mime='text/csv'
        )
else:
    st.error("❌ 無法載入分類表或藥價表，請確認 CSV 檔名與路徑。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
