import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品分類量價全項串接系統", layout="wide")

# --- 數據讀取工具函式 ---
def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    if not file or not os.path.exists(file):
        return None
    for enc in encodings:
        try:
            # 讀取時將標題中的換行符號替換掉，確保程式對齊
            df = pd.read_csv(file, encoding=enc)
            df.columns = df.columns.str.replace(r'\n', ' ', regex=True).str.strip()
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

# --- 預計算字典 (對應一萬三千筆資料的關鍵) ---
def prepare_price_dict(price_df, years=[2022, 2023, 2024]):
    price_map = {year: {} for year in years}
    if price_df is None: return price_map
    
    pdf = price_df.copy()
    pdf['起'] = pdf['有效起日'].apply(parse_roc_date)
    pdf['迄'] = pdf['有效迄日'].apply(parse_roc_date)
    pdf['支付價'] = pd.to_numeric(pdf['支付價'], errors='coerce').fillna(0.0)
    
    for year in years:
        start_dt = datetime(year, 1, 1)
        end_dt = datetime(year, 12, 31)
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
            # 確保用量表的欄位名稱對齊
            col_code, col_qty = '藥品代碼', '含包裹支付的醫令量_合計'
            if col_code in df.columns and col_qty in df.columns:
                usage_map[yr] = dict(zip(df[col_code].astype(str).str.strip(), pd.to_numeric(df[col_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 載入資料 ---
@st.cache_data
def load_all_files():
    # 指定您上傳的正確檔名
    master_file = "現行健保收載藥品重新分類項目表_1141229_1140673262.csv"
    m_df = try_read_csv(master_file)
    
    # 讀取藥價表
    p1 = try_read_csv('Price_ATC1.csv')
    p2 = try_read_csv('Price_ATC2.csv')
    p_df = pd.concat([p1, p2], ignore_index=True) if (p1 is not None and p2 is not None) else None
    
    # 讀取用量表
    u22 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    u23 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    u24 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    
    return m_df, p_df, u22, u23, u24, master_file

# --- 主介面 ---
st.title("💊 健保藥品分類全項量價附加系統")
st.markdown("針對「重新分類項目表」中的每一項目，自動補上 2022-2024 的銷售量、健保單價與總價。")

m_df, p_df, u22, u23, u24, master_name = load_all_files()

if m_df is not None and p_df is not None:
    st.success(f"✅ 已讀取分類表: {master_name} (共 {len(m_df)} 筆)")
    
    if st.button("🚀 開始執行全量資料標註 (標註 13,950 筆)", type="primary"):
        with st.status("正在進行高效率價量串接...", expanded=True) as status:
            # 預建立 Mapping
            p_map = prepare_price_dict(p_df)
            u_map = prepare_usage_dict(u22, u23, u24)
            
            # 計算價量附加資訊
            results = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                item_data = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    price = p_map[yr].get(code, 0.0)
                    usage = u_map[yr].get(code, 0.0)
                    total = round(price * usage, 1)
                    
                    item_data[f'{yr} 銷售量'] = usage
                    item_data[f'{yr} 當年健保單價'] = price
                    item_data[f'{yr} 當年健保總價'] = total
                results.append(item_data)
            
            # 合併回原始 DataFrame
            stats_df = pd.DataFrame(results)
            final_df = pd.merge(m_df, stats_df, on='藥品代碼', how='left')
            
            status.update(label="✅ 標註完成！", state="complete", expanded=False)
            
        st.subheader("標註結果預覽 (前50筆)")
        # 欄位較多，使用表格橫向展開
        st.dataframe(final_df.head(50), use_container_width=True)
        
        # 準備下載檔案
        output = io.StringIO()
        final_df.to_csv(output, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載附加價量標註之完整分析表",
            data=output.getvalue(),
            file_name=f"健保重新分類項目_價量標註_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
else:
    st.error("❌ 無法載入必要檔案。請確認 `現行健保收載藥品重新分類項目表_1141229_1140673262.csv` 與藥價、用量檔皆存在。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
