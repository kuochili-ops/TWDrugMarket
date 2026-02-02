import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品價量分析系統", layout="wide")

# --- 數據讀取工具函式 ---
def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    if not file or not os.path.exists(file):
        return None
    for enc in encodings:
        try:
            df = pd.read_csv(file, encoding=enc)
            # 強制清理所有欄位名稱的空白與換行
            df.columns = df.columns.str.replace(r'\n', '', regex=True).str.strip()
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

# --- 預計算優化邏輯 ---
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
            col_code, col_qty = '藥品代碼', '含包裹支付的醫令量_合計'
            if col_code in df.columns and col_qty in df.columns:
                usage_map[yr] = dict(zip(df[col_code].astype(str).str.strip(), pd.to_numeric(df[col_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 檔案自動偵測 ---
def find_master_file():
    target_keyword = "重新分類項目表"
    files = [f for f in os.listdir('.') if f.endswith('.csv')]
    for f in files:
        if target_keyword in f:
            return f
    return None

# --- 載入資料 ---
@st.cache_data
def load_data_all():
    master_file = find_master_file()
    m_df = try_read_csv(master_file)
    
    p1 = try_read_csv('Price_ATC1.csv')
    p2 = try_read_csv('Price_ATC2.csv')
    p_df = pd.concat([p1, p2], ignore_index=True) if (p1 is not None and p2 is not None) else None
    
    u22 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    u23 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    u24 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    
    return m_df, p_df, u22, u23, u24, master_file

# --- 主畫面 ---
st.title("💊 健保藥品分類量價全項串接系統")

m_df, p_df, u22, u23, u24, master_name = load_data_all()

# 側邊欄：如果自動偵測失敗，手動排除
if m_df is None:
    st.sidebar.error("⚠️ 找不到分類表 CSV")
    manual_name = st.sidebar.text_input("請手動輸入分類表完整檔名 (需含 .csv):")
    if manual_name:
        m_df = try_read_csv(manual_name)
        master_name = manual_name

# 檢查狀態
with st.expander("📂 當前目錄檔案清單與檢查"):
    st.write("目錄下的 CSV 檔案：", [f for f in os.listdir('.') if f.endswith('.csv')])
    st.write(f"偵測到的分類表：`{master_name}`")
    st.divider()
    status_map = {"分類表": m_df, "藥價表": p_df, "2022量": u22, "2023量": u23, "2024量": u24}
    for k, v in status_map.items():
        st.write(f"{k}: {'✅ OK' if v is not None else '❌ Missing'}")

if m_df is not None and p_df is not None:
    st.success(f"✅ 準備就緒！分類表共有 {len(m_df)} 筆藥品。")
    
    if st.button("🚀 執行全項量價串接 (約需 5-10 秒)", type="primary"):
        with st.status("數據處理中...", expanded=True) as status:
            p_map = prepare_price_dict(p_df)
            u_map = prepare_usage_dict(u22, u23, u24)
            
            results = []
            # 使用藥品代碼作為 Key
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                item = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    pr = p_map[yr].get(code, 0.0)
                    qt = u_map[yr].get(code, 0.0)
                    item[f'{yr}單價'] = pr
                    item[f'{yr}用量'] = qt
                    item[f'{yr}總價'] = round(pr * qt, 1)
                results.append(item)
            
            final_df = pd.merge(m_df, pd.DataFrame(results), on='藥品代碼', how='left')
            status.update(label="運算完成！", state="complete")
            
            st.dataframe(final_df.head(50))
            
            csv = final_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button("📥 下載完整分析報表 (CSV)", csv, 
                               file_name=f"全項分析_{datetime.now().strftime('%m%d')}.csv", 
                               mime='text/csv')
else:
    st.info("💡 請確保您的分類表檔名包含『重新分類項目表』字樣，且副檔名為 `.csv`。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
