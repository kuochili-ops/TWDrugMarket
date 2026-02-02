import pandas as pd
import streamlit as st
from datetime import datetime
import io

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品價量分析系統", layout="wide")

# --- 數據讀取工具函式 ---
def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    for enc in encodings:
        try:
            df = pd.read_csv(file, encoding=enc)
            df.columns = df.columns.str.strip()
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

# --- 預計算優化邏輯 (讓一萬多筆資料能跑得動) ---
def prepare_price_dict(price_df, years=[2022, 2023, 2024]):
    price_map = {year: {} for year in years}
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

# --- 載入資料 ---
@st.cache_data
def load_and_merge_master():
    # 讀取分類主表
    m_df = try_read_csv('現行健保收載藥品重新分類項目表_1141229_1140673262.xlsx - 現行健保收載藥品重新分類項目表.csv')
    # 讀取藥價表
    p1 = try_read_csv('Price_ATC1.csv')
    p2 = try_read_csv('Price_ATC2.csv')
    p_df = pd.concat([p1, p2], ignore_index=True) if (p1 is not None and p2 is not None) else None
    # 讀取用量表
    u22 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    u23 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    u24 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    return m_df, p_df, u22, u23, u24

# --- 主程式介面 ---
st.title("💊 健保藥品分類量價全項串接系統")

m_df, p_df, u22, u23, u24 = load_and_merge_master()

if m_df is not None and p_df is not None:
    st.success(f"✅ 已成功載入分類表 ({len(m_df)} 筆資料)")
    
    # 關鍵的執行按鈕
    if st.button("🚀 開始執行全項目價量串接運算", type="primary"):
        with st.status("正在進行大量資料運算...", expanded=True) as status:
            st.write("步驟 1: 建立快速對照字典...")
            p_map = prepare_price_dict(p_df)
            u_map = prepare_usage_dict(u22, u23, u24)
            
            st.write("步驟 2: 計算每項藥品之年總價...")
            # 建立計算結果列表
            results = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                item_res = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    price = p_map[yr].get(code, 0.0)
                    qty = u_map[yr].get(code, 0.0)
                    item_res[f'{yr}單價'] = price
                    item_res[f'{yr}用量'] = qty
                    item_res[f'{yr}總價'] = round(price * qty, 1)
                results.append(item_res)
            
            st.write("步驟 3: 合併報表...")
            stats_df = pd.DataFrame(results)
            # 合併原始分類表與計算結果
            final_df = pd.merge(m_df, stats_df, on='藥品代碼', how='left')
            
            status.update(label="✅ 運算完成！", state="complete", expanded=False)
        
        st.subheader("分析結果預覽")
        st.dataframe(final_df.head(50), use_container_width=True)
        
        # 下載按鈕
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載完整分析報表 (Excel CSV)",
            data=csv_buffer.getvalue(),
            file_name=f"全項價量分析_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv'
        )
else:
    st.error("❌ 檔案讀取失敗，請確認 'Price_ATC1.csv' 與 '分類項目表.csv' 等檔案在同一目錄下。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
