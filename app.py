import pandas as pd
import streamlit as st
from datetime import datetime
import io

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品分類量價分析工具", layout="wide")

# --- 數據讀取工具函式 ---
def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    for enc in encodings:
        try:
            # 針對可能包含檔案路徑或上傳對象的情況處理
            df = pd.read_csv(file, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    return None

def parse_roc_date(s):
    try:
        s = str(int(float(s))) # 處理可能出現的浮點數或字串
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

# --- 核心優化運算邏輯 ---
@st.cache_data
def prepare_price_dict(price_df, years=[2022, 2023, 2024]):
    """預先計算所有藥品每一年的代表藥價 (年度內天數最長者)"""
    price_map = {year: {} for year in years}
    
    # 預處理日期
    pdf = price_df.copy()
    pdf['起'] = pdf['有效起日'].apply(parse_roc_date)
    pdf['迄'] = pdf['有效迄日'].apply(parse_roc_date)
    pdf['支付價'] = pd.to_numeric(pdf['支付價'], errors='coerce').fillna(0.0)
    
    for year in years:
        start_dt = datetime(year, 1, 1)
        end_dt = datetime(year, 12, 31)
        
        # 篩選該年份有效的藥價
        mask = (pdf['起'] <= end_dt) & (pdf['迄'] >= start_dt)
        temp_df = pdf[mask].copy()
        
        if temp_df.empty:
            continue
            
        temp_df['區間起'] = temp_df['起'].apply(lambda d: max(d, start_dt))
        temp_df['區間迄'] = temp_df['迄'].apply(lambda d: min(d, end_dt))
        temp_df['天數'] = (temp_df['區間迄'] - temp_df['區間起']).dt.days + 1
        
        # 依藥品代號分組，取天數最大的那一筆
        idx = temp_df.groupby('藥品代號')['天數'].idxmax()
        final_prices = temp_df.loc[idx, ['藥品代號', '支付價']]
        price_map[year] = dict(zip(final_prices['藥品代號'], final_prices['支付價']))
        
    return price_map

@st.cache_data
def prepare_usage_dict(u22, u23, u24):
    """預先計算醫令量字典"""
    usage_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            # 確保欄位名稱正確
            col_code = '藥品代碼'
            col_qty = '含包裹支付的醫令量_合計'
            if col_code in df.columns and col_qty in df.columns:
                usage_map[yr] = dict(zip(df[col_code].astype(str).str.strip(), pd.to_numeric(df[col_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 側邊欄：載入檔案 ---
st.sidebar.header("資料檔案設定")
st.sidebar.info("請確保以下檔案放在程式同一目錄下：")

# --- 載入主要資料 ---
def load_all_data():
    price1 = try_read_csv('Price_ATC1.csv')
    price2 = try_read_csv('Price_ATC2.csv')
    if price1 is None or price2 is None:
        st.error("找不到 Price_ATC1.csv 或 Price_ATC2.csv")
        st.stop()
    price_df = pd.concat([price1, price2], ignore_index=True)
    
    u22 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    u23 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    u24 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    
    master_df = try_read_csv('現行健保收載藥品重新分類項目表_1141229_1140673262.xlsx - 現行健保收載藥品重新分類項目表.csv')
    
    return price_df, u22, u23, u24, master_df

# 執行載入
price_df, use_22, use_23, use_24, master_df = load_all_data()

# --- 主畫面 ---
st.title("💊 健保藥品分類暨年度價量分析系統")
st.markdown("本工具將「重新分類項目表」與 2022-2024 健保價量資料進行全項串接。")

if master_df is not None:
    if st.button("開始執行全項目串接運算"):
        with st.status("正在處理大量數據，請稍候...", expanded=True) as status:
            st.write("1. 建立藥價快取字典...")
            p_map = prepare_price_dict(price_df)
            
            st.write("2. 建立醫令量快取字典...")
            u_map = prepare_usage_dict(use_22, use_23, use_24)
            
            st.write("3. 執行全量項目計算...")
            
            # 使用列表推導式快速建立結果，避免在 DataFrame 裡使用 apply (這會很慢)
            target_data = []
            for idx, row in master_df.iterrows():
                code = str(row['藥品代碼']).strip()
                
                res = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    price = p_map[yr].get(code, 0.0)
                    qty = u_map[yr].get(code, 0.0)
                    total = price * qty
                    res[f'{yr}單價'] = price
                    res[f'{yr}用量'] = qty
                    res[f'{yr}總價'] = total
                target_data.append(res)
            
            stats_df = pd.DataFrame(target_data)
            
            st.write("4. 合併分類資訊表...")
            # 避免重複欄位
            final_df = pd.merge(master_df, stats_df, on='藥品代碼', how='left')
            
            status.update(label="計算完成！", state="complete", expanded=False)

        st.subheader("分析結果預覽 (前100筆)")
        # 設定顯示格式
        st.dataframe(
            final_df.head(100), 
            use_container_width=True
        )

        # 匯出 CSV
        csv = final_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="💾 下載完整分析報表 (CSV)",
            data=csv,
            file_name=f"健保藥品價量分析_{datetime.now().strftime('%Y%m%d')}.csv",
            mime='text/csv',
        )
else:
    st.warning("請確認 '現行健保收載藥品重新分類項目表...csv' 是否已放置於正確目錄。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
