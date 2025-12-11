import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np

# --- 數據讀取與日期解析工具函式 (保持不變) ---

def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    """嘗試使用多種編碼讀取 CSV 檔案，並移除欄位名稱的空白。"""
    for enc in encodings:
        try:
            # 檔案路徑需在 Streamlit Cloud 部署時保持正確
            df = pd.read_csv(file, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    raise ValueError(f"{file} 無法用常見編碼讀取，請確認檔案格式。")

def parse_roc_date(s):
    """將民國日期字串轉換為 Python datetime 物件。"""
    try:
        s = str(int(s))
    except Exception:
        return None
    if len(s) == 7:
        year = int(s[:3]) + 1911
        month = int(s[3:5])
        day = int(s[5:7])
    elif len(s) == 6:
        year = int(s[:2]) + 1911
        month = int(s[2:4])
        day = int(s[4:6])
    else:
        return None
    try:
        return datetime(year, month, day)
    except Exception:
        return None

# --- 核心計算函式 (保持原有邏輯，不變動) ---

def get_longest_price(price_df, code, year):
    """計算特定藥品在特定年度的最長有效支付價格及其中文名。"""
    df = price_df[price_df['藥品代號'] == code].copy()
    df['起'] = df['有效起日'].apply(parse_roc_date)
    df['迄'] = df['有效迄日'].apply(parse_roc_date)
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    
    # 篩選在該年度有效的價格
    df = df[((df['起'] <= end) & (df['迄'] >= start)) | (df['起'].isnull() & df['迄'].isnull())].copy()
    
    if df.empty:
        return 0, 'N/A'
    
    # 計算該價格在該年度的有效天數
    df['Effective_Start'] = df['起'].apply(lambda x: max(x, start) if pd.notna(x) else start)
    df['Effective_End'] = df['迄'].apply(lambda x: min(x, end) if pd.notna(x) else end)
    df['Days'] = (df['Effective_End'] - df['Effective_Start']).dt.days + 1
    
    # 取得天數最長的價格紀錄
    longest_record = df.sort_values(by='Days', ascending=False).iloc[0]
    
    return longest_record['支付價'], longest_record['藥品中文名稱']

def calc_annual_payment(price_df, use_df, code, year):
    """計算特定藥品在特定年度的加總支付金額 (支付價 * 醫令量)。"""
    price, cname = get_longest_price(price_df, code, year)
    
    if use_df is not None and not use_df.empty:
        # 假設 use_df 中的 '藥品代碼' 和 '醫令量_合計' 是用於計算的關鍵欄位
        usage_record = use_df[use_df['藥品代碼'] == code]
        
        if not usage_record.empty:
            # 取得醫令量_合計 (用 .iloc[0] 避免 Series)
            total_usage = usage_record['醫令量_合計'].iloc[0]
            total_payment = price * total_usage
            return total_payment, cname
    
    return 0, cname

# --- 載入所有數據 (包含 37_2.csv) ---

@st.cache_data
def load_data():
    """載入所有藥價、使用量和適應症數據。"""
    
    # 1. 載入並合併藥價數據 Price_ATC1/2.csv
    price_df = pd.concat([try_read_csv("Price_ATC1.csv"), try_read_csv("Price_ATC2.csv")])
    price_df.columns = price_df.columns.str.strip()
    
    # 2. 資料清理/準備 (計算日期和建立 ATC 5碼)
    price_df['起'] = price_df['有效起日'].apply(parse_roc_date)
    price_df['迄'] = price_df['有效迄日'].apply(parse_roc_date)
    price_df['ATC代碼_5碼'] = price_df['ATC代碼'].str[:5] 
    
    # 3. 載入使用量數據
    use_2022 = try_read_csv("A21030000I-E41005-001 (2022).csv")
    use_2023 = try_read_csv("A21030000I-E41005-002 (2023).csv")
    use_2024 = try_read_csv("A21030000I-E41005-003 (2024).csv")

    # 4. 載入適應症數據 (37_2.csv)
    indications_df = try_read_csv("37_2.csv")
    indications_df.columns = indications_df.columns.str.strip()
    
    # 將價格欄位轉為數值，確保計算正常
    price_df['支付價'] = pd.to_numeric(price_df['支付價'], errors='coerce')

    return price_df, use_2022, use_2023, use_2024, indications_df

# --- 適應症查詢函式 (新增) ---

def get_indication_by_chinese_name(chinese_name, indications_df):
    """依據【藥品中文名稱】查詢適應症。"""
    if not isinstance(chinese_name, str): return "無資料"
    
    result = indications_df[indications_df['中文品名'].str.strip() == chinese_name.strip()]
    
    if not result.empty:
        return "<br>".join(result['適應症'].unique().tolist())
    
    return "適應症資料庫無此品項 (中文名)"

def get_indication_by_english_name(english_name, indications_df):
    """依據【藥品英文名稱】查詢適應症。"""
    if not isinstance(english_name, str): return "無資料"
    
    result = indications_df[indications_df['英文品名'].str.strip().str.upper() == english_name.strip().upper()]
    
    if not result.empty:
        return "<br>".join(result['適應症'].unique().tolist())
    
    return "適應症資料庫無此品項 (英文名)"


# --- Main Streamlit App Logic ---

st.set_page_config(layout="wide")
st.title("藥品市場分析工具 (TW Market Analysis)")

# 載入所有數據
try:
    price_df, use_2022, use_2023, use_2024, indications_df = load_data()
except Exception as e:
    st.error(f"數據載入失敗，請確認所有 CSV 檔案是否與 app.py 放在同一目錄且編碼正確。錯誤訊息: {e}")
    st.stop()


# 側邊欄輸入 (沿用您的原有邏輯)
atc_code_5 = st.sidebar.text_input('ATC 5碼', 'N05AH03') # 使用一個完整的範例
drug_code = st.sidebar.text_input('藥品代號', 'AC52617100') 

# ----------------------------------------------------------------------
# 主成分搜尋結果 (還原原有功能 + 實作 需求 2)
# ----------------------------------------------------------------------

if len(atc_code_5) >= 5:
    atc_code_4 = atc_code_5[:4] # 取得 ATC 4碼

    st.markdown(f"## 主成分搜尋結果 - 同規格藥品比較 (ATC {atc_code_5})")
    
    # 篩選 ATC 5碼
    sub_df_atc5 = price_df[price_df['ATC代碼_5碼'] == atc_code_5].copy()
    # 移除重複的藥品代號，保留最新生效的價格，以便計算總支付金額
    sub_df_atc5 = sub_df_atc5.sort_values(by='起', ascending=False).drop_duplicates(subset=['藥品代號'], keep='first')

    if not sub_df_atc5.empty:
        
        # 1. 計算同規格藥品中各年度加總支付金額 (沿用您的原有邏輯)
        years = [2022, 2023, 2024]
        use_dfs = {2022: use_2022, 2023: use_2023, 2024: use_2024}
        
        # 計算每個藥品代號的總支付金額 (跨年度總和)
        sub_df_atc5['總加總支付金額'] = sub_df_atc5['藥品代號'].apply(
            lambda code: sum(calc_annual_payment(price_df, use_dfs[year], code, year)[0] for year in years)
        )
        
        # 2. 找出加總支付金額最高者，取得其中文名稱
        if sub_df_atc5['總加總支付金額'].max() > 0:
            max_payment_drug = sub_df_atc5.loc[sub_df_atc5['總加總支付金額'].idxmax()]
            highest_paid_chinese_name = max_payment_drug['藥品中文名稱']
        else:
            highest_paid_chinese_name = "N/A"
        
        # 3. 以其中文名稱查詢適應症 【新增】
        indication_for_main_component = get_indication_by_chinese_name(highest_paid_chinese_name, indications_df)
        
        # 4. 在「同規格藥品各年度加總支付金額」表頭後新增「適應症」收合欄位 (依需求 2) 【新增】
        st.markdown("#### 同規格藥品各年度加總支付金額") 
        with st.expander(f"適應症 (以金額最高者: **{highest_paid_chinese_name}** 查詢)"):
            st.markdown(indication_for_main_component, unsafe_allow_html=True)
        
        # 顯示結果表
        st.dataframe(sub_df_atc5[['藥品代號', '藥品中文名稱', '總加總支付金額']].sort_values(by='總加總支付金額', ascending=False), 
                    hide_index=True, 
                    column_config={"總加總支付金額": st.column_config.NumberColumn(format="%.2f")})

        
        # ----------------------------------------------------------------------
        # 【還原】: 整個 ATC 4碼 的市場總結 (還原原有功能)
        # ----------------------------------------------------------------------
        
        st.markdown("---")
        st.markdown(f"#### 整個 ATC 4 碼 ({atc_code_4}) 的市場總結")
        
        # 篩選 ATC 4碼 (確保只計算一次)
        sub_df_atc4 = price_df[price_df['ATC代碼'].str.startswith(atc_code_4)].drop_duplicates(subset=['藥品代號'], keep='first').copy()
        
        # 計算各年度的總支付金額
        total_payments_atc4 = {}
        for year in years:
            # 計算該年度所有 ATC 4 藥品的總支付金額
            annual_payment = sub_df_atc4.apply(
                lambda r: calc_annual_payment(price_df, use_dfs[year], r['藥品代號'], year)[0], axis=1
            ).sum()
            total_payments_atc4[year] = annual_payment

        # 顯示總結表
        summary_data = {
            '年份': years,
            '加總支付金額': [total_payments_atc4[y] for y in years]
        }
        
        st.dataframe(pd.DataFrame(summary_data), hide_index=True, 
                    column_config={"加總支付金額": st.column_config.NumberColumn(format="%.2f")})
    else:
        st.warning(f"在 {atc_code_5} 找不到藥品代號。")


# ----------------------------------------------------------------------
# 以商品名搜尋結果 (實作 需求 3 & 1)
# ----------------------------------------------------------------------

# 篩選單一藥品代號
target_df = price_df[price_df['藥品代號'] == drug_code].copy()

if not target_df.empty:
    # 取得最新一筆的中文名和英文名，用於查詢
    target_info = target_df.sort_values(by='起', ascending=False).iloc[0]
    target_chinese_name = target_info['藥品中文名稱']
    target_english_name = target_info['藥品英文名稱']

    st.markdown(f"## 商品名搜尋結果 - {target_chinese_name} ({drug_code})")
    
    # 1. 以其 藥品中文名稱 查詢 適應症 (需求 3: 商品名搜尋) 【新增】
    indication_for_trade_name = get_indication_by_chinese_name(target_chinese_name, indications_df)
    
    # 2. 在「各商品 (藥品代號) 之 各時間階段藥價調整與調整率」表頭後新增「適應症」收合欄位 (依需求 3) 【新增】
    st.markdown("#### 各商品 (藥品代號) 之 各時間階段藥價調整與調整率") 
    
    with st.expander(f"適應症 (以中文名: **{target_chinese_name}** 查詢)"):
        st.markdown(indication_for_trade_name, unsafe_allow_html=True)
        
    # 顯示價格調整表 (沿用原有數據處理)
    price_adjustment_df = target_df[['有效起日', '有效迄日', '支付價']].sort_values(by='有效起日').copy()
    
    # 計算調整率
    price_adjustment_df['調整率'] = price_adjustment_df['支付價'].pct_change().fillna(0)
    
    price_adjustment_df.rename(columns={'支付價': '支付價格', '調整率': '價格調整率'}, inplace=True)
    
    st.dataframe(price_adjustment_df[['有效起日', '有效迄日', '支付價格', '價格調整率']], hide_index=True,
                 column_config={
                     "支付價格": st.column_config.NumberColumn(format="%.2f"),
                     "價格調整率": st.column_config.NumberColumn(format="%.2%")
                 })

    
    # ----------------------------------------------------------------------
    # 額外滿足：個別品項的英文名查詢 (實作 需求 1) 【新增】
    # ----------------------------------------------------------------------
    st.markdown("---")
    st.markdown("#### 額外資訊: 個別品項適應症 (英文名查詢)")
    
    # 查詢該品項的適應症 (依需求 1: 以英文名查詢)
    indication_for_individual_item = get_indication_by_english_name(target_english_name, indications_df)

    with st.expander(f"適應症 (以英文名: *{target_english_name}* 查詢)"):
        st.markdown(indication_for_individual_item, unsafe_allow_html=True)
else:
    if drug_code:
        st.warning(f"在藥價資料中找不到藥品代號: {drug_code}。")
