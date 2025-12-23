
import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np

# 讀取適應症資料
indication_df = pd.read_csv('37_2.csv')
indication_df.columns = indication_df.columns.str.strip()
indication_map = dict(zip(
    indication_df['英文品名'].astype(str).str.strip(),
    indication_df['適應症'].astype(str).str.strip()
))

def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    for enc in encodings:
        try:
            df = pd.read_csv(file, encoding=enc)
            df.columns = df.columns.str.strip()
            return df
        except Exception:
            continue
    raise ValueError(f"{file} 無法用常見編碼讀取，請確認檔案格式。")

def parse_roc_date(s):
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

@st.cache_data
def load_atc4_to_subclass():
    try:
        atc4_df = try_read_csv('ATC4_Subclass_Map.csv')
        return dict(zip(
            atc4_df['ATC4代碼'].astype(str).str.strip().str[:5],
            atc4_df['化學/藥理學子分類(英文)'].astype(str).str.strip()
        ))
    except Exception:
        st.error("ATC4_Subclass_Map.csv 載入失敗，將使用空白字典。")
        return {}

@st.cache_data
def load_atc5_to_ingredient():
    try:
        atc5_df = try_read_csv('ATC5_Ingredient_Map.csv')
        return dict(zip(
            atc5_df['ATC代碼'].astype(str).str.strip().str[:7],
            atc5_df['成分'].astype(str).str.strip()
        ))
    except Exception:
        st.error("ATC5_Ingredient_Map.csv 載入失敗，將使用空白字典。")
        return {}

atc4_to_subclass = load_atc4_to_subclass()
atc5_to_ingredient = load_atc5_to_ingredient()

def get_longest_price(price_df, code, year):
    df = price_df[price_df['藥品代號'] == code].copy()
    df['起'] = df['有效起日'].apply(parse_roc_date)
    df['迄'] = df['有效迄日'].apply(parse_roc_date)
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    valid_mask = ((df['起'] <= end) & (df['迄'] >= start))
    df = df[valid_mask].copy()
    if df.empty:
        return 0.0
    df['區間起'] = df['起'].apply(lambda d: max(d, start) if pd.notna(d) else start)
    df['區間迄'] = df['迄'].apply(lambda d: min(d, end) if pd.notna(d) else end)
    df['天數'] = (df['區間迄'] - df['區間起']).dt.days + 1
    df = df[df['天數'] > 0].copy()
    if df.empty:
        return 0.0
    row = df.loc[df['天數'].idxmax()]
    try:
        price = float(row['支付價'])
    except Exception:
        price = 0.0
    return price

def calc_annual_payment(price_df, use_df, code, year):
    price = get_longest_price(price_df, code, year)
    qty = 0.0
    if not use_df.empty:
        use_df.columns = use_df.columns.str.strip()
        if '藥品代碼' in use_df.columns and '含包裹支付的醫令量_合計' in use_df.columns:
            row = use_df[use_df['藥品代碼'] == code]
            if not row.empty:
                qty = row['含包裹支付的醫令量_合計'].values[0]
    try:
        qty = float(qty)
    except Exception:
        qty = 0.0
    amt = price * qty
    return amt, price, qty

st.title("健保藥品 2022~2024 年度價量分析")
@st.cache_data
def load_main_data():
    price1 = try_read_csv('Price_ATC1.csv')
    price2 = try_read_csv('Price_ATC2.csv')
    price_df = pd.concat([price1, price2], ignore_index=True)
    price_df['ATC代碼'] = price_df['ATC代碼'].astype(str).str.strip().fillna('')
    price_df['ATC5'] = price_df['ATC代碼'].str[:7]
    price_df['ATC4'] = price_df['ATC代碼'].str[:5]
    use_2022 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    use_2023 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    use_2024 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    price_df.columns = price_df.columns.str.strip()
    use_2022.columns = use_2022.columns.str.strip()
    use_2023.columns = use_2023.columns.str.strip()
    use_2024.columns = use_2024.columns.str.strip()
    return price_df, use_2022, use_2023, use_2024

try:
    price_df, use_2022, use_2023, use_2024 = load_main_data()
except Exception as e:
    st.error(f"資料讀取失敗，請確認檔案存在且編碼正確。錯誤訊息：{e}")
    st.stop()

# ------- 主成分/商品名查詢 -------
df_product = pd.DataFrame()
sub_df_ingredient = pd.DataFrame()
df_ingredient_table = pd.DataFrame()
keyword = st.text_input('請輸入主成分或商品英文名稱（如 VENLAFAXINE 或 ARCOXIA）')
if keyword:
    sub_df_ingredient = price_df[price_df['成分'].str.contains(keyword, case=False, na=False)].copy()
    if not sub_df_ingredient.empty:
        df_ingredient_table = show_ingredient_tables(sub_df_ingredient, keyword)
        # --- 成分查詢後，補回商品選擇與藥價調整 ---
        product_options = sub_df_ingredient['藥品英文名稱'] + " (" + sub_df_ingredient['藥品代號'] + ")"
        selected_product = st.selectbox("選擇要顯示藥價調整的藥品：", product_options, key="ingredient_price_select")
        if selected_product:
            selected_code = selected_product.split('(')[-1].replace(')', '').strip()
            df_price = price_df[price_df['藥品代號'] == selected_code].copy()
            df_price['起'] = df_price['有效起日'].apply(parse_roc_date)
            df_price['迄'] = df_price['有效迄日'].apply(parse_roc_date)
            df_price['支付價'] = pd.to_numeric(df_price['支付價'], errors='coerce')
            df_price = df_price.sort_values('起')
            df_price['調整率'] = df_price['支付價'].pct_change().fillna(0) * 100
            st.subheader(f"{selected_product} 各時間階段藥價調整與調整率")
            st.dataframe(df_price[['起','迄','支付價','調整率']],
                use_container_width=True,
                column_config={
                    "支付價": st.column_config.NumberColumn("支付價", format="%.2f"),
                    "調整率": st.column_config.NumberColumn("調整率 (%)", format="%.2f"),
                }
            )
    else:
        sub_df_product = price_df[price_df['藥品英文名稱'].str.contains(keyword, case=False, na=False)].copy()
        if not sub_df_product.empty:
            df_product = show_product_tables(sub_df_product, keyword)
            ingredient_list = df_product['成分'].dropna().unique().tolist()
            if ingredient_list:
                if len(ingredient_list) == 1:
                    ingredient_name = ingredient_list[0]
                else:
                    ingredient_name = st.selectbox("此商品包含多個成分，請選擇要查詢的成分：", ingredient_list, key="select_ing_1")
                if st.button(f"是否要以成分「{ingredient_name}」進行查詢？", key="atc_search_button"):
                    sub_df_ingredient = price_df[price_df['成分'].str.contains(ingredient_name, case=False, na=False)].copy()
                    if not sub_df_ingredient.empty:
                        df_ingredient_table = show_ingredient_tables(sub_df_ingredient, ingredient_name)
                        # --- 成分查詢後，補回商品選擇與藥價調整 ---
                        product_options = sub_df_ingredient['藥品英文名稱'] + " (" + sub_df_ingredient['藥品代號'] + ")"
                        selected_product = st.selectbox("選擇要顯示藥價調整的藥品：", product_options, key="ingredient_price_select2")
                        if selected_product:
                            selected_code = selected_product.split('(')[-1].replace(')', '').strip()
                            df_price = price_df[price_df['藥品代號'] == selected_code].copy()
                            df_price['起'] = df_price['有效起日'].apply(parse_roc_date)
                            df_price['迄'] = df_price['有效迄日'].apply(parse_roc_date)
                            df_price['支付價'] = pd.to_numeric(df_price['支付價'], errors='coerce')
                            df_price = df_price.sort_values('起')
                            df_price['調整率'] = df_price['支付價'].pct_change().fillna(0) * 100
                            st.subheader(f"{selected_product} 各時間階段藥價調整與調整率")
                            st.dataframe(df_price[['起','迄','支付價','調整率']],
                                use_container_width=True,
                                column_config={
                                    "支付價": st.column_config.NumberColumn("支付價", format="%.2f"),
                                    "調整率": st.column_config.NumberColumn("調整率 (%)", format="%.2f"),
                                }
                            )
                    else:
                        st.warning(f"查無成分「{ingredient_name}」的資料")
            else:
                st.warning(f"查無 {keyword} 的成分名或商品名資料")
``

# ------- 藥商查詢功能 -------
vendor_keyword = st.text_input('請輸入藥商名稱查詢（如 台灣羅氏、台灣默沙東等）*')
if vendor_keyword:
    sub_df_vendor = price_df[price_df['藥商'].str.contains(vendor_keyword, case=False, na=False)]
    if not sub_df_vendor.empty:
        result_vendor = []
        for _, row in sub_df_vendor.drop_duplicates('藥品代號').iterrows():
            code = row['藥品代號']
            name_en = row['藥品英文名稱']
            name_zh = row['藥品中文名稱']
            ingredient = row['成分']
            amt22, _, _ = calc_annual_payment(price_df, use_2022, code, 2022)
            amt23, _, _ = calc_annual_payment(price_df, use_2023, code, 2023)
            amt24, _, _ = calc_annual_payment(price_df, use_2024, code, 2024)
            result_vendor.append({
                '藥品代號': code,
                '藥品英文名稱': name_en,
                '藥品中文名稱': name_zh,
                '成分': ingredient,
                '2022支付金額': amt22,
                '2023支付金額': amt23,
                '2024支付金額': amt24
            })
        df_vendor = pd.DataFrame(result_vendor)
        df_vendor.index = range(1, len(df_vendor)+1)
        st.subheader(f"{vendor_keyword} 各產品各年度支付金額")
        st.dataframe(df_vendor, use_container_width=True,
            column_config={
                "2022支付金額": st.column_config.NumberColumn("2022支付金額", format="%.1f"),
                "2023支付金額": st.column_config.NumberColumn("2023支付金額", format="%.1f"),
                "2024支付金額": st.column_config.NumberColumn("2024支付金額", format="%.1f"),
            }
        )
        total_22 = df_vendor['2022支付金額'].sum()
        total_23 = df_vendor['2023支付金額'].sum()
        total_24 = df_vendor['2024支付金額'].sum()
        st.subheader(f"{vendor_keyword} 所有藥品各年度支付金額加總")
        st.write(f"2022年：{total_22:,.1f} 元")
        st.write(f"2023年：{total_23:,.1f} 元")
        st.write(f"2024年：{total_24:,.1f} 元")
        product_options = df_vendor['藥品英文名稱'] + " (" + df_vendor['藥品代號'] + ")"
        selected_product = st.selectbox("選擇要顯示藥價調整的藥品：", product_options, key="vendor_price_select")
        if selected_product:
            selected_code = selected_product.split('(')[-1].replace(')', '').strip()
            df_price = price_df[price_df['藥品代號'] == selected_code].copy()
            df_price['起'] = df_price['有效起日'].apply(parse_roc_date)
            df_price['迄'] = df_price['有效迄日'].apply(parse_roc_date)
            df_price['支付價'] = pd.to_numeric(df_price['支付價'], errors='coerce')
            df_price = df_price.sort_values('起')
            df_price['調整率'] = df_price['支付價'].pct_change().fillna(0) * 100
            st.subheader(f"{selected_product} 各時間階段藥價調整與調整率")
            st.dataframe(df_price[['起','迄','支付價','調整率']],
                use_container_width=True,
                column_config={
                    "支付價": st.column_config.NumberColumn("支付價", format="%.2f"),
                    "調整率": st.column_config.NumberColumn("調整率 (%)", format="%.2f"),
                }
            )
    else:
        st.warning(f"查無藥商「{vendor_keyword}」的資料")

# ------- 白六圖片 -------
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
