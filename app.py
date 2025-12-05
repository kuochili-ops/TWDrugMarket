
import pandas as pd
import streamlit as st
from datetime import datetime

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

def get_longest_price(price_df, code, year):
    df = price_df[price_df['藥品代號'] == code].copy()
    df['起'] = df['有效起日'].apply(parse_roc_date)
    df['迄'] = df['有效迄日'].apply(parse_roc_date)
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)
    df = df[(df['起'] <= end) & (df['迄'] >= start)]
    if df.empty:
        return 0.0
    df['區間起'] = df['起'].apply(lambda d: max(d, start))
    df['區間迄'] = df['迄'].apply(lambda d: min(d, end))
    df['天數'] = (df['區間迄'] - df['區間起']).dt.days + 1
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
def load_data():
    price1 = try_read_csv('Price_ATC1.csv')
    price2 = try_read_csv('Price_ATC2.csv')
    price_df = pd.concat([price1, price2], ignore_index=True)
    use_2022 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    use_2023 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    use_2024 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    price_df.columns = price_df.columns.str.strip()
    use_2022.columns = use_2022.columns.str.strip()
    use_2023.columns = use_2023.columns.str.strip()
    use_2024.columns = use_2024.columns.str.strip()
    return price_df, use_2022, use_2023, use_2024

try:
    price_df, use_2022, use_2023, use_2024 = load_data()
except Exception as e:
    st.error(f"資料讀取失敗，請確認檔案存在且編碼正確。錯誤訊息：{e}")
    st.stop()

def show_product_tables(sub_df_product, keyword):
    # 年度金額表
    result_product = []
    for _, row in sub_df_product.drop_duplicates('藥品代號').iterrows():
        code = row['藥品代號']
        name_en = row['藥品英文名稱']
        name_zh = row['藥品中文名稱']
        ingredient = row['成分']
        vendor = row['藥商']
        atc = row['ATC代碼']
        amt22, _, _ = calc_annual_payment(price_df, use_2022, code, 2022)
        amt23, _, _ = calc_annual_payment(price_df, use_2023, code, 2023)
        amt24, _, _ = calc_annual_payment(price_df, use_2024, code, 2024)
        result_product.append({
            '藥品代號': code,
            '藥品英文名稱': name_en,
            '藥品中文名稱': name_zh,
            '成分': ingredient,
            '藥商': vendor,
            '2022支付金額': amt22,
            '2023支付金額': amt23,
            '2024支付金額': amt24,
            'ATC代碼': atc
        })
    df_product = pd.DataFrame(result_product)
    df_product.index = range(1, len(df_product)+1)
    st.subheader(f"{keyword.upper()} 不同規格產品各年度支付金額")
    st.dataframe(df_product[['藥品代號','藥品英文名稱','藥品中文名稱','成分','藥商',
                             '2022支付金額','2023支付金額','2024支付金額']],
                 use_container_width=True,
                 column_config={
                     "2022支付金額": st.column_config.NumberColumn("2022支付金額", format="%.1f"),
                     "2023支付金額": st.column_config.NumberColumn("2023支付金額", format="%.1f"),
                     "2024支付金額": st.column_config.NumberColumn("2024支付金額", format="%.1f"),
                 }
    )
    # 各規格價格調整表
    for _, row in sub_df_product.drop_duplicates('藥品代號').iterrows():
        code = row['藥品代號']
        name_en = row['藥品英文名稱']
        df_price = price_df[price_df['藥品代號'] == code].copy()
        df_price['起'] = df_price['有效起日'].apply(parse_roc_date)
        df_price['迄'] = df_price['有效迄日'].apply(parse_roc_date)
        df_price['支付價'] = pd.to_numeric(df_price['支付價'], errors='coerce')
        df_price = df_price.sort_values('起')
        df_price['調整率'] = df_price['支付價'].pct_change().fillna(0) * 100
        st.subheader(f"{name_en} ({code}) 各時間階段藥價調整與調整率")
        st.dataframe(df_price[['起','迄','支付價','調整率']],
                     use_container_width=True,
                     column_config={
                         "支付價": st.column_config.NumberColumn("支付價", format="%.2f"),
                         "調整率": st.column_config.NumberColumn("調整率 (%)", format="%.2f"),
                     }
        )
    return df_product  # 回傳以便後續取得成分

def show_ingredient_tables(sub_df, keyword):
    result = []
    for _, row in sub_df.drop_duplicates('藥品代號').iterrows():
        code = row['藥品代號']
        name_en = row['藥品英文名稱']
        name_zh = row['藥品中文名稱']
        ingredient = row['成分']
        vendor = row['藥商']
        atc = row['ATC代碼']
        amt22, _, _ = calc_annual_payment(price_df, use_2022, code, 2022)
        amt23, _, _ = calc_annual_payment(price_df, use_2023, code, 2023)
        amt24, _, _ = calc_annual_payment(price_df, use_2024, code, 2024)
        result.append({
            '藥品代號': code,
            '藥品英文名稱': name_en,
            '藥品中文名稱': name_zh,
            '成分': ingredient,
            '藥商': vendor,
            '2022支付金額': amt22,
            '2023支付金額': amt23,
            '2024支付金額': amt24,
            'ATC代碼': atc
        })
    df = pd.DataFrame(result)
    df.index = range(1, len(df)+1)
    # 表1：各藥品支付金額
    st.subheader("各藥品支付金額")
    st.dataframe(df, use_container_width=True,
                 column_config={
                     "2022支付金額": st.column_config.NumberColumn("2022支付金額", format="%.1f"),
                     "2023支付金額": st.column_config.NumberColumn("2023支付金額", format="%.1f"),
                     "2024支付金額": st.column_config.NumberColumn("2024支付金額", format="%.1f"),
                 }
    )
    # 表2：同規格藥品加總
    summary = df.groupby('成分', as_index=False)[['2022支付金額','2023支付金額','2024支付金額']].sum()
    summary.index = range(1, len(summary)+1)
    st.subheader(f"{keyword.upper()} 同規格藥品各年度加總支付金額")
    st.dataframe(summary, use_container_width=True,
                 column_config={
                     "2022支付金額": st.column_config.NumberColumn("2022支付金額", format="%.1f"),
                     "2023支付金額": st.column_config.NumberColumn("2023支付金額", format="%.1f"),
                     "2024支付金額": st.column_config.NumberColumn("2024支付金額", format="%.1f"),
                 }
    )
    # 表3：同藥商加總
    df['主成分'] = df['成分'].str.split().str[0]
    summary_vendor = df.groupby(['主成分','藥商'], as_index=False)[['2022支付金額','2023支付金額','2024支付金額']].sum()
    summary_vendor = summary_vendor[['藥商','2022支付金額','2023支付金額','2024支付金額']]
    summary_vendor.index = range(1, len(summary_vendor)+1)
    st.subheader(f"{keyword.upper()} 同藥商產品各年度加總支付金額")
    st.dataframe(summary_vendor, use_container_width=True,
                 column_config={
                     "2022支付金額": st.column_config.NumberColumn("2022支付金額", format="%.1f"),
                     "2023支付金額": st.column_config.NumberColumn("2023支付金額", format="%.1f"),
                     "2024支付金額": st.column_config.NumberColumn("2024支付金額", format="%.1f"),
                 }
    )

# ------- 主成分/商品名查詢 -------
keyword = st.text_input('請輸入主成分或商品英文名稱（如 VENLAFAXINE 或 ARCOXIA）')

if keyword:
    # 先查成分名
    sub_df_ingredient = price_df[price_df['成分'].str.contains(keyword, case=False, na=False)]
    if not sub_df_ingredient.empty:
        show_ingredient_tables(sub_df_ingredient, keyword)
    else:
        # 再查商品名
        sub_df_product = price_df[price_df['藥品英文名稱'].str.contains(keyword, case=False, na=False)]
        if not sub_df_product.empty:
            df_product = show_product_tables(sub_df_product, keyword)
            # 取得所有商品的成分（去重）
            ingredient_list = df_product['成分'].dropna().unique().tolist()
            if ingredient_list:
                # 若有多個成分，讓使用者選擇
                if len(ingredient_list) == 1:
                    ingredient_name = ingredient_list[0]
                else:
                    ingredient_name = st.selectbox("此商品包含多個成分，請選擇要查詢的成分：", ingredient_list)
                if st.button(f"是否要以成分「{ingredient_name}」進行查詢？"):
                    sub_df_ingredient2 = price_df[price_df['成分'].str.contains(ingredient_name, case=False, na=False)]
                    if not sub_df_ingredient2.empty:
                        show_ingredient_tables(sub_df_ingredient2, ingredient_name)
                    else:
                        st.warning(f"查無成分「{ingredient_name}」的資料")
        else:
            st.warning(f"查無 {keyword} 的成分名或商品名資料")

# ------- 藥商查詢 -------
vendor_keyword = st.text_input('請輸入藥商名稱查詢（如 台灣羅氏、台灣默沙東等）')

if vendor_keyword:
    # 只查藥商欄位
    sub_df_vendor = price_df[price_df['藥商'].str.contains(vendor_keyword, case=False, na=False)]
    if not sub_df_vendor.empty:
        # 各產品各年度支付金額
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
        # 加總該藥商所有藥品的各年度支付金額
        total_22 = df_vendor['2022支付金額'].sum()
        total_23 = df_vendor['2023支付金額'].sum()
        total_24 = df_vendor['2024支付金額'].sum()
        st.subheader(f"{vendor_keyword} 所有藥品各年度支付金額加總")
        st.write(f"2022年：{total_22:,.1f} 元")
        st.write(f"2023年：{total_23:,.1f} 元")
        st.write(f"2024年：{total_24:,.1f} 元")
        # 選擇要顯示哪項藥品的藥價調整
        product_options = df_vendor['藥品英文名稱'] + " (" + df_vendor['藥品代號'] + ")"
        selected_product = st.selectbox("選擇要顯示藥價調整的藥品：", product_options)
        if selected_product:
            # 取得選擇的藥品代號
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

# ------- 最下面顯示白六的圖 -------
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手")
