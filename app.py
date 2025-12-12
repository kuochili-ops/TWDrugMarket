import pandas as pd
import streamlit as st
from datetime import datetime
import numpy as np # 確保 numpy 存在

# 讀取適應症資料
indication_df = pd.read_csv('37_2.csv')
indication_df.columns = indication_df.columns.str.strip()
indication_map = dict(zip(
    indication_df['英文品名'].astype(str).str.strip(),
    indication_df['適應症'].astype(str).str.strip()
))

# --- 數據讀取工具函式 ---

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

# --- 載入 ATC 輔助資料 ---

@st.cache_data
def load_atc4_to_subclass():
    try:
        atc4_df = try_read_csv('ATC4_Subclass_Map.csv')
        return dict(zip(
            atc4_df['ATC4代碼'].astype(str).str.strip(),
            atc4_df['化學/藥理學子分類(英文)'].astype(str).str.strip()
        ))
    except Exception:
        # 為了讓程式碼能跑，遇到錯誤時使用空的字典
        st.error("ATC4_Subclass_Map.csv 載入失敗，將使用空白字典。")
        return {}

@st.cache_data
def load_atc5_to_ingredient():
    try:
        atc5_df = try_read_csv('ATC5_Ingredient_Map.csv')
        return dict(zip(
            atc5_df['ATC代碼'].astype(str).str.strip(),
            atc5_df['成分'].astype(str).str.strip()
        ))
    except Exception:
        # 為了讓程式碼能跑，遇到錯誤時使用空的字典
        st.error("ATC5_Ingredient_Map.csv 載入失敗，將使用空白字典。")
        return {}

atc4_to_subclass = load_atc4_to_subclass()
atc5_to_ingredient = load_atc5_to_ingredient()

# --- 核心計算函式 ---

def get_longest_price(price_df, code, year):
    df = price_df[price_df['藥品代號'] == code].copy()
    # 這裡的邏輯必須處理可能為 NaN 的狀況，以確保 .copy() 後不報錯
    df['起'] = df['有效起日'].apply(parse_roc_date)
    df['迄'] = df['有效迄日'].apply(parse_roc_date)
    start = datetime(year, 1, 1)
    end = datetime(year, 12, 31)

    # 處理無效起迄日的情形 (假設沒有起迄日則永遠有效)
    valid_mask = ((df['起'] <= end) & (df['迄'] >= start))
    df = df[valid_mask].copy()

    if df.empty:
        return 0.0
    
    # 處理 NaN 的日期欄位
    df['區間起'] = df['起'].apply(lambda d: max(d, start) if pd.notna(d) else start)
    df['區間迄'] = df['迄'].apply(lambda d: min(d, end) if pd.notna(d) else end)
    df['天數'] = (df['區間迄'] - df['區間起']).dt.days + 1
    
    # 確保天數大於 0
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

# --- 載入主要資料 ---

st.title("健保藥品 2022~2024 年度價量分析")

@st.cache_data
def load_main_data():
    price1 = try_read_csv('Price_ATC1.csv')
    price2 = try_read_csv('Price_ATC2.csv')
    price_df = pd.concat([price1, price2], ignore_index=True)
    
    # 確保 ATC 代碼欄位存在
    price_df['ATC代碼'] = price_df['ATC代碼'].astype(str).str.strip().fillna('')
    price_df['ATC5'] = price_df['ATC代碼'].str[:5] # 新增 ATC5 欄位方便篩選
    price_df['ATC4'] = price_df['ATC代碼'].str[:4] # 新增 ATC4 欄位方便篩選
    
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

# --- 顯示表格函式 ---

def show_product_tables(sub_df_product, keyword):
    # 年度金額表
    result_product = []
    # 使用 drop_duplicates('藥品代號') 確保只計算一次
    for _, row in sub_df_product.sort_values('有效起日', ascending=False).drop_duplicates('藥品代號').iterrows():
        code = row['藥品代號']
        name_en = row['藥品英文名稱']
        name_zh = row['藥品中文名稱']
        ingredient = row['成分']
        vendor = row['藥商']
        atc = row['ATC代碼']
        # 確保價格和使用量計算使用正確的 price_df, use_df
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
    for _, row in sub_df_product.sort_values('有效起日', ascending=False).drop_duplicates('藥品代號').iterrows():
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
    return df_product # 回傳以便後續取得成分

def show_ingredient_tables(sub_df, keyword):
    result = []
    # 使用 drop_duplicates('藥品代號') 確保只計算一次
    for _, row in sub_df.sort_values('有效起日', ascending=False).drop_duplicates('藥品代號').iterrows():
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
    return df # 回傳包含 ATC 代碼的 DataFrame

# ------- 主成分/商品名查詢 -------
# 【修正點 1】初始化用於跨區塊使用的 DataFrame，避免 NameError
df_product = pd.DataFrame() 
sub_df_ingredient = pd.DataFrame()
df_ingredient_table = pd.DataFrame() # 新增：用於保存成分查詢的結果 DataFrame

keyword = st.text_input('請輸入主成分或商品英文名稱（如 VENLAFAXINE 或 ARCOXIA）')

if keyword:
    # 先查成分名
    sub_df_ingredient = price_df[price_df['成分'].str.contains(keyword, case=False, na=False)].copy()
    if not sub_df_ingredient.empty:
        # 【修正點 2】捕獲成分查詢的結果 DataFrame
        df_ingredient_table = show_ingredient_tables(sub_df_ingredient, keyword) 
    else:
        # 再查商品名
        sub_df_product = price_df[price_df['藥品英文名稱'].str.contains(keyword, case=False, na=False)].copy()
        if not sub_df_product.empty:
            df_product = show_product_tables(sub_df_product, keyword)
            # 取得所有商品的成分（去重）
            ingredient_list = df_product['成分'].dropna().unique().tolist()
            if ingredient_list:
                # 若有多個成分，讓使用者選擇
                if len(ingredient_list) == 1:
                    ingredient_name = ingredient_list[0]
                else:
                    ingredient_name = st.selectbox("此商品包含多個成分，請選擇要查詢的成分：", ingredient_list, key="select_ing_1") # 加入 key
                
                if st.button(f"是否要以成分「{ingredient_name}」進行查詢？", key="atc_search_button"):
                    # 這是次要的成分查詢
                    sub_df_ingredient = price_df[price_df['成分'].str.contains(ingredient_name, case=False, na=False)].copy()
                    if not sub_df_ingredient.empty:
                        # 【修正點 2】捕獲次要成分查詢的結果 DataFrame
                        df_ingredient_table = show_ingredient_tables(sub_df_ingredient, ingredient_name)
                    else:
                        st.warning(f"查無成分「{ingredient_name}」的資料")
        else:
            st.warning(f"查無 {keyword} 的成分名或商品名資料")


# ===== 商品適應症查詢功能 =====

with st.expander("商品適應症查詢", expanded=False):
    product_names = []
    # 【修正點 3】檢查 df_product (商品查詢結果) 或 df_ingredient_table (成分查詢結果)
    if not df_product.empty:
        # 使用商品名查詢結果的英文名
        product_names = df_product['藥品英文名稱'].dropna().unique().tolist()
    elif not df_ingredient_table.empty:
        # 使用成分查詢結果的英文名 (df_ingredient_table 是 show_ingredient_tables 回傳的 DataFrame)
        product_names = df_ingredient_table['藥品英文名稱'].dropna().unique().tolist()
    else:
        st.info("請先使用上方欄位進行 **商品名** 或 **主成分** 查詢，結果將顯示於此選單。")


    if product_names:
        # 確保選單 key 獨立
        selected_product = st.selectbox("選擇商品以查看適應症：", product_names, key="indication_select")
        if selected_product:
            # 由於 indication_map 是 英文品名: 適應症 (字串) 的對應
            indication = indication_map.get(selected_product.strip(), "查無適應症資料")
            st.write(f"**{selected_product}** 的適應症：")
            # 【修正點 4】使用 markdown 配合 <br> 標籤來顯示多行的適應症
            st.markdown(indication.replace('\n', '<br>'), unsafe_allow_html=True)

# ------- 藥商查詢 -------
# ... (此部分保持不變) ...
vendor_keyword = st.text_input('請輸入藥商名稱查詢（如 台灣羅氏、台灣默沙東等）*Serena 要的')

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
        selected_product = st.selectbox("選擇要顯示藥價調整的藥品：", product_options, key="vendor_price_select")
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


# ===== 延伸分析函式 (ATC 占比 - 尚未啟動) =====
def show_top_atc5_and_products(atc_code_4):
    subclass_name = atc4_to_subclass.get(atc_code_4, '')
    st.subheader(f"該 ATC4 分類 ({atc_code_4} {subclass_name}) 中各年度金額與佔比最高的前三 ATC5")
    # 使用新欄位 ATC4
    sub_df_atc4 = price_df[price_df['ATC4'] == atc_code_4].copy()

    # 計算每個 ATC5 的年度金額
    atc5_summary = []
    # 使用新欄位 ATC5 進行分組
    for atc5, group in sub_df_atc4.groupby('ATC5'):
        # 確保 ATC5 是有效的 5碼
        if len(atc5) != 5: continue 
            
        # 這裡的 apply 效率可能較低，但為保持邏輯一致性，暫時保留
        amt22 = group.apply(lambda r: calc_annual_payment(price_df, use_2022, r['藥品代號'], 2022)[0], axis=1).sum()
        amt23 = group.apply(lambda r: calc_annual_payment(price_df, use_2023, r['藥品代號'], 2023)[0], axis=1).sum()
        amt24 = group.apply(lambda r: calc_annual_payment(price_df, use_2024, r['藥品代號'], 2024)[0], axis=1).sum()
        ingredient_name = atc5_to_ingredient.get(atc5, '')
        atc5_summary.append({'ATC5代碼': atc5, '主成分/規格': ingredient_name, '2022支付金額': amt22, '2023支付金額': amt23, '2024支付金額': amt24})

    df_atc5 = pd.DataFrame(atc5_summary)
    
    # 移除支付金額全為 0 的列
    df_atc5 = df_atc5[(df_atc5['2022支付金額'] > 0) | (df_atc5['2023支付金額'] > 0) | (df_atc5['2024支付金額'] > 0)].copy()

    for year in [2022, 2023, 2024]:
        year_col = f'{year}支付金額'
        st.write(f"### {year} 年度 Top 3 ATC5")
        
        # 總金額
        total_amt = df_atc5[year_col].sum()
        
        df_sorted = df_atc5.sort_values(year_col, ascending=False).head(3).copy() # 使用 copy
        df_sorted[f'{year}佔比(%)'] = (df_sorted[year_col] / total_amt * 100).round(2)
        
        st.dataframe(df_sorted.rename(columns={year_col: f'{year}金額'}), 
                     column_config={
                          f'{year}金額': st.column_config.NumberColumn(f'{year}金額', format="%.1f"),
                     })

        # 顯示每個 Top ATC5 中金額最高的商品
        for _, row in df_sorted.iterrows():
            atc5_code = row['ATC5代碼']
            # 篩選出該 ATC5 的所有藥品
            sub_df_atc5_products = sub_df_atc4[sub_df_atc4['ATC5'] == atc5_code].copy() # 使用 copy
            
            # 計算該年度各產品的金額
            sub_df_atc5_products['年度金額'] = sub_df_atc5_products.apply(
                lambda r: calc_annual_payment(price_df, use_2022 if year==2022 else (use_2023 if year==2023 else use_2024), r['藥品代號'], year)[0], axis=1
            )
            
            # 找到金額最高的商品
            if not sub_df_atc5_products.empty:
                top_product_row = sub_df_atc5_products.sort_values('年度金額', ascending=False).iloc[0]
                st.write(f"**ATC5 {atc5_code}** 中最高金額商品：{top_product_row['藥品英文名稱']} ({top_product_row['藥品代號']})，金額：**{top_product_row['年度金額']:.1f}** 元")
            else:
                st.write(f"**ATC5 {atc5_code}** 查無 {year} 年度支付資料。")


# ===== 新增功能：ATC 金額占比分析（商品名查詢） =====
# 由於 df_product 和 sub_df_ingredient 已在上方初始化，故不會有 NameError
if not df_product.empty:
    enable_atc_calc_product = st.checkbox("啟動 ATC 金額占比計算（商品名查詢）")
    if enable_atc_calc_product:
        # 確保 atc_code_5 是 5碼
        atc_code_5 = df_product['ATC代碼'].dropna().str[:5].iloc[0] 
        atc_code_4 = atc_code_5[:4]
        subclass_name = atc4_to_subclass.get(atc_code_4, '')
        st.subheader("ATC 金額占比分析（商品名）")
        st.write(f"第五層 ATC Code：**{atc_code_5}** {atc5_to_ingredient.get(atc_code_5, '')}")
        st.write(f"第四層 ATC Code：**{atc_code_4}** {subclass_name}")
        show_top_atc5_and_products(atc_code_4)


# ===== 新增功能：ATC 金額占比分析（主成分查詢） =====
if not sub_df_ingredient.empty:
    enable_atc_calc_ing = st.checkbox("啟動 ATC 金額占比計算（主成分查詢）", key="atc_ing_checkbox")
    if enable_atc_calc_ing:
        # 確保 atc_code_5 是 5碼
        atc_code_5 = sub_df_ingredient['ATC代碼'].dropna().str[:5].iloc[0]
        atc_code_4 = atc_code_5[:4]
        subclass_name = atc4_to_subclass.get(atc_code_4, '')
        st.subheader("ATC 金額占比分析（主成分）")
        st.write(f"第五層 ATC Code：**{atc_code_5}** {atc5_to_ingredient.get(atc_code_5, '')}")
        st.write(f"第四層 ATC Code：**{atc_code_4}** {subclass_name}")
        show_top_atc5_and_products(atc_code_4)
