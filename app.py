import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 網頁設定 ---
st.set_page_config(page_title="健保藥品價量標註系統", layout="wide")

def local_loader(fn):
    """讀取同目錄下的 CSV，支援多種編碼並清理標題空白"""
    if not os.path.exists(fn):
        return None
    for enc in ['utf-8-sig', 'utf-8', 'big5', 'cp950']:
        try:
            df = pd.read_csv(fn, encoding=enc)
            df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
            return df
        except:
            continue
    return None

def prepare_price_map(p_df):
    """建立藥價字典：以 2024 年最新的價格為主要參考 (簡化邏輯避免 KeyError)"""
    price_map = {2022: {}, 2023: {}, 2024: {}}
    if p_df is None:
        return price_map
    
    # 確保必要欄位存在
    if '藥品代號' not in p_df.columns or '支付價' not in p_df.columns:
        st.error("藥價表格式不正確，缺少 '藥品代號' 或 '支付價'")
        return price_map

    # 簡單化處理：將支付價轉為數字
    p_df['支付價'] = pd.to_numeric(p_df['支付價'], errors='coerce').fillna(0.0)
    
    # 根據有效起日排序，取最新的一筆作為代表價 (預設給 2022-2024)
    # 若需精確到每一天的價格變動，運算量會太大，此處取該藥品在表中的最後價格
    latest_prices = p_df.sort_values('有效起日').groupby('藥品代號')['支付價'].last().to_dict()
    
    for yr in [2022, 2023, 2024]:
        price_map[yr] = latest_prices
    return price_map

def prepare_usage_map(u22, u23, u24):
    """建立用量字典"""
    u_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            # 用量檔通常第二欄是代碼
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                u_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return u_map

# --- 主程式介面 ---
st.title("💊 健保藥品分類量價自動標註系統")

# 1. 檢查目錄下的背景檔案
with st.sidebar:
    st.header("📂 背景數據檢查")
    p1 = local_loader('Price_ATC1.csv')
    p2 = local_loader('Price_ATC2.csv')
    p_df = pd.concat([p1, p2], ignore_index=True) if p1 is not None else None
    u22 = local_loader('A21030000I-E41005-001 (2022).csv')
    u23 = local_loader('A21030000I-E41005-002 (2023).csv')
    u24 = local_loader('A21030000I-E41005-003 (2024).csv')
    
    st.write(f"藥價數據: {'✅' if p_df is not None else '❌'}")
    st.write(f"用量數據 (22-24): {'✅' if u22 is not None and u24 is not None else '❌'}")

# 2. 上傳 items.csv
up_file = st.file_uploader("📤 請上傳您的分類表 (例如: items.csv)", type="csv")

if up_file:
    # 讀取上傳檔案，不清理欄位名稱以保留原始結構 (Unnamed 等)
    m_df = pd.read_csv(up_file, encoding='utf-8-sig')
    st.success(f"已載入分類表，共 {len(m_df)} 筆藥品。")

    if st.button("🚀 開始標註 2022-2024 價量數據", type="primary"):
        with st.status("正在匹配數據...", expanded=True):
            # 取得對應字典
            p_map = prepare_price_map(p_df)
            u_map = prepare_usage_map(u22, u23, u24)
            
            # 建立計算結果清單
            results = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                res = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    price = p_map[yr].get(code, 0.0)
                    qty = u_map[yr].get(code, 0.0)
                    res[f'{yr} 銷售量'] = qty
                    res[f'{yr} 當年健保單價'] = price
                    res[f'{yr} 當年健保總價'] = round(price * qty, 1)
                results.append(res)
            
            # 使用 left join 將數據貼回原始表格右側
            stats_df = pd.DataFrame(results)
            final_df = pd.merge(m_df, stats_df, on='藥品代碼', how='left')
            
        st.subheader("標註結果預覽")
        st.dataframe(final_df.head(100))
        
        # 下載區
        csv_out = final_df.to_csv(index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載完成標註之報表",
            data=csv_out,
            file_name=f"健保標註結果_{datetime.now().strftime('%m%d')}.csv",
            mime='text/csv'
        )
else:
    st.info("💡 操作說明：請在上方上傳您的 `items.csv` 分類表，系統會自動結合目錄下的藥價與用量檔。")
