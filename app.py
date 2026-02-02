import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品價量標註系統", layout="wide")

def try_read_csv(file, encodings=['utf-8-sig', 'utf-8', 'big5', 'cp950']):
    if not os.path.exists(file):
        return None
    for enc in encodings:
        try:
            # 讀取時不進行過濾，保留原始結構 (包含 Unnamed 欄位)
            df = pd.read_csv(file, encoding=enc)
            # 清理標題文字中的換行，但保留欄位位置
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

# --- 建立查找字典 (一萬多筆資料秒開的關鍵) ---
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
            idx = temp_df.groupby('藥品代號')['天數'].idxmax()
            final_prices = temp_df.loc[idx, ['藥品代號', '支付價']]
            price_map[year] = dict(zip(final_prices['藥品代號'], final_prices['支付價']))
    return price_map

def prepare_usage_dict(u22, u23, u24):
    usage_map = {2022: {}, 2023: {}, 2024: {}}
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            # 針對用量檔抓取 藥品代碼 欄位
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                usage_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    return usage_map

# --- 載入區 ---
@st.cache_data
def load_all_data():
    # 優先找 items.csv，找不到就找包含「重新分類」關鍵字的檔案
    m_file = 'items.csv'
    if not os.path.exists(m_file):
        for f in os.listdir('.'):
            if '重新分類' in f and f.endswith('.csv'):
                m_file = f
                break
    
    m_df = try_read_csv(m_file)
    p_df = pd.concat([try_read_csv('Price_ATC1.csv'), try_read_csv('Price_ATC2.csv')], ignore_index=True)
    u22 = try_read_csv('A21030000I-E41005-001 (2022).csv')
    u23 = try_read_csv('A21030000I-E41005-002 (2023).csv')
    u24 = try_read_csv('A21030000I-E41005-003 (2024).csv')
    return m_df, p_df, u22, u23, u24, m_file

# --- 主介面 ---
st.title("💊 健保藥品價量自動標註系統")

m_df, p_df, u22, u23, u24, final_master_name = load_all_data()

# 側邊欄狀態檢查
with st.sidebar:
    st.header("📂 檔案檢查")
    st.write(f"分類表 ({final_master_name}): {'✅' if m_df is not None else '❌'}")
    st.write(f"藥價表: {'✅' if p_df is not None else '❌'}")
    st.write(f"用量表: {'✅' if u22 is not None else '❌'}")

if m_df is not None:
    st.success(f"已連結分類表，準備標註 {len(m_df)} 筆藥品資料。")
    
    if st.button("🚀 執行全項標註 (附加 2022-2024 價量)", type="primary"):
        with st.status("正在串接大數據...", expanded=True):
            p_map = prepare_price_dict(p_df)
            u_map = prepare_usage_dict(u22, u23, u24)
            
            stats = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                res = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    p = p_map[yr].get(code, 0.0)
                    q = u_map[yr].get(code, 0.0)
                    res[f'{yr} 銷售量'] = q
                    res[f'{yr} 當年健保單價'] = p
                    res[f'{yr} 當年健保總價'] = round(p * q, 1)
                stats.append(res)
            
            # 使用 left join，確保 items.csv 的原始結構(含空欄位)完全保留
            final_df = pd.merge(m_df, pd.DataFrame(stats), on='藥品代碼', how='left')
            
        st.subheader("結果預覽")
        st.dataframe(final_df.head(100), use_container_width=True)
        
        csv_buffer = io.StringIO()
        final_df.to_csv(csv_buffer, index=False, encoding='utf-8-sig')
        st.download_button(
            label="📥 下載標註完成報表",
            data=csv_buffer.getvalue(),
            file_name=f"健保標註結果_{datetime.now().strftime('%m%d')}.csv",
            mime='text/csv'
        )
else:
    st.error("找不到分類表檔案。請確認資料夾內有 `items.csv`。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
