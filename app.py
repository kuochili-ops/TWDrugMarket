import pandas as pd
import streamlit as st
from datetime import datetime
import io
import os

# --- 基礎設定 ---
st.set_page_config(page_title="健保藥品價量標註系統", layout="wide")

def parse_roc_date(s):
    try:
        s = str(int(float(s)))
    except: return None
    if len(s) == 7:
        y, m, d = int(s[:3]) + 1911, int(s[3:5]), int(s[5:7])
    elif len(s) == 6:
        y, m, d = int(s[:2]) + 1911, int(s[2:4]), int(s[4:6])
    else: return None
    try: return datetime(y, m, d)
    except: return None

def local_loader(fn):
    if not os.path.exists(fn): return None
    for enc in ['utf-8-sig', 'utf-8', 'big5', 'cp950']:
        try:
            df = pd.read_csv(fn, encoding=enc)
            # 清理標題，但不改動資料順序
            df.columns = [str(c).replace('\n', ' ').strip() for c in df.columns]
            return df
        except: continue
    return None

# --- 建立字典 ---
def prepare_maps(p_df, u22, u23, u24):
    p_map = {2022: {}, 2023: {}, 2024: {}}
    u_map = {2022: {}, 2023: {}, 2024: {}}
    
    # 處理單價
    if p_df is not None:
        p_df['起'] = p_df['有效起日'].apply(parse_roc_date)
        p_df['迄'] = p_df['有效迄日'].apply(parse_roc_date)
        p_df['支付價'] = pd.to_numeric(p_df['支付價'], errors='coerce').fillna(0.0)
        for yr in [2022, 2023, 2024]:
            s_dt, e_dt = datetime(yr, 1, 1), datetime(yr, 12, 31)
            mask = (p_df['起'] <= e_dt) & (p_df['迄'] >= s_dt)
            tmp = p_df[mask].copy()
            if not tmp.empty:
                tmp['天數'] = (tmp[['迄', pd.Timestamp(e_dt)]].min(axis=1) - tmp[['起', pd.Timestamp(s_dt)]].max(axis=1)).dt.days + 1
                idx = tmp.groupby('藥品代號')['天數'].idxmax()
                p_map[yr] = dict(zip(tmp.loc[idx, '藥品代號'], tmp.loc[idx, '支付價']))

    # 處理用量
    for yr, df in zip([2022, 2023, 2024], [u22, u23, u24]):
        if df is not None:
            c_code = '藥品代碼' if '藥品代碼' in df.columns else df.columns[1]
            c_qty = '含包裹支付的醫令量_合計'
            if c_qty in df.columns:
                u_map[yr] = dict(zip(df[c_code].astype(str).str.strip(), pd.to_numeric(df[c_qty], errors='coerce').fillna(0.0)))
    
    return p_map, u_map

# --- 主介面 ---
st.title("💊 健保藥品分類量價自動標註系統")

# 側邊欄檢查本地背景檔
with st.sidebar:
    st.header("📂 背景數據檢查 (本地目錄)")
    p1, p2 = local_loader('Price_ATC1.csv'), local_loader('Price_ATC2.csv')
    p_df = pd.concat([p1, p2], ignore_index=True) if p1 is not None else None
    u22 = local_loader('A21030000I-E41005-001 (2022).csv')
    u23 = local_loader('A21030000I-E41005-002 (2023).csv')
    u24 = local_loader('A21030000I-E41005-003 (2024).csv')
    
    st.write(f"藥價數據: {'✅' if p_df is not None else '❌'}")
    st.write(f"2022用量: {'✅' if u22 is not None else '❌'}")
    st.write(f"2023用量: {'✅' if u23 is not None else '❌'}")
    st.write(f"2024用量: {'✅' if u24 is not None else '❌'}")

# 上傳 items.csv
up_file = st.file_uploader("📤 請上傳原始分類表 (items.csv)", type="csv")

if up_file:
    # 讀取時不清理 Unnamed 欄位，保留原始物理位置
    m_df = pd.read_csv(up_file, encoding='utf-8-sig')
    
    # 標題做基本換行處理，以便在 UI 顯示，但合併時會用原本的 column list
    raw_cols = m_df.columns.tolist()
    
    st.success(f"已讀取上傳檔案，共 {len(m_df)} 筆項目。")

    if st.button("🚀 執行全項標註 (精準對齊版)", type="primary"):
        with st.status("正在匹配量價數據...", expanded=True):
            p_map, u_map = prepare_maps(p_df, u22, u23, u24)
            
            # 建立附加欄位
            new_data = []
            for _, row in m_df.iterrows():
                code = str(row['藥品代碼']).strip()
                res = {'藥品代碼': code}
                for yr in [2022, 2023, 2024]:
                    p = p_map[yr].get(code, 0.0)
                    q = u_map[yr].get(code, 0.0)
                    res[f'{yr} 銷售量'] = q
                    res[f'{yr} 當年健保單價'] = p
                    res[f'{yr} 當年健保總價'] = round(p * q, 1)
                new_data.append(res)
            
            # 以「藥品代碼」為 Key 合併
            # 注意：這裡使用 left join 確保原始表格的所有 Unnamed 欄位順序完全不變
            stats_df = pd.DataFrame(new_data)
            final_df = pd.merge(m_df, stats_df, on='藥品代碼', how='left')
            
        st.subheader("結果預覽")
        st.dataframe(final_df.head(50))
        
        # 下載
        out = io.StringIO()
        final_df.to_csv(out, index=False, encoding='utf-8-sig')
        st.download_button("📥 下載對齊後之完整報表", out.getvalue(), 
                           file_name=f"標註結果_{datetime.now().strftime('%m%d')}.csv", 
                           mime='text/csv')
else:
    st.info("請上傳您的原始 items.csv 開始運算。")

st.divider()
st.image("S__38543373.jpg", caption="白六-健保資料查詢小幫手", width=100)
