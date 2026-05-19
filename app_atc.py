import pandas as pd

# 1. 讀取並整合藥品 ATC 與價格檔
print("正在載入藥品對照表...")
price_cols = ['藥品代號', '藥品中文名稱', '成分', '支付價', 'ATC代碼']
price1 = pd.read_csv('Price_ATC1.csv', usecols=price_cols)
price2 = pd.read_csv('Price_ATC2.csv', usecols=price_cols)
price_df = pd.concat([price1, price2], ignore_index=True)

# 轉換型態並清理重複值，保留中位數價格與第一個 ATC
price_df['支付價'] = pd.to_numeric(price_df['支付價'], errors='coerce')
price_df = price_df.dropna(subset=['藥品代號', 'ATC代碼'])
price_clean = price_df.groupby('藥品代號').agg({
    '支付價': 'median', 
    'ATC代碼': 'first',
    '藥品中文名稱': 'first',
    '成分': 'first'
}).reset_index()

# 2. 篩選出 ATC 為 N (神經系統) 的藥品
price_n_class = price_clean[price_clean['ATC代碼'].str.startswith('N', na=False)].copy()

# 3. 分塊讀取健保申報檔案 (以 2022 年為例，您可自行更換為 2023 或 2024)
declaration_file = 'A21030000I-E41005-001 (2022).csv'
print(f"正在分析 {declaration_file} 中的神經系統藥物...")

# 建立字典來存儲 ATC3 (子類) 與 ATC5 (具體成分) 的總金額
atc3_amounts = {}
atc5_amounts = {}

for chunk in pd.read_csv(declaration_file, chunksize=50000):
    chunk = chunk.dropna(subset=['藥品代碼', '含包裹支付的醫令量_合計'])
    chunk['含包裹支付的醫令量_合計'] = pd.to_numeric(chunk['含包裹支付的醫令量_合計'], errors='coerce').fillna(0)
    
    # 僅與 N 類藥品進行內連鎖 (Inner Join)
    merged = pd.merge(chunk, price_n_class, left_on='藥品代碼', right_on='藥品代號', how='inner')
    merged['給付總金額'] = merged['含包裹支付的醫令量_合計'] * merged['支付價']
    
    # 擷取 ATC 第三碼 (子分類，如 N02B, N05A) 與 第五碼 (化學成分)
    merged['ATC_L3'] = merged['ATC代碼'].str[:4]
    merged['ATC_L5'] = merged['ATC代碼'].str[:7]
    
    # 累加至子分類字典
    for code, amt in merged.groupby('ATC_L3')['給付總金額'].sum().items():
        atc3_amounts[code] = atc3_amounts.get(code, 0) + amt
        
    # 累加至具體成分字典 (同時帶上成分名稱方便閱讀)
    for _, row in merged.iterrows():
        code5 = row['ATC_L5']
        if code5 not in atc5_amounts:
            atc5_amounts[code5] = {'金額': 0, '成分': row['成分'], '中文名': row['藥品中文名稱']}
        atc5_amounts[code5]['金額'] += row['給付總金額']

# 4. 建立 ATC3 中文對照表
atc3_mapping = {
    'N01A': '全身麻醉劑 (Anesthetics, systemic)',
    'N02A': '鴉片類止痛藥 (Opioids)',
    'N02B': '其他止痛解熱劑 (Other analgesics and antipyretics，如普拿疼)',
    'N03A': '抗癲癇藥 (Antiepileptics)',
    'N04A': '抗膽鹼精神異常藥 (Anticholinergic agents)',
    'N04B': '多巴胺精神異常藥 (Dopaminergic agents，如巴金森氏症藥)',
    'N05A': '抗精神病藥 (Antipsychotics)',
    'N05B': '抗焦慮藥 (Anxiolytics)',
    'N05C': '安眠藥及鎮靜劑 (Hypnotics and sedatives)',
    'N06A': '抗憂鬱劑 (Antidepressants)',
    'N06B': '精神興奮劑/過動症用藥 (Psychostimulants, agents used for ADHD)',
    'N06D': '抗失智症藥 (Anti-dementia drugs)',
    'N07A': '副交感神經興奮藥 (Parasympathomimetics)',
    'N07C': '抗眩暈藥 (Antivertigo preparations)'
}

# 5. 輸出統計結果
print("\n--- 📊 統計結果 1：神經系統 (N) 健保給付金額最高的【子分類 (ATC3)】 ---")
df_atc3 = pd.DataFrame(list(atc3_amounts.items()), columns=['ATC3代碼', '總金額'])
df_atc3['分類名稱'] = df_atc3['ATC3代碼'].map(atc3_mapping).fillna('其他神經系統藥物')
df_atc3 = df_atc3.sort_values(by='總金額', ascending=False).reset_index(drop=True)
df_atc3['總金額(元)'] = df_atc3['總金額'].apply(lambda x: f"{x:,.0f}")
print(df_atc3[['ATC3代碼', '分類名稱', '總金額(元)']].head(5))

print("\n--- 🏆 統計結果 2：神經系統 (N) 健保給付金額最高的【前 5 大具體成分 (ATC5)】 ---")
df_atc5 = pd.DataFrame([{'ATC5': k, '總金額': v['金額'], '成分英文': v['成分'], '中文藥名範例': v['中文名']} for k, v in atc5_amounts.items()])
df_atc5 = df_atc5.sort_values(by='總金額', ascending=False).reset_index(drop=True)
df_atc5['總金額(元)'] = df_atc5['總金額'].apply(lambda x: f"{x:,.0f}")
print(df_atc5[['ATC5', '成分英文', '總金額(元)']].head(5))
