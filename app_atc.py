import pandas as pd

# 1. 讀取並合併藥品與 ATC 支付價對照表
print("正在載入藥品支付價與 ATC 對照表...")
price_cols = ['藥品代號', '支付價', 'ATC代碼']
price1 = pd.read_csv('Price_ATC1.csv', usecols=price_cols)
price2 = pd.read_csv('Price_ATC2.csv', usecols=price_cols)
price_df = pd.concat([price1, price2], ignore_index=True)

# 因同一藥品可能有不同時期的支付價，轉成數值並取最新的單價（或平均價），此處以最新/最大值為代表
price_df['支付價'] = pd.to_numeric(price_df['支付價'], errors='coerce')
price_df = price_df.dropna(subset=['藥品代號', 'ATC代碼'])
# 移除重複，保留每個藥品代號最新的 ATC 與價格
price_clean = price_df.groupby('藥品代號').agg({'支付價': 'median', 'ATC代碼': 'first'}).reset_index()

# 2. 分塊讀取大型的健保申報量檔案 (以 2022 年為例)
declaration_file = 'A21030000I-E41005-001 (2022).csv'
print(f"正在分塊處理健保申報檔案: {declaration_file} ...")

atc1_amounts = {} # 用來儲存 ATC 大類的總金額

# 每次讀取 50,000 列，避免記憶體不足
for chunk in pd.read_csv(declaration_file, chunksize=50000):
    # 清理申報欄位
    chunk = chunk.dropna(subset=['藥品代碼', '含包裹支付的醫令量_合計'])
    chunk['含包裹支付的醫令量_合計'] = pd.to_numeric(chunk['含包裹支付的醫令量_合計'], errors='coerce').fillna(0)
    
    # 串聯價格與 ATC 碼
    merged = pd.merge(chunk, price_clean, left_on='藥品代碼', right_on='藥品代號', how='inner')
    
    # 計算該項藥品的健保總給付金額 = 醫令量 * 支付價
    merged['給付總金額'] = merged['含包裹支付的醫令量_合計'] * merged['支付價']
    
    # 擷取 ATC 第一碼 (大分類，如 A, C, L)
    merged['ATC_L1'] = merged['ATC代碼'].str[0].str.upper()
    
    # 按大類加總金額並累加到字典
    grouped = merged.groupby('ATC_L1')['給付總金額'].sum()
    for atc_code, total_money in grouped.items():
        atc1_amounts[atc_code] = atc1_amounts.get(atc_code, 0) + total_money

# 3. 轉換成 DataFrame 並對應中文名稱
atc_mapping = {
    'A': '消化道及新陳代謝用藥 (Alimentary tract and metabolism)',
    'B': '血液及造血器官用藥 (Blood and blood forming organs)',
    'C': '心血管系統用藥 (Cardiovascular system)',
    'D': '皮膚用藥 (Dermatologicals)',
    'G': '生殖泌尿系統及性激素 (Genito urinary system and sex hormones)',
    'H': '全身性激素製劑 (Systemic hormonal preparations, excl. sex hormones and insulins)',
    'J': '全身性抗感染藥 (Antiinfectives for systemic use)',
    'L': '抗腫瘤及免疫調節劑 (Antineoplastic and immunomodulating agents)',
    'M': '肌肉骨骼系統 (Musculo-skeletal system)',
    'N': '神經系統用藥 (Nervous system)',
    'P': '抗寄生蟲藥、殺蟲劑及驅蟲劑 (Antiparasitic products, insecticides and repellents)',
    'R': '呼吸系統用藥 (Respiratory system)',
    'S': '感官器官用藥 (Sensory organs)',
    'V': '其它各種藥物 (Various)'
}

result_df = pd.DataFrame(list(atc1_amounts.items()), columns=['ATC大類代碼', '健保給付總金額(元)'])
result_df['分類中文名稱'] = result_df['ATC大類代碼'].map(atc_mapping).fillna('其他/未分類')

# 排序並格式化輸出
result_df = result_df.sort_values(by='健保給付總金額(元)', ascending=False).reset_index(drop=True)
result_df['健保給付總金額(元)'] = result_df['健保給付總金額(元)'].apply(lambda x: f"{x:,.0f}")

print("\n🏆 計算完成！獲得最多健保給付金額的 ATC 分類排名：")
print(result_df)
