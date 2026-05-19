import pandas as pd
import os

def process_yearly_data(file_path, year):
    print(f"正在預處理 {year} 年數據...")
    # 預計健保數據欄位沒有標題，我們依欄位順序命名，請根據實際 CSV 欄位調整
    # 假設：第1欄是年度，第2欄是藥品代碼，最後一欄是總醫令量
    # 這裡強烈建議在 usecols 限制只讀取需要的欄位（例如代碼與數量），省記憶體

    chunks = pd.read_csv(file_path, header=None, chunksize=50000, encoding='utf-8')

    atc_counts = {}

    for chunk in chunks:
        # 假設第二欄 (索引 1) 是藥品代碼，最後一欄 (例如索引 10 或 -1) 是醫令量
        # 這裡先以常見的健保公開格式為例：
        # 欄位順序：申報年、藥品代碼、藥品名稱、醫令起、醫令迄、...門診量、住院量、合計量
        # 如果您知道欄位名稱，也可以直接用名稱。這裡假設最後一欄為總醫令量

        chunk.columns = [f'col_{i}' for i in range(len(chunk.columns))]

        # 假設 col_1 是藥品代碼，col_10 是含包裹支付的醫令量_合計
        # 請根據您的實際欄位位置修改下方索引（例如代碼在第2欄=col_1，合計在最後一欄）
        code_col = chunk.columns[1]
        vol_col = chunk.columns[-1] 

        # 轉成字串並確保數字型態
        chunk[code_col] = chunk[code_col].astype(str).str.strip()
        chunk[vol_col] = pd.to_numeric(chunk[vol_col], errors='coerce').fillna(0)

        # 擷取首碼作為 ATC 大類
        chunk['ATC_Group'] = chunk[code_col].str[0].str.upper()

        # 只保留英文字母 A-Z 的部分
        chunk = chunk[chunk['ATC_Group'].str.isalpha()]

        # 該 chunk 分組加總
        summary = chunk.groupby('ATC_Group')[vol_col].sum()

        for atc, val in summary.items():
            atc_counts[atc] = atc_counts.get(atc, 0) + val

    df_res = pd.DataFrame(list(atc_counts.items()), columns=['ATC_Group', 'Total_Volume'])
    df_res['Year'] = year
    return df_res

# 讀取本機的三個檔案
df_2022 = process_yearly_data('A21030000I-E41005-001 (2022).csv', 2022)
df_2023 = process_yearly_data('A21030000I-E41005-002 (2023).csv', 2023)
df_2024 = process_yearly_data('A21030000I-E41005-003 (2024).csv', 2024)

# 合併並匯出成超小結果檔
final_summary = pd.concat([df_2022, df_2023, df_2024], ignore_index=True)
final_summary.to_csv('atc_summary_results.csv', index=False)
print("預處理完成！已產生 atc_summary_results.csv")
