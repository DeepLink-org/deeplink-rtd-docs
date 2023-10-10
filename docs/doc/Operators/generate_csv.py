import pandas as pd

# 原始的xlsx文件不能直接用于展示，需要进行处理后保存为csv文件进行展示

src_path = "./download_op.xlsx"
save_csv_path = "./processed_op.csv"
df = pd.read_excel(src_path)

# fill the blank
df["标准分类"].fillna(method='ffill', inplace=True)

# add <> for url link
for i in range(len(df["PyTorch link"])):
    if not (pd.isna(df["PyTorch link"][i])):
        url = df["PyTorch link"][i]
        url = "<" + url + ">"
        df["PyTorch link"][i] = url

# save the processed dataframe
df.to_csv(save_csv_path, index=False, encoding="utf-8")