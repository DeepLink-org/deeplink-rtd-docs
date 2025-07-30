import pandas as pd
import re

src_path = "./operators.xlsx"
save_csv_path = "./processed_op.csv"
df = pd.read_excel(src_path)

# fill the blank
df["标准分类"].fillna(method='ffill', inplace=True)

# 处理多个链接：将换行或空格分隔的多个 URL 分别包裹
def wrap_urls(cell):
    if pd.isna(cell):
        return cell
    # 用正则提取所有 http/https 链接
    urls = re.findall(r'https?://[^\s\n]+', str(cell))
    # 包裹每个链接并拼接
    return ' '.join([f"<{url}>" for url in urls])

df["PyTorch link"] = df["PyTorch link"].apply(wrap_urls)

# 保存处理后的 CSV
df.to_csv(save_csv_path, index=False, encoding="utf-8")