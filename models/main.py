import pandas as pd

df = pd.read_csv("../data/processed/paper_01.csv")
drop_cols = ["Text_Title", "Cutoff_Year", "Cutoff_Session", "Yrs_Since_Last"]
X = df.drop(columns=["App_Next"])
y = df[["App_Next"]]
X = df.drop(columns=drop_cols)

num_cols = X.select_dtypes(include=["int64", "float64"]).columns
cat_cols = X.select_dtypes(include=["object", "category"]).columns