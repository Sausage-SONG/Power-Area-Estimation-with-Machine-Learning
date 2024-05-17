import pandas as pd

con = pd.read_csv("contest.csv", sep='    ')
df = pd.read_csv("data.csv")
df = df.sort_values('id').reset_index(drop=True)
con.columns = [f"arch{i}" for i in range(29)] + ["gt_pf", "gt_pw", "gt_area", "gt_t"]
df = df.join(con[["gt_pf", "gt_pw", "gt_area", "gt_t"]])
df.to_csv("temp.csv", index=False)
