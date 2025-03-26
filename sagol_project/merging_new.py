import pandas as pd

df_main = pd.read_excel("New_motion_dataset_all_final.xlsx")
df_main.rename(columns={"SDAN": "ID"}, inplace=True)
df_missing = pd.read_excel("missing_variables.xlsx")
file_name = "Group_Motion_Info_1_Final.xlsx"

df_aux = pd.read_excel(file_name)
df_aux = df_aux[df_aux["ID"].isin(df_main["ID"])]

relevant_columns = df_missing.iloc[:, 0].tolist()
relevant_columns.insert(0, "ID") 
df_aux = df_aux[relevant_columns]

df_main = df_main.merge(df_aux, on="ID", how="left")

df_main.to_excel("Merged_dataset.xlsx", index=False)
