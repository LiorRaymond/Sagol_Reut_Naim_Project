import pandas as pd
#adding missing variables
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

df_main.to_csv("Merged_dataset.csv", index=False)

#replacing hand grip data
df_merged = pd.read_csv("Merged_dataset.csv")
df_grip = pd.read_csv("GripForce_Final.csv")

df_merged.set_index('ID', inplace=True)
df_grip.set_index('ID', inplace=True)

columns_to_override = ['Calm_Direct_FinalMax',	'Calm_Avert_FinalMax','Angry_Direct_FinalMax',	'Angry_Avert_FinalMax']

grip_subset = df_grip[df_grip.columns.intersection(columns_to_override)]

df_merged.update(grip_subset)
df_merged.reset_index(inplace=True)
df_merged.to_csv("Updated_merged.csv", index=False)