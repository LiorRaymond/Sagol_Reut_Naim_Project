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

#df_main.to_csv("Merged_dataset.csv", index=False)

#replacing hand grip data
df_grip = pd.read_csv("GripForce_Final.csv")

df_main.set_index('ID', inplace=True)
df_grip.set_index('ID', inplace=True)

columns_to_override = ['Calm_Direct_FinalMax','Calm_Avert_FinalMax','Angry_Direct_FinalMax','Angry_Avert_FinalMax']

grip_subset = df_grip[df_grip.columns.intersection(columns_to_override)]

df_main.update(grip_subset)
df_main.reset_index(inplace=True)

#replacing dempgraphic info
df_demograph = pd.read_excel("Clinical_Data.xlsx")
df_demograph.rename(columns={"SDAN": "ID"}, inplace=True)

df_main.set_index('ID', inplace=True)
df_demograph.set_index('ID', inplace=True)

columns_to_override = ['Dx','Age','Sex']

dempograph_subset = df_demograph[df_demograph.columns.intersection(columns_to_override)]

df_main.update(dempograph_subset)
df_main.reset_index(inplace=True)
#df_main.to_csv("Updated_data.csv", index=False)

#adding clinical info
columns_to_exclude = [ 'excluded_eyegaze_data', 'excluded_pupil_data','Dx','Age','Sex']
clinical_data = df_demograph.drop(columns=columns_to_exclude)

df_main = df_main.merge(clinical_data, on="ID", how="left")
#df_main.to_csv("Updated_data.csv", index=False)

#adding 2 missing ID's
missing_ids = ['21666', '23559']
relevant_excels = [
    'GripForce_Final.csv',
    'Motion level 2_Final_Final.xlsx',
    'Group_Motion_Info_2_Final.xlsx',
    'Group_Motion_Info_1_Final.xlsx'
    ]

variables_transform = pd.read_excel("variables_transform.xlsx")
rename_dict = dict(zip(variables_transform["variable"], variables_transform["new"]))

dfs = []
for file in relevant_excels:
    file_path = file
    if file.endswith(".csv"):
        df = pd.read_csv(file_path)
    else:
        df = pd.read_excel(file_path)

    if "SDAN" in df.columns:
        df.rename(columns={"SDAN": "ID"}, inplace=True)

    df["ID"] = df["ID"].astype(str)
    df_filtered = df[df["ID"].isin(missing_ids)]
    
    df_filtered = df_filtered.rename(columns=rename_dict)
    dfs.append(df_filtered)

df_missing_data = pd.concat(dfs, ignore_index=True) if dfs else pd.DataFrame()
df_missing_data = df_missing_data.groupby("ID", as_index=False).first()

df_main = pd.concat([df_main, df_missing_data], ignore_index=True)

df_main.to_csv("Updated_data.csv", index=False)

                


