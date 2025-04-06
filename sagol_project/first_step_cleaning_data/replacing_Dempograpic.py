import pandas as pd

df_main = pd.read_excel("Updated_data_04.06.xlsx")

#replacing dempgraphic info
df_demograph = pd.read_csv("Motion_Dx_Age_Sex_Final.csv")
df_demograph.rename(columns={"SDAN": "ID"}, inplace=True)

df_main.set_index('ID', inplace=True)
df_demograph.set_index('ID', inplace=True)

columns_to_override = ['Dx','Age','Sex']

dempograph_subset = df_demograph[df_demograph.columns.intersection(columns_to_override)]

df_main.update(dempograph_subset)
df_main.reset_index(inplace=True)
df_main.to_csv("Cleaned_data.csv", index=False)


