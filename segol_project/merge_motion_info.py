import pandas as pd

def merge_excel_files(file1, file2, output_file, key_column='ID'):

    df1 = pd.read_excel(file1)
    df2 = pd.read_excel(file2)

    merged_df = pd.merge(df1, df2, on=key_column, how='outer')
    merged_df = merged_df.loc[:, ~merged_df.columns.duplicated()]

    merged_df = merged_df[sorted(merged_df.columns)]
    
    merged_df.to_excel(output_file, index=False)
    print ("merged")

#merge 
merge_excel_files('Group_Motion_Info_1_Final.xlsx', 'Group_Motion_Info_2_Final.xlsx', 'Group_Motion_Info_merged.xlsx')