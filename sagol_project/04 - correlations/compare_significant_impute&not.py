import pandas as pd
from pathlib import Path

# טען את שני הקבצים
BASE_DIR = Path(__file__).parent
df1 = pd.read_csv(BASE_DIR / "results/all_significant_correlations.csv")
df2 = pd.read_csv(BASE_DIR / "results_imuted_data/all_significant_correlations_im.csv")

# קח רק את העמודות הרלוונטיות
pairs1 = df1[['Feature_1', 'Feature_2']]
pairs2 = df2[['Feature_1', 'Feature_2']]

# ודא שאין כפילויות
pairs1 = pairs1.drop_duplicates().reset_index(drop=True)
pairs2 = pairs2.drop_duplicates().reset_index(drop=True)

# בדוק אילו זוגות מהקובץ הראשון לא נמצאים בשני
missing_pairs = pairs1[~pairs1.apply(tuple, axis=1).isin(pairs2.apply(tuple, axis=1))]

# הצג את הזוגות החסרים
print("Missing pairs from file2:")
print(missing_pairs)
