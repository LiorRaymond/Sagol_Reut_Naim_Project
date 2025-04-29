import pandas as pd
import openpyxl
from openpyxl.styles import PatternFill
import numpy as np

# Load the correlation matrix from CSV
csv_path = 'results/correlation_matrix.csv'
df = pd.read_csv(csv_path, index_col=0)

# Create a new Excel workbook and sheet
wb = openpyxl.Workbook()
ws = wb.active
ws.title = "Correlation Matrix"

# Write headers
headers = [''] + list(df.columns)  # Empty cell for top-left corner
for col_idx, header in enumerate(headers, start=1):
    ws.cell(row=1, column=col_idx).value = header

# Write data with row indices
for row_idx, (idx_name, row_data) in enumerate(df.iterrows(), start=2):
    # Write row index name
    ws.cell(row=row_idx, column=1).value = idx_name
    
    # Write row data
    for col_idx, value in enumerate(row_data, start=2):
        cell = ws.cell(row=row_idx, column=col_idx)
        cell.value = value
        
        # Apply conditional formatting - skip coloring if value is exactly 1 (self-correlation)
        if abs(value) > 0.9 and not np.isclose(value, 1.0, rtol=1e-5):
            if value > 0.9:
                red_hex = f"{int(255 * value):02X}"
                cell.fill = PatternFill(start_color=f"{red_hex}AAAA", end_color=f"{red_hex}AAAA", fill_type="solid")
            elif value < -0.9:
                blue_hex = f"{int(255 * abs(value)):02X}"
                cell.fill = PatternFill(start_color=f"AAAA{blue_hex}", end_color=f"AAAA{blue_hex}", fill_type="solid")

ws.freeze_panes = 'B2' 

# Save to Excel in the main folder
output_path = 'correlation_colored.xlsx'
wb.save(output_path)

print(f"✅ Excel file saved successfully as: {output_path}")