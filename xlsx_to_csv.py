import pandas as pd
import sys
import os

def xlsx_to_csv(xlsx_path, output_folder="data/csv_translations"):
    os.makedirs(output_folder, exist_ok=True)

    filename = os.path.splitext(os.path.basename(xlsx_path))[0]
    output_path = os.path.join(output_folder, filename + ".csv")

    if os.path.exists(output_path):
        print(f"Skipped (already exists): {filename}.csv")
        return

    df = pd.read_excel(xlsx_path)
    df.to_csv(output_path, index=False)

    print(f"Saved: {output_path}")

if __name__ == "__main__":
    input_folder = "data/excel translations"

    xlsx_files = [f for f in os.listdir(input_folder) if f.endswith(".xlsx")]

    if not xlsx_files:
        print("No .xlsx files found in 'data/excel translations'")
        sys.exit(1)

    for file in xlsx_files:
        xlsx_to_csv(os.path.join(input_folder, file))
