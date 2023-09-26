# write a script to find all csvs in a folder and merge them

import os
import pandas as pd
import argparse

def combine_csvs(folder_path, out_path):
    # Create empty DataFrame to store combined data
    combined_data = pd.DataFrame()

    # Loop through all files in folder with .csv extension
    for file in os.listdir(folder_path):
        if file.endswith(".csv"):
            file_path = os.path.join(folder_path, file)
            # Read CSV file and append to combined_data
            csv_data = pd.read_csv(file_path)
            combined_data = combined_data.append(csv_data, ignore_index=True)

    # Write combined data to new CSV file
    combined_data.to_csv(out_path, index=False)
    print(f"Combined CSV file saved to {out_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--folder_path', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot/')
    parser.add_argument('--out_path', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot/combined.csv')
    args = parser.parse_args()
    folder_path = args.folder_path
    out_path = args.out_path
    combine_csvs(folder_path, out_path)



