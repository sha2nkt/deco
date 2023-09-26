#python scripts/datascripts/add_imgname_column_to_deco_csv.py --csv_path /ps/scratch/ps_shared/stripathi/deco/4agniv/DCA/mturk_csvs_combined_temp.csv --out_path /ps/scratch/ps_shared/stripathi/deco/4agniv/DCA/mturk_csvs_combined_temp_with_imgnames.csv

import argparse
import pandas as pd
import os

def add_imagename_column(csv_path, out_path):
    # Load csv
    csv_data = pd.read_csv(csv_path)

    # Add column with image name from vertices column, where every element is a dictionary with key the image names as keys and values as the vertices
    csv_data['imgnames'] = csv_data['vertices'].apply(lambda x: os.path.basename(list(eval(x).keys())[0]))

    # Write combined data to new CSV file
    csv_data.to_csv(out_path, index=False)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--csv_path', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot/dca.csv')
    parser.add_argument('--out_path', type=str, default='/ps/scratch/ps_shared/stripathi/deco/4agniv/hot/dca_with_imgname.csv')
    args = parser.parse_args()
    add_imagename_column(args.csv_path, args.out_path)

