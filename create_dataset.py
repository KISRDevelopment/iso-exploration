import numpy as np 
import pandas as pd 
import sys 
import glob 
import os 

def main():

    rows_to_aliases = get_rows_to_extract()
    
    files = glob.glob("datasets/*.xltm")

    selected_rows = list(rows_to_aliases.keys())

    sdfs = []
    for file in files:
        if os.path.basename(file).startswith('~'):
            continue 
    
        try:
            df = pd.read_excel(file).set_index('Unnamed: 0')
        except:

            print("Ignoring %s" % file)
            continue 
        
        df.index = df.index.str.strip()
        
        first_row = df.loc[selected_rows[0]]
        col_ix = np.array([isinstance(v, np.floating) for v in first_row]) & ~pd.isnull(first_row)
        
        r = set(selected_rows) - set(df.index) 
        if len(r) > 0:
            print(file)
            print("Not found in df: %d" % len(r))
            print("\n".join(r))
            exit(0)

        df = df.loc[selected_rows, col_ix].copy()
        df.index = [rows_to_aliases[r] for r in selected_rows]
        df.columns = np.arange(len(df.columns))
        
        print("Processed %s" % file)
        sdfs.append(df.T)

    final_df = pd.concat(sdfs, axis=0)
    final_df.to_csv("iso.csv", index=False)

def get_rows_to_extract():

    with open('rows_to_extract.txt', 'r') as f:
        rows = [l.strip() for l in f]
    

    with open('row_aliases.txt', 'r') as f:
        aliases = [l.strip() for l in f]
    
    rows_to_aliases = dict(zip(rows, aliases))

    return rows_to_aliases

if __name__ == "__main__":
    main()
