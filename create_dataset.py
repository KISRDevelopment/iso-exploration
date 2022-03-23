import numpy as np 
import pandas as pd 
import sys 
import glob 
import os 
import re 
import xmltodict
def main(input_dir, output_file):

    files = glob.glob("%s/*.xltm" % input_dir) + glob.glob("%s/*.xlsx"  % input_dir)
    files = ['tmp/T1=130.xltm', 'tmp/T1=128.xltm']
    files = ['datasets/T1=130.xltm', 'datasets/T1=128.xltm']
    
    final_df = create_df(files)
    
    final_df.to_csv(output_file, index=False)

    print("Dataset size: %d" % final_df.shape[0])

def create_df(files):
    
    rows_spec_df = read_rows_spec()

    sdfs = []
    for file in files:
        
        print("Processing %s" % file)
        if os.path.basename(file).startswith('~'):
            continue 
    
        df = pd.read_excel(file)
        df.index = read_tags(df['Unnamed: 0'])
        
        col_ix = ~pd.isnull(df.iloc[4]) & (np.arange(df.shape[1]) >= 4)
        print("Number of configs tested: %d" % np.sum(col_ix))
        #print(df.loc[rows_spec_df.index]) 

        sdf = df.loc[rows_spec_df.index, df.columns[col_ix]].copy()

        empty_ix = np.sum(sdf == '<empty>', axis=0) > 0
        
        if np.sum(empty_ix) == 0:
            sdf = sdf.astype(np.floating)
        else:
            
            sdf = sdf[sdf.columns[~empty_ix]].astype(np.floating)

        assert np.all(sdf.index == rows_spec_df.index)

        sdf.columns = np.arange(len(sdf.columns))
        sdf.index = rows_spec_df['alias']

        print(" Done")
        sdfs.append(sdf.T)

    final_df = pd.concat(sdfs, axis=0)

    perc_cols = list(rows_spec_df[rows_spec_df['perc_to_frac'] == 1]['alias'])

    final_df[perc_cols] = final_df[perc_cols] / 100
    
    return final_df 

def read_tags(rows):
    new_rows = []
    for i, row in enumerate(rows):
        if type(row) != str:
            new_rows.append((None, None, i))
            continue 

        o = xmltodict.parse(row)
        tag_name = ''
        attr_name = ''
        attr_value = ''
        obj_path = ''
        if 'UnisimTag' in o:
            tag_name = 'UnisimTag'
        elif 'UnisimElement' in o:
            tag_name = 'UnisimElement'
            
        attr_name = '@uopUnisimObjectName'
        attr_value = o[tag_name][attr_name]
        obj_path_attr = '@uopUnisimObjectPath'
        obj_path = o[tag_name][obj_path_attr]

        new_rows.append((tag_name, attr_name, attr_value, obj_path))
    
    s = pd.Series(new_rows)
    for r in s[s.duplicated()]:
        print(r)

    
    assert len(new_rows) == len(set(new_rows))
    return new_rows 

def read_rows_spec():

    df = pd.read_csv('rows.csv')
    df.index = list(zip(df['tag_name'], df['attr_name'], df['attr_value'], df['obj_path']))

    return df 

def is_float(element):
    try:
        float(element)
        return True
    except ValueError:
        return False

if __name__ == "__main__":
    import sys 
    main(sys.argv[1], sys.argv[2])
