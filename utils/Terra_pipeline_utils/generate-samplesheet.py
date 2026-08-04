# generate a bclconvert-compliant sampleshet

# script takes in a comma-delimited list of lanes (ints [1,8]),
# and a comma-delimited list of indexes (strings),
# and a flowcell name (string)

# saves a Macosko-Pipelines/bcl-convert/samplesheets/<name>-samplesheet.csv
# containing lane info and the appropiate sequence indexes for each library

# todo: optionally take in index names that are user-defined 
# i.e. human-readable lib names as oppsosed to SI-.*

# to check: make valid samplesheets for ND7 and ND5 indexes

# FROM: https://github.com/MacoskoLab/Macosko-Pipelines/blob/main/bcl-convert/generate-samplesheet.py


"""
python generate-samplesheet.py \
  --lanes 1,2,3,4,5,6,7,8 \
  --indexes SI-TT-E1,SI-TT-E2,SI-TT-E3,SI-TT-E4,SI-TT-E5,SI-TT-E6,SI-TT-E7,SI-TT-F1,SI-TT-F2,SI-TT-F3,SI-TT-F4,SI-TT-F5,SI-TT-F6,SI-TT-G1,SI-TT-G2,SI-TT-G3,SI-TT-G4,SI-TT-G5,SI-TT-G6,SI-TT-G7,SI-TT-H1,SI-TT-H2,SI-TT-H3,SI-TT-H4,SI-TT-H5,SI-TT-H6 \
  --name 260721_SL-EXB_0892_B23Y3GKLT4
"""


import argparse

import os
import re
import csv
import glob
import pandas as pd
from os.path import basename
from itertools import product

indexes_root = "https://raw.githubusercontent.com/MacoskoLab/Macosko-Pipelines/main/bcl-convert/indexes/"
NN = pd.read_csv(indexes_root+"SI-NN.csv")
NT = pd.read_csv(indexes_root+"SI-NT.csv")
TS = pd.read_csv(indexes_root+"SI-TS.csv")
TT = pd.read_csv(indexes_root+"SI-TT.csv")
ND7 = pd.read_csv(indexes_root+"ND7.csv")
ND5 = pd.read_csv(indexes_root+"ND5.csv")
NA = pd.read_csv(indexes_root+"SI-NA.csv", header=None)
GA = pd.read_csv(indexes_root+"SI-GA.csv", header=None)

# output directory for samplesheets
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), 'samplesheets')
os.makedirs(OUTPUT_DIR, exist_ok=True)

def dict2df(indexes):
    rows = []
    assert isinstance(indexes, dict)
    for lane in sorted(indexes.keys()):
        assert lane in [1,2,3,4,5,6,7,8]
        assert isinstance(indexes[lane], list)
        assert all(isinstance(e, str) for e in indexes[lane])
        assert len(indexes[lane])==len(set(indexes[lane])) # no duplicates
        for index in indexes[lane]:
            if bool(re.match(r"^[ND]7\d{2}[ND]5\d{2}$", index)): # [ND]7xx[ND]5yy
                row1 = ND7.loc[ND7['I7_Index_ID'] == index[:4]]
                seq1 = row1["index"].iloc[0]
                row2 = ND5.loc[ND5['I5_Index_ID'] == index[-4:]]
                seq2 = row2["index2_workflow_a"].iloc[0]
            elif index in ND7["I7_Index_ID"].values: # D7xx
                row = ND7.loc[ND7['I7_Index_ID'] == index]
                seq1 = row["index"].iloc[0]
                seq2 = "CGAGATCT"
            elif index in TT["index_name"].values: # SI-TT-xx
                row = TT.loc[TT['index_name'] == index]
                seq1 = row["index(i7)"].iloc[0]
                seq2 = row["index2_workflow_a(i5)"].iloc[0]
            elif index in NT["index_name"].values: # SI-NT-xx
                row = NT.loc[NT['index_name'] == index]
                seq1 = row["index(i7)"].iloc[0]
                seq2 = row["index2_workflow_a(i5)"].iloc[0]
            elif index in NN["index_name"].values: # SI-NN-xx
                row = NN.loc[NN['index_name'] == index]
                seq1 = row["index(i7)"].iloc[0]
                seq2 = row["index2_workflow_a(i5)"].iloc[0]
            elif index in TS["index_name"].values: # SI-TS-xx
                row = TS.loc[TS['index_name'] == index]
                seq1 = row["index(i7)"].iloc[0]
                seq2 = row["index2_workflow_a(i5)"].iloc[0]
            else:
                raise IndexError(f"ERROR: index {index} not found")

            rows.append([lane, index, seq1, seq2])
    
    df = pd.DataFrame(rows, columns=['Lane', 'Sample_ID', 'index', 'index2'])

    # add padding to 8bp indexes if some indexes are 10bp
    if any(df['index'].str.len() == 10) or any(df['index2'].str.len() == 10):
        df['index'] = df['index'].apply(lambda x: x+'AT' if len(x) == 8 else x)
        df['index2'] = df['index2'].apply(lambda x: 'AC'+x if len(x) == 8 else x)

    # bcl-convert requires all indexes to have the same length
    assert df['index'].str.len().nunique() == 1
    assert df['index2'].str.len().nunique() == 1
    
    return df

def dict2df_A(indexes, XA):
    rows = []
    assert isinstance(indexes, dict)
    for lane in sorted(indexes.keys()):
        assert lane in [1,2,3,4,5,6,7,8]
        assert isinstance(indexes[lane], list)
        assert all(isinstance(e, str) for e in indexes[lane])
        assert len(indexes[lane])==len(set(indexes[lane]))
        for index in indexes[lane]:
            assert index in XA[0].values # SI-NA-xx or SI-GA-xx
            row = XA.loc[XA[0] == index]
            rows.append([lane, index, row[1].iloc[0]])
            rows.append([lane, index, row[2].iloc[0]])
            rows.append([lane, index, row[3].iloc[0]])
            rows.append([lane, index, row[4].iloc[0]])
    
    df = pd.DataFrame(rows, columns=['Lane', 'Sample_ID', 'index'])
    assert df['index'].str.len().nunique() == 1
    return df

def generate_samplesheet(lanes, vals):

    indexes = {lane: vals for lane in lanes}

    # if GA or NA indexes, use dict2df_A
    # otherwise use dict2df
    # if a mix, throw an error
    if any(index in NA[0].values or index in GA[0].values for index in vals):
        assert all(index in NA[0].values for index in vals) or all(index in GA[0].values for index in vals)
        samplesheet = dict2df_A(indexes, NA if any(index in NA[0].values for index in vals) else GA)
    else:
        samplesheet = dict2df(indexes)

    assert len(samplesheet) == len(samplesheet.drop_duplicates())

    return samplesheet

# python script should take as arguments: a comma-delimited list of lanes, a comma-delimited list of indexes, and a path
# the comma -delimited strings should be converted to lists

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description='Generate a SampleSheet.csv for bcl-convert.')
    parser.add_argument('--lanes', type=str, required=True, help='Comma-delimited list of lanes (1-8)')
    parser.add_argument('--indexes', type=str, required=True, help='Comma-delimited list of indexes')
    parser.add_argument('--name', type=str, required=True, help='Name for SampleSheet.csv')

    args = parser.parse_args()
    lanes = [int(lane) for lane in args.lanes.split(',')]
    vals = args.indexes.split(',')

    output_path = os.path.join(OUTPUT_DIR, f"{args.name}-samplesheet.csv")
    
    print(f"Generating SampleSheet for lanes {lanes} with indexes {vals}")
    samplesheet = generate_samplesheet(lanes, vals)

    print(f"Writing SampleSheet to {output_path}")
    with open(output_path, 'w') as f:
        f.write("[Settings]\n")
        f.write("CreateFastqForIndexReads,0\n") # default: 0
        f.write("NoLaneSplitting,false\n") # default: false
        f.write("\n[Data]\n")

    # if indexes contain -GA or -NA, ['Lane', 'Sample_ID', 'index']
    # otherwise ['Lane', 'Sample_ID', 'index', 'index2']
    if any('-GA' in index or '-NA' in index for index in vals):
        column_list = ['Lane', 'Sample_ID', 'index']
    else:
        column_list = ['Lane', 'Sample_ID', 'index', 'index2']

    samplesheet[column_list].to_csv(output_path, mode='a', index=False)
    