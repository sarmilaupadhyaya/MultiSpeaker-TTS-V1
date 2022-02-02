# -*- coding: utf-8 -*-
"""
Created on Sat Jan 29 13:42:36 2022

@author: Etudiant
"""

import pandas as pd
df = pd.read_csv("final_merge.csv",  sep="\t")
merge_dict = {row["symbol"]:row["merge"] for _, row in df.iterrows()}
merge_values = set(merge_dict.values())
merge_ids = {p : i for i, p in enumerate(merge_values)}
row_ids = [merge_ids[r] for r in df["merge"]]
df["id"] = row_ids
print(df.head())
print(len(merge_values))
df.to_csv("final_phonemes.csv", sep='\t')
