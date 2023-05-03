'''
This file for any functions that modify the raw dataset

Author: Kaniel Vicencio
'''

import os
import csv
import tqdm

def create_annotation_file(fp): 
    with open('../dataset/annotations.csv', 'w', newline='') as file:
        writer = csv.writer(file)
        field = ["type", "filename"]
        writer.writerow(field)

        for dir in tqdm(os.listdir(fp)):
            for idx, filename in enumerate(os.listdir(f"{fp}" + f"{dir}")): 
                row = [dir, filename, fp + dir + "/" + filename]
                writer.writerow(row)
    return None