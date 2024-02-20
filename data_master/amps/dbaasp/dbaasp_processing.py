#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Oct 19 16:11:30 2023

@author: anupkumar
"""
# Python program to read
# json file

import json

# Opening JSON file
f = open('/media/anupkumar/Backup Plus/project/ML_project/peptides-complete.json')

# returns JSON object as a dictionary
datasets = json.load(f)
# Closing file
f.close()
data_list = datasets['peptides']
aa = ["A","C", "D", "E","F","G", "H","I","K", "L","M","N", "P","Q","R", "S","T","V", "W","Y"]
features = ['dbaaspId', 'sequence', "cTerminus", "physicoChemicalProperties"]
extracted_data = []
for data in data_list:
    avail_data = {}
    data_keys = data.keys()

    try:
        if len(data["sequence"])>3: #len(data["physicoChemicalProperties"])>6:
            for f in features:
                if f in data_keys:
                    avail_data[f] = data[f]
    except:
        pass
    if len(avail_data) > 0:
        extracted_data.append(avail_data)
    
            

### remove unusall seq, duplicate and do amidation check
temp = []
peptides_NH2 = {}
c = 0
for i, data in enumerate(extracted_data):
    pc = data["sequence"]
    pc = pc.upper()
    if pc not in temp:
        temp.append(pc)
        for a in pc:
            if a not in aa:
                break    
        try:
            amide = data["cTerminus"]["name"]
            if amide == "AMD":
                pc = pc +"J"
                c = c+1
        except:
            pc=pc + "Z"
        peptides_NH2["seq" + str(i)] = pc

   
print("Total amidated peptides: " + str(c))




