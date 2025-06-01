import pandas as pd
import csv

# Load data
file_path = "/home/drosophila-lab/Documents/Genomics Project/snp-data/DATA/2025_03_13_genomic_trajectory_SNP_table_withPvalues.csv"
df = pd.read_csv(file_path)

POPS_DICT = dict()
INDIVS = ['CACO1_18', 'CACO2_18', 'CACO3_18', 'CACO4_18', 'CACO5_18', 'CAO1_18', 'CAO2_18', 'CAO3_18', 'CAO4_18', 'CAO5_18', 'NACO1_18', 'NACO2_18', 'NACO3_18', 'NACO4_18', 'NACO5_18', 'ANCO1_18', 'ANCO2_18', 'ANCO3_18', 'ANCO4_18', 'ANCO5_18', 'NACO1_19', 'NACO2_19', 'NACO3_19', 'NACO4_19', 'NACO5_19', 'ANCO1_19', 'ANCO2_19', 'ANCO3_19', 'ANCO4_19', 'ANCO5_19', 'CAO1_19', 'CAO2_19', 'CAO3_19', 'CAO4_19', 'CAO5_19', 'CACO1_19', 'CACO2_19', 'CACO3_19', 'CACO4_19', 'CACO5_19', 'NACO1_20', 'NACO2_20', 'NACO3_20', 'NACO4_20', 'NACO5_20', 'ANCO1_20', 'ANCO2_20', 'ANCO3_20', 'ANCO4_20', 'ANCO5_20', 'CAO1_20', 'CAO2_20', 'CAO3_20', 'CAO4_20', 'CAO5_20', 'CACO1_20', 'CACO2_20', 'CACO3_20', 'CACO4_20', 'CACO5_20', 'NACO1_24', 'NACO2_24', 'NACO3_24', 'NACO4_24', 'NACO5_24', 'ANCO1_24', 'ANCO2_24', 'ANCO3_24', 'ANCO4_24', 'ANCO5_24', 'CAO1_24', 'CAO2_24', 'CAO3_24', 'CAO4_24', 'CAO5_24', 'CACO1_24', 'CACO2_24', 'CACO3_24', 'CACO4_24', 'CACO5_24']
SCALING_FACTOR = 10**-1
P_VALUE = 0.3504 * SCALING_FACTOR

def make_indivs_to_dict_of_pops():
    for name in INDIVS:
        pop = name.split('_')[0]
        if pop not in POPS_DICT:
            POPS_DICT[pop] = [name]
        else:
            POPS_DICT[pop].append(name)

def get_sel_pop(name):
    sel = name.split('_')[0][-1]
    pop = name.split('_')[0][0:-1]
    return sel, pop

def is_evolving(p_value):
    return float(format(float(p_value), 'f')) < (P_VALUE)

make_indivs_to_dict_of_pops()
print('Populated dictionary.')

while P_VALUE > 10**-20:
    with open(f'SNP_CSV_threshold_{P_VALUE}.csv', 'w', newline="") as file:
        writer = csv.writer(file)
        writer.writerow(['Chr', 'Pos', 'Sel', 'Pop', 'Evolving', 'Freq1', 'Freq2', 'Freq3', 'Freq4'])
        for index, row in df.iterrows(): # 80 x 1426594 = 114127520 total rows ~ O(n^2) runtime
            for gen in POPS_DICT:
                writer.writerow([row['chr'], row['pos'], get_sel_pop(gen)[0], get_sel_pop(gen)[1], is_evolving(row['pvalue']), row[POPS_DICT[gen][0]], row[POPS_DICT[gen][1]], row[POPS_DICT[gen][2]], row[POPS_DICT[gen][3]]])
    
    P_VALUE = P_VALUE * SCALING_FACTOR

# chr,pos,ref,alt,CACO1_18,CACO2_18,CACO3_18,CACO4_18,CACO5_18,CAO1_18,CAO2_18,CAO3_18,CAO4_18,CAO5_18,NACO1_18,NACO2_18,NACO3_18,NACO4_18,NACO5_18,ANCO1_18,ANCO2_18,ANCO3_18,ANCO4_18,ANCO5_18,NACO1_19,NACO2_19,NACO3_19,NACO4_19,NACO5_19,ANCO1_19,ANCO2_19,ANCO3_19,ANCO4_19,ANCO5_19,CAO1_19,CAO2_19,CAO3_19,CAO4_19,CAO5_19,CACO1_19,CACO2_19,CACO3_19,CACO4_19,CACO5_19,NACO1_20,NACO2_20,NACO3_20,NACO4_20,NACO5_20,ANCO1_20,ANCO2_20,ANCO3_20,ANCO4_20,ANCO5_20,CAO1_20,CAO2_20,CAO3_20,CAO4_20,CAO5_20,CACO1_20,CACO2_20,CACO3_20,CACO4_20,CACO5_20,NACO1_24,NACO2_24,NACO3_24,NACO4_24,NACO5_24,ANCO1_24,ANCO2_24,ANCO3_24,ANCO4_24,ANCO5_24,CAO1_24,CAO2_24,CAO3_24,CAO4_24,CAO5_24,CACO1_24,CACO2_24,CACO3_24,CACO4_24,CACO5_24,convertedPosition,pvalue