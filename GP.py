import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from scipy.stats import mannwhitneyu

df_glu_insulin_8 = pd.read_csv('/home/zhi/nas/Diet_challenge/df_biochem_insulin_Cpeptid_filtered.csv')
df_asthma18 = pd.read_excel('/home/zhi/nas/Diet_challenge/CP0155_Asthma_cross_18yr.xlsx', sheet_name = 'Data')
df_asthma19 = pd.read_excel('/home/zhi/nas/Diet_challenge/CP0156_Asthma_diagnoses_0-19yr.xlsx', sheet_name = 'Data')

df_asthma19['copsacno'] = df_asthma19['copsacno'].astype('str')
df_asthma19['icd10'] = df_asthma19['icd10'].astype('str')
df_asthma19['copsacno_icd10'] = df_asthma19['copsacno'] + '_' + df_asthma19['icd10']
df_glu_insulin_8['copsacno'] = df_glu_insulin_8['copsacno'].astype('str')

df_asthma = df_glu_insulin_8[['copsacno']].drop_duplicates(subset = ['copsacno'], keep = 'first')
df_asthma.reset_index(drop = True, inplace = True)
df_asthma['asthma'] = 0
df_asthma['copsacno'] = df_asthma['copsacno'].astype('str')

#mark df_asthma as 1 if its copsacno is in df_asthma19
for i in range(len(df_asthma)):
    if df_asthma['copsacno'][i] in df_asthma19['copsacno'].values:
        df_asthma['asthma'][i] = 1
#find is there any missing value in df_glucose_insulin_8 and find where are them
na = pd.DataFrame(df_glu_insulin_8.isnull().sum())
na[na[0] != 0]

df_2000 = pd.read_csv('/home/zhi/nas/Diet_challenge/df2000.csv')
df_2000['copsacno'] = df_2000['COPSACNO'].astype('str')
df_asthma_sex = pd.merge(df_asthma, df_2000[['Sex', 'copsacno']], on = 'copsacno', how = 'left')

#find the row and column with nan from df_glu_insulin_8
df_glu_insulin_8[df_glu_insulin_8.isnull().any(axis = 1)][['copsacno', 'TIMEPOINT', 'Gly', 'Tyr', 'Lactate', 'Pyruvate', 'bOHbutyrate', 'Creatinine']]
#drop rows with more than 2 nan
df_glu_insulin_8.drop(df_glu_insulin_8[df_glu_insulin_8['copsacno'] == '182'].index, inplace = True)
df_glu_insulin_8.drop(df_glu_insulin_8[df_glu_insulin_8['copsacno'] == '316'].index, inplace = True)
df_glu_insulin_8.reset_index(drop = True, inplace = True)

#drop TAG
na1 = pd.DataFrame(df_glu_insulin_8.applymap(lambda x: 'TAG' in str(x)).sum())
na1[na1[0] != 0]

df_glu_insulin_8[df_glu_insulin_8.applymap(lambda x: 'TAG' in str(x)).any(axis = 1)][['copsacno', 'TIMEPOINT'] + na1[na1[0] != 0].index.tolist()]
df_glu_insulin_8.drop(df_glu_insulin_8[df_glu_insulin_8['copsacno'] == '142'].index, inplace = True)
df_glu_insulin_8.drop(df_glu_insulin_8[df_glu_insulin_8['copsacno'] == '79'].index, inplace = True)
df_glu_insulin_8.reset_index(drop = True, inplace = True)

df_biochem_asthma = pd.merge(df_asthma_sex, df_glu_insulin_8, on = 'copsacno', how = 'right')
df_biochem_asthma['copsacno'] = df_biochem_asthma['copsacno'].astype('str')
df_biochem_asthma['asthma'] = df_biochem_asthma['asthma'].astype('str')
len(np.unique(df_biochem_asthma['copsacno']))

df_phenotype = df_biochem_asthma.copy(deep = True).drop_duplicates(subset = ['copsacno'], keep = 'first')[['copsacno', 'asthma', 'Sex', 'age']]
np.unique(df_phenotype['asthma'], return_counts = True)
np.unique(df_phenotype['Sex'], return_counts = True)
np.unique(df_phenotype[df_phenotype['Sex'] == 0]['asthma'], return_counts = True)
np.unique(df_phenotype[df_phenotype['Sex'] == 1]['asthma'], return_counts = True)

#pivot df_biochem_asthma after column age to get the dataframe with only one row for each copsacno
df_biochem = df_biochem_asthma.copy(deep = True)
df_biochem.drop(columns = ['asthma', 'age', 'Sex'], inplace = True)
df_biochem_pivot = df_biochem.pivot(index='copsacno', columns='TIMEPOINT')
df_biochem_pivot.columns = ['{}_{}'.format(x[0], x[1]) for x in df_biochem_pivot.columns]
df_biochem_pivot = pd.merge(df_phenotype, df_biochem_pivot, on = 'copsacno')
df_biochem_pivot.reset_index(drop = True, inplace = True)

row, cols = np.where(df_biochem_pivot.applymap(lambda x: 'TAG' in str(x)))
row1, cols1 = np.where(pd.isnull(df_biochem_pivot))

def impute_nan(r, c, df):
    for id in r:
        gender = df.loc[id]['Sex']
        for ix in c:
            df1 = df[df['Sex'] == gender].iloc[:,ix]
            df.iloc[id, ix] = np.nanmedian(df1[df1 != 'TAG'].astype('float'))
    return df

df_biochem_pivot_noTAG = impute_nan(row, cols, df_biochem_pivot)
df_biochem_pivot_noTAG_nonan = impute_nan(row1, cols1, df_biochem_pivot)

#df_biochem_pivot_noTAG_nonan.to_csv('/home/zhi/data/GP_diet_challenge/df_biochem_pheno_imputed.csv')
--------------------------------------------------------------------------------------------------------------------------------
df_biochem_pivot_noTAG_nonan1 = pd.read_csv('/home/zhi/nas/Diet_challenge/df_biochem_pheno_imputed.csv', index_col = 0)
df_biochem_pivot_noTAG_nonan1['copsacno'] = df_biochem_pivot_noTAG_nonan1['copsacno'].astype('str')
df_biochem_pivot_noTAG_nonan1['asthma'] = df_biochem_pivot_noTAG_nonan1['asthma'].astype('str')
df_biochem_pivot_noTAG_nonan1[df_biochem_pivot_noTAG_nonan1.columns[4:]] = df_biochem_pivot_noTAG_nonan1[df_biochem_pivot_noTAG_nonan1.columns[4:]].astype('float')

