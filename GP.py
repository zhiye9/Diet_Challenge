import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import stats
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import LinearSVC, SVC
from sklearn.model_selection import GridSearchCV
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import roc_curve, auc, roc_auc_score, precision_recall_curve, average_precision_score, f1_score
from sklearn.metrics import confusion_matrix
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import RocCurveDisplay
import pickle
from sklearn.manifold import TSNE
import os
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

def CV(p_grid, out_fold, in_fold, model, X, y, rand):
    outer_cv = StratifiedKFold(n_splits = out_fold, shuffle = True, random_state = rand)
    inner_cv = StratifiedKFold(n_splits = in_fold, shuffle = True, random_state = rand)
    auc_train = []
    auc_test = []
    accuracy = []
    f1 = []
    recall = []
    precision_plot = []
    recall_plot = []
    precision = []
    spcificty = []
    AUPR = []
    AUROC = []
    y_true = []
    y_proba = []

    tprs = []
    fprs = []

    for j, (train, test) in enumerate(outer_cv.split(X, y)):
        #split dataset to decoding set and test set
        x_train, x_test = X[train], X[test]
        y_train, y_test = y[train], y[test]
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "neg_mean_squared_error")
        #clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "f1")
        clf =  GridSearchCV(estimator = model, param_grid = p_grid, cv = inner_cv, scoring = "roc_auc")
        clf.fit(x_train, y_train)
        print(j)
        
        #predict labels on the train set
        y_pred = clf.predict(x_train)
        #r2train.append(mean_squared_error(y_train, y_pred))
        auc_train.append(roc_auc_score(y_train, y_pred))
        
        #predict labels on the test set
        y_pred = clf.predict(x_test)
        #r2test.append(mean_squared_error(y_test, y_pred))
        auc_test.append(roc_auc_score(y_test, y_pred))
        a = metrics.accuracy_score(y_test, y_pred)
        f = metrics.f1_score(y_test, y_pred)
        r = metrics.recall_score(y_test, y_pred, average='binary')
        p = metrics.precision_score(y_test, y_pred, average='binary')
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        s = tn / (tn+fp)
        #print(a)
        #print(f)
        accuracy.append(a)
        f1.append(f)
        recall.append(r)
        precision.append(p)
        spcificty.append(s)
        y_score = clf.predict_proba(x_test)[:,1]
        y_true.append(y_test) 
        y_proba.append(y_score)
        #pre, re, thresholds = precision_recall_curve(y_test, y_score, pos_label = 1)
        #AUPR.append(auc(re, pre))
        #fpr, tpr, thresholds = roc_curve(y_test, y_score, drop_intermediate = False, pos_label = 1)
        #fprs.append(fpr)
        #tprs.append(tpr)
        #AUROC.append(auc(fpr, tpr))
        
    return auc_train, auc_test, accuracy, f1, recall, precision, spcificty, y_true, y_proba

p_grid_rf = {'max_depth': [10, 50, 100, 200, None]}
model_rf = RandomForestClassifier()
p_grid_lsvm = {'base_estimator__C': [0.01, 0.5, 0.1, 1]}
model_lsvm = CalibratedClassifierCV(base_estimator=LinearSVC(max_iter = 10000))
p_grid_svm = {'C': [0.1, 1, 10], 'gamma': [1e-4, 1e-3, 'scale']}
model_svm =  SVC(kernel = 'rbf',  probability = True)

X = df_biochem_pivot_noTAG_nonan1.iloc[:, 4:].values
y = df_biochem_pivot_noTAG_nonan1['Sex'].values.astype('int')
RF_train_wrinkle, RF_test_wrinkle, acc_RF_wrinkle, f1_RF_wrinkle, recall_RF_wrinkle, precision_RF_wrinkle, spcificty_RF_wrinkle, ytrue_RF_wrinkle, y_proba_RF_wrinkle = CV(p_grid = p_grid_rf, out_fold = 5, in_fold = 5, model = model_rf, X = X, y =y, rand = 9)
RF_results = [RF_train_wrinkle, RF_test_wrinkle, acc_RF_wrinkle, f1_RF_wrinkle, recall_RF_wrinkle, precision_RF_wrinkle, spcificty_RF_wrinkle, ytrue_RF_wrinkle, y_proba_RF_wrinkle]
with open('/home/zhi/data/GP_diet_challenge/GP_resultRF.txt', 'w') as file:
    for item_list in RF_results:
        row = ' '.join(map(str, item_list))  # Convert list to a space-separated string
        file.write(row + '\n')

import warnings
warnings.filterwarnings("ignore")
LinSVM_train_wrinkle, LinSVM_test_wrinkle, acc_LinSVM_wrinkle, f1_LinSVM_wrinkle, recall_LinSVM_wrinkle, precision_LinSVM_wrinkle, spcificty_LinSVM_wrinkle, ytrue_LinSVM_wrinkle, y_proba_LinSVM_wrinkle = CV(p_grid = p_grid_lsvm, out_fold = 5, in_fold = 5, model = model_lsvm, X = X, y =y, rand = 9)
LinSVM_results = [LinSVM_train_wrinkle, LinSVM_test_wrinkle, acc_LinSVM_wrinkle, f1_LinSVM_wrinkle, recall_LinSVM_wrinkle, precision_LinSVM_wrinkle, spcificty_LinSVM_wrinkle, ytrue_LinSVM_wrinkle, y_proba_LinSVM_wrinkle]
with open('/home/zhi/data/GP_diet_challenge/GP_resultLinSVM.txt', 'w') as file:
    for item_list in LinSVM_results:
        row = ' '.join(map(str, item_list))  # Convert list to a space-separated string
        file.write(row + '\n')

SVM_train_wrinkle, SVM_test_wrinkle, acc_SVM_wrinkle, f1_SVM_wrinkle, recall_SVM_wrinkle, precision_SVM_wrinkle, spcificty_SVM_wrinkle, ytrue_SVM_wrinkle, y_proba_SVM_wrinkle = CV(p_grid = p_grid_svm, out_fold = 5, in_fold = 5, model = model_svm, X = X, y =y, rand = 9)
SVM_results = [SVM_train_wrinkle, SVM_test_wrinkle, acc_SVM_wrinkle, f1_SVM_wrinkle, recall_SVM_wrinkle, precision_SVM_wrinkle, spcificty_SVM_wrinkle, ytrue_SVM_wrinkle, y_proba_SVM_wrinkle]
with open('/home/zhi/data/GP_diet_challenge/GP_resultSVM.txt', 'w') as file:
    for item_list in SVM_results:
        row = ' '.join(map(str, item_list))  # Convert list to a space-separated string
        file.write(row + '\n')

fig, ax = plt.subplots()
ax.plot([0, 1], [1, 0], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
y_real_rf = np.concatenate(ytrue_RF_wrinkle)
y_proba_rf = np.concatenate(y_proba_RF_wrinkle)
precision_rf, recall_rf, _ = precision_recall_curve(y_real_rf, y_proba_rf)
y_real_svm = np.concatenate(ytrue_SVM_wrinkle)
y_proba_svm = np.concatenate(y_proba_SVM_wrinkle)
precision_svm, recall_svm, _ = precision_recall_curve(y_real_svm, y_proba_svm)
y_real_lsvm = np.concatenate(ytrue_LinSVM_wrinkle)
y_proba_lsvm = np.concatenate(y_proba_LinSVM_wrinkle)
precision_lsvm, recall_lsvm, _ = precision_recall_curve(y_real_lsvm, y_proba_lsvm)
lab1 = 'RF AUC=%.4f' % (auc(recall_rf, precision_rf))
lab2 = 'SVM with Gaussian AUC=%.4f' % (auc(recall_svm, precision_svm))
lab3 = 'Linear SVM AUC=%.4f' % (auc(recall_lsvm, precision_lsvm))
ax.step(recall_rf, precision_rf, label=lab1, lw=2, color='blue')
ax.step(recall_svm, precision_svm, label=lab2, lw=2, color='orange')
ax.step(recall_lsvm, precision_lsvm, label=lab3, lw=2, color='brown')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall (PR) Curve')
ax.legend(loc='lower left', fontsize='small')


fig, ax = plt.subplots()
ax.plot([0, 1], [0, 1], linestyle="--", lw=2, color="r", label="Chance", alpha=0.8)
y_real_rf = np.concatenate(ytrue_RF_wrinkle)
y_proba_rf = np.concatenate(y_proba_RF_wrinkle)
fpr_rf, tpr_rf, _ = roc_curve(y_real_rf, y_proba_rf)
y_real_svm = np.concatenate(ytrue_SVM_wrinkle)
y_proba_svm = np.concatenate(y_proba_SVM_wrinkle)
fpr_svm, tpr_svm, _ = roc_curve(y_real_svm, y_proba_svm)
y_real_lsvm = np.concatenate(ytrue_LinSVM_wrinkle)
y_proba_lsvm = np.concatenate(y_proba_LinSVM_wrinkle)
fpr_lsvm, tpr_lsvm, _ = roc_curve(y_real_lsvm, y_proba_lsvm)
lab1 = 'RF AUC=%.4f' % (auc(fpr_rf, tpr_rf))
lab2 = 'SVM with Gaussian AUC=%.4f' % (auc(fpr_svm, tpr_svm))
lab3 = 'Linear SVM AUC=%.4f' % (auc(fpr_lsvm, tpr_lsvm))
ax.step(fpr_rf, tpr_rf, label=lab1, lw=2, color='blue')
ax.step(fpr_svm, tpr_svm, label=lab2, lw=2, color='orange')
ax.step(fpr_lsvm, tpr_lsvm, label=lab3, lw=2, color='brown')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('Receiver Operating Characteristic (ROC) Curve')
ax.legend(loc='lower right', fontsize='small')     