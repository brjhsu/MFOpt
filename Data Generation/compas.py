#!/usr/bin/env python3
# -*- coding: utf-8 -*-

# Use AIF360 package (prepare_data) class to load in preprocesed compas data for replicability 
from prepare_data import prepare_compas
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


def validate_bins(df, n_bins):
    df = train.groupby(['bin','g']).agg({'s': ['mean', 'count']}).reset_index()
    nonoverlap_bins = []
    for i in range(1, n_bins+1):
        df_sub = df.loc[df['bin']==i]
        if df_sub.shape[0] < 2:
            nonoverlap_bins.append(i)
    return nonoverlap_bins
    
folder = r'/users/bhsu/Documents/Prototypes/MultipleFairness/Data/'
task_name = 'compas'
model_type = "random_forest"
seed = 42

dataA,dataY,dataX,perm = prepare_compas()
X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(dataX,dataY,dataA+1,test_size=0.05,random_state=seed)

pipe = Pipeline([('scaler', StandardScaler())])
X_train = pipe.fit_transform(X_train)
X_test = pipe.transform(X_test)
    
if model_type == "random_forest":
    # also train a more powerful model based on ERM 
    tune_grid = {"n_estimators": [100],
                 "max_features":['auto',0.5],
                 "max_depth":[3, 4, 8, 10],
                 "min_samples_leaf":[1, 2, 4],
                 "min_samples_split":[2, 4, 6]}
    rf = RandomForestClassifier(random_state=seed)
    rf_grid = RandomizedSearchCV(estimator = rf, param_distributions = tune_grid,
                                 n_iter = 10, cv = 2, verbose=2, random_state=seed, n_jobs = -1)
    rf_grid.fit(X_train, y_train)
    best_params = rf_grid.best_params_
    print(best_params)
    model = rf_grid.best_estimator_

else:
    model = LogisticRegression()
    model.fit(X_train, y_train)

yhat_train = model.predict_proba(X_train)[:,1]
yhat_test = model.predict_proba(X_test)[:,1]

train = pd.DataFrame({'s':yhat_train, 'y': y_train.astype('int'), 'g': A_train})
test = pd.DataFrame({'s':yhat_test, 'y': y_test.astype('int'), 'g': A_test})
roc_auc_score(train['y'], train['s'])
roc_auc_score(test['y'], test['s'])

n_bins = 50 # use 50 bins across all experiments
bins, cuts = pd.qcut(train['s'], q=n_bins, retbins=True, duplicates='drop')
cut_midpoints = [cuts[i]+(cuts[i+1] - cuts[i])/2 for i in range(len(cuts)-1)]
cuts[0], cuts[-1] = 0.0, 1.0
train['bin']=pd.cut(train['s'], bins = cuts, include_lowest=True, labels = False)
train['bin']=train['bin']+1
test['bin']=pd.cut(test['s'], bins = cuts, include_lowest=True, labels = False)
test['bin']=test['bin']+1
assert(validate_bins(train,n_bins) == [])
assert(validate_bins(test,n_bins) == [])

# Save results
pd.DataFrame(cut_midpoints, columns=['bin_midpoints']).to_csv(folder+task_name+'_bin_midpoints.csv', index=False)
train.to_csv(folder+task_name+'_train.csv', index=False)
test.to_csv(folder+task_name+'_test.csv', index=False)