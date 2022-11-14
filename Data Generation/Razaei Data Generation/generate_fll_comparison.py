from __future__ import print_function

import os
os.chdir(r"/.../fair-logloss-classification-master")

from scipy.io import arff
import functools
import numpy as np
import pandas as pd
import sys
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.metrics import roc_auc_score
from fair_logloss import DP_fair_logloss_classifier, EOPP_fair_logloss_classifier, EODD_fair_logloss_classifier
from sklearn.model_selection import train_test_split
from sklearn.model_selection import PredefinedSplit
import folktables
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline

def validate_bins(df, n_bins):
    df = train.groupby(['bin','g']).agg({'s': ['mean', 'count']}).reset_index()
    nonoverlap_bins = []
    for i in range(1, n_bins+1):
        df_sub = df.loc[df['bin']==i]
        if df_sub.shape[0] < 2:
            nonoverlap_bins.append(i)
    return nonoverlap_bins
    
data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
west_states = ["CA","OR","WA", "NV", "AZ"]
east_states = ['ME','NH','MA','RI','CT','NY','NJ','DE','MD','VA','NC','SC','GA','FL']
acs_data = data_source.get_data(states=east_states, download=True)

# We again use folktables data. We picked the mobility dataset for no particular reason 
acs_task, task_name, seed = folktables.ACSMobility, "acs_west_mobility", 4

group_var = acs_task.group
target_var = acs_task.target
acs_data.groupby(group_var).agg({group_var:'count'})
groups_to_keep = [1,2]
acs_data = acs_data.loc[acs_data[group_var].isin(groups_to_keep)]

# PROCESS DATASET
dataX, dataY, dataA = acs_task.df_to_numpy(acs_data, drop_group = True)
dataY = dataY.astype('float64')
dataA = dataA.astype('float64')
dataA = dataA-1

# Run using the following parameters
# We selected eqopps since during testing, "dp" and "eqodds" frequently yielded datasets without overlapping score distributions while eqopps provides consistent overlap 
# We explain in our paper that non-overlap is not a practical issue, but its interpretation is convoluted and not relevant for our demonstration purposes
n_trials = 20
C = .005
criteria = "eqopp" 
n_bins = 50

# Generate datasets over 20 trials where in each trial we make a random trian/test split, fit a model, and apply the model and methodology on the testing set
for i in range(n_trials): 
    seed = i*i*100
    folder = '/.../MultipleFairness/Data/FairLogloss/Trial_'+str(i)+'/'
    os.mkdir(folder)
    
    X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
        dataX, dataY, dataA, test_size=0.3, random_state=seed)
    pipe = Pipeline([('scaler', StandardScaler())])
    X_train = pipe.fit_transform(X_train)
    X_test = pipe.transform(X_test)
    poly = PolynomialFeatures(2, interaction_only=True)
    X_train_int = poly.fit_transform(X_train)
    X_test_int = poly.transform(X_test)
    
    if criteria == 'dp':
        h = DP_fair_logloss_classifier(C=C, random_initialization=False, verbose=False)
    elif criteria == 'eqopp':
        h = EOPP_fair_logloss_classifier(C=C, random_initialization=False, verbose=False)
    elif criteria == 'eqodd':
        h = EODD_fair_logloss_classifier(C=C, random_initialization=False, verbose=False)    
    else:
        raise ValueError('Invalid second arg')
        
    # h.fit(X_train, y_train, A_train)
    # yhat_train = h.predict_proba(X_train, A_train)
    # yhat_test = h.predict_proba(X_test, A_test)
    h.fit(X_train_int, y_train, A_train)
    yhat_train = h.predict_proba(X_train_int, A_train)
    yhat_test = h.predict_proba(X_test_int, A_test)
    
    train = pd.DataFrame({'s':yhat_train, 'y': y_train.astype('int'), 'g': A_train+1})
    test = pd.DataFrame({'s':yhat_test, 'y': y_test.astype('int'), 'g': A_test+1})
    
    bins, cuts = pd.qcut(train['s'], q=n_bins, retbins=True, duplicates='drop')
    cut_midpoints = [cuts[i]+(cuts[i+1] - cuts[i])/2 for i in range(len(cuts)-1)]
    cuts[0], cuts[-1] = 0.0, 1.0
    # Cut both train and test scores into bins based on cuts from the training data
    train['bin']=pd.cut(train['s'], bins = cuts, include_lowest=True, labels = False)
    train['bin']=train['bin']+1
    test['bin']=pd.cut(test['s'], bins = cuts, include_lowest=True, labels = False)
    test['bin']=test['bin']+1
    assert(validate_bins(train,n_bins) == [])
    assert(validate_bins(test,n_bins) == [])
    
    # also train a more powerful model based on ERM 
    tune_grid = {"n_estimators": [100],
                 "max_features":['auto',0.5],
                 "max_depth":[3, 4, 8, 10],
                 "min_samples_leaf":[1, 2, 4],
                 "min_samples_split":[2, 4, 6]}
    rf = RandomForestClassifier(random_state=42)
    rf_grid = RandomizedSearchCV(estimator = rf, param_distributions = tune_grid,
                                 n_iter = 10, cv = 2, verbose=2, random_state=seed, n_jobs = -1)
    rf_grid.fit(np.concatenate([X_train, A_train.reshape(-1,1)], axis=1), y_train)
    
    best_params = rf_grid.best_params_
    print(best_params)
    
    model = rf_grid.best_estimator_
    
    yhat_train_erm = model.predict_proba(np.concatenate([X_train, A_train.reshape(-1,1)], axis=1))[:,1]
    yhat_test_erm = model.predict_proba(np.concatenate([X_test, A_test.reshape(-1,1)], axis=1))[:,1]                          
    
    train_erm = pd.DataFrame({'s':yhat_train_erm, 'y': y_train.astype('int'), 'g': A_train+1})
    test_erm = pd.DataFrame({'s':yhat_test_erm, 'y': y_test.astype('int'), 'g': A_test+1})
    roc_auc_score(train_erm['y'], train_erm['s'])
    roc_auc_score(test_erm['y'], test_erm['s'])
    
    bins, cuts = pd.qcut(train_erm['s'], q=n_bins, retbins=True, duplicates='drop')
    cut_midpoints = [cuts[i]+(cuts[i+1] - cuts[i])/2 for i in range(len(cuts)-1)]
    cuts[0], cuts[-1] = 0.0, 1.0
    # Cut both train_erm and test_erm scores into bins based on cuts from the train_erming data
    train_erm['bin']=pd.cut(train_erm['s'], bins = cuts, include_lowest=True, labels = False)
    train_erm['bin']=train_erm['bin']+1
    test_erm['bin']=pd.cut(test_erm['s'], bins = cuts, include_lowest=True, labels = False)
    test_erm['bin']=test_erm['bin']+1
    assert(validate_bins(train_erm,n_bins) == [])
    assert(validate_bins(test_erm,n_bins) == [])
    
    train.to_csv(folder+task_name+'_train_fll.csv', index=False)
    test.to_csv(folder+task_name+'_test_fll.csv', index=False)
    
    pd.DataFrame(cut_midpoints, columns=['bin_midpoints']).to_csv(folder+task_name+'_bin_midpoints.csv', index=False)
    train_erm.to_csv(folder+task_name+'_train.csv', index=False)
    test_erm.to_csv(folder+task_name+'_test.csv', index=False)
