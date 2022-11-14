#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os 
os.chdir(r'/.../MultipleFairness/equalized_odds_and_calibration-master/')
from eq_odds import Model
import numpy as np
import pandas as pd
import calibration as cal
import folktables
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

def validate_bins(df, n_bins):
    df = df.groupby(['bin','g']).agg({'s': ['mean', 'count']}).reset_index()
    nonoverlap_bins = []
    for i in range(1, n_bins+1):
        df_sub = df.loc[df['bin']==i]
        if df_sub.shape[0] < 2:
            nonoverlap_bins.append(i)
    return nonoverlap_bins


folder = r'/.../MultipleFairness/Data/Pleiss/'
data_source = folktables.ACSDataSource(survey_year='2018', horizon='1-Year', survey='person')
west_states = ["CA","OR","WA"]
east_states = ['ME','NH','MA','RI','CT','NY','NJ','DE','MD','VA','NC','SC','GA','FL']
acs_data = data_source.get_data(states=west_states, download=True)

# We again use folktables data. We picked the income dataset for no particular reason 
acs_task, task_name, seed = folktables.ACSIncome, "acs_west_income", 8
model_type = "random_forest"
group_var = acs_task.group
target_var = acs_task.target
groups_to_keep = [1,2]
acs_data = acs_data.loc[acs_data[group_var].isin(groups_to_keep)]

# PROCESS DATASET
dataX, dataY, dataA = acs_task.df_to_numpy(acs_data)

X_train, X_test, y_train, y_test, A_train, A_test = train_test_split(
    dataX, dataY, dataA, test_size=0.3, random_state=seed)

# TRAIN MODEL
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


n_bins = 50
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

pd.DataFrame(cut_midpoints, columns=['bin_midpoints']).to_csv(folder+task_name+'_bin_midpoints.csv', index=False)
train.to_csv(folder+task_name+'_train.csv', index=False)
test.to_csv(folder+task_name+'_test.csv', index=False)


# APPLY PLEISS METHODOLOGY
# Create model objects - one for each group, validation and test
group_0_val_data = train[train['g'] == 1].copy(deep=True)
group_1_val_data = train[train['g'] == 2].copy(deep=True)
group_0_test_data = test[test['g'] == 1].copy(deep=True)
group_1_test_data = test[test['g'] == 2].copy(deep=True)

# Construct eodds model
group_0_val_model = Model(group_0_val_data['s'].values, group_0_val_data['y'].values)
group_1_val_model = Model(group_1_val_data['s'].values, group_1_val_data['y'].values)
group_0_test_model = Model(group_0_test_data['s'].values, group_0_test_data['y'].values)
group_1_test_model = Model(group_1_test_data['s'].values, group_1_test_data['y'].values)

# Find mixing rates for equalized odds models
_, _, mix_rates = Model.eq_odds(group_0_val_model, group_1_val_model)

# Apply the mixing rates to the val models
eq_odds_group_0_val_model, eq_odds_group_1_val_model = Model.eq_odds(group_0_val_model,
                                                                     group_1_val_model,
                                                                     mix_rates)
# Apply the mixing rates to the test models
eq_odds_group_0_test_model, eq_odds_group_1_test_model = Model.eq_odds(group_0_test_model,
                                                                       group_1_test_model,
                                                                       mix_rates)

# Print results on test model
print('Original group 0 model:\n%s\n' % repr(group_0_test_model))
print('Original group 1 model:\n%s\n' % repr(group_1_test_model))
print('Equalized odds group 0 model:\n%s\n' % repr(eq_odds_group_0_test_model))
print('Equalized odds group 1 model:\n%s\n' % repr(eq_odds_group_1_test_model))

group_0_val_data['s'] = eq_odds_group_0_val_model.pred
group_1_val_data['s'] = eq_odds_group_1_val_model.pred

group_0_test_data['s'] = eq_odds_group_0_test_model.pred
group_1_test_data['s'] = eq_odds_group_1_test_model.pred
val_df = pd.concat([group_0_val_data, group_1_val_data], axis=0)
test_df = pd.concat([group_0_test_data, group_1_test_data], axis=0)

bins, cuts = pd.qcut(val_df['s'], q=n_bins, retbins=True, duplicates='drop')
cut_midpoints = [cuts[i]+(cuts[i+1] - cuts[i])/2 for i in range(len(cuts)-1)]
cuts[0], cuts[-1] = 0.0, 1.0
val_df['bin']=pd.cut(val_df['s'], bins = cuts, include_lowest=True, labels = False)
val_df['bin']=val_df['bin']+1
test_df['bin']=pd.cut(test_df['s'], bins = cuts, include_lowest=True, labels = False)
test_df['bin']=test_df['bin']+1

assert(validate_bins(val_df,n_bins) == [])
assert(validate_bins(test_df,n_bins) == [])

val_df.to_csv(folder+task_name+'_train_pleiss.csv', index=False)
test_df.to_csv(folder+task_name+'_test_pleiss.csv', index=False)
