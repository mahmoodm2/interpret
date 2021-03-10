# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Setup a classification experiment

# %%
import pickle
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.base import is_classifier

from interpret import show
from interpret.data import ClassHistogram
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression, ClassificationTree, DecisionListClassifier

import numpy as np
import copy


# %%
if True:
    df = pd.read_csv(
        "http://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
        header=None)
    df.columns = [
        "Age", "WorkClass", "fnlwgt", "Education", "EducationNum",
        "MaritalStatus", "Occupation", "Relationship", "Race", "Gender",
        "CapitalGain", "CapitalLoss", "HoursPerWeek", "NativeCountry", "Income"
    ]
    # df = df.sample(frac=0.1, random_state=1)
    train_cols = df.columns[0:-1]
    label = df.columns[-1]
    X = df[train_cols]
    y = df[label].apply(lambda x: 0 if x == " <=50K" else 1) #Turning response into 0 and 1

    seed = 1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)

    # %% [markdown]
    # ## Explore the dataset

    # %%  

    # hist = ClassHistogram().explain_data(X_train, y_train, name = 'Train Data')
    # show(hist)

    # %% [markdown]
    # ## Train the Explainable Boosting Machine (EBM)
# %%
# from interpret import show
# from interpret.data import ClassHistogram

# hist = ClassHistogram().explain_data(X_train, y_train, name = 'Train Data')
# show(hist)
    # %%
# %%
from sklearn.model_selection import KFold
from interpret.glassbox import ExplainableBoostingClassifier

rng = np.random.RandomState(0)
cv = KFold(n_splits=3, shuffle=True, random_state=rng)
models =[]
seed =1
for train_index, test_index in cv.split(X):    

    X_train, y_train = X.iloc[train_index], y.iloc[train_index]

    print( len(X_train) , len(y_train))
    
    ebm = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
    ebm.fit(X_train, y_train)  
    models.append(ebm)
    seed +=10
# %%

# %%
%load_ext autoreload
%autoreload 2
from interpret.glassbox.ebm.utils import *

merged_ebm = EBMUtils.merge_models(models=models)
# %%
ebm_global = merged_ebm.explain_global(name='EBM')

show(ebm_global)
# %%
ebm1 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
ebm1.fit(X_train, y_train)  
# %%

# %%
seed =1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
ebm1 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
ebm1.fit(X_train, y_train)  

    # print( len(ebm1.additive_terms_))


    # %%
    
seed =1
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
ebm1 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
ebm1.fit(X_train, y_train)  

print( len(ebm1.additive_terms_))

seed +=10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
ebm2 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
ebm2.fit(X_train, y_train)  
print( len(ebm2.additive_terms_))


seed +=10
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
ebm3 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
ebm3.fit(X_train, y_train)  
print( len(ebm3.additive_terms_))


    # %%
pickle.dump(ebm1,open("D:\\Projects\\interpret\\examples\\python\\notebooks\\ebm1.pickle", 'wb'))
pickle.dump(ebm2,open("D:\\Projects\\interpret\\examples\\python\\notebooks\\ebm2.pickle", 'wb'))
pickle.dump(ebm3,open("D:\\Projects\\interpret\\examples\\python\\notebooks\\ebm3.pickle", 'wb'))


# %%
ebm1 = pickle.load(open("D:\\Projects\\interpret\\examples\\python\\notebooks\\ebm1.pickle" ,'rb'))
ebm2 = pickle.load(open("D:\\Projects\\interpret\\examples\\python\\notebooks\\ebm2.pickle" ,'rb'))
ebm3 = pickle.load(open("D:\\Projects\\interpret\\examples\\python\\notebooks\\ebm3.pickle" ,'rb'))


# %%
ebm2.feature_names


# %%
ebm1.preprocessor_.col_bin_counts_[0] , ebm2.preprocessor_.col_bin_counts_[0]


# %%
(ebm1.preprocessor_.col_bin_counts_[0])


# %%
for index,_ in enumerate(ebm1.feature_groups_):
    if index in ebm1.preprocessor_.col_bin_edges_.keys():
        print(index, len(ebm1.preprocessor_.col_bin_edges_[index]))
# %% 

def weighted_std(a, axis, weights):

    average = np.average(a, axis , weights)
    
    variance = np.average((a - average)**2, axis , weights)

    return np.sqrt(variance)
# %%
# ebm2.preprocessor_.col_bin_edges_[ebm1.feature_groups_[0]]
from sklearn.base import clone
models=[ebm1, ebm2, ebm3]
ebm = clone(models[0])
# ebm = copy.deepcopy(ebm1)

ebm.preprocessor_ = clone(models[0].preprocessor_)
ebm.pair_preprocessor_ = clone(models[0].pair_preprocessor_)

if not all([  model.preprocessor_.col_types_ == ebm.preprocessor_.col_types_ for model in models]):
            raise Exception("All models should have the same types of features. Probably the models are trained using different datasets")
       
if not all([  model.preprocessor_.col_bin_edges_.keys() == ebm.preprocessor_.col_bin_edges_.keys() for model in models]):
            raise Exception("All models should have the same types of features. Probably the models are trained using different datasets")


if not all([  model.preprocessor_.col_mapping_.keys() == ebm.preprocessor_.col_mapping_.keys() for model in models]):
            raise Exception("All models should have the same types of features. Probably the models are trained using different datasets")

if is_classifier(ebm):
        if not all([is_classifier(model) for model in models]):
            raise Exception("All models should be the same type.")
        # else:
        #     if not all([ebm.classes_ == model.classes_ for model in models]):
        #             raise Exception("All models should have the same number of classes.")
else:
        #ebm is not a classifier, checking for at least one classifier in other models
        if any([is_classifier(model) for model in models]):
            raise Exception("All models should be the same type.")

new_feature_groups = []
merged_interactions =set()

main_feature_len = len(ebm.preprocessor_.feature_names)

ebm.additive_terms_ = []
ebm.term_standard_deviations_ = []

for model in models:  
    for index, feature_group in enumerate(model.feature_groups_):

        if len(feature_group) != 1:
            merged_interactions.add( (feature_group, model.feature_names[index] , model.feature_types[index]))
       

for index, feature_group in enumerate(ebm.feature_groups_):           

        # interction tuples
        if len(feature_group) != 1:
            # Exluding interction fwarures from merging
            continue

        log_odds_tensors = []
        bin_weights = []
        # numeric features
        if index in ebm.preprocessor_.col_bin_edges_.keys():           
                            
            merged_bin_edges = sorted(set().union(*[ set(model.preprocessor_.col_bin_edges_[index]) for model in models]))
            
            ebm.preprocessor_.col_bin_edges_[index] = np.array(merged_bin_edges)
          
            for model in models:            
            # Merging the bin edges for different models for each feature group
                model_bin_edges = model.preprocessor_.col_bin_edges_[index]

                bin_indexs = np.searchsorted(model_bin_edges, merged_bin_edges + [np.inf])
                
                bin_counts = model.preprocessor_.col_bin_counts_
                # All the estimators of one ebm model share the same bin edges
                for estimator in model.bagged_models_: 

                    # if have different bin_edges for this fearture group:       
                    # ignoring the the first element as reserved for futur.                              
                    mvalues = estimator.model_[index][1:] 

                    # expanding the model_ values to cover all the new merged bin edges
                    # x represents the index of the merged bin edge in the new merged bin edges
                    # new_model_ = [ mvalues[x-1] if x > 0 and x <=len(mvalues) else np.nan for x in bin_indexs[1:] ]
                    new_model_ = [ mvalues[x-1] if x > 0 and x <=len(mvalues) else 0. for x in bin_indexs[1:] ]

                    wvalues = bin_counts[index][1:] 
                    new_weights =[ wvalues[x-1] if x > 0 and x <=len(mvalues) else 0. for x in bin_indexs[1:] ]
                    
                    log_odds_tensors.append(new_model_)
                    bin_weights.append( new_weights)
        else:
            # Categorical features
            merged_col_mapping = sorted(set().union(*[ set(model.preprocessor_.col_mapping_[index]) for model in models]))
           
            ebm.preprocessor_.col_mapping_[index] = dict( (key, idx +1) for idx, key in enumerate(merged_col_mapping))
            
            for model in models: 
                
                bin_counts = model.preprocessor_.col_bin_counts_

                mask = [ model.preprocessor_.col_mapping_[index].get(col, None ) for col in merged_col_mapping]

                for estimator in model.bagged_models_:

                    mvalues = estimator.model_[index]  
                    # new_model_ =  [ mvalues[i] if i else np.nan for i in mask]
                    new_model_ =  [ mvalues[i] if i else 0.0 for i in mask]

                    wvalues = bin_counts[index]
                    new_weights =[ wvalues[i] if i else 0.0 for i in mask ]
                   
                    log_odds_tensors.append(new_model_)
                    bin_weights.append( new_weights)


        # Using nan versions to exlude nan values in calculating mean/std values
        # averaged_model = np.nanmean(np.array(log_odds_tensors), axis=0)
        # # model_errors = np.nanstd(np.array(log_odds_tensors), axis=0)
        # for v in log_odds_tensors:
        #     print(len(v))

        # for v in bin_weights:
        #     print(len(v))
        
        averaged_model = np.average(log_odds_tensors, axis=0 , weights=bin_weights )
        model_errors = weighted_std(np.array(log_odds_tensors), axis=0, weights= np.array(bin_weights) )

        averaged_model = np.append(0., averaged_model)
        ebm.additive_terms_.append(averaged_model)

        model_errors = np.append(0. , model_errors )
        ebm.term_standard_deviations_.append(model_errors)

        print(index, len(averaged_model) , len(model_errors))


# main_indices + new_pair_indices
ebm.feature_groups_ = ebm.feature_groups_[:main_feature_len] #+ [ feature[0] for feature in merged_interactions]
ebm.feature_names = ebm.feature_names[:main_feature_len] #+ [ feature[1] for feature in merged_interactions]
ebm.feature_types = ebm.feature_types[:main_feature_len] #+ [ feature[2] for feature in merged_interactions]

print(ebm.feature_groups_ , ebm.feature_names , ebm.feature_types)
# %%
from interpret.glassbox.ebm.utils import *


# %%
EBMUtils.merge_models(models)


# %%
len(set(ebm2.preprocessor_.col_bin_edges_[2]).union(set((ebm1.preprocessor_.col_bin_edges_[2]))))


# %%
ebm.preprocessor_.col_bin_edges_

# %% [markdown]
# ## Global Explanations: What the model learned overall

# %%
ebm_global = ebm.explain_global(name='EBM')
show(ebm_global)

# %% [markdown]
# ## Local Explanations: How an individual prediction was made

# %%
ebm_local = ebm.explain_local(X_test[:5], y_test[:5], name='EBM')
show(ebm_local)

# %% [markdown]
# ## Evaluate EBM performance

# %%
from interpret.perf import ROC

ebm_perf = ROC(ebm.predict_proba).explain_perf(X_test, y_test, name='EBM')
show(ebm_perf)

# %% [markdown]
# ## Let's test out a few other Explainable Models

# %%
from interpret.glassbox import LogisticRegression, ClassificationTree

# We have to transform categorical variables to use Logistic Regression and Decision Tree
X_enc = pd.get_dummies(X, prefix_sep='.')
feature_names = list(X_enc.columns)
X_train_enc, X_test_enc, y_train, y_test = train_test_split(X_enc, y, test_size=0.20, random_state=seed)

lr = LogisticRegression(random_state=seed, feature_names=feature_names, penalty='l1', solver='liblinear')
lr.fit(X_train_enc, y_train)

tree = ClassificationTree()
tree.fit(X_train_enc, y_train)

# %% [markdown]
# ## Compare performance using the Dashboard

# %%
lr_perf = ROC(lr.predict_proba).explain_perf(X_test_enc, y_test, name='Logistic Regression')
tree_perf = ROC(tree.predict_proba).explain_perf(X_test_enc, y_test, name='Classification Tree')

show(lr_perf)
show(tree_perf)
show(ebm_perf)

# %% [markdown]
# ## Glassbox: All of our models have global and local explanations

# %%
lr_global = lr.explain_global(name='Logistic Regression')
tree_global = tree.explain_global(name='Classification Tree')

show(lr_global)
show(tree_global)
show(ebm_global)

# %% [markdown]
# ## Dashboard: look at everything at once

# %%
# Do everything in one shot with the InterpretML Dashboard by passing a list into show

show([hist, lr_global, lr_perf, tree_global, tree_perf, ebm_global, ebm_perf], share_tables=True)


