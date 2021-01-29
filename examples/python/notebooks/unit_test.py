# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %% [markdown]
# ## Setup a classification experiment

# %%
import pickle


# %%
import pandas as pd
from sklearn.model_selection import train_test_split

from interpret import show
from interpret.data import ClassHistogram
from interpret.glassbox import ExplainableBoostingClassifier, LogisticRegression, ClassificationTree, DecisionListClassifier

import numpy as np


if False:
    df = pd.read_csv(
        "https://archive.ics.uci.edu/ml/machine-learning-databases/adult/adult.data",
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

    hist = ClassHistogram().explain_data(X_train, y_train, name = 'Train Data')
    # show(hist)

    # %% [markdown]
    # ## Train the Explainable Boosting Machine (EBM)

    # %%
    
    # ebm = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
    # ebm.fit(X_train, y_train)   #Works on dataframes and numpy arrays


    # %%
    
    seed =1
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    ebm1 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
    ebm1.fit(X_train, y_train)  

    seed +=10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    ebm2 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
    ebm2.fit(X_train, y_train)  


    seed +=10
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=seed)
    ebm3 = ExplainableBoostingClassifier(random_state=seed, n_jobs=-1)
    ebm3.fit(X_train, y_train)  


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
# ebm2.preprocessor_.col_bin_edges_[ebm1.feature_groups_[0]]
models=[ebm1, ebm2, ebm3]
ebm = ebm1
new_feature_groups = []

new_interactions =set()
new_features =[]

feature_len = len(ebm.preprocessor_.feature_names)

# for index, feature_group in enumerate(ebm.feature_groups_[:]):

#         if len(feature_group) == 1:         
#             new_features.append((feature_group, ebm.feature_names[index] , ebm.feature_types[index]))

for model in models:  
    for index, feature_group in enumerate(model.feature_groups_):

        if len(feature_group) != 1:
            new_interactions.add( (feature_group, model.feature_names[index] , model.feature_types[index]))
       


for index, feature_group in enumerate(ebm.feature_groups_):           

            # interction tuples
            if len(feature_group) != 1:
                # Exluding interction tuples from bin edge merges              
                continue

            log_odds_tensors = []

            # Normalzing the bin edges for different models for each feature group
            if index in ebm.preprocessor_.col_bin_edges_.keys():

                print('---------', index) #, len(ebm1.preprocessor_.col_bin_edges_[index]))
                # new_bin_edges = ebm.preprocessor_.col_bin_edges_[index]

                # list(ebm1.preprocessor_.col_bin_edges_.keys())[0]
                # for model in models:
                #     print(len(model.preprocessor_.col_bin_edges_[index]))
                #     new_bin_edges = set(new_bin_edges).union(set(model.preprocessor_.col_bin_edges_[index]))
               
                new_bin_edges = sorted(set().union(*[ set(model.preprocessor_.col_bin_edges_[index]) for model in models]))

                print(' -->' ,len(new_bin_edges))
            
            # Exisitng approach
            for model in models:

                # ??, do we need to support multi level merges (merge of merged models)? 
                # ebm.bagged_models_.extend(model.bagged_models_)

                for estimator in model.bagged_models_:
                    # if have different bin_edges for this fearture group:                    
                    model_bin_edges = model.preprocessor_.col_bin_edges_[index]

                    ne = np.searchsorted(model_bin_edges, new_bin_edges + [np.inf])

                    mvalues = estimator.model_[index][1:]

                    # expanding the exsiting model_ values to cover the expanded new bin edges
                    new_model_ = [ mvalues[x-1] if x > 0 and x <=len(mvalues) else np.nan for x in ne[1:] ]

                    new_model_ = [estimator.model_[index][0]] + new_model_

                    # new_model_ = expand( estimator.model_[index] , new_bin_edges )
                    # log_odds_tensors.append(estimator.model_[index])
                    log_odds_tensors.append(new_model_)

            averaged_model = np.nanmean(np.array(log_odds_tensors), axis=0)
            model_errors = np.std(np.array(log_odds_tensors), axis=0)

            ebm.additive_terms_.append(averaged_model)
            ebm.term_standard_deviations_.append(model_errors)


# main_indices + new_pair_indices
ebm.feature_groups_ = ebm.feature_groups_[:feature_len]+ [ feature[0] for feature in new_interactions]
ebm.feature_names = ebm.feature_names[:feature_len] + [ feature[1] for feature in new_interactions]
ebm.feature_types = ebm.feature_types[:feature_len] + [ feature[2] for feature in new_interactions]


# %%
from interpret.glassbox.ebm.utils import *


# %%
EBMUtils.merge_models(models)


# %%
len(set(ebm2.preprocessor_.col_bin_edges_[2]).union(set((ebm1.preprocessor_.col_bin_edges_[2]))))


# %%
ebm1.preprocessor_.col_bin_edges_

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


