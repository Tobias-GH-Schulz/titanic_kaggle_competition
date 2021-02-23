# %%

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
# %%
# NOTE: Make sure that the outcome column is labeled 'target' in the data file
df = pd.read_csv('Collection/titanic_comp/Train_clean.csv')
test = pd.read_csv('Collection/titanic_comp/Data/Test_clean.csv')

X = df.drop(['class','Unnamed: 0'], axis=1)
y = df['class']

# Average CV score on the training set was: 0.8249615975422427
exported_pipeline = GradientBoostingClassifier(learning_rate=0.1, 
                                               max_depth=1, 
                                               max_features=0.35000000000000003, 
                                               min_samples_leaf=1, 
                                               min_samples_split=8, 
                                               n_estimators=100, 
                                               subsample=0.35000000000000003)
# Fix random state in exported estimator
if hasattr(exported_pipeline, 'random_state'):
    setattr(exported_pipeline, 'random_state', 42)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)
