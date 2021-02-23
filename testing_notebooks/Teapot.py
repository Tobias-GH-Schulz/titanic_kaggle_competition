# %%
import tpot
from tpot import TPOTClassifier
from sklearn.model_selection import train_test_split
import pandas as pd 
import numpy as np
import dask.dataframe as dd
df = pd.read_csv('Collection/titanic_comp/Data/Train_clean (1).csv')

# %%
df.head()

# %%
df = df.rename(columns={'Survived':'class'})



# %%
pipeline_optimizer = TPOTClassifier(generations=25, population_size=40, cv=10,
                                    random_state=42, verbosity=2, max_eval_time_mins=10,)
# %%
X_Train = df.drop(['class','Unnamed: 0'], axis=1)
y = df['class']


# %%
from dask_ml.preprocessing import Categorizer, DummyEncoder
from sklearn.pipeline import make_pipeline
pipe = make_pipeline(
   Categorizer(),
   DummyEncoder(),
)
X = pipe.fit(X)

# %%
from sklearn.preprocessing import OneHotEncoder
enc = OneHotEncoder(handle_unknown='ignore')
X = enc.fit_transform(X)

# %%
X

# %%
from dask_ml.model_selection import train_test_split, GridSearchCV

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.70, test_size=0.30,shuffle=True)

# %%


# %%
X_test.head()

#  %%
y_test.head()

# %%
pipeline_optimizer.fit(X_train, y_train)

# %%


print(pipeline_optimizer.score(X_test, y_test))
# %%

pipeline_optimizer.export('Collection/titanic_comp/New_pipeline.py')

# %%
