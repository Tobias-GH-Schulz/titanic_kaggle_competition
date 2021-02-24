# %%
%load_ext autotime

# %%
from sklearn.model_selection import train_test_split
from autoPyTorch import (AutoNetClassification,
                         AutoNetMultilabel,
                         AutoNetRegression,
                         AutoNetImageClassification,
                         AutoNetImageClassificationMultipleDatasets)
import pandas as pd
import numpy as np
import os as os
import json

from sklearn.metrics import accuracy_score


# %%
data = pd.read_csv('Collection/titanic_comp/Data/Train_clean_1.csv')
X = data.drop(columns=['Survived','Unnamed: 0'],axis=1)
y = data.Survived
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=25, test_size=0.2)
# %%
data.head()

# %%
X_test.head()



# %%
X_train.head()
# %%
# running Auto-PyTorch
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=300,
                                    min_budget=30,
                                    max_budget=90)

autoPyTorch.fit(X_train, y_train, validation_split=0.3)
y_pred = autoPyTorch.predict(X_test)


# %%




# %%
autonet.print_help()


# %%
autonet = AutoNetClassification(config_preset="full_cs", result_logger_dir="Collection/titanic_comp/logs/")
# %%
results_fit = autonet.fit(X_train,
                          y_train,
                          validation_split=0.3,
                          refit=True)

# %%

y_pred = autoPyTorch.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))
# %%
with open("logs/results_fit.json", "w") as file:
    json.dump(results_fit, file)
# %%
import torch
torch.cuda.is_available()
# %%
from autoPyTorch import AutoNetClassification
# %%
# data and metric imports
import sklearn.model_selection
import sklearn.datasets
import sklearn.metrics

# %%
X, y = sklearn.datasets.load_digits(return_X_y=True)
X_train, X_test, y_train, y_test = \
        sklearn.model_selection.train_test_split(X, y, random_state=1)

# running Auto-PyTorch
autoPyTorch = AutoNetClassification("tiny_cs",  # config preset
                                    log_level='info',
                                    max_runtime=300,
                                    min_budget=30,
                                    max_budget=90)

autoPyTorch.fit(X_train, y_train, validation_split=0.3)
y_pred = autoPyTorch.predict(X_test)

print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))


# %%
test_passenger_df = pd.read_csv('Collection/titanic_comp/Data/test_clean.csv')
test_passenger_df = test_passenger_df.drop(columns=['Unnamed: 0'],axis=1)
test_passenger_df.head()
y_pred = autoPyTorch.predict(test_passenger_df)
# %%
print(y_pred)

# print("Accuracy score", accuracy_score(y_test, y_pred))

test_passenger_df = pd.read_csv('Collection/titanic_comp/Data/test_clean.csv')
tree = test_passenger_df.PassengerId

submission = pd.DataFrame()
submission['PassengerID'] = tree.tolist()
submission['Survived'] = y_pred.Survived

print(submission.head(5))

submission.to_csv("Collection/titanic_comp/Previous_submissions/NN-submission-1.csv", index=False)
# %%
test_passenger_df = pd.read_csv('Collection/titanic_comp/Data/test.csv')
tree = test_passenger_df.PassengerId

submission = pd.DataFrame()
submission['PassengerID'] = tree.tolist()
submission['Survived'] = y_pred

print(submission.head(5))

submission.to_csv("Collection/titanic_comp/Previous_submissions/NN-submission-1-bak.csv", index=False)
# %%

submission.head()
# %%
from tensorboardX import SummaryWriter
#SummaryWriter encapsulates everything
writer = SummaryWriter('Collection/titanic_comp/runs/exp-1')
#creates writer object. The log will be saved in 'runs/exp-1'
writer2 = SummaryWriter()
#creates writer2 object with auto generated file name, the dir will be something like 'runs/Aug20-17-20-33'
writer3 = SummaryWriter(comment='3x learning rate')
#creates writer3 object with auto generated file name, the comment will be appended to the filename. The dir will be something like 'runs/Aug20-17-20-33-3xlearning rate'
# %%
from tensorboard_logger import configure, log_value

configure("runs/run-1234", flush_secs=5)
# %%
tensorboard --logdir=runs
# %%
from autoPyTorch import AutoNetEnsemble
autoPyTorchEnsemble = AutoNetEnsemble(AutoNetClassification, "tiny_cs", max_runtime=600, min_budget=15, max_budget=120)
hope = autoPyTorchEnsemble.fit(X_train, y_train, validation_split=0.3)

# %%
X_test


# %%
print(hope)
y_pred = autoPyTorchEnsemble.predict(X_test, metrics=True)
# print("Accuracy score", sklearn.metrics.accuracy_score(y_test, y_pred))

# score = sklearn.metrics.accuracy_score(y_test, y_pred)

# if (score > .77):
#     test_passenger_df = pd.read_csv('Collection/titanic_comp/Data/test_clean.csv')
#     tree = test_passenger_df.PassengerId
#     submission = pd.DataFrame()
#     submission['PassengerID'] = tree.tolist()
#     submission['Survived'] = y_pred.Survived
#     print(submission.head(5))
#     submission.to_csv("Collection/titanic_comp/Previous_submissions/NN-submission-2.csv", index=False)
# else:
#     Print ('Too low!!!')
# %%
