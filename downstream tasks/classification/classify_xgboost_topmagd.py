import json
import gensim
import collections
import numpy as np
import pandas as pd
from xgboost.sklearn import XGBClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score


def print_precison_recall_f1(y_true, y_pre):
    """打印精准率、召回率和F1值"""
    print("打印精准率、召回率和F1值")
    print(classification_report(y_true, y_pre))
    a = round(accuracy_score(y_true, y_pre), 4)
    f1 = round(f1_score(y_true, y_pre, average='weighted'), 4)
    p = round(precision_score(y_true, y_pre, average='weighted'), 4)
    r = round(recall_score(y_true, y_pre, average='weighted'), 4)
    print("Accuracy: {}, Precision: {}, Recall: {}, F1: {} ".format(a, p, r, f1))


def importance_features_top(model_str, model, x_train):
    """打印模型的重要指标，排名top10指标"""
    print("打印XGBoost重要指标")
    feature_importances_ = model.feature_importances_
    feature_names = list(range(len(x_train)))
    importance_col = pd.DataFrame([*zip(feature_names, feature_importances_)],
                                  columns=['a', 'b'])
    importance_col_desc = importance_col.sort_values(by='b', ascending=False)
    print(importance_col_desc.iloc[:10, :])


with open('melody.json') as f:
    melody = json.load(f)
with open('top_magd.json') as f:
    top_magd = json.load(f)
with open('masd.json') as f:
    masd = json.load(f)

model = gensim.models.Doc2Vec.load('d2v.bin')

X = [model.docvecs[sample[-1]] for sample in top_magd]
Y = [sample[-2] for sample in top_magd]

class_names = list(sorted(set(Y)))
print(len(set(Y)))  # 13
print([(i, j) for i, j in enumerate(class_names)])
word_counts = collections.Counter(Y)
print(word_counts)

for i, name in enumerate(Y):
    Y[i] = class_names.index(name)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=1,
    stratify=Y
)

xgb = XGBClassifier(min_child_weight=5, max_depth=20, objective='multi:softmax', num_class=13)
xgb.fit(np.array(X_train), Y_train)

importance_features_top('xgboost', xgb, X_train)

Y_pred = xgb.predict(X_val)
print_precison_recall_f1(Y_val, Y_pred)

# ----------------------------

X = [model.docvecs[sample[-1]] for sample in masd]
Y = [sample[-2] for sample in masd]

class_names = list(sorted(set(Y)))
print(len(set(Y)))  # 25
print([(i, j) for i, j in enumerate(class_names)])
word_counts = collections.Counter(Y)
print(word_counts)

for i, name in enumerate(Y):
    Y[i] = class_names.index(name)

X_train, X_val, Y_train, Y_val = train_test_split(
    X, Y,
    test_size=0.2,
    random_state=1,
    stratify=Y
)

xgb = XGBClassifier(min_child_weight=5, max_depth=20, objective='multi:softmax', num_class=25)
xgb.fit(np.array(X_train), Y_train)

importance_features_top('xgboost', xgb, X_train)

Y_pred = xgb.predict(X_val)
print_precison_recall_f1(Y_val, Y_pred)
