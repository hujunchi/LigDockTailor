import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.datasets import make_multilabel_classification
from sklearn.metrics import make_scorer, hamming_loss
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.multioutput import MultiOutputClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import learning_curve
from mlxtend.feature_selection import SequentialFeatureSelector as SFS
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score,hamming_loss,roc_auc_score
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler,MinMaxScaler,RobustScaler

# read data
df = pd.read_csv(r"../data/train_val.csv")
scaler = MinMaxScaler()
op_row = df.iloc[:,3:15].values
scaled_data = scaler.fit_transform(op_row)
df.iloc[:,3:15] = scaled_data
selected_rows =df.iloc[:, 3:].values
X = np.array(selected_rows)
df.iloc[:, 1] = df.iloc[:, 1].apply(lambda x: [int(i) for i in str(x).split(',')])
first_column_list = df.iloc[:, 1].tolist()
mlb = MultiLabelBinarizer()
y = mlb.fit_transform(first_column_list)

#split data
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# define parameter grid for GridSearchCV
param_grid = {
    'estimator__max_depth': [50,100,200,400,None],
    'estimator__n_estimators': [20,40,60,80,100,200,400,800,1600,2000],
    'estimator__min_samples_leaf': [1,2,4,8,16,32,64],
    'estimator__min_samples_split': [1,2,4,8,16,32,64],
    'estimator__max_leaf_nodes': [None,50,100,200],
    'estimator__class_weight': [None, 'balanced', 'balanced_subsample']    
}
# define scorer
def custom_scoring_gridsearch(y_true, y_pred):
    precision = precision_score(y_true, y_pred,average='weighted')
    recall = recall_score(y_true, y_pred,average='weighted')
    if precision == 0 and recall == 0:
        return 0.0
    beta = 0.5
    f05 = (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)
    if np.isnan(f05):
        return 0.0
    return f05
custom_scorer = make_scorer(custom_scoring_gridsearch, greater_is_better=True)

# model training
rf_classifier = RandomForestClassifier(random_state=42)
multi_output_classifier = MultiOutputClassifier(rf_classifier)
grid_search = GridSearchCV(multi_output_classifier, param_grid, cv=5, n_jobs=80,scoring=custom_scorer)
grid_search.fit(X_train,y_train)

