import pandas as pd
from sklearn.feature_extraction import DictVectorizer
import xgboost as xgb
import pickle

df_train = pd.read_csv('data/customer_conversion_traing_dataset .csv')
y_train = df_train['Conversion (Target)']
del df_train['Conversion (Target)']
del df_train['LeadID']


train_dict = df_train.to_dict(orient='records')
dv = DictVectorizer(sparse=False)
dv.fit(train_dict)
X_train = dv.transform(train_dict)


features = dv.get_feature_names_out().tolist()
dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=features)

xgb_params = {
    'eta': 0.9, 
    'max_depth': 10,
    'min_child_weight': 10,
    
    'objective': 'binary:logistic',
    'nthread': 8,
    
    'seed': 1,
    'verbosity': 0,
}

model = xgb.train(xgb_params, dtrain, num_boost_round=5)

with open('model.bin', 'wb') as f_out: # 'wb' means write-binary
    pickle.dump((dv, model), f_out)