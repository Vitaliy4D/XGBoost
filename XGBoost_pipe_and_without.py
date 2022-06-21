import pandas as pd
import numpy as np
import matplotlib.pyplot as plt



from sklearn.model_selection import train_test_split

# Read the data
X_full = pd.read_csv('train.csv', index_col='Id')
X_test_full = pd.read_csv('test.csv', index_col='Id')

# Remove rows with missing target, separate target from predictors
X_full.dropna(axis=0, subset=['SalePrice'], inplace=True)
y = X_full.SalePrice
X_full.drop(['SalePrice'], axis=1, inplace=True)

# Break off validation set from training data
X_train_full, X_valid_full, y_train, y_valid = train_test_split(X_full, y, 
                                                                train_size=0.8, test_size=0.2,
                                                                random_state=0)

# "Cardinality" means the number of unique values in a column
# Select categorical columns with relatively low cardinality (convenient but arbitrary)
categorical_cols = [cname for cname in X_train_full.columns if
                    X_train_full[cname].nunique() < 10 and 
                    X_train_full[cname].dtype == "object"]

# Select numerical columns
numerical_cols = [cname for cname in X_train_full.columns if 
                X_train_full[cname].dtype in ['int64', 'float64']]

# Keep selected columns only
my_cols = categorical_cols + numerical_cols
X_train = X_train_full[my_cols].copy()
X_valid = X_valid_full[my_cols].copy()
X_test = X_test_full[my_cols].copy()


# "Cardinality" means the number of unique values in a column
# Get number of unique entries in each column with categorical data
object_nunique = list(map(lambda col: X_train[col].nunique(), categorical_cols))
d = dict(zip(categorical_cols, object_nunique))

# Print number of unique entries by column, in ascending order
sorted(d.items(), key=lambda x: x[1])


# find na in categorical vars
cn=[]
qna=[]
for i in X_train[categorical_cols]:
    if X_train[i].isna().sum()>0:
        cn.append(i)
        qna.append(X_train[i].isna().sum())
r=list(zip(cn,qna))
print(r)

## Find best params with GridSearchCV
from sklearn.model_selection import GridSearchCV

est = GradientBoostingRegressor()

param_grid = {
    'random_state': [0],
    'learning_rate': [0.1],
    'max_features': [4, 10, 14, 26],
    'n_estimators': [350, 400, 450],
    'subsample': [0.2, 0.3, 0.4, 0.5],  
    'min_samples_split': [4, 10, 14, 26],
    'max_depth': [4, 10, 14, 26],
    'max_leaf_nodes': [4, 10, 14, 26]
}

gs_cv = GridSearchCV(est, param_grid, n_jobs=4).fit(OH_X_train, y_train)

gs_cv.best_params_


####  ML with pipeline  ####
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
numerical_transformer = SimpleImputer(strategy='mean')

# Preprocessing for categorical data
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

# Bundle preprocessing for numerical and categorical data
preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_cols),
        ('cat', categorical_transformer, categorical_cols)
    ])

# Define model
params = {
    "n_estimators": 700,
    "random_state": 0
}
model = GradientBoostingRegressor(**params)

# Bundle preprocessing and modeling code in a pipeline
reg = Pipeline(steps=[('preprocessor', preprocessor),
                      ('model', model)
                     ])

# Preprocessing of training data, fit model 
reg.fit(X_train, y_train)

# Preprocessing of validation data, get predictions
y_pred = reg.predict(X_valid)

print('MAE:', mean_absolute_error(y_valid, y_pred))






#### ML without pipeline ####

from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

# Preprocessing for numerical data
imp_mean = SimpleImputer(strategy='mean') #  strategy='mean' / 'median' / 'most_frequent' / 'constant'

imp_mean.fit(X_train[numerical_cols])

imputed_X_train1 = imp_mean.transform(X_train[numerical_cols])
imputed_X_valid1 = imp_mean.transform(X_valid[numerical_cols])

# Fill in the lines below: imputation removed column names; put them back
imputed_num_X_train = pd.DataFrame(imputed_X_train1,
                                   columns=X_train[numerical_cols].columns.tolist())
imputed_num_X_valid = pd.DataFrame(imputed_X_valid1,
                                   columns=X_valid[numerical_cols].columns.tolist())


# X_train[categorical_cols]

imp_const = SimpleImputer(strategy='constant')

imp_const.fit(X_train[categorical_cols])

imputed_X_train2 = imp_const.transform(X_train[categorical_cols])
imputed_X_valid2 = imp_const.transform(X_valid[categorical_cols])

# Fill in the lines below: imputation removed column names; put them back
imputed_cat_X_train = pd.DataFrame(imputed_X_train2,
                                   columns=X_train[categorical_cols].columns.tolist())

imputed_cat_X_valid = pd.DataFrame(imputed_X_valid2,
                                   columns=X_valid[categorical_cols].columns.tolist())


# cat_no_na
# cat_no_na=[i for i in categorical_cols if i not in cn]
# print(cat_no_na)

# Apply one-hot encoder to each column with categorical data
OH_encoder = OneHotEncoder(handle_unknown='ignore', sparse=False)
OH_cols_train = pd.DataFrame(OH_encoder.fit_transform(imputed_cat_X_train))
OH_cols_valid = pd.DataFrame(OH_encoder.transform(imputed_cat_X_valid))

# One-hot encoding removed index; put it back
OH_cols_train.index = imputed_cat_X_train.index
OH_cols_valid.index = imputed_cat_X_valid.index

# Add one-hot encoded columns to numerical features
OH_X_train = pd.concat([imputed_num_X_train, OH_cols_train], axis=1)
OH_X_valid = pd.concat([imputed_num_X_valid, OH_cols_valid], axis=1)


# Define model
params = {
    'random_state': 0,
    'learning_rate': 0.1,
    'loss': 'huber',
    'max_features': 10,
    'n_estimators': 400
}

reg = GradientBoostingRegressor(**params)

# Preprocessing of training data, fit model 
reg.fit(OH_X_train, y_train)

# Preprocessing of validation data, get predictions
y_pred = reg.predict(OH_X_valid)

print('MAE:', mean_absolute_error(y_valid, y_pred))

# OH_X_train.rename(columns={})
# rename columns in OH_X_train to be only str instead of str + digit(dummy of categoricals)
OH_col_names=OH_X_train.columns.tolist()
OH_col_names_2=[]
for i in OH_col_names:
    OH_col_names_2.append(str(i))
print(OH_col_names_2)


## In statistics, deviance is a goodness-of-fit statistic for a statistical model; 
# it is often used for statistical hypothesis testing. 
# It is a generalization of the idea of using the sum of squares of residuals (RSS) 
# in ordinary least squares to cases where model-fitting is achieved by maximum likelihood. 
# It plays an important role in exponential dispersion models and generalized linear models. 

test_score = np.zeros((params["n_estimators"],), dtype=np.float64)
for i, y_pred in enumerate(reg.staged_predict(OH_X_valid)):
    test_score[i] = reg.loss_(y_valid, y_pred)

fig = plt.figure(figsize=(6, 6))
plt.subplot(1, 1, 1)
plt.title("Deviance")
plt.plot(
    np.arange(params["n_estimators"]) + 1,
    reg.train_score_,
    "b-",
    label="Training Set Deviance",
)
plt.plot(
    np.arange(params["n_estimators"]) + 1, test_score, "r-", label="Test Set Deviance"
)
plt.legend(loc="upper right")
plt.xlabel("Boosting Iterations")
plt.ylabel("Deviance")
fig.tight_layout()
plt.show()



## Feature Importance (MDI) and Permutation Importance (test set)


feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(18, 10))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(numerical_cols)[sorted_idx])
plt.vlines(0.05,25,36, color='red')
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    reg, imputed_num_X_valid, y_valid, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(numerical_cols)[sorted_idx],
)
plt.vlines(0.05,25,36, color='red')
plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()


# feature impotance other way

indices = np.argsort(reg.feature_importances_)
# plot as bar chart
plt.figure(figsize=(10, 8))
plt.axvline(x=0.05, color='red')
plt.barh(np.arange(len(numerical_cols)), reg.feature_importances_[indices])
plt.yticks(np.arange(len(numerical_cols)) + 0.25, np.array(numerical_cols)[indices])
_ = plt.xlabel('Relative importance')


## NOT FINISHED feature importance of categorical vars

# I used DictVectorizer instead and it fixed the problem:
# Now fit the model and plot the feature importances:
X, y = df.iloc[:,:-1], df.iloc[:,-1]

# Import DictVectorizer
from sklearn.feature_extraction import DictVectorizer

# Convert df into a dictionary using .to_dict(): df_dict
df_dict = X.to_dict("records")

# Create the DictVectorizer object: dv
dv = DictVectorizer(sparse=False)

# Apply dv on df_dict: df_encoded
X_encoded = dv.fit_transform(df_dict)

X_encoded = pd.DataFrame(X_encoded)

X_train, X_test, y_train, y_test= train_test_split(X_encoded, y, test_size=0.2, random_state=123)

# Finally, you have to look up the names:
# Use pprint to make the vocabulary easier to read
# import pprint
# pprint.pprint(dv.vocabulary_)

feature_importance = reg.feature_importances_
sorted_idx = np.argsort(feature_importance)
pos = np.arange(sorted_idx.shape[0]) + 0.5
fig = plt.figure(figsize=(18, 10))
plt.subplot(1, 2, 1)
plt.barh(pos, feature_importance[sorted_idx], align="center")
plt.yticks(pos, np.array(categorical_cols)[sorted_idx])
plt.title("Feature Importance (MDI)")

result = permutation_importance(
    reg, OH_cols_valid, y_valid, n_repeats=10, random_state=42, n_jobs=2
)
sorted_idx = result.importances_mean.argsort()
plt.subplot(1, 2, 2)
plt.boxplot(
    result.importances[sorted_idx].T,
    vert=False,
    labels=np.array(categorical_cols)[sorted_idx],
)

plt.title("Permutation Importance (test set)")
fig.tight_layout()
plt.show()


## instead of gridsearch find out best param and lot it

def get_score(n):   
    # Define model
    params = {'random_state':0,
              'n_estimators': n
             }

    reg = GradientBoostingRegressor(**params)

    # Preprocessing of training data, fit model 
    reg.fit(OH_X_train, y_train)

    # Preprocessing of validation data, get predictions
    y_pred = reg.predict(OH_X_valid)

    return mean_absolute_error(y_valid, y_pred)
    print('MAE:', mean_absolute_error(y_valid, y_pred))

n=list(range(50,600,50))
print(n)

# n=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0]

f={}
for i in n:
    name=i
    f.update({name:get_score(i)})
print(f)


# plot and print min result

plt.plot(f.keys(),f.values())
plt.axvline(x=[u for u in f.keys() if f.get(u)== min(f.values())],color='red')
[u for u in f.keys() if f.get(u)== min(f.values())]

