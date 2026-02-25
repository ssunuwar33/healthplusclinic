import joblib
import pandas as pd
import matplotlib.pyplot as plt

from imblearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay,confusion_matrix
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


from feature_eng import feature_engineering

#load data
df = pd.read_csv('D:/healthplusclinic/data/01-rawdata/rawdata.csv')
df = feature_engineering(df)

X = df.drop('is_no_show_0_1', axis=1)
y = df['is_no_show_0_1']

# splitting data by stratifying as we have imbalance data
X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2, random_state=42, stratify=y)
print(X_train.shape)
print(X_test.shape)



# ------------------------
# Column Groups
# ------------------------



one_cat = ['insurance_type','clinic_assignment','city','is_weekend','is_weekdays']
label_cat = ['time_of_day','age_group','patient_clinic_frequency_visit']

categorical_cols = one_cat + label_cat

num_features = [
    col for col in X.columns
    if col not in categorical_cols
]


# Ensure categorical columns are string
X_train[one_cat + label_cat] = X_train[one_cat + label_cat].astype(str)
X_test[one_cat + label_cat] = X_test[one_cat + label_cat].astype(str)

# ------------------------
# Pipelines for Each Type
# ------------------------

num_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="median"))
])

onehot_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent",fill_value='missing')),
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])

ordinal_pipeline = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent",fill_value='missing')),
    ("encoder", OrdinalEncoder())
])


# ------------------------
# Final Preprocessor
# ------------------------

preprocessor = ColumnTransformer(
    transformers=[
        ("num", num_pipeline, num_features),
        ("one_cat", onehot_pipeline, one_cat),
        ("label_cat", ordinal_pipeline, label_cat)
    ],
    remainder="drop"   
)


# Model Development
# model instantiate
xgb_model = XGBClassifier(
    max_depth= 3,
    learning_rate= 0.059948, # Search across orders of magnitude
    n_estimators= 100,
    subsample= 0.6,
    colsample_bytree= 0.8,
    tree_method = 'hist'
)


#pipeline for preprocessing and handling imbalance data
pipeline = Pipeline([
        ('preprocessing', preprocessor),
        ('smote', SMOTE(random_state=42)),
        ('xgb', xgb_model)

])



#fitting to train and test data
pipeline.fit(X_train,y_train)




y_train_pred = pipeline.predict(X_train)
y_pred = pipeline.predict(X_test)


# model evalution

print('Accuracy on train data: ', accuracy_score(y_train,y_train_pred))
print('Accuracy on test data: ',accuracy_score(y_test,y_pred))
print(classification_report(y_test,y_pred))
confusion_mat = confusion_matrix(y_test,y_pred, normalize='all')
print(f'Confusion Matrix \n',confusion_mat)
ConfusionMatrixDisplay(confusion_mat,display_labels=['show','no_show']).plot(cmap=plt.cm.Blues)
plt.grid(False)


##########
#Save Model
##########

model = (preprocessor,xgb_model)
joblib.dump(model,'D:/healthplusclinic/models/healthplus.pkl')