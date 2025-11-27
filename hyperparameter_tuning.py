import glob
import os
import numpy as np
import librosa
import openl3

from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from xgboost import XGBClassifier

################### Load Data #####################
files = glob.glob(r"C:\\Users\\Asus\\Desktop\\Akarsh\\BabyCryingSounds\\**\\*.wav", recursive=True)
labels = [os.path.basename(os.path.dirname(f)) for f in files]

features1, features2, features3 = [], [], []

for f in files:
    y, sr = librosa.load(f, sr=None, mono=True)

    # MFCC
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    features1.append(np.mean(mfcc.T, axis=0))

    # Mel-spectrogram
    melspc = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    features2.append(np.mean(melspc.T, axis=0))

    # OpenL3 embeddings
    emb, ts = openl3.get_audio_embedding(y, sr, content_type="env", embedding_size=512)
    features3.append(emb.mean(axis=0))

X1, X2, X3 = np.vstack(features1), np.vstack(features2), np.vstack(features3)

le = LabelEncoder()
y = le.fit_transform(np.array(labels))

X1_train, X1_test, y1_train, y1_test = train_test_split(X1, y, test_size=0.6, stratify=y)
X2_train, X2_test, y2_train, y2_test = train_test_split(X2, y, test_size=0.6, stratify=y)
X3_train, X3_test, y3_train, y3_test = train_test_split(X3, y, test_size=0.6, stratify=y)

scaler1, scaler2, scaler3 = StandardScaler(), StandardScaler(), StandardScaler()
X1_train_scaled, X1_test_scaled = scaler1.fit_transform(X1_train), scaler1.transform(X1_test)
X2_train_scaled, X2_test_scaled = scaler2.fit_transform(X2_train), scaler2.transform(X2_test)
X3_train_scaled, X3_test_scaled = scaler3.fit_transform(X3_train), scaler3.transform(X3_test)


################### Hyperparameter Tuning #####################

def tune_model(model, param_grid, X, y, search_type="grid"):
    if search_type == "grid":
        search = GridSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2)
    else:
        search = RandomizedSearchCV(model, param_grid, cv=3, scoring="accuracy", n_jobs=-1, verbose=2, n_iter=20, random_state=42)
    search.fit(X, y)
    print(f"Best parameters for {model.__class__.__name__}: {search.best_params_}")
    print(f"Best CV Score: {search.best_score_}")
    return search.best_params_


# Random Forest
rf_params = {
    'n_estimators': [100, 200, 400, 600],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10]
}
tune_model(RandomForestClassifier(random_state=42), rf_params, X1_train, y1_train, search_type="random")

# XGBoost
xgb_params = {
    'n_estimators': [100, 200, 400],
    'max_depth': [3, 6, 10],
    'learning_rate': [0.01, 0.05, 0.1, 0.2],
    'subsample': [0.7, 0.8, 1.0]
}
tune_model(XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42), xgb_params, X1_train, y1_train, search_type="random")

# Logistic Regression
logreg_params = {
    'C': [0.01, 0.1, 1, 10],
    'l1_ratio': [0.1, 0.5, 0.7, 0.9],
    'max_iter': [500, 1000]
}
tune_model(LogisticRegression(penalty="elasticnet", solver="saga", random_state=42), logreg_params, X1_train_scaled, y1_train, search_type="grid")

# SVM
svm_params = {
    'C': [0.1, 1, 10],
    'kernel': ['linear', 'rbf', 'poly'],
    'gamma': ['scale', 'auto']
}
tune_model(SVC(probability=True, random_state=42), svm_params, X1_train_scaled, y1_train, search_type="grid")

print("Hyperparameter tuning complete for all models.")