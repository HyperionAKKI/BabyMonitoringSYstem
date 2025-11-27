############################################################
#  Audio Classification (Baby Cry Detection) using OpenL3
#  Features: OpenL3 Embeddings (512-d)
#  Models: Random Forest, SVM, and optional MLP (PyTorch)
############################################################

import glob
import os
import librosa
import numpy as np
import openl3

############################################################
# Step 1: Load all audio files and labels
############################################################
files = glob.glob(r"C:\Users\akars\OneDrive\Desktop\Akarsh\Akarsh\BabyCryingSounds\**\*.wav", recursive=True)
labels = [os.path.basename(os.path.dirname(f)) for f in files]

############################################################2
# Step 2: Extract OpenL3 Embeddings
############################################################
features = []

for f in files:
    y, sr = librosa.load(f, sr=None, mono=True)
    emb, ts = openl3.get_audio_embedding(y, sr, content_type="env", embedding_size=512)
    vec = emb.mean(axis=0)
    features.append(vec)

X = np.vstack(features)
y = np.array(labels)

############################################################
# Step 3: Encode Labels & Split Data
############################################################
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

le = LabelEncoder()
y_encoded = le.fit_transform(y)

X_train, X_test,                                                                                                                                                    y_train, y_test = train_test_split(
    X, y_encoded, test_size=0.1, random_state=42
)

# Standardize embeddings
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

############################################################
# Step 4: Dimensionality Reduction (PCA)
############################################################
from sklearn.decomposition import PCA

n_components = min(256, X_train_scaled.shape[0], X_train_scaled.shape[1])
pca = PCA(n_components=n_components, random_state=42)
X_train_pca = pca.fit_transform(X_train_scaled)
X_test_pca = pca.transform(X_test_scaled)


############################################################
# Step 5: Classification Models
############################################################
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score

# --- Random Forest ---
rf = RandomForestClassifier(n_estimators=300, random_state=42)
rf.fit(X_train_pca, y_train)
rf_acc = accuracy_score(y_test, rf.predict(X_test_pca))
print("Random Forest (OpenL3)")
print(classification_report(y_test, rf.predict(X_test_pca)))

# --- SVM (Linear Kernel) ---
svm = SVC(kernel='linear', C=1.0, probability=True, random_state=42)
svm.fit(X_train_pca, y_train)
svm_acc = accuracy_score(y_test, svm.predict(X_test_pca))
print("SVM (OpenL3)")
print(classification_report(y_test, svm.predict(X_test_pca)))

############################################################
# Step 6: Save the Best Model
############################################################
import joblib

if rf_acc >= svm_acc:
    best_model = rf
    best_name = "RandomForest_OpenL3"
    best_acc = rf_acc
else:
    best_model = svm
    best_name = "SVM_OpenL3"
    best_acc = svm_acc

print(f"\n✅ Best Model: {best_name}  |  Accuracy: {best_acc:.4f}")

os.makedirs("saved_models", exist_ok=True)
joblib.dump(best_model, f"saved_models/{best_name}.pkl")
joblib.dump(le, "saved_models/label_encoder.pkl")
joblib.dump(scaler, "saved_models/scaler.pkl")
joblib.dump(pca, "saved_models/pca.pkl")

print("✅ Saved model, label encoder, scaler, and PCA for inference.")
