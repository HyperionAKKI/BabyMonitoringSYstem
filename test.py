############################################################
#  Test Script for Baby Cry Detection (OpenL3)
#  Loads: Model, Scaler, PCA, LabelEncoder
#  Input: Any .wav file
#  Output: Predicted Label + Confidence Score
############################################################

import os
import joblib
import librosa
import numpy as np
import openl3

# ----------------------------------------------------------
# Step 1: Load saved model components
# ----------------------------------------------------------
MODEL_DIR = "saved_models"

model_path = os.path.join(MODEL_DIR, "SVM_OpenL3.pkl")  # or RandomForest_OpenL3.pkl
label_encoder_path = os.path.join(MODEL_DIR, "label_encoder.pkl")
scaler_path = os.path.join(MODEL_DIR, "scaler.pkl")
pca_path = os.path.join(MODEL_DIR, "pca.pkl")

model = joblib.load(model_path)
le = joblib.load(label_encoder_path)
scaler = joblib.load(scaler_path)
pca = joblib.load(pca_path)

print(f"✅ Loaded model from: {model_path}")

# ----------------------------------------------------------
# Step 2: Define prediction function
# ----------------------------------------------------------
def predict_audio(filepath):
    if not os.path.exists(filepath):
        print("❌ File not found:", filepath)
        return

    # Load audio
    y, sr = librosa.load(filepath, sr=None, mono=True)

    # Extract OpenL3 embedding
    emb, ts = openl3.get_audio_embedding(y, sr, content_type="env", embedding_size=512)
    vec = emb.mean(axis=0).reshape(1, -1)

    # Scale + PCA
    
    vec_scaled = scaler.transform(vec)
    vec_pca = pca.transform(vec_scaled)

    # Predict
    pred = model.predict(vec_pca)[0]
    pred_label = le.inverse_transform([pred])[0]

    # Confidence (if supported)
    if hasattr(model, "predict_proba"):
        conf = np.max(model.predict_proba(vec_pca))
        print(f"🎯 Prediction: {pred_label} (Confidence: {conf:.2f})")
    else:
        print(f"🎯 Prediction: {pred_label}")

# ----------------------------------------------------------
# Step 3: Test with a new file
# ----------------------------------------------------------
if __name__ == "__main__":
    # Example test file
    test_file1= r"C:\Users\akars\OneDrive\Desktop\Akarsh\Akarsh\test.wav"
    test_file2=r"C:\Users\akars\OneDrive\Desktop\Akarsh\Akarsh\BabyCryingSounds\discomfort\196c.wav"
    test_file3=r"C:\Users\akars\OneDrive\Desktop\Akarsh\Akarsh\BabyCryingSounds\burping\522c.wav"
    test_file4=r"C:\Users\akars\OneDrive\Desktop\Akarsh\Akarsh\BabyCryingSounds\tired\306c.wav"
    test_file5=r"C:\Users\akars\OneDrive\Desktop\Akarsh\Akarsh\BabyCryingSounds\laugh\laugh_1.m4a_52.wav"
    predict_audio(test_file2)
