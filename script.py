from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_score, recall_score, f1_score, roc_curve, auc
import sklearn
import joblib
import boto3
import pathlib
from io import StringIO 
import argparse
import joblib
import os
import numpy as np
import pandas as pd

def model_fn(model_dir):
    """
    Load the model for inference
    """
    clf = joblib.load(os.path.join(model_dir, "model.joblib"))
    return clf

if __name__ == "__main__":

    print("[INFO] Extracting arguments")
    parser = argparse.ArgumentParser()

    # Hyperparameters
    parser.add_argument("--n_estimators", type=int, default=100)
    parser.add_argument("--random_state", type=int, default=0)

    # Directories
    parser.add_argument("--model-dir", type=str, default=os.environ.get("SM_MODEL_DIR"))
    parser.add_argument("--train", type=str, default=os.environ.get("SM_CHANNEL_TRAIN"))
    parser.add_argument("--test", type=str, default=os.environ.get("SM_CHANNEL_TEST"))

    args, _ = parser.parse_known_args()

    print("[INFO] Reading data")
    
    # --- SMART LOADER START ---
    # Instead of assuming the filename is "train.csv", we find whatever CSV is there.
    # This fixes 99% of FileNotFoundError issues in SageMaker.
    
    print(f"Files in Train Directory: {os.listdir(args.train)}")
    print(f"Files in Test Directory: {os.listdir(args.test)}")

    train_file = [f for f in os.listdir(args.train) if f.endswith('.csv')][0]
    test_file = [f for f in os.listdir(args.test) if f.endswith('.csv')][0]

    print(f"Loading Train File: {train_file}")
    train_df = pd.read_csv(os.path.join(args.train, train_file), header=None)
    
    print(f"Loading Test File: {test_file}")
    test_df = pd.read_csv(os.path.join(args.test, test_file), header=None)
    # --- SMART LOADER END ---

    # Split Data (Column 0 is target)
    y_train = train_df.iloc[:, 0]
    X_train = train_df.iloc[:, 1:]
    
    y_test = test_df.iloc[:, 0]
    X_test = test_df.iloc[:, 1:]

    print("[INFO] Training Model")
    model = RandomForestClassifier(
        n_estimators=args.n_estimators, 
        random_state=args.random_state, 
        verbose=True
    )
    
    model.fit(X_train, y_train)
    print("[INFO] Model Trained")

    # Evaluate
    print("[INFO] Evaluating Model")
    y_pred_test = model.predict(X_test)
    test_acc = accuracy_score(y_test, y_pred_test)
    test_rep = classification_report(y_test, y_pred_test)

    print(f"---- METRICS RESULTS FOR TESTING DATA ----")
    print(f"Total Rows are: {X_test.shape[0]}")
    print(f"[TESTING] Model Accuracy is: {test_acc}")
    print(f"[TESTING] Testing Report: \n{test_rep}")

    # Save Model
    print("[INFO] Saving Model")
    path = os.path.join(args.model_dir, "model.joblib")
    joblib.dump(model, path)
    print(f"Model saved to {path}")