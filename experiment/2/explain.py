import os
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import shap
from lime import lime_tabular

# Configuration
DATA_DIR = "../../data"
EXPERIMENT_DIR = os.path.dirname(os.path.abspath(__file__))
STAGE0_MODEL = os.path.join(EXPERIMENT_DIR, "../0/stage0_model.joblib")
TRAIN_PATH = os.path.join(DATA_DIR, "train_engineered.parquet")
TEST_PATH = os.path.join(DATA_DIR, "test_engineered.parquet")
PLOTS_DIR = os.path.join(EXPERIMENT_DIR, "plots")


def load_data():
    X_train = pd.read_parquet(TRAIN_PATH).drop(columns=['entry_id', 'patient_id', 'date', 'migraine_target'])
    X_test = pd.read_parquet(TEST_PATH).drop(columns=['entry_id', 'patient_id', 'date', 'migraine_target'])
    return X_train, X_test


def explain_shap(model, X_train, X_test, model_name):
    print(f"[{model_name}] Computing SHAP values...")
    # For ensembles/calibrated models, we use KernelExplainer or TreeExplainer
    # depending on the underlying estimator. Kernel is safer for calibrated wrappers.
    # We use a background summary to speed up KernelExplainer
    background = shap.kmeans(X_train, 10)
    explainer = shap.KernelExplainer(model.predict_proba, background)

    # Explain the first 20 instances of the test set for summary
    shap_values = explainer.shap_values(X_test.iloc[:20])

    # Save Summary Plot
    plt.figure()
    shap.summary_plot(shap_values[1], X_test.iloc[:20], show=False)
    plt.title(f"SHAP Global Importance - {model_name}")
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_shap_summary.png"))
    plt.close()


def explain_lime(model, X_train, X_test, model_name):
    print(f"[{model_name}] Generating LIME explanation...")
    explainer = lime_tabular.LimeTabularExplainer(
        training_data=X_train.values,
        feature_names=X_train.columns.tolist(),
        class_names=['No Migraine', 'Migraine'],
        mode='classification'
    )

    # Explain a random instance from the test set
    idx = 0
    exp = explainer.explain_instance(
        data_row=X_test.iloc[idx].values,
        predict_fn=model.predict_proba
    )

    # Save LIME plot
    fig = exp.as_pyplot_figure()
    plt.title(f"LIME Local Explanation (Idx {idx}) - {model_name}")
    plt.tight_layout()
    plt.savefig(os.path.join(PLOTS_DIR, f"{model_name}_lime_local.png"))
    plt.close()


def main():
    os.makedirs(PLOTS_DIR, exist_ok=True)
    X_train, X_test = load_data()

    # Explain Stage 0
    if os.path.exists(STAGE0_MODEL):
        model0 = joblib.load(STAGE0_MODEL)
        explain_shap(model0, X_train, X_test, "Stage0_Stacked")
        explain_lime(model0, X_train, X_test, "Stage0_Stacked")
    else:
        print("Stage 0 model not found. Run stage 0 training first.")


if __name__ == "__main__":
    main()