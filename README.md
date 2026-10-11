# ChurnSense AI — Customer Churn Prediction

ChurnSense AI is a Streamlit dashboard for exploring the IBM Telco Customer Churn dataset and assessing a customer with a Decision Tree model trained in Google Colab. It does not contain a trained model or sample evaluation scores yet. Until you add your real model bundle, prediction controls stay disabled and evaluation values are shown as unavailable.

> **See the risk. Understand the why. Retain the relationship.**

## Run in this Codespace

Open a terminal in the project folder and run:

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
streamlit run app.py
```

Streamlit prints a local URL (usually `http://localhost:8501`). In a Codespace, open the forwarded port 8501 when prompted. To stop the app, press `Ctrl+C` in the terminal.

## Add project data

Upload your IBM Telco Customer Churn CSV from the sidebar, or place it at `data/telco_customer_churn.csv`. The dashboard requires the columns `Churn`, `Contract`, `tenure`, and `MonthlyCharges`; the standard IBM dataset includes these. Dataset metrics and insight charts are calculated from the loaded CSV, not from hard-coded example values.

## Add the trained model

Create a `models/` folder and place the model bundle at:

```text
models/churn_model_bundle.joblib
```

The app loads this bundle in `app.py` using `joblib.load`. It expects a dictionary containing a fitted scikit-learn estimator under `model`. Export a trusted artifact you created yourself; joblib files can execute code when loaded.

For a scikit-learn Pipeline that accepts the original IBM Telco fields, a bundle can look like:

```python
bundle = {
	"model": fitted_pipeline,
	"input_mode": "raw",
	"feature_columns": ["gender", "SeniorCitizen", "Partner", "Dependents",
						"tenure", "PhoneService", "MultipleLines",
						"InternetService", "OnlineSecurity", "OnlineBackup",
						"DeviceProtection", "TechSupport", "StreamingTV",
						"StreamingMovies", "Contract", "PaperlessBilling",
						"PaymentMethod", "MonthlyCharges", "TotalCharges"],
	"metrics": {"accuracy": actual_accuracy, "precision": actual_precision,
				"recall": actual_recall, "f1": actual_f1},
	"evaluation": {"test_size": actual_test_fraction},
	"confusion_matrix": actual_confusion_matrix.tolist(),
}
```

Then save it from the same compatible Python/scikit-learn environment with:

```python
import joblib
joblib.dump(bundle, "models/churn_model_bundle.joblib")
```

For a classifier trained on one-hot encoded columns rather than a raw-input Pipeline, set `input_mode` to `"encoded"` and include the exact ordered training columns in `feature_columns`. The app creates IBM-style one-hot columns from the form and aligns them to this list. Ensure the model was trained with the same field names, categories, and preprocessing. `metrics`, `evaluation`, and `confusion_matrix` are optional and should only contain results calculated from your actual held-out test data. Accuracy, precision, recall, and F1 are expected as fractions from 0 to 1 (or as the original numeric score); the app does not derive or invent them.

The model must expose `predict_proba` and two classes to produce the requested estimated churn probability. Its positive class should be labelled `Yes`, `1`, `True`, `churn`, or `churned` (otherwise the second class is treated as churn). Predictions are model estimates, not real-world guarantees.

## Pages

- **Overview:** Project context, dataset-derived portfolio KPIs, and churn distribution.
- **Predict Customer:** Telco profile form; enabled only after the saved model bundle loads successfully.
- **Customer Insights:** Dataset churn patterns and, if available, fitted model feature importance.
- **Model Lab:** Decision Tree explanation, saved evaluation scores, training/testing split, and confusion matrix when included in the bundle.

If the model fails to load or its input schema does not match the form, the app reports the issue in the sidebar or prediction view. Confirm your artifact path, Python/scikit-learn compatibility, and training-time feature preparation.