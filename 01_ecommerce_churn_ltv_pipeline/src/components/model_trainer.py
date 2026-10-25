import os
import json
import numpy as np
from dataclasses import dataclass
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score, mean_absolute_error

from src.logger import logger
from src.utils import save_object

@dataclass
class ModelTrainerConfig:
    project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    churn_model_path: str = os.path.join(project_root, "artifacts", "churn_model.pkl")
    ltv_model_path: str = os.path.join(project_root, "artifacts", "ltv_model.pkl")
    metrics_path: str = os.path.join(project_root, "artifacts", "metrics.json")
    feature_meta_path: str = os.path.join(project_root, "artifacts", "feature_importance.json")

class ModelTrainer:
    def __init__(self):
        self.config = ModelTrainerConfig()

    def initiate_model_trainer(self, train_array: np.ndarray, test_array: np.ndarray, feature_names: list = None):
        try:
            logger.info("Splitting train and test input arrays into features and dual targets (Churn & LTV).")
            X_train = train_array[:, :-2]
            y_churn_train = train_array[:, -2]
            y_ltv_train = train_array[:, -1]

            X_test = test_array[:, :-2]
            y_churn_test = test_array[:, -2]
            y_ltv_test = test_array[:, -1]

            # 1. Train Churn Classifier (GradientBoosting as primary enterprise model)
            logger.info("Training Churn Classifier models...")
            churn_models = {
                "GradientBoosting": GradientBoostingClassifier(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
                "RandomForest": RandomForestClassifier(n_estimators=100, max_depth=8, random_state=42),
                "LogisticRegression": LogisticRegression(max_iter=1000, random_state=42)
            }

            best_churn_score = -1.0
            best_churn_model = None
            best_churn_name = None
            churn_metrics = {}

            for name, model in churn_models.items():
                model.fit(X_train, y_churn_train)
                y_pred = model.predict(X_test)
                y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else y_pred
                
                roc_auc = roc_auc_score(y_churn_test, y_proba)
                f1 = f1_score(y_churn_test, y_pred)
                acc = accuracy_score(y_churn_test, y_pred)
                prec = precision_score(y_churn_test, y_pred)
                rec = recall_score(y_churn_test, y_pred)

                churn_metrics[name] = {
                    "accuracy": round(float(acc), 4),
                    "roc_auc": round(float(roc_auc), 4),
                    "f1_score": round(float(f1), 4),
                    "precision": round(float(prec), 4),
                    "recall": round(float(rec), 4)
                }

                if roc_auc > best_churn_score:
                    best_churn_score = roc_auc
                    best_churn_model = model
                    best_churn_name = name

            logger.info(f"Best Churn Model: {best_churn_name} with ROC-AUC: {best_churn_score:.4f}")
            save_object(self.config.churn_model_path, best_churn_model)

            # Feature Importance / Explainable AI (XAI)
            feature_importance_dict = {}
            if hasattr(best_churn_model, "feature_importances_"):
                importances = best_churn_model.feature_importances_
                names = feature_names if feature_names and len(feature_names) == len(importances) else [f"feature_{i}" for i in range(len(importances))]
                sorted_idx = np.argsort(importances)[::-1]
                feature_importance_dict = {names[i]: round(float(importances[i]), 4) for i in sorted_idx}
            elif hasattr(best_churn_model, "coef_"):
                coefs = np.abs(best_churn_model.coef_[0])
                names = feature_names if feature_names and len(feature_names) == len(coefs) else [f"feature_{i}" for i in range(len(coefs))]
                sorted_idx = np.argsort(coefs)[::-1]
                feature_importance_dict = {names[i]: round(float(coefs[i]), 4) for i in sorted_idx}

            with open(self.config.feature_meta_path, "w") as f:
                json.dump(feature_importance_dict, f, indent=2)

            # 2. Train LTV Regressor
            logger.info("Training Lifetime Value (LTV) Regressor...")
            ltv_models = {
                "GradientBoostingRegressor": GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42),
                "Ridge": Ridge()
            }

            best_ltv_score = -1.0
            best_ltv_model = None
            best_ltv_name = None
            ltv_metrics = {}

            for name, model in ltv_models.items():
                model.fit(X_train, y_ltv_train)
                y_pred = model.predict(X_test)

                r2 = r2_score(y_ltv_test, y_pred)
                mae = mean_absolute_error(y_ltv_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_ltv_test, y_pred))

                ltv_metrics[name] = {
                    "r2_score": round(float(r2), 4),
                    "mae": round(float(mae), 2),
                    "rmse": round(float(rmse), 2)
                }

                if r2 > best_ltv_score:
                    best_ltv_score = r2
                    best_ltv_model = model
                    best_ltv_name = name

            logger.info(f"Best LTV Model: {best_ltv_name} with R²: {best_ltv_score:.4f}")
            save_object(self.config.ltv_model_path, best_ltv_model)

            all_metrics = {
                "churn_models": churn_metrics,
                "best_churn_model": best_churn_name,
                "top_feature_importances": feature_importance_dict,
                "ltv_models": ltv_metrics,
                "best_ltv_model": best_ltv_name
            }

            with open(self.config.metrics_path, "w") as f:
                json.dump(all_metrics, f, indent=2)

            return all_metrics

        except Exception as e:
            logger.error(f"Error in model training: {e}")
            raise e
