import os
import pandas as pd
import numpy as np
from dataclasses import dataclass
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from src.logger import logger
from src.utils import save_object

@dataclass
class DataTransformationConfig:
    project_root: str = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    preprocessor_obj_file_path: str = os.path.join(project_root, "artifacts", "preprocessor.pkl")

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()
        self.numerical_columns = [
            "tenure_months",
            "monthly_charges",
            "total_charges",
            "support_tickets",
            "app_logins_per_month",
            "cart_abandonment_rate"
        ]
        self.categorical_columns = [
            "contract",
            "payment_method",
            "discount_applied"
        ]

    def get_data_transformer_object(self):
        """Constructs and returns the scikit-learn ColumnTransformer object."""
        try:
            num_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler())
                ]
            )

            cat_pipeline = Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="most_frequent")),
                    ("one_hot_encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False))
                ]
            )

            preprocessor = ColumnTransformer(
                transformers=[
                    ("num_pipeline", num_pipeline, self.numerical_columns),
                    ("cat_pipeline", cat_pipeline, self.categorical_columns)
                ]
            )

            return preprocessor
        except Exception as e:
            logger.error(f"Error initializing preprocessor: {e}")
            raise e

    def initiate_data_transformation(self, train_path: str, test_path: str):
        try:
            train_df = pd.read_csv(train_path)
            test_df = pd.read_csv(test_path)

            logger.info("Reading train and test data for transformation.")
            preprocessing_obj = self.get_data_transformer_object()

            target_churn = "churn"
            target_clv = "clv"
            drop_cols = ["customer_id", target_churn, target_clv]

            input_feature_train_df = train_df.drop(columns=drop_cols, errors="ignore")
            target_churn_train = train_df[target_churn]
            target_clv_train = train_df[target_clv]

            input_feature_test_df = test_df.drop(columns=drop_cols, errors="ignore")
            target_churn_test = test_df[target_churn]
            target_clv_test = test_df[target_clv]

            logger.info("Applying preprocessing pipeline on training and testing feature sets.")
            input_feature_train_arr = preprocessing_obj.fit_transform(input_feature_train_df)
            input_feature_test_arr = preprocessing_obj.transform(input_feature_test_df)

            train_arr = np.c_[input_feature_train_arr, target_churn_train.values, target_clv_train.values]
            test_arr = np.c_[input_feature_test_arr, target_churn_test.values, target_clv_test.values]

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessing_obj
            )
            logger.info(f"Saved preprocessor object to {self.data_transformation_config.preprocessor_obj_file_path}")

            return train_arr, test_arr, self.data_transformation_config.preprocessor_obj_file_path

        except Exception as e:
            logger.error(f"Error in data transformation: {e}")
            raise e
