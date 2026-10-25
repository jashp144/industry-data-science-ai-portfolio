import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List, Optional
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import RobustScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_percentage_error

class DragonRealEstateEngine:
    def __init__(self, data_path: Optional[str] = None, model_path: Optional[str] = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if data_path is None:
            data_path = os.path.join(project_root, "data", "dragon_housing_data.csv")
        if model_path is None:
            model_path = os.path.join(project_root, "data", "dragon_model.pkl")

        self.data_path = data_path
        self.model_path = model_path
        self.pipeline = None
        self.metrics = {}
        self.feature_columns = [
            "CRIM", "ZN", "INDUS", "CHAS", "NOX", "RM", "AGE", "DIS", "RAD", "TAX", "PTRATIO", "B", "LSTAT"
        ]

    def generate_synthetic_housing_data(self, n_samples: int = 1500, random_state: int = 42) -> pd.DataFrame:
        """
        Generates realistic housing records based on classic Dragon Real Estate schema
        with non-linear valuation interactions and missing data.
        """
        np.random.seed(random_state)
        crim = np.random.exponential(scale=3.5, size=n_samples).clip(0.01, 80.0).round(4)
        zn = np.random.choice([0.0, 12.5, 25.0, 50.0, 80.0], size=n_samples, p=[0.7, 0.1, 0.1, 0.05, 0.05])
        indus = np.random.uniform(1.0, 25.0, size=n_samples).round(2)
        chas = np.random.choice([0, 1], size=n_samples, p=[0.93, 0.07])
        nox = np.random.uniform(0.38, 0.87, size=n_samples).round(3)
        rm = np.random.normal(loc=6.3, scale=0.7, size=n_samples).clip(3.5, 9.0).round(2)
        age = np.random.uniform(5.0, 100.0, size=n_samples).round(1)
        dis = np.random.exponential(scale=3.8, size=n_samples).clip(1.1, 12.5).round(2)
        rad = np.random.choice([1, 2, 4, 5, 8, 24], size=n_samples)
        tax = (rad * 15 + np.random.normal(300, 50, size=n_samples)).clip(180, 711).round(0)
        ptratio = np.random.uniform(13.0, 22.0, size=n_samples).round(1)
        b = np.random.uniform(250.0, 396.9, size=n_samples).round(2)
        lstat = np.random.exponential(scale=11.0, size=n_samples).clip(1.5, 38.0).round(2)

        # Non-linear House Price Valuation (in $1,000s)
        medv = (
            25.0
            + (rm - 6.0) * 8.5
            - (lstat * 0.45)
            - (crim * 0.25)
            + (chas * 4.5)
            - (ptratio * 0.6)
            - (age * 0.05)
            + np.random.normal(0, 2.5, size=n_samples)
        ).clip(5.0, 50.0).round(2)

        df = pd.DataFrame({
            "CRIM": crim, "ZN": zn, "INDUS": indus, "CHAS": chas, "NOX": nox,
            "RM": rm, "AGE": age, "DIS": dis, "RAD": rad, "TAX": tax,
            "PTRATIO": ptratio, "B": b, "LSTAT": lstat, "MEDV": medv
        })

        # Inject realistic 2% missing values in RM (rooms) to demonstrate robust imputation
        missing_mask = np.random.rand(n_samples) < 0.02
        df.loc[missing_mask, "RM"] = np.nan

        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False)
        return df

    def train_and_evaluate(self) -> Dict[str, Any]:
        if not os.path.exists(self.data_path):
            self.generate_synthetic_housing_data()
        df = pd.read_csv(self.data_path)

        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        X_train = train_df[self.feature_columns]
        y_train = train_df["MEDV"]
        X_test = test_df[self.feature_columns]
        y_test = test_df["MEDV"]

        # Production Scikit-Learn Pipeline with Outlier-Resistant RobustScaler
        self.pipeline = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", RobustScaler()),
            ("model", GradientBoostingRegressor(n_estimators=150, learning_rate=0.08, max_depth=4, random_state=42))
        ])

        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))
        mape = mean_absolute_percentage_error(y_test, preds) * 100

        # Extract feature importances
        gbdt_model = self.pipeline.named_steps["model"]
        importances = {
            col: round(float(imp), 4)
            for col, imp in zip(self.feature_columns, gbdt_model.feature_importances_)
        }
        sorted_importances = dict(sorted(importances.items(), key=lambda x: x[1], reverse=True))

        self.metrics = {
            "r2_score": round(float(r2), 4),
            "rmse": round(float(rmse), 2),
            "mape_pct": round(float(mape), 2),
            "top_drivers": sorted_importances
        }

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "metrics": self.metrics}, f)

        return self.metrics

    def load(self):
        if self.pipeline is None:
            if not os.path.exists(self.model_path):
                self.train_and_evaluate()
            else:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.pipeline = data["pipeline"]
                    self.metrics = data["metrics"]

    def predict_valuation_and_roi(self, property_features: dict, down_payment_pct: float = 0.20, interest_rate_pct: float = 6.5) -> dict:
        self.load()
        df = pd.DataFrame([property_features])[self.feature_columns]
        price_in_k = float(self.pipeline.predict(df)[0])
        estimated_property_value = round(price_in_k * 1000.0, 2)

        # Financial Investment & ROI Calculations
        # 1. Gross Annual Rent Estimate (typical 0.7% monthly yield of property value)
        monthly_rent = round(estimated_property_value * 0.007, 2)
        annual_gross_rent = monthly_rent * 12

        # 2. Operating Expenses (Property tax, insurance, maintenance ~ 30% of gross rent)
        operating_expenses = round(annual_gross_rent * 0.30, 2)
        net_operating_income_noi = annual_gross_rent - operating_expenses

        # 3. Capitalization Rate (Cap Rate)
        cap_rate_pct = round((net_operating_income_noi / estimated_property_value) * 100.0, 2)

        # 4. Cash-on-Cash Return with specified down payment
        down_payment = estimated_property_value * down_payment_pct
        loan_amount = estimated_property_value - down_payment
        # Annual mortgage payment (30-year fixed amortization)
        r = (interest_rate_pct / 100.0) / 12
        n = 360
        monthly_mortgage = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)
        annual_debt_service = monthly_mortgage * 12
        cash_flow = net_operating_income_noi - annual_debt_service
        cash_on_cash_roi_pct = round((cash_flow / down_payment) * 100.0, 2)
        dscr = round(net_operating_income_noi / annual_debt_service, 2) if annual_debt_service > 0 else 99.0

        # Investment Grade Rating
        if cap_rate_pct >= 6.0 and cash_on_cash_roi_pct >= 7.0 and dscr >= 1.25:
            investment_grade = "TIER_A_STRONG_BUY"
        elif cap_rate_pct >= 4.5 and dscr >= 1.10:
            investment_grade = "TIER_B_MODERATE_HOLD"
        else:
            investment_grade = "TIER_C_OVERVALUED_LOW_YIELD"

        return {
            "estimated_property_value_usd": estimated_property_value,
            "monthly_estimated_rent_usd": monthly_rent,
            "net_operating_income_noi_usd": round(net_operating_income_noi, 2),
            "cap_rate_pct": cap_rate_pct,
            "monthly_mortgage_usd": round(monthly_mortgage, 2),
            "annual_debt_service_usd": round(annual_debt_service, 2),
            "net_annual_cash_flow_usd": round(cash_flow, 2),
            "cash_on_cash_roi_pct": cash_on_cash_roi_pct,
            "debt_service_coverage_ratio_dscr": dscr,
            "investment_grade": investment_grade
        }

    def calculate_interest_rate_sensitivity(self, property_features: dict, rates: Optional[List[float]] = None, down_payment_pct: float = 0.20) -> List[Dict[str, Any]]:
        """
        Stress-tests property cash flows and Debt Service Coverage Ratio (DSCR)
        across variable macroeconomic interest rate regimes (e.g. 5.0% to 8.5%).
        """
        if rates is None:
            rates = [5.0, 5.5, 6.0, 6.5, 7.0, 7.5, 8.0, 8.5]

        base_res = self.predict_valuation_and_roi(property_features, down_payment_pct=down_payment_pct, interest_rate_pct=6.5)
        property_val = base_res["estimated_property_value_usd"]
        noi = base_res["net_operating_income_noi_usd"]
        down_payment = property_val * down_payment_pct
        loan_amount = property_val - down_payment
        n = 360

        sensitivity = []
        for rate in rates:
            r = (rate / 100.0) / 12
            monthly_mortgage = loan_amount * (r * (1 + r)**n) / ((1 + r)**n - 1)
            annual_debt_service = monthly_mortgage * 12
            net_cash_flow = noi - annual_debt_service
            coc_roi = round((net_cash_flow / down_payment) * 100.0, 2)
            dscr = round(noi / annual_debt_service, 2) if annual_debt_service > 0 else 99.0

            sensitivity.append({
                "interest_rate_pct": rate,
                "monthly_mortgage_usd": round(monthly_mortgage, 2),
                "annual_debt_service_usd": round(annual_debt_service, 2),
                "net_annual_cash_flow_usd": round(net_cash_flow, 2),
                "cash_on_cash_roi_pct": coc_roi,
                "debt_service_coverage_ratio_dscr": dscr,
                "is_dscr_healthy": bool(dscr >= 1.25)
            })

        return sensitivity
