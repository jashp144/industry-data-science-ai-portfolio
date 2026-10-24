import os
import pickle
import numpy as np
import pandas as pd
from typing import Dict, Any, List
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import mean_squared_error, r2_score

class CodersLaborMarketEngine:
    def __init__(self, data_path: str = None, model_path: str = None):
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        if data_path is None:
            data_path = os.path.join(project_root, "data", "coders_delhi_bangalore.csv")
        if model_path is None:
            model_path = os.path.join(project_root, "data", "salary_model.pkl")

        self.data_path = data_path
        self.model_path = model_path
        self.pipeline = None
        self.metrics = {}

    def generate_synthetic_coder_data(self, n_samples: int = 3000, random_state: int = 42) -> pd.DataFrame:
        """Generates realistic tech labor market survey data across Delhi-NCR and Bangalore."""
        np.random.seed(random_state)
        
        cities = np.random.choice(["Bangalore", "Delhi-NCR", "Remote"], size=n_samples, p=[0.45, 0.40, 0.15])
        exp_years = np.random.exponential(scale=4.5, size=n_samples).clip(0.5, 20.0).round(1)
        primary_stacks = np.random.choice(["Python/AI", "Java/Spring", "TypeScript/Node", "Go/Distributed", "C++/Systems"], size=n_samples)
        has_cloud_skill = np.random.choice([0, 1], size=n_samples, p=[0.4, 0.6])
        has_genai_skill = np.random.choice([0, 1], size=n_samples, p=[0.75, 0.25])
        company_tier = np.random.choice(["Product/Unicorn", "Service/IT", "Early-Stage Startup"], size=n_samples, p=[0.35, 0.45, 0.20])

        # Base Compensation in INR Lakhs per Annum (LPA)
        base = 4.5
        city_boost = {"Bangalore": 1.25, "Delhi-NCR": 1.10, "Remote": 1.15}
        tier_boost = {"Product/Unicorn": 1.8, "Early-Stage Startup": 1.2, "Service/IT": 0.9}
        stack_boost = {"Python/AI": 1.35, "Go/Distributed": 1.30, "Java/Spring": 1.15, "TypeScript/Node": 1.10, "C++/Systems": 1.25}

        salaries = []
        for i in range(n_samples):
            city_mult = city_boost[cities[i]]
            t_mult = tier_boost[company_tier[i]]
            s_mult = stack_boost[primary_stacks[i]]
            exp_val = exp_years[i]
            
            # Non-linear salary progression: Base + (Exp * 2.2)^1.1 * multipliers
            sal = (base + (exp_val * 2.6) + (has_cloud_skill[i] * 3.5) + (has_genai_skill[i] * 5.5)) * city_mult * t_mult * s_mult
            sal += np.random.normal(0, sal * 0.12)
            salaries.append(max(3.2, round(sal, 2)))

        df = pd.DataFrame({
            "city": cities,
            "experience_years": exp_years,
            "primary_stack": primary_stacks,
            "has_cloud_skill": has_cloud_skill,
            "has_genai_skill": has_genai_skill,
            "company_tier": company_tier,
            "annual_salary_lpa": salaries
        })

        os.makedirs(os.path.dirname(self.data_path), exist_ok=True)
        df.to_csv(self.data_path, index=False)
        return df

    def train(self) -> Dict[str, Any]:
        if not os.path.exists(self.data_path):
            self.generate_synthetic_coder_data()
        df = pd.read_csv(self.data_path)

        train_size = int(len(df) * 0.8)
        train_df = df.iloc[:train_size]
        test_df = df.iloc[train_size:]

        cat_cols = ["city", "primary_stack", "company_tier"]
        num_cols = ["experience_years", "has_cloud_skill", "has_genai_skill"]

        preprocessor = ColumnTransformer(transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), cat_cols),
            ("num", "passthrough", num_cols)
        ])

        self.pipeline = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("regressor", GradientBoostingRegressor(n_estimators=120, learning_rate=0.08, max_depth=4, random_state=42))
        ])

        X_train = train_df.drop(columns=["annual_salary_lpa"])
        y_train = train_df["annual_salary_lpa"]
        X_test = test_df.drop(columns=["annual_salary_lpa"])
        y_test = test_df["annual_salary_lpa"]

        self.pipeline.fit(X_train, y_train)
        preds = self.pipeline.predict(X_test)

        r2 = r2_score(y_test, preds)
        rmse = np.sqrt(mean_squared_error(y_test, preds))

        self.metrics = {
            "r2_score": round(float(r2), 4),
            "rmse_lpa": round(float(rmse), 2)
        }

        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        with open(self.model_path, "wb") as f:
            pickle.dump({"pipeline": self.pipeline, "metrics": self.metrics}, f)

        return self.metrics

    def load(self):
        if self.pipeline is None:
            if not os.path.exists(self.model_path):
                self.train()
            else:
                with open(self.model_path, "rb") as f:
                    data = pickle.load(f)
                    self.pipeline = data["pipeline"]
                    self.metrics = data["metrics"]

    def analyze_profile_and_recommend(self, candidate_profile: dict) -> dict:
        self.load()
        df = pd.DataFrame([candidate_profile])
        predicted_salary = round(float(self.pipeline.predict(df)[0]), 2)

        # Calculate Skill Uplift Options:
        # What happens if candidate adds Cloud / MLOps?
        profile_with_cloud = candidate_profile.copy()
        profile_with_cloud["has_cloud_skill"] = 1
        salary_with_cloud = round(float(self.pipeline.predict(pd.DataFrame([profile_with_cloud]))[0]), 2)

        # What happens if candidate adds GenAI / LLMs?
        profile_with_genai = candidate_profile.copy()
        profile_with_genai["has_genai_skill"] = 1
        salary_with_genai = round(float(self.pipeline.predict(pd.DataFrame([profile_with_genai]))[0]), 2)

        # Both
        profile_with_both = candidate_profile.copy()
        profile_with_both["has_cloud_skill"] = 1
        profile_with_both["has_genai_skill"] = 1
        salary_with_both = round(float(self.pipeline.predict(pd.DataFrame([profile_with_both]))[0]), 2)

        cloud_gain_pct = round(((salary_with_cloud - predicted_salary) / predicted_salary) * 100.0, 1)
        genai_gain_pct = round(((salary_with_genai - predicted_salary) / predicted_salary) * 100.0, 1)
        total_gain_pct = round(((salary_with_both - predicted_salary) / predicted_salary) * 100.0, 1)

        recommendations = []
        if candidate_profile["has_genai_skill"] == 0:
            recommendations.append(f"Master Generative AI & RAG pipelines: Projected salary uplift of +{genai_gain_pct}% (+INR {round(salary_with_genai - predicted_salary, 1)} LPA).")
        if candidate_profile["has_cloud_skill"] == 0:
            recommendations.append(f"Acquire Cloud (AWS/Azure/Docker) deployment skills: Projected salary uplift of +{cloud_gain_pct}% (+INR {round(salary_with_cloud - predicted_salary, 1)} LPA).")
        if not recommendations:
            recommendations.append("Profile is well-optimized with Cloud & GenAI skills! Target Tier-1 Product / Unicorn roles.")

        return {
            "predicted_salary_lpa": predicted_salary,
            "market_salary_range_lpa": {
                "25th_percentile": round(predicted_salary * 0.85, 2),
                "50th_percentile_median": predicted_salary,
                "75th_percentile": round(predicted_salary * 1.20, 2),
                "90th_percentile": round(predicted_salary * 1.45, 2)
            },
            "projected_salary_with_both_skills_lpa": salary_with_both,
            "total_potential_uplift_pct": total_gain_pct,
            "strategic_skill_recommendations": recommendations
        }
