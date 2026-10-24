import pandas as pd
from typing import Dict, Any, List

class ExecutiveInsightsGenerator:
    def generate_brief(self, query_result: Dict[str, Any]) -> Dict[str, Any]:
        """
        Translates raw SQL result sets into executive takeaways and actionable business insights.
        """
        df: pd.DataFrame = query_result.get("dataframe")
        if df is None or df.empty:
            return {
                "summary": "No data returned for this query.",
                "key_findings": [],
                "actionable_recommendations": []
            }

        cols = df.columns.tolist()
        findings = []
        recommendations = []

        # 1. MoM Analysis Insights
        if "mom_growth_pct" in cols:
            valid_growth = df["mom_growth_pct"].dropna()
            avg_growth = valid_growth.mean()
            best_month = df.loc[df["total_revenue"].idxmax()]
            
            findings.append(f"Average Month-over-Month (MoM) revenue growth is {avg_growth:.2f}%.")
            findings.append(f"Peak sales recorded in {best_month['sales_month']} with ${best_month['total_revenue']:,.2f} in net revenue.")
            
            recommendations.append("Replicate promotional campaigns from peak months to counter seasonal dips.")
            recommendations.append("Investigate negative growth quarters for supply-chain or discount cannibalization issues.")

        # 2. Customer Spend & Ranking Insights
        elif "spend_rank" in cols and "total_spend" in cols:
            top_client = df.iloc[0]
            total_top_10 = df["total_spend"].sum()
            findings.append(f"Top 1 customer ({top_client['name']}) contributed ${top_client['total_spend']:,.2f} across {top_client['total_orders']} orders.")
            findings.append(f"Top 10 enterprise accounts account for ${total_top_10:,.2f} of gross sales.")
            
            recommendations.append("Assign dedicated Customer Success Managers (CSMs) to customers in the top 3 spend ranks.")
            recommendations.append("Implement high-touch VIP retention contracts to mitigate enterprise churn risk.")

        # 3. Product Profitability Insights
        elif "profit_margin_pct" in cols:
            highest_margin = df.loc[df["profit_margin_pct"].idxmax()]
            lowest_margin = df.loc[df["profit_margin_pct"].idxmin()]
            findings.append(f"Highest profit margin category is '{highest_margin['category']}' with {highest_margin['profit_margin_pct']}% gross margin.")
            findings.append(f"Lowest profit margin category is '{lowest_margin['category']}' at {lowest_margin['profit_margin_pct']}%.")
            
            recommendations.append(f"Increase ad-spend allocation toward high-margin '{highest_margin['category']}' products.")
            recommendations.append(f"Review supplier unit costs for '{lowest_margin['category']}' to renegotiate volume discount tiers.")

        # 4. Regional Distribution
        elif "regional_revenue" in cols:
            top_region = df.iloc[0]
            findings.append(f"Leading market is {top_region['region']} ({top_region['segment']} segment) generating ${top_region['regional_revenue']:,.2f}.")
            recommendations.append("Expand sales operations and local cloud availability zones in high-performing regions.")

        return {
            "executive_summary": "Automated SQL Business Intelligence Synthesis",
            "analyzed_rows": len(df),
            "key_findings": findings,
            "actionable_recommendations": recommendations
        }
