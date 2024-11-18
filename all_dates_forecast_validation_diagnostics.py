import pandas as pd
import logging
from typing import Dict, Optional
import os
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy import stats


class ForecastDiagnostics:
    def __init__(self,
                 country_code: str,
                 original_timestamp: str,
                 data_path: str,
                 results_path: Optional[str] = None):
        """Initialize diagnostic tool with proper data type handling"""
        # [Previous initialization code remains the same]
        self.country_code = country_code
        self.original_timestamp = original_timestamp
        self.data_path = data_path
        self.results_path = results_path or f"./results/{country_code}_{original_timestamp}/comprehensive_results.csv"

        # Setup logging
        self.output_dir = f"./diagnostics/{country_code}_{original_timestamp}"
        os.makedirs(self.output_dir, exist_ok=True)

        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(message)s',
            handlers=[
                logging.FileHandler(f"{self.output_dir}/diagnostic_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger('ForecastDiagnostics')

        # Load and validate data at initialization
        try:
            self.data = pd.read_csv(self.data_path)
            self.data['BranchId'] = self.data['BranchId'].astype(str)
            self.data['ProductId'] = self.data['ProductId'].astype(str)
            self.data['EffectiveDate'] = pd.to_datetime(self.data['EffectiveDate'])

            self.logger.info(f"Successfully loaded data from {self.data_path}")
            self.logger.info(f"Data shape: {self.data.shape}")
            self.logger.info(f"Date range: {self.data['EffectiveDate'].min()} to {self.data['EffectiveDate'].max()}")
        except Exception as e:
            self.logger.error(f"Error loading data: {str(e)}")
            raise

    def analyze_autocorrelation(self, historical: pd.DataFrame, max_lags: int = 7) -> None:
        """Analyze autocorrelation in demand"""
        if historical.empty:
            return

        self.logger.info("\n=== Autocorrelation Analysis ===")
        demand_series = historical.set_index('EffectiveDate')['Demand'].sort_index()

        # Calculate and plot autocorrelation
        plt.figure(figsize=(12, 6))
        acf_values = []
        for lag in range(1, max_lags + 1):
            correlation = demand_series.autocorr(lag=lag)
            acf_values.append(correlation)
            self.logger.info(f"Lag {lag} autocorrelation: {correlation:.3f}")

            if abs(correlation) > 0.7:
                self.logger.warning(f"Strong autocorrelation at lag {lag} - check if model captures this pattern")

        plt.bar(range(1, max_lags + 1), acf_values)
        plt.axhline(y=0, color='r', linestyle='-')
        plt.axhline(y=1.96 / np.sqrt(len(demand_series)), color='r', linestyle='--')
        plt.axhline(y=-1.96 / np.sqrt(len(demand_series)), color='r', linestyle='--')
        plt.title('Autocorrelation Function')
        plt.xlabel('Lag')
        plt.ylabel('Correlation')
        plt.savefig(f"{self.output_dir}/autocorrelation.png")
        plt.close()

    def check_outliers(self, historical: pd.DataFrame) -> None:
        """Identify potential outliers in historical data"""
        if historical.empty:
            return

        self.logger.info("\n=== Outlier Analysis ===")

        # Calculate outlier bounds
        Q1 = historical['Demand'].quantile(0.25)
        Q3 = historical['Demand'].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Identify outliers
        outliers = historical[
            (historical['Demand'] < lower_bound) |
            (historical['Demand'] > upper_bound)
            ]

        if not outliers.empty:
            self.logger.warning(
                f"\nFound {len(outliers)} potential outliers ({len(outliers) / len(historical) * 100:.1f}% of data):")
            self.logger.warning(f"Dates: {sorted(outliers['EffectiveDate'].dt.date.unique())}")
            self.logger.warning(f"Values range: {outliers['Demand'].min():.2f} to {outliers['Demand'].max():.2f}")

            # Analyze outliers by day of week
            dow_outliers = outliers.groupby('DayOfWeekName').size()
            self.logger.warning("\nOutliers by day of week:")
            self.logger.warning(f"{dow_outliers}")

            # Plot outliers
            plt.figure(figsize=(15, 8))
            plt.scatter(historical['EffectiveDate'], historical['Demand'], alpha=0.5, label='Normal')
            plt.scatter(outliers['EffectiveDate'], outliers['Demand'], color='red', alpha=0.7, label='Outlier')
            plt.axhline(y=upper_bound, color='r', linestyle='--', label='Upper bound')
            plt.axhline(y=lower_bound, color='r', linestyle='--', label='Lower bound')
            plt.title('Demand with Outliers Highlighted')
            plt.xlabel('Date')
            plt.ylabel('Demand')
            plt.legend()
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/outliers.png")
            plt.close()

    def check_forecast_consistency(self, forecast: pd.DataFrame) -> None:
        """Check if forecasts are consistent across steps"""
        if forecast.empty:
            return

        self.logger.info("\n=== Forecast Consistency Analysis ===")

        # Analyze each date that appears in multiple forecast steps
        dates = forecast['EffectiveDate'].unique()
        inconsistent_dates = []

        for date in dates:
            date_forecasts = forecast[forecast['EffectiveDate'] == date]
            if len(date_forecasts) > 1:
                p50_values = date_forecasts['p50'].unique()
                p50_std = date_forecasts['p50'].std()
                p50_range = date_forecasts['p50'].max() - date_forecasts['p50'].min()

                if len(p50_values) > 1:
                    inconsistent_dates.append({
                        'date': date,
                        'std': p50_std,
                        'range': p50_range,
                        'values': p50_values
                    })

                    self.logger.warning(f"\nInconsistent forecasts for date {date}:")
                    self.logger.warning(f"Standard deviation: {p50_std:.2f}")
                    self.logger.warning(f"Range: {p50_range:.2f}")
                    for _, row in date_forecasts.iterrows():
                        self.logger.warning(f"Step {row['ForecastStep']}: p50 = {row['p50']:.2f}")

        if inconsistent_dates:
            # Plot inconsistency analysis
            plt.figure(figsize=(15, 8))
            dates = [d['date'] for d in inconsistent_dates]
            ranges = [d['range'] for d in inconsistent_dates]
            plt.bar(dates, ranges)
            plt.title('Forecast Inconsistency by Date')
            plt.xlabel('Date')
            plt.ylabel('Range of p50 predictions')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(f"{self.output_dir}/forecast_inconsistency.png")
            plt.close()

    def analyze_historical_data(self, combination: Dict) -> pd.DataFrame:
        """Analyze historical data with enhanced diagnostics"""
        # [Previous historical analysis code remains the same]
        # After the existing analysis, add:

        historical = super().analyze_historical_data(combination)
        if not historical.empty:
            # Add new analyses
            self.check_outliers(historical)
            self.analyze_autocorrelation(historical)

        return historical

    def analyze_forecast_results(self, combination: Dict) -> pd.DataFrame:
        """Analyze forecast results with enhanced diagnostics"""
        # [Previous forecast analysis code remains the same]
        # After the existing analysis, add:

        forecast = super().analyze_forecast_results(combination)
        if not forecast.empty:
            self.check_forecast_consistency(forecast)

        return forecast

    def run_full_diagnosis(self, branch_id: str, product_id: str, currency: str) -> None:
        """Run complete diagnostic analysis with all enhancements"""
        combination = {
            'BranchId': str(branch_id),
            'ProductId': str(product_id),
            'Currency': currency
        }

        self.logger.info(f"\nStarting full diagnosis for combination: {combination}")

        if not self.validate_combination(branch_id, product_id, currency):
            return

        self.analyze_branch_patterns(branch_id)
        historical = self.analyze_historical_data(combination)

        if not historical.empty:
            self.analyze_seasonal_patterns(historical)
            self.check_outliers(historical)
            self.analyze_autocorrelation(historical)

            try:
                if os.path.exists(self.results_path):
                    forecast = self.analyze_forecast_results(combination)
                    if not forecast.empty:
                        self.check_forecast_consistency(forecast)

                    # Save detailed results
                    output_base = f"{self.output_dir}/branch{branch_id}_product{product_id}_{currency}"
                    historical.to_csv(f"{output_base}_historical.csv", index=False)
                    if not forecast.empty:
                        forecast.to_csv(f"{output_base}_forecast.csv", index=False)
                else:
                    self.logger.warning(f"Forecast results file not found: {self.results_path}")
                    self.logger.info("Only historical analysis was performed")
            except Exception as e:
                self.logger.error(f"Error analyzing forecast results: {str(e)}")


def main():
    diagnostics = ForecastDiagnostics(
        country_code="ZM",
        original_timestamp="20241103-163028",
        data_path="data/cash/ZM.csv"
    )

    diagnostics.run_full_diagnosis(
        branch_id="11",
        product_id="18",
        currency="USD"
    )


if __name__ == "__main__":
    main()