# Save this as forecast_visualizer.py
import plotly.graph_objects as go
import pandas as pd
from pathlib import Path
import os
from datetime import datetime
from typing import List, Tuple, Optional
import concurrent.futures
import logging
from functools import partial


class AutomatedForecastComparison:
    def __init__(self, data: pd.DataFrame, output_dir: str = "forecast_visualizations"):
        """Initialize with forecast data and output directory"""
        self.data = data.copy()
        self.data['EffectiveDate'] = pd.to_datetime(self.data['EffectiveDate'])
        self.output_dir = Path(output_dir)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def get_unique_combinations(self) -> List[Tuple[str, str, str]]:
        """Get all unique combinations of ProductId, BranchId, and Currency"""
        combinations = self.data[['ProductId', 'BranchId', 'Currency']].drop_duplicates()
        return [tuple(row) for row in combinations.values]

    def create_single_visualization(self, combination: Tuple[str, str, str]) -> dict:
        """Process a single combination"""
        product_id, branch_id, currency = combination

        try:
            # Filter data
            mask = (
                    (self.data['ProductId'] == product_id) &
                    (self.data['BranchId'] == branch_id) &
                    (self.data['Currency'] == currency)
            )
            plot_data = self.data[mask].copy()

            if plot_data.empty:
                raise ValueError(f"No data found for combination {combination}")

            # Create figure
            fig = go.Figure()

            # Add confidence interval
            fig.add_trace(
                go.Scatter(
                    x=plot_data['EffectiveDate'],
                    y=plot_data['p90'],
                    mode='lines',
                    line=dict(width=0),
                    showlegend=False,
                    hovertemplate='Date: %{x}<br>P90: %{y:,.0f}<extra></extra>'
                )
            )

            fig.add_trace(
                go.Scatter(
                    x=plot_data['EffectiveDate'],
                    y=plot_data['p10'],
                    mode='lines',
                    line=dict(width=0),
                    fill='tonexty',
                    fillcolor='rgba(173, 216, 230, 0.3)',
                    name='80% Confidence Interval',
                    hovertemplate='Date: %{x}<br>P10: %{y:,.0f}<extra></extra>'
                )
            )

            # Add p50 forecast line
            fig.add_trace(
                go.Scatter(
                    x=plot_data['EffectiveDate'],
                    y=plot_data['p50'],
                    mode='lines',
                    name='Forecast (P50)',
                    line=dict(color='rgb(31, 119, 180)', width=2),
                    hovertemplate='Date: %{x}<br>P50: %{y:,.0f}<extra></extra>'
                )
            )

            # Add actual values
            fig.add_trace(
                go.Scatter(
                    x=plot_data['EffectiveDate'],
                    y=plot_data['Demand'],
                    mode='markers',
                    name='Actual',
                    marker=dict(color='rgb(44, 160, 44)', size=8, symbol='diamond'),
                    hovertemplate='Date: %{x}<br>Actual: %{y:,.0f}<extra></extra>'
                )
            )

            # Calculate metrics
            mape = (abs(plot_data['Demand'] - plot_data['p50']) / plot_data['Demand']).mean() * 100
            coverage = ((plot_data['Demand'] >= plot_data['p10']) &
                        (plot_data['Demand'] <= plot_data['p90'])).mean() * 100

            # Update layout
            fig.update_layout(
                title=dict(
                    text=f'Demand Forecast Comparison<br>' +
                         f'Product: {product_id} | Branch: {branch_id} | Currency: {currency}<br>' +
                         f'MAPE: {mape:.1f}% | 80% Coverage: {coverage:.1f}%',
                    x=0.5,
                    xanchor='center'
                ),
                xaxis_title='Date',
                yaxis_title=f'Demand ({currency})',
                template='plotly_white',
                height=600,
                hovermode='x unified',
                margin=dict(t=120)
            )

            # Save plot
            filename = f"forecast_{product_id}_{branch_id}_{currency}.html"
            filepath = self.output_dir / filename
            fig.write_html(filepath)

            return {
                'product_id': product_id,
                'branch_id': branch_id,
                'currency': currency,
                'mape': mape,
                'coverage': coverage,
                'filename': filename,
                'status': 'success'
            }

        except Exception as e:
            self.logger.error(f"Error processing combination {combination}: {str(e)}")
            return {
                'product_id': product_id,
                'branch_id': branch_id,
                'currency': currency,
                'status': 'error',
                'error': str(e)
            }

    def generate_all_visualizations(self, max_workers: int = 4) -> pd.DataFrame:
        """Generate visualizations for all combinations"""
        self.logger.info("Starting visualization generation...")

        combinations = self.get_unique_combinations()
        self.logger.info(f"Found {len(combinations)} unique combinations")

        results = []

        # Sequential processing (more reliable than parallel for debugging)
        for combo in combinations:
            result = self.create_single_visualization(combo)
            results.append(result)

            if result['status'] == 'success':
                self.logger.info(
                    f"Processed: Product {result['product_id']}, "
                    f"Branch {result['branch_id']}, "
                    f"Currency {result['currency']}"
                )
            else:
                self.logger.error(
                    f"Failed: Product {result['product_id']}, "
                    f"Branch {result['branch_id']}, "
                    f"Currency {result['currency']}"
                )

        # Create summary DataFrame
        summary_df = pd.DataFrame(results)

        # Save summary
        summary_path = self.output_dir / 'visualization_summary.csv'
        summary_df.to_csv(summary_path, index=False)

        self.logger.info(f"Generated {len(results)} visualizations")
        self.logger.info(f"Summary saved to {summary_path}")

        return summary_df


# Example usage
def main():
    """Run the forecast visualization automation"""
    try:
        # Load data
        data_path = "data/comprehensive_results.csv"  # Update this path to your actual data file
        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Data file not found at {data_path}")

        data = pd.read_csv(data_path)
        required_columns = ['ProductId', 'BranchId', 'Currency', 'EffectiveDate',
                            'Demand', 'p10', 'p50', 'p90']

        # Validate data
        missing_columns = [col for col in required_columns if col not in data.columns]
        if missing_columns:
            raise ValueError(f"Missing required columns: {missing_columns}")

        # Create output directory with timestamp
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"forecast_visualizations_{timestamp}"

        # Initialize and run
        viz_generator = AutomatedForecastComparison(data, output_dir)
        summary = viz_generator.generate_all_visualizations()

        # Print summary statistics
        successful = summary['status'] == 'success'

        print("\n=== Generation Summary ===")
        print(f"Output Directory: {output_dir}")
        print(f"Total combinations processed: {len(summary)}")
        print(f"Successful generations: {successful.sum()}")
        print(f"Failed generations: {(~successful).sum()}")

        if successful.any():
            print("\n=== Performance Metrics ===")
            print(f"Average MAPE: {summary[successful]['mape'].mean():.1f}%")
            print(f"Average Coverage: {summary[successful]['coverage'].mean():.1f}%")

            # Additional statistics
            print("\n=== Detailed Metrics ===")
            mape_stats = summary[successful]['mape'].describe()
            coverage_stats = summary[successful]['coverage'].describe()

            print("\nMAPE Statistics:")
            print(f"Minimum: {mape_stats['min']:.1f}%")
            print(f"Maximum: {mape_stats['max']:.1f}%")
            print(f"Median: {mape_stats['50%']:.1f}%")

            print("\nCoverage Statistics:")
            print(f"Minimum: {coverage_stats['min']:.1f}%")
            print(f"Maximum: {coverage_stats['max']:.1f}%")
            print(f"Median: {coverage_stats['50%']:.1f}%")

            # Save detailed statistics
            stats_path = os.path.join(output_dir, 'performance_metrics.txt')
            with open(stats_path, 'w') as f:
                f.write("=== Forecast Performance Metrics ===\n\n")
                f.write(f"Analysis Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write(f"Data Source: {data_path}\n\n")
                f.write("--- Summary Statistics ---\n")
                f.write(f"Total Combinations: {len(summary)}\n")
                f.write(f"Successful: {successful.sum()}\n")
                f.write(f"Failed: {(~successful).sum()}\n\n")
                f.write("--- Performance Metrics ---\n")
                f.write(f"Average MAPE: {summary[successful]['mape'].mean():.1f}%\n")
                f.write(f"Average Coverage: {summary[successful]['coverage'].mean():.1f}%\n\n")
                f.write("--- MAPE Statistics ---\n")
                for stat, value in mape_stats.items():
                    f.write(f"{stat}: {value:.1f}%\n")
                f.write("\n--- Coverage Statistics ---\n")
                for stat, value in coverage_stats.items():
                    f.write(f"{stat}: {value:.1f}%\n")

        print(f"\nDetailed metrics saved to: {output_dir}/performance_metrics.txt")
        print(f"Visualizations saved in: {output_dir}/")

    except Exception as e:
        print(f"\nError running forecast automation: {str(e)}")
        raise


if __name__ == "__main__":
    main()