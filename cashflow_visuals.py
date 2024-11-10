import plotly.graph_objects as go
import pandas as pd
from typing import Optional


class ForecastComparison:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize with forecast data

        Args:
            data: DataFrame with columns [ProductId, BranchId, Currency,
                  EffectiveDate, Demand, p10, p50, p90]
        """
        self.data = data.copy()
        self.data['EffectiveDate'] = pd.to_datetime(self.data['EffectiveDate'])

    def create_comparison_plot(
            self,
            product_id: str,
            branch_id: str,
            currency: str,
            start_date: Optional[str] = None,
            end_date: Optional[str] = None
    ) -> go.Figure:
        """
        Create detailed comparison plot for specific product/branch/currency

        Args:
            product_id: Product identifier
            branch_id: Branch identifier
            currency: Currency code
            start_date: Optional start date for filtering (YYYY-MM-DD)
            end_date: Optional end date for filtering (YYYY-MM-DD)
        """
        # Filter data for specific combination
        mask = (
                (self.data['ProductId'] == product_id) &
                (self.data['BranchId'] == branch_id) &
                (self.data['Currency'] == currency)
        )

        plot_data = self.data[mask].copy()

        # Apply date filters if provided
        if start_date:
            plot_data = plot_data[plot_data['EffectiveDate'] >= pd.to_datetime(start_date)]
        if end_date:
            plot_data = plot_data[plot_data['EffectiveDate'] <= pd.to_datetime(end_date)]

        # Sort by date
        plot_data = plot_data.sort_values('EffectiveDate')

        # Create figure
        fig = go.Figure()

        # Add confidence interval (p10 to p90)
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
                fillcolor='rgba(173, 216, 230, 0.3)',  # Light blue with transparency
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

        # Add actual values as markers
        fig.add_trace(
            go.Scatter(
                x=plot_data['EffectiveDate'],
                y=plot_data['Demand'],
                mode='markers',
                name='Actual',
                marker=dict(
                    color='rgb(44, 160, 44)',
                    size=8,
                    symbol='diamond'
                ),
                hovertemplate='Date: %{x}<br>Actual: %{y:,.0f}<extra></extra>'
            )
        )

        # Calculate accuracy metrics
        mape = (
                       abs(plot_data['Demand'] - plot_data['p50']) /
                       plot_data['Demand']
               ).mean() * 100

        coverage = (
                ((plot_data['Demand'] >= plot_data['p10']) &
                 (plot_data['Demand'] <= plot_data['p90'])).mean() * 100
        )

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
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            ),
            margin=dict(t=120)  # Increase top margin for title
        )

        # Add date range selector
        fig.update_xaxes(
            rangeslider_visible=True,
            rangeselector=dict(
                buttons=list([
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(step="all")
                ])
            )
        )

        return fig


def main():
    """Example usage"""
    # Sample data loading (replace with your actual data)
    data = pd.read_csv('data/comprehensive_results.csv')

    # Create visualizer
    viz = ForecastComparison(data)

    # Create comparison plot for specific combination
    fig = viz.create_comparison_plot(
        product_id='341',
        branch_id='13',
        currency='BWP',
        start_date='2024-07-05',
        end_date='2024-07-23'
    )

    # Save interactive plot
    fig.write_html('forecast_comparison.html')

    # Display in notebook
    fig.show()


if __name__ == "__main__":
    main()