import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from typing import Dict, List
import plotly.express as px
import plotly.figure_factory as ff


class ForecastVisualizer:
    def __init__(self, data: pd.DataFrame):
        """
        Initialize visualizer with forecast data

        Args:
            data: DataFrame with columns [ProductId, BranchId, CountryCode, Currency,
                  EffectiveDate, ForecastStep, Demand, p10, p50, p90,
                  AbsoluteError, PercentageError]
        """
        self.data = data.copy()
        self.data['EffectiveDate'] = pd.to_datetime(self.data['EffectiveDate'])
        self.colors = {
            'actual': '#2E7D32',  # Dark green
            'forecast': '#1976D2',  # Blue
            'bounds': '#90CAF9',  # Light blue
            'error': '#FF9800',  # Orange
            'grid': '#E0E0E0'  # Light grey
        }

    def create_time_series_plot(self, product_id=None, branch_id=None) -> go.Figure:
        """Create time series plot showing forecast vs actuals with confidence bounds"""
        # Filter data if needed
        plot_data = self.data
        if product_id:
            plot_data = plot_data[plot_data['ProductId'] == product_id]
        if branch_id:
            plot_data = plot_data[plot_data['BranchId'] == branch_id]

        # Create figure
        fig = go.Figure()

        # Add actual values
        fig.add_trace(
            go.Scatter(
                x=plot_data['EffectiveDate'],
                y=plot_data['Demand'],
                mode='markers',
                name='Actual',
                marker=dict(color=self.colors['actual'], size=8),
                hovertemplate='Date: %{x}<br>Actual: %{y}<extra></extra>'
            )
        )

        # Add forecast (P50)
        fig.add_trace(
            go.Scatter(
                x=plot_data['EffectiveDate'],
                y=plot_data['p50'],
                mode='lines',
                name='Forecast (P50)',
                line=dict(color=self.colors['forecast'], width=2),
                hovertemplate='Date: %{x}<br>Forecast: %{y}<extra></extra>'
            )
        )

        # Add confidence interval
        fig.add_trace(
            go.Scatter(
                x=plot_data['EffectiveDate'],
                y=plot_data['p90'],
                mode='lines',
                line=dict(width=0),
                showlegend=False,
                hovertemplate='Date: %{x}<br>P90: %{y}<extra></extra>'
            )
        )

        fig.add_trace(
            go.Scatter(
                x=plot_data['EffectiveDate'],
                y=plot_data['p10'],
                mode='lines',
                line=dict(width=0),
                fill='tonexty',
                fillcolor=f'rgba(144, 202, 249, 0.2)',  # Light blue with transparency
                name='80% Confidence Interval',
                hovertemplate='Date: %{x}<br>P10: %{y}<extra></extra>'
            )
        )

        # Update layout
        title = 'Demand Forecast vs Actual'
        if product_id:
            title += f' - Product {product_id}'
        if branch_id:
            title += f' - Branch {branch_id}'

        fig.update_layout(
            title=dict(
                text=title,
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Date',
            yaxis_title='Demand',
            template='plotly_white',
            hovermode='x unified',
            height=600,
            legend=dict(
                yanchor="top",
                y=0.99,
                xanchor="left",
                x=0.01
            )
        )

        return fig

    def create_error_distribution_plot(self) -> go.Figure:
        """Create histogram of forecast errors"""
        fig = go.Figure()

        # Add histogram of percentage errors
        fig.add_trace(
            go.Histogram(
                x=self.data['PercentageError'],
                nbinsx=30,
                name='Error Distribution',
                marker_color=self.colors['error']
            )
        )

        # Add mean line
        mean_error = self.data['PercentageError'].mean()
        fig.add_vline(
            x=mean_error,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Mean Error: {mean_error:.1f}%",
            annotation_position="top right"
        )

        fig.update_layout(
            title=dict(
                text='Forecast Error Distribution',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Percentage Error (%)',
            yaxis_title='Frequency',
            template='plotly_white',
            height=400,
            showlegend=False
        )

        return fig

    def create_step_performance_plot(self) -> go.Figure:
        """Create plot showing forecast accuracy by step"""
        # Calculate MAPE by step
        mape_by_step = self.data.groupby('ForecastStep')['PercentageError'].agg(
            ['mean', 'std', 'count']
        ).reset_index()

        fig = go.Figure()

        # Add MAPE line
        fig.add_trace(
            go.Scatter(
                x=mape_by_step['ForecastStep'],
                y=mape_by_step['mean'],
                mode='lines+markers',
                name='MAPE',
                line=dict(color=self.colors['forecast'], width=2),
                error_y=dict(
                    type='data',
                    array=mape_by_step['std'] / np.sqrt(mape_by_step['count']),
                    visible=True
                )
            )
        )

        fig.update_layout(
            title=dict(
                text='Forecast Accuracy by Step',
                x=0.5,
                xanchor='center'
            ),
            xaxis_title='Forecast Step',
            yaxis_title='Mean Absolute Percentage Error (%)',
            template='plotly_white',
            height=400,
            showlegend=False
        )

        return fig

    def create_dashboard(self, product_id=None, branch_id=None) -> go.Figure:
        """Create complete dashboard with all plots"""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Demand Forecast vs Actual',
                'Forecast Error Distribution',
                'Forecast Accuracy by Step',
                'Product/Branch Performance'
            ),
            specs=[[{'colspan': 2}, None],
                   [{}, {}]],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # Add time series plot
        ts_fig = self.create_time_series_plot(product_id, branch_id)
        for trace in ts_fig.data:
            fig.add_trace(trace, row=1, col=1)

        # Add error distribution
        err_fig = self.create_error_distribution_plot()
        for trace in err_fig.data:
            fig.add_trace(trace, row=2, col=1)

        # Add step performance
        step_fig = self.create_step_performance_plot()
        for trace in step_fig.data:
            fig.add_trace(trace, row=2, col=2)

        # Update layout
        fig.update_layout(
            height=1000,
            title=dict(
                text='Forecast Quality Dashboard',
                x=0.5,
                xanchor='center',
                y=0.95
            ),
            template='plotly_white',
            showlegend=True,
            legend=dict(
                yanchor="top",
                y=0.95,
                xanchor="left",
                x=0.01
            )
        )

        return fig


def main():
    """Example usage"""
    # Load your data
    data = pd.read_csv('data/comprehensive_results.csv')

    # Create visualizer
    viz = ForecastVisualizer(data)

    # Create and save individual plots
    ts_fig = viz.create_time_series_plot()
    ts_fig.write_html('time_series_forecast.html')

    err_fig = viz.create_error_distribution_plot()
    err_fig.write_html('error_distribution.html')

    step_fig = viz.create_step_performance_plot()
    step_fig.write_html('step_performance.html')

    # Create and save complete dashboard
    dashboard = viz.create_dashboard()
    dashboard.write_html('forecast_dashboard.html')


if __name__ == "__main__":
    main()