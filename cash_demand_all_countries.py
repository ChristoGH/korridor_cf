import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly
import holidays
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from prophet.serialize import model_to_json, model_from_json
from prophet.diagnostics import cross_validation, performance_metrics

historical = pd.read_csv('data/ZM.csv')
inference = pd.read_csv('data/ZM.csv.out')
country_code = 'ZM'
cv_horizon = '14 days'

historical['EffectiveDate'].nunique()
inference['EffectiveDate'].nunique()
