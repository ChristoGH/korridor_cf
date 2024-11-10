import pandas as pd

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from prophet import Prophet
from prophet.plot import plot_plotly, plot_components_plotly
import holidays
import plotly.graph_objects as go
from plotly.subplots import make_subplots

plt.style.use('seaborn')

BW = pd.read_csv('data/BW.csv')