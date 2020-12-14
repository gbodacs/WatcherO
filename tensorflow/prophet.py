
import pandas as pd
from fbprophet import Prophet
from fbprophet.plot import plot_plotly, plot_components_plotly
from matplotlib import pyplot

df = pd.read_csv('VIX_daily.csv', usecols=[0,4])
df.head()

m = Prophet(weekly_seasonality=False,yearly_seasonality=False,daily_seasonality=False)
m.add_seasonality('self_define_cycle',period=27,fourier_order=8,mode='additive')
m.fit(df)
future = m.make_future_dataframe(periods=365)
future.tail()

forecast = m.predict(future)
# summarize the forecast
print(forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].head())
# plot forecast
m.plot(forecast)
pyplot.show()

# forecast[['ds', 'yhat', 'yhat_lower', 'yhat_upper']].tail()
# fig1 = m.plot(forecast)
# fig2 = m.plot_components(forecast)


# plot_plotly(m, forecast)
# #plot_plotly.show()

# plot_components_plotly(m, forecast)
# pyplot.show()