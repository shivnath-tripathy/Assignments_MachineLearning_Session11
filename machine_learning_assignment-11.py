
# coding: utf-8

# In this assignment students have to make ARIMA model over shampoo sales data and
# check the MSE between predicted and actual value.
# Student can download data in .csv format from the following link:
# https://datamarket.com/data/set/22r0/sales-of-shampoo-over-a-three-year-period#!ds
# =22r0&display=line

# In[1]:


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot as plt
import statsmodels.api as sm
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error


# In[2]:


def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')

series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0,
squeeze=True, date_parser=parser)


# In[3]:


series[:]


# In[4]:


# show plots in the notebook
get_ipython().magic('matplotlib inline')
series.plot(figsize=(12,8));


# In[5]:


sm.stats.durbin_watson(series) #Below results says Postive Corelation as per Durbin Watson Statics

# Another popular test for serial correlation is the Durbin-Watson statistic. The DW statistic will lie in
# the 0-4 range, with a value near two indicating no first-order serial correlation. Positive serial
# correlation is associated with DW values below 2 and negative serial correlation with DW values
# above 2.


# In[6]:


series.values.squeeze()


# In[7]:


series.values


# In[13]:


# show plots in the notebook
get_ipython().magic('matplotlib inline')
fig = plt.figure(figsize=(12,8))
ax1 = fig.add_subplot(211)
fig = sm.graphics.tsa.plot_acf(series, lags=35, ax=ax1)
ax2 = fig.add_subplot(212)
fig = sm.graphics.tsa.plot_pacf(series, lags=35, ax=ax2)


# In[24]:


dta = series


# In[25]:


from pandas.tools.plotting import autocorrelation_plot
# show plots in the notebook
get_ipython().magic('matplotlib inline')
dta = (series - series.mean()) / (series.std())
plt.acorr(dta,maxlags = len(dta) -1, linestyle = "solid", usevlines = False, marker='')
plt.show()
autocorrelation_plot(series)
plt.show()


# In[29]:


from pandas import datetime
from pandas import DataFrame
from statsmodels.tsa.arima_model import ARIMA
# fit the model
arima_mod = ARIMA(series,order=(5,1,0))
arima_mod_fit = arima_mod.fit(disp=0)
print(arima_mod_fit.summary())
#Residual Errors
residuals = DataFrame(arima_mod_fit.resid)
residuals.plot()
plt.show()
residuals.plot(kind='kde')
plt.show()
print(residuals.describe())


# In[30]:


from pandas import read_csv
from pandas import datetime
from matplotlib import pyplot
from statsmodels.tsa.arima_model import ARIMA
from sklearn.metrics import mean_squared_error
 
def parser(x):
	return datetime.strptime('190'+x, '%Y-%m')
 
series = read_csv('shampoo-sales.csv', header=0, parse_dates=[0], index_col=0, squeeze=True, date_parser=parser)
X = series.values
size = int(len(X) * 0.66)
train, test = X[0:size], X[size:len(X)]
history = [x for x in train]
predictions = list()
for t in range(len(test)):
    model = ARIMA(history, order=(5,1,0))
    model_fit = model.fit(disp=0)
    output = model_fit.forecast()
    yhat = output[0]
    predictions.append(yhat)
    obs = test[t]
    history.append(obs)
    print('predicted=%f, expected=%f' % (yhat, obs))
error = mean_squared_error(test, predictions)
print('Test MSE: %.3f' % error)
# plot
pyplot.plot(test)
pyplot.plot(predictions, color='red')
pyplot.show()

