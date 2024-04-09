import yfinance as yf
import pandas as pd
import numpy as np
from scipy.optimize import Bounds
import matplotlib.pyplot as plt
from scipy.optimize import minimize
from scipy.optimize import LinearConstraint




#################################
########## FUNCTIONS ############
#################################
'''
Function 'create' creates dataframe with date and close prices from
the list of stocks imported from Wikipedia List of S&P 500 companies
'''

def create(tickers, StartDate, EndDate):
    global Companies
    Companies = tickers['Symbol'].to_list()
    global CompaniesDf
    CompaniesDf = yf.download(Companies)['Adj Close']
    CompaniesDf = CompaniesDf.loc[(CompaniesDf.index > StartDate) & (CompaniesDf.index < EndDate)]
    #CompaniesDf = CompaniesDf.iloc[:,0:10]
    return CompaniesDf


'''
Function 'select_companys' allows to select only the companys which 
contain data for the whole period used for creating dataframe
'''
def select_companys (df, NullsRate):
    for i in df:
    #Calculate how much null values are in vector of each company comparing to
    #amount of all rows
        a = df[i].isnull().sum()
        b = len(df)
        c = a/b
    #If there is more nulls than 'NullsRate' of all rows, delete company
        if c > NullsRate:
            del df[i]
    return CompaniesDf



'''
Function 'statistics' calculates returns for each stock in DF  
and returns Covariance Matrix, and Expected value
'''
def statistics(CompaniesDf):
    #Calculate log returns for each stock in every period
    global LogReturn
    LogReturn = np.log(1+CompaniesDf.pct_change())
    LogReturn = LogReturn.dropna()

    #Calculate the covariance matrix for all stocks
    #Wariancja na przekatnych
    global CovMatrix
    CovMatrix = LogReturn.cov()

    #Calculate expected values of each company (equal weights)
    global ExpectedValue
    ExpectedValue = np.mean(LogReturn,axis = 0)
    
    return LogReturn, CovMatrix, ExpectedValue


#Returns of the whole portfolio
def ret(ExpectedValue,w):
    return np.dot(ExpectedValue,w)

#Volatility - standard deviation of return (level of risk)
def vol(w,CovMatrix):
    return np.sqrt(np.dot(w,np.dot(w,CovMatrix)))


#Select beginning and ending periods
StartDate = '2000-1-1'
EndDate = '2023-12-31'

#Import S&P500 tickers and close prices from Wikipedia
table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]

create(tickers, StartDate, EndDate) 

#% rate of acceptable NaN
NullsRate = 0.0
select_companys(CompaniesDf, NullsRate)

statistics(CompaniesDf)

#All weights have to have values between 0 and 1
bounds = Bounds(0, 1)
#All weights need to sum to 0
linear_constraint = LinearConstraint(np.ones((LogReturn.shape[1],), dtype=int),1,1)

#Make the initial guess of weights: 1/number of companies
weights = np.ones(LogReturn.shape[1])
x0 = weights/np.sum(weights)

#Minimize volatility and find portfolio with lowest risk possible
fun1 = lambda w: np.sqrt(np.dot(w,np.dot(w,CovMatrix)))
res = minimize(fun1,x0,method='trust-constr',constraints = linear_constraint,bounds = bounds)
w_min = res.x

np.set_printoptions(suppress = True, precision=4)
print(list(CompaniesDf.columns))
print(w_min)
print('return: % .3f'% (ret(ExpectedValue,w_min)*100)) 
print('risk: % .3f'% vol(w_min,CovMatrix))
print('Sharpe ratio: ', ret(ExpectedValue,w_min)/vol(w_min,CovMatrix))

#Maximum efficiency of portfolio when Sharpe ratio is maximized
#We can achieve that by finding the minimum of 1/sharpe ratio

fun2 = lambda w: np.sqrt(np.dot(w,np.dot(w,CovMatrix)))/ExpectedValue.dot(w)
res_sharpe = minimize(fun2,x0,method='trust-constr',constraints = linear_constraint,bounds = bounds)
w_sharpe = res_sharpe.x

print('companies: ',list(CompaniesDf.columns))
print('portfolio: ', w_sharpe)
print('return: % .3f'% (ret(ExpectedValue,w_sharpe)*100))
print('risk: % .3f'% vol(w_sharpe,CovMatrix))
print('Sharpe ratio: ', (ret(ExpectedValue,w_sharpe))/vol(w_sharpe,CovMatrix))



#plot
w = w_min
num_ports = 100
gap = (np.amax(ExpectedValue) - ret(ExpectedValue,w_min))/num_ports

all_weights = np.zeros((num_ports, len(LogReturn.columns)))
all_weights[0],all_weights[1]=w_min,w_sharpe
ret_arr = np.zeros(num_ports)
ret_arr[0],ret_arr[1]=ret(ExpectedValue,w_min),ret(ExpectedValue,w_sharpe)
vol_arr = np.zeros(num_ports)
vol_arr[0],vol_arr[1]=vol(w_min,CovMatrix),vol(w_sharpe,CovMatrix)

for i in range(num_ports):
    port_ret = ret(ExpectedValue,w) + i*gap
    double_constraint = LinearConstraint([np.ones(LogReturn.shape[1]),ExpectedValue],[1,port_ret],[1,port_ret])
    x0 = w_min
    #Define a function for portfolio volatility.
    fun = lambda w: np.sqrt(np.dot(w,np.dot(w,CovMatrix)))
    a = minimize(fun,x0,method='trust-constr',constraints = double_constraint,bounds = bounds)

    all_weights[i,:]=a.x
    ret_arr[i]=port_ret
    vol_arr[i]=vol(a.x,CovMatrix)

sharpe_arr = ret_arr/vol_arr  

plt.figure(figsize=(20,10))
plt.scatter(vol_arr, ret_arr, c=sharpe_arr, cmap='viridis')
plt.scatter(vol(w_min,CovMatrix),ret(ExpectedValue,w_min),marker='*',s=350,label='Portfolio with lowest risk possible',c='blue')
plt.scatter(vol(w_sharpe,CovMatrix),ret(ExpectedValue,w_sharpe), marker='*', s=350, label='Max Sharpe ratio',c='red')
plt.colorbar(label='Sharpe Ratio')
plt.xlabel('Volatility',fontsize=22)
plt.ylabel('Return',fontsize=22)
plt.legend(fontsize=22)
plt.show()

