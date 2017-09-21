# -*- coding: utf-8 -*-
"""
Created on Fri Sep 26 15:35:44 2014

@author: Administrator
"""
from datetime import datetime, date
import numpy as np
import pandas as pd
import urllib2
import scipy.optimize as syop
import os
import scipy as sp
import pylab as ply

# def no.1
def print_full(x):
    pd.set_option('display.height', 2*len(x))
    pd.set_option('display.max_rows', len(x))
    print(x)
    pd.reset_option('display.height')
    pd.reset_option('display.max_rows')

# def no.2
def port_mean(W, R):
        return np.sum(np.array(R)*(np.transpose(np.array(W))))
def port_var(W, C):
    A=np.dot((W).T,C)
    AA=np.dot(A,W)
    return float(AA)
def port_std(W, C):
    return np.sqrt(W.T.dot(C).dot(W))
def port_mean_var(W, R, C):
        return port_mean(W, R), port_var(W, C)
def port_mean_std(W, R, C):
        return port_mean(W, R), port_std(W, C)
        
def solve_frontier(R, C, rf):
    def fitness(W, R, C, r):
        # For given level of return r, find weights which minimizes
        # portfolio variance.
        mean, var = port_mean_var(W, R, C)
        # Big penalty for not meeting stated portfolio return 
        # effectively serves as optimization constraint
        penalty = 100*abs(mean-r)       
        return var + penalty
    frontier_mean, frontier_var, frontier_weights = [], [], []
    n = len(R)  # Number of assets in the portfolio
    # Iterate through the range of returns on Y axis
    for r in np.linspace(min(R), max(R), num=40): 
        W = np.ones([n])/n     # start optimization with equal weights
        b_ = [(0,1) for i in range(n)]
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })
        optimized = syop.minimize(fitness, W, (R, C, r), 
                             method='SLSQP', constraints=c_, bounds=b_)   
        if not optimized.success: 
            raise BaseException(optimized.message)
        # add point to the min-var frontier [x,y] = [optimized.x, r]
        frontier_mean.append(r)             
        frontier_var.append(port_var(optimized.x, C))
        frontier_weights.append(optimized.x)
    return np.array(frontier_mean), np.array(frontier_var), frontier_weights

def solve_weights(R, C, rf,lower,upper):
        def fitness(W, R, C, rf):
                mean, var = port_mean_var(W, R, C)      # calculate mean/variance of the portfolio
                sharp_ratio = (mean - rf) / np.sqrt(var)       # utility = Sharpe ratio
                return 1/sharp_ratio                           # maximize the utility, minimize its inverse value
        n = len(R)
        W = np.ones([n])/n                                         # start optimization with equal weights
        b_ = [(lower,upper) for i in range(n)]        # weights for boundaries between 0%..100%. No leverage, no shorting
        c_ = ({'type':'eq', 'fun': lambda W: sum(W)-1. })       # Sum of weights must be 100%
        optimized = syop.minimize(fitness, W, (R, C, rf), method='SLSQP', constraints=c_, bounds=b_)  
        if not optimized.success: 
                raise BaseException(optimized.message)
        return optimized.x 

def prepare_views_and_link_matrix(names, views):
        r, c = len(views), len(names)
        Q = [views[i][3] for i in range(r)]     # view matrix
        P = np.zeros([r, c])                    # link matrix
        nameToIndex = dict()
        for i, n in enumerate(names):
                nameToIndex[n] = i
        for i, v in enumerate(views):
                name1, name2 = views[i][0], views[i][2]
                P[i, nameToIndex[name1]] = +1 if views[i][1]=='>' else -1
                P[i, nameToIndex[name2]] = -1 if views[i][1]=='>' else +1
        return np.array(Q), P
# def no.3
''' calculate annual sharpe ration assuming 252 trading days per year. annual_sharpe ((daily) excess_retruns, (daily) excess_std, N=252 by default)
'''
def getHistoricData(symbol, sDate=(1990,1,1),eDate=date.today().timetuple()[0:3],verbose=True):
    """ 
    get data from Yahoo finance and return pandas dataframe

    symbol: Yahoo finanance symbol
    sDate: start date (y,m,d)
    eDate: end date (y,m,d)
    """

    urlStr = 'http://ichart.finance.yahoo.com/table.csv?s={0}&a={1}&b={2}&c={3}&d={4}&e={5}&f={6}'.\
    format(symbol,sDate[1]-1,sDate[2],sDate[0],eDate[1]-1,eDate[2],eDate[0])
 
    try:
        lines = urllib2.urlopen(urlStr).readlines()
    except Exception, e:
        s = "Failed to download:\n{0}".format(e);
        print s

    dates = []
    data = [[] for i in range(6)]
    #high
    
    # header : Date,Open,High,Low,Close,Volume,Adj Close
    for line in lines[1:]:
        #print line
        fields = line.rstrip().split(',')
        dates.append(datetime.strptime( fields[0],'%Y-%m-%d'))
        for i,field in enumerate(fields[1:]):
            data[i].append(float(field))
       
    idx = Index(dates)
    data = dict(zip(['open','high','low','close','volume','adj_close'],data))
    
    # create a pandas dataframe structure   
    df = DataFrame(data,index=idx).sort()
    
    if verbose:
        print 'Got %i days of data' % len(df)
    
    return df


def get_historical_adj_close_price(symbols,sdate,edate):
    '''get historical adjust close price online (Yahoo)'''
    naming=locals()
    plist = pd.DataFrame()
    for x in symbols:
        plist['{}'.format(x)] = (getHistoricData(x,sdate,edate)).adj_close   
    return plist
    
def get_market_cap(symbols):
    caps= list() 
    clist= list()
    for x in (symbols):
        caps.append(yq.get_market_cap(x))
    for s in caps:
        if 'B' in s:
            clist.append(float(s.replace('B', '')))
        else:
            clist.append(float(s.replace('M', ''))/1000)  
    return pd.DataFrame(data=clist,index=symbols,columns=['Market_cap']) 

def BL_optimizer(capital,rf,Symbol,sDate, eDate, sDate_out, eDate_out, a_, b_):
    #['AAPL','BRK-B','DPS','JNJ','LMT','SNDK','TJX','MSFT','CSCO','GD','GILD','ORCL','WDC','TWX','RAI','ATVI','RHT']
    symbol=Symbol
    symbol.sort()
    
    plist=get_historical_adj_close_price(symbol, sDate, eDate)
    cap=get_market_cap(symbol)
    capw = cap.Market_cap / float(cap.sum())    
    eqw = 1.0/len(symbol)*np.ones(len(symbol))   # create weights
    rlist = plist.pct_change()
    
    cov, cor = rlist.cov()*250, rlist.corr()
    r = (1+rlist.mean())**250-1
    lmb = (port_mean(capw,r) - rf) / ft.port_var(capw,cov)             # Calculate return/risk trade-off
    adj_r = np.dot(np.dot(lmb, cov), capw) # Compute equilibrium excess returns
    r_report=pd.DataFrame({'r' : r, "capw" : capw, 'adj_r' : adj_r})
    
    # Determine views to the equilibrium returns and prepare views (Q) and link (P) matrices
    views = [
            ('JNJ', '<', 'RHT', 0), 
            ('JNJ', '<', 'BRK-B', 0)
            ]
    
    Q, P = prepare_views_and_link_matrix(symbol, views)
    print('Views Matrix')
    print(Q)
    print('Link Matrix')
    print(P)
    
    tau = .025 # scaling factor
    
    # Calculate omega - uncertainty matrix about views
    omega = np.dot(np.dot(np.dot(tau, P), cov), np.transpose(P)) # 0.025 * P * C * transpose(P)
    # Calculate equilibrium excess returns with views incorporated
    sub_a = np.linalg.inv(np.dot(tau, cov))
    sub_b = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), P)
    sub_c = np.dot(np.linalg.inv(np.dot(tau, cov)), adj_r)
    sub_d = np.dot(np.dot(np.transpose(P), np.linalg.inv(omega)), Q)
    Pi = np.dot(np.linalg.inv(sub_a + sub_b), (sub_c + sub_d))
    r_report['view adj_r']=Pi
         
    weight=np.around(solve_weights(np.transpose(Pi+rf),cov,rf, a_, b_),decimals=4)
    r_report['view weight']=weight
    nshare=np.round(capital*weight/plist.tail(1))
    report=np.around(port_mean_std(weight, r, cov),decimals=4)
    sharp_adj=(report[0]-rf)/report[1]
    sharp_raw=(r.mean()-rf)/port_std(eqw,cov)
    print 'return_stas:', report
    print 'weight :', nshare.transpose() 
    print 'sharp_adj:', sharp_adj
    print 'sharp_raw:', sharp_raw
    
    # backtesting the weights
    plist_out = get_historical_adj_close_price(symbol,(2012,1,1),(2014,12,24))
    rlist_out = plist_out.pct_change()
    cov_out, cor_out = rlist_out.cov()*250, rlist_out.corr()
    r_out = (1+rlist_out.mean())**250-1
    
    mean_std_adj= port_mean_std(weight,r_out,cov_out)
    mean_std_raw= port_mean_std(eqw,r_out,cov_out)
    print 'mean_std_adj:', mean_std_adj
    print 'mean_std_raw:', mean_std_raw
    sharp_out_adj= (mean_std_adj[0]-rf)/mean_std_adj[1]
    sharp_out_raw= (mean_std_raw[0]-rf)/mean_std_raw[1]
    print 'sharp_out_adj:', sharp_out_adj
    print 'sharp_out_raw:', sharp_out_raw

def annual_sharpe(excess, N=252):
    return np.sqrt(N) * excess.mean() / excess.std()
    
# def no.3
    ''' do quicker rounding'''
def rounding(number,dig):
    return round(float(number),dig)
    

from numpy import log, polyfit, sqrt, std, subtract

def hurst(time_series):
    """Returns the Hurst Exponent of the time series vector ts"""
# Create the range of lag values
    ts=log(time_series)
    lags = range(2, 50)
    # Calculate the array of the variances of the lagged differences
    tau = [sqrt(std(subtract(ts[lag:], ts[:-lag]))) for lag in lags]
    # Use a linear fit to estimate the Hurst Exponent
    poly = polyfit(log(lags), log(tau), 1)
    # Return the Hurst exponent from the polyfit output
    return poly[0]*2.0
    
import statsmodels.tsa.stattools as tes
def cadfuller(y,x, intercept_if = False):
    res = (pd.ols(y = y,x = x,intercept = intercept_if)).resid
    Cadf = tes.adfuller(res)
    return Cadf

def half_life(y,x):
    """calculate the half_life for a pair"""
    model = pd.ols(y=y,x=x,intercept = False)
    res = model.resid
    dres = res.shift(1)[1:]-res[1:]
    resmodel = pd.ols(y=dres,x=res, intercept = False)
    half = -np.log(2)/resmodel.beta[0]
    return half

def KF(y, x, delta = 0.0000007, Ve = 0.001, n_iter = 50):
    """(y, x, delta = 0.0000007, Ve = 0.001, n_iter = 1000)"""
    x= np.transpose(np.array((x,np.ones((len(x))))))
    yhat, e, Q = np.zeros(len(y))*np.NaN, np.zeros(len(y))*np.NaN, np.zeros(len(y))*np.NaN
    P, R = np.zeros((2,2)),np.zeros((2,2))
    beta = np.zeros((2,len(y)))*np.NaN
    Vw = delta/(1-delta)*np.eye(2)    
    beta[:,0] = 0
    
    for t in range(len(y)):   
        if t>0:     
            beta[:,t] = beta[:,t-1]
            R= P+Vw            
        yhat[t] = np.dot(x[t,:],beta[:,t])
        Q[t] = np.dot(np.dot(x[t,:],R),x[t,:].T)+Ve
        e[t] = y[t]-yhat[t]
        K = (np.dot(R,x[t,:].T))/Q[t]
        beta[:,t] = beta[:,t] +np.dot(K,e[t])
        P = R-np.dot(np.dot(K,x[t,:]),R)
    return e[n_iter:],Q[n_iter:]**0.5

def get_order_book(symbols = ["MDY","IJH"], delta = 0.0000003, Ve = 0.001, n_iter = 50):
    
    """y =y, x = x, delta = 0.0000003, Ve = 0.001, n_iter = 50"""    
    naming = locals()
    for sym in symbols:
        naming["%s" % sym] = pd.io.parsers.read_csv(os.path.join('C:\Dropbox\Python Scripts\Trading Engine', 
        '%s.csv' % sym),header=None, index_col=0, parse_dates= True,
        names=['date-','high-'+'{}'.format(sym), 'low-'+'{}'.format(sym),'open-'+'{}'.format(sym),
                        'close-'+'{}'.format(sym),'volume-'+'{}'.format(sym),'oi-'+'{}'.format(sym)])
        
    y = naming['%s' % symbols[0]]['close-'+'{}'.format(symbols[0])]
    x = naming['%s' % symbols[1]]['close-'+'{}'.format(symbols[1])]    
    long_market, short_market = False, False
    e, Q = ft.KF(y, x, delta, Ve, n_iter)
    order_book = [None]*len(Q)
    for i in range(len(Q)):
        if e[i] < -Q[i] and not long_market and not short_market:
            long_market = True
            order_book[i] = "long_enter"
        if e[i] > -Q[i] and long_market:
            long_market = False
            order_book[i] = "long_exit"
        if e[i] > Q[i] and not short_market and not long_market:
            short_market = True
            order_book[i] = "short_enter"
        if e[i] < Q[i] and short_market:
            short_market = False
            order_book[i] = "short_exit"
    return order_book, y
    
def his(x, binss, color = "b", alpha = 0.75):
    """
    plot the histogram of an array
    """
    n, bins, patches = ply.hist(x, binss, normed=1, histtype='stepfilled')
    ply.setp(patches, 'facecolor', color, 'alpha', alpha)
        
def onettest(data, confidence=0.95):
    """
    return t and p value and confidence level of a one sample t-test
    """
    a = 1.0*np.array(data)
    n = len(a)
    m, se = np.mean(a), sp.stats.sem(a)
    h = se * sp.stats.t._ppf((1+confidence)/2., n-1)
    tp = sp.stats.ttest_1samp(data,0)
    tplist = []
    for i in tp:
        tplist.append(round(i,4))
    print "T,P value:", tplist, ", " "%s" % confidence + " confidence interval:", round(m-h,4), round(m+h,4)
    
def spliter(df, year_no = 1, month_s = 0, month_e = 3):
    """
    split an input dateframe into a few months data, using year no and month no to contrrol the output
    """
    ylist = []
    mlist = []
    for group in df.groupby(df.index.year):
        ylist.append(group[1])
    for i in range(len(ylist)):
        mlist.append([])    
        for group in ylist[i].groupby(ylist[i].index.month):
            mlist[i].append(group[1])    
    return pd.concat(mlist[year_no-1][month_s-1:month_e])