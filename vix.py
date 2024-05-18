import pandas as pd
from io import StringIO
import numpy as np
import math
from datetime import datetime
import pytz
import seaborn as sns
import matplotlib.pyplot as plt
import os
from scipy.stats import norm
import py_vollib.black.implied_volatility
from curl_cffi import requests as req

# Set the timezone to Asia/Kolkata
kolkata = pytz.timezone('Asia/Kolkata')
s = req.Session()



def black_scholes_call(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    call_price = (S * norm.cdf(d1) - K * np.exp(-r * T) * norm.cdf(d2))
    return call_price

def black_scholes_put(S, K, T, r, sigma):
    d1 = (np.log(S / K) + (r + 0.5 * sigma ** 2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    put_price = (K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1))
    return put_price

def implied_volatility(S, K, T, r, market_price,isCall=True):
    if isCall:
      objective_function = lambda sigma: black_scholes_call(S, K, T, r, sigma) - market_price
    else:
      objective_function = lambda sigma: black_scholes_put(S, K, T, r, sigma) - market_price

    initial_guess = 0.2  # Starting guess for volatility, usually around 20%
    return newton(objective_function, initial_guess)



def getVariance(date_string,df,script,k,calcBasisVix=True):
  s=script
  if script==s:
    target_date = datetime.strptime(date_string, "%d-%b-%Y")
    target_date = kolkata.localize(target_date)
    today_date = datetime.now(kolkata)
    daysInBetween = (target_date - today_date).days - 1
    if daysInBetween<0:
      daysInBetween=0

    now_kolkata = datetime.now(kolkata)
    end_of_today_kolkata = now_kolkata.replace(hour=23, minute=59, second=59, microsecond=999999)
    remaining_time_today = end_of_today_kolkata - now_kolkata
    remaining_hours_today = remaining_time_today.seconds / 3600

    daysInYear = 365
    hoursInDay = 24
    expiryDayInMins=930



    #mibor rate
    R = 0.0717
    #mins left
    Tmins=remaining_hours_today*60+daysInBetween*hoursInDay*60+expiryDayInMins

    #clean option chain
    # with open(file_path, 'r') as file:
    #     lines = file.readlines()
    # df = pd.read_csv(StringIO(''.join([','.join(line.split(',')[1:]) for line in lines[1:]])))
    df=df[["CE.bidprice","CE.askPrice","strikePrice","PE.askPrice","PE.bidprice"]]
    df=df.rename(columns={"CE.bidprice":"CALLBID","CE.askPrice":"CALLASK","strikePrice":"STRIKE","PE.askPrice":"PUTASK","PE.bidprice":"PUTBID"})
    df.replace('-', np.nan, inplace=True)
    df.dropna(inplace=True)
    df['CALLBID'] = df['CALLBID'].replace(',', '', regex=True).astype(float)
    df['CALLASK'] = df['CALLASK'].replace(',', '', regex=True).astype(float)
    df['PUTBID'] = df['PUTBID'].replace(',', '', regex=True).astype(float)
    df['PUTASK'] = df['PUTASK'].replace(',', '', regex=True).astype(float)
    df['STRIKE'] = df['STRIKE'].replace(',', '', regex=True).astype(float)
    df.reset_index(inplace=True)

    #forward price and strike diff calculation
    df['CALLMID'] = (df['CALLBID'] + df['CALLASK']) / 2
    df['PUTMID'] = (df['PUTBID'] + df['PUTASK']) / 2
    df['SPREAD'] = ((df['CALLASK'] - df['CALLBID'])+(df['PUTASK'] - df['PUTBID']))/ 2
    fDf=df.iloc[df[['SPREAD']].idxmin()]
    F=fDf["STRIKE"].to_numpy()[0]+fDf["CALLMID"].to_numpy()[0]-fDf["PUTMID"].to_numpy()[0]
    # df["DIFF"]=df["STRIKE"]-df["STRIKE"].shift(1)
    # kDf=df.iloc[df[['DIFF']].idxmin()]
    # k=kDf["DIFF"].to_numpy()[0]
    # k=50

    #vix calculation
    T = Tmins / (daysInYear * hoursInDay * 60)
    K0 = math.floor(F/k)*k
    if calcBasisVix:
      calls_otm = df[(df['STRIKE'] > F)].copy()
      puts_otm = df[(df['STRIKE'] < F) ].copy()
      calls_otm['TERM'] = (k / (calls_otm['STRIKE']**2)) * np.exp(R * T) * calls_otm['CALLMID']
      puts_otm['TERM'] = (k / (puts_otm['STRIKE'] **2)) * np.exp(R * T)* puts_otm['PUTMID']
      variance = 2  * (calls_otm['TERM'].sum() +puts_otm['TERM'].sum()) -  ((F / K0 - 1) ** 2)
    else:
      try:
        cp=df[df["STRIKE"]==K0]['CALLMID'].to_numpy()[0]
        pp=df[df["STRIKE"]==K0]['PUTMID'].to_numpy()[0]
      except Exception as e:
        cp=df[df["STRIKE"]==(K0-k)]['CALLMID'].to_numpy()[0]
        pp=df[df["STRIKE"]==(K0-k)]['PUTMID'].to_numpy()[0]
        
      # civ=implied_volatility(F, K0, T, R, cp,True)
      # piv=implied_volatility(F, K0, T, R, pp,False)
      civ=py_vollib.black.implied_volatility.implied_volatility(cp, F, K0, T, R, 'c')
      piv=py_vollib.black.implied_volatility.implied_volatility(pp, F, K0, T, R, 'p')
      # print(civ,piv)
      variance=(((civ+piv)/2)**2)*T

    return date_string,variance,T
  return "",0,0

    
for stockname in ["NIFTY","BANKNIFTY","FINNIFTY","MIDCPNIFTY"]:
  #fetch data and calculate variance
  dataSet=[]
  

  url = "https://www.nseindia.com/companies-listing/corporate-filings-event-calendar"
  r = s.get(url, impersonate="chrome")

  url = "https://www.nseindia.com/api/option-chain-indices?symbol="+stockname
  r = s.get(url, impersonate="chrome")
  data=r.json()["records"]
  df = pd.json_normalize(data["data"])

  unique_expiry_dates = data['expiryDates']
  for expiry in unique_expiry_dates:
    filtered_data = df[df['expiryDate'] == expiry]
    try:
      diff=50
      if stockname=='MIDCPNIFTY':
         diff=25
      if stockname=="BANKNIFTY":
         diff=100
         
      date_string,variance,T=getVariance(expiry,filtered_data,stockname,diff,False)
      if date_string!="":
        dataSet.append([date_string,variance,T])
    except Exception as e:
      print(e)


  #plot term structure
  df=pd.DataFrame(dataSet,columns=["date","variance","time"])
  df["vol"]=np.sqrt(df["variance"]/df["time"])*100
  df=df.sort_values(by=['time'])
  dates = df["date"]
  values = df["vol"]
  fig, ax = plt.subplots()
  ax.plot(dates, values)
  plt.xticks(rotation=45) 
  plt.title('Term Structure')
  plt.xlabel('Date')
  plt.ylabel('Vol')
  plt.tight_layout() 
  plt.savefig(stockname+"_term.png")

  #plot fvol heatmap
  df.reset_index(inplace = True)
  ivol_matrix = pd.DataFrame(index=df['date'], columns=df['date'])
  for i in range(len(df)):
      for j in range(len(df)):
          if i != j:
              var_diff = df['variance'].iloc[j] - df['variance'].iloc[i]
              time_diff = df['time'].iloc[j] - df['time'].iloc[i]
              if time_diff != 0: 
                  ivol = np.sqrt(abs(var_diff / time_diff)) * 100
              else:
                  ivol = np.nan 
              ivol_matrix.iloc[i, j] = ivol
  ivol_matrix_float = ivol_matrix.astype(float)
  plt.figure(figsize=(13, 11))
  sns.heatmap(ivol_matrix_float, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Forward Volatility'})
  plt.title('Forward Volatility Between Expiries')
  plt.xlabel('Expiry Date')
  plt.ylabel('Expiry Date')
  plt.savefig(stockname+"_fv.png")
  # plt.show()