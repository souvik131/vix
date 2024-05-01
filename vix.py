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
from scipy.optimize import newton

# Set the timezone to Asia/Kolkata
kolkata = pytz.timezone('Asia/Kolkata')




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



def getVariance(file_path,calcBasisVix=True):
  file_arr=file_path.split(".")[0].split("-")

  date_string = "-".join(file_arr[len(file_arr)-3:])
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
  with open(file_path, 'r') as file:
      lines = file.readlines()
  df = pd.read_csv(StringIO(''.join([','.join(line.split(',')[1:]) for line in lines[1:]])))
  df=df[["BID","ASK","STRIKE","ASK.1","BID.1"]]
  df=df.rename(columns={"BID":"CALLBID","ASK":"CALLASK","STRIKE":"STRIKE","ASK.1":"PUTASK","BID.1":"PUTBID"})
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
  df["DIFF"]=df["STRIKE"]-df["STRIKE"].shift(1)
  kDf=df.iloc[df[['DIFF']].idxmin()]
  k=kDf["DIFF"].to_numpy()[0]

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
    cp=df[df["STRIKE"]==K0]['CALLMID'].to_numpy()[0]
    pp=df[df["STRIKE"]==K0]['PUTMID'].to_numpy()[0]
    civ=implied_volatility(F, K0, T, R, cp,True)
    piv=implied_volatility(F, K0, T, R, pp,False)
    variance=(((civ+piv)/2)**2)*T

  return date_string,variance,T


dataSet=[]



directory_path = './'
files_in_directory = os.listdir(directory_path)
csv_files = [file for file in files_in_directory if file.endswith('.csv')]
for file_path in csv_files:
  try:
    date_string,variance,T=getVariance(file_path,True)
    dataSet.append([date_string,variance,T])
  except Exception as e:
    print(file_path,e)

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
plt.show()

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

plt.figure(figsize=(10, 8))
sns.heatmap(ivol_matrix_float, annot=True, fmt=".2f", cmap="viridis", cbar_kws={'label': 'Forward Volatility'})
plt.title('Forward Volatility Between Expiries')
plt.xlabel('Expiry Date')
plt.ylabel('Expiry Date')
plt.show()