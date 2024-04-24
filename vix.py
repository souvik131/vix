import pandas as pd
from io import StringIO
import numpy as np
import math

#parameters

#future price 
F = 22509.75 
#mibor rate 
R = 0.0717 
#strike difference
k=50
#mins left
Tmins=36*24*60+930 
#nse option chain file
file_path = 'option-chain-ED-NIFTY-30-May-2024.csv' 

#clean option chain
with open(file_path, 'r') as file:
    lines = file.readlines()
df = pd.read_csv(StringIO(''.join([','.join(line.split(',')[1:]) for line in lines[1:]])))
df=df[["BID","ASK","STRIKE","ASK.1","BID.1"]]
df=df.rename(columns={"BID":"CALLBID","ASK":"CALLASK","STRIKE":"STRIKE","ASK.1":"PUTASK","BID.1":"PUTBID"})
df.replace('-', np.nan, inplace=True)
df.dropna(inplace=True)

#vix calculation
K0 = math.floor(F/k)*k
T = Tmins / (356 * 24 * 60) 
df['CALLBID'] = df['CALLBID'].replace(',', '', regex=True).astype(float)
df['CALLASK'] = df['CALLASK'].replace(',', '', regex=True).astype(float)
df['PUTBID'] = df['PUTBID'].replace(',', '', regex=True).astype(float)
df['PUTASK'] = df['PUTASK'].replace(',', '', regex=True).astype(float)
df['STRIKE'] = df['STRIKE'].replace(',', '', regex=True).astype(float)
df['CALLMID'] = (df['CALLBID'] + df['CALLASK']) / 2
df['PUTMID'] = (df['PUTBID'] + df['PUTASK']) / 2
calls_otm = df[(df['STRIKE'] > F)].copy()
puts_otm = df[(df['STRIKE'] < F) ].copy()
calls_otm['TERM'] = (k / (calls_otm['STRIKE']**2)) * np.exp(R * T) * calls_otm['CALLMID']
puts_otm['TERM'] = (k / (puts_otm['STRIKE'] **2)) * np.exp(R * T)* puts_otm['PUTMID']
variance = (2 / T) * (calls_otm['TERM'].sum() +puts_otm['TERM'].sum()) - (1 / T) * ((F / K0 - 1) ** 2)
vix = 100 * np.sqrt(variance)
print(vix)