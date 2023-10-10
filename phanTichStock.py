import pandas as pd
import itertools
import numpy as np
from math import sqrt
from matplotlib import pyplot as plt

file = "CafeF.HSX.Upto15.03.2022.csv"
df = pd.read_csv("CafeF.HSX.Upto15.03.2022.csv").drop(columns=['<High>', '<Low>'])
df = df[['<Volume>', '<Ticker>', '<DTYYYYMMDD>', '<Open>', '<Close>']]
df.columns = ['Volume', 'Stock', 'Date', 'Open', 'Close']
df['Date'] = pd.to_datetime(df['Date'], format='%Y%m%d')
filter = (df['Date'] >= '2021-04-01')
df = df[filter]
df = df.set_index('Date').groupby('Stock').resample('W').agg({
    'Volume': sum, 'Open': 'first', 'Close': 'last'}).reset_index()[
    ['Volume', 'Stock', 'Date', 'Open', 'Close']]
df = df.sort_values(by=['Stock', 'Date'])
df['HPY'] = df['Close'] / df['Open'] - 1
df2 = df[['Stock', 'HPY', 'Date']]
df2['Date'] = pd.to_datetime(df2['Date'])
df2 = df2.set_index('Date').groupby('Stock').mean().reset_index()
df2.columns = ['Stock', 'Mean']
df = df.merge(df2)
df3 = df.groupby(by='Stock')['HPY'].describe()[['count', 'std']].reset_index()
df3.columns = ['Stock', 'count', 'std']
df = df.merge(df3)
df['Mean/Std'] = df['Mean'] / df['std']
df['M HPY-std'] = df['Mean'] - df['std']
df['M HPY+std'] = df['Mean'] + df['std']
df.to_excel('data.xlsx', index=False)

# get top 3 stock
n = 3
df2 = df.groupby('Stock')['Mean/Std'].describe()[['count', 'mean']]
df2 = df2.sort_values('mean', ascending=False)[df2['count'] >= 20]
df2.to_excel('top.xlsx')
stocks = [i[0] for i in df2.reset_index()[['Stock']][:n].values]

# get correlation
filter = (df['Stock'].isin(stocks))
df0 = df[filter]
df0.to_excel('chose.xlsx', index=False)
df_corr = df0.pivot(index='Date', columns='Stock', values='HPY').corr()
df_corr.to_excel('correlation.xlsx')
df_corr = pd.read_excel('correlation.xlsx')
max = 100
step = 1
min = 0
numbers = 100
weights = [i / numbers for i in range(min, max + step, step)]

def possible_weightings(weights=weights, n = 3, target = 1):
    for perm in itertools.product(weights, repeat=n):
        if sum(perm) == target:
            yield perm

class stock_class():
    def __init__(self, name, er, std):
        self.name = name
        self.w = 0
        self.er = er
        self.std = std

    def set_weight(self, weight):
        self.w = weight

def expected_return(stocks):
    er = 0
    for stock in stocks:
        er += stock.w * stock.er
    return er

def r(df, stock1, stock2):
    return df.reset_index()[[stock1.name]][df['Stock'] == stock2.name].iat[0,0]

def expected_std(stocks, df=df):
    es = 0
    for stock in stocks:
        es += stock.w**2 * stock.std**2
    for stock_coms in itertools.combinations(stocks, r=2):
        es += 2 * stock_coms[0].w  * stock_coms[1].w * stock_coms[0].std * stock_coms[1].std * r(
            df, stock_coms[0], stock_coms[1])
    return sqrt(es)

print(stocks)
stock_list = []
for stock in stocks:
    name = stock
    er = df[df['Stock'] == name][['Mean']].iat[0,0]
    std = df[df['Stock'] == name][['std']].iat[0,0]
    stock_list.append(stock_class(name, er, std))

def main(stock_list=stock_list, ws=weights):
    for weight_list in possible_weightings(ws):
        results = []
        for i, weight in enumerate(weight_list):
            stock_list[i].set_weight(weight)
        for stock in stock_list:
            results.append(stock.w)
        results.append(expected_return(stock_list))
        results.append(expected_std(stock_list, df=df_corr))
        yield results

df_return = pd.DataFrame(np.array([result for result in main()]), columns=[f'w-{stocks[0]}', f'w-{stocks[1]}', f'w-{stocks[2]}', 'er', 'std'])
df_return['er/std'] = df_return['er'] / df_return['std']
df_return = df_return.sort_values('er/std', ascending=False)
df_return.to_excel('main.xlsx', index=False)

#run for plot png
max = 100
step = 10
min = 0
numbers = 100
weights2 = [i / numbers for i in range(min, max + step, step)]
df_return2 = pd.DataFrame(np.array([result for result in main(ws=weights2)]), columns=[f'w-{stocks[0]}', f'w-{stocks[1]}', f'w-{stocks[2]}', 'er', 'std'])
df_return2['er/std'] = df_return2['er'] / df_return2['std']
df_return2 = df_return2.sort_values('er/std', ascending=False)
plt.scatter(df_return2['er'], df_return2['std'], c='blue')
plt.xlabel('er')
plt.ylabel('std')
plt.savefig('result.png')
plt.show()
