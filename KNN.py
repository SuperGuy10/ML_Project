import datetime
import pandas as pd
import talib as ta
from pandas_datareader import data as pandasData
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier

start = datetime.datetime(2017,01,04)
end = datetime.datetime(2017,12,01)
nameList=[]
f = open("stock.txt")
for i in range(50):
    nameList.append(f.readline().split("\n")[0])
f.close()

def getData(code,start,end):
    data = pandasData.DataReader(code,'yahoo',start,end)
    # print len(data)
    f_ema = ta.EMA(data['Close'].values, timeperiod=30).tolist()
    # print f_ema
    f_ma = ta.MA(data['Close'].values, timeperiod=30, matype=0).tolist()
    f_wma = ta.WMA(data['Close'].values, timeperiod=30).tolist()
    f_momentum = ta.MOM(data['Close'].values, timeperiod=10).tolist()
    f_roc = ta.ROC(data['Close'].values, timeperiod=10).tolist()
    # f_cycle = ta.HT_DCPERIOD(data['Close'].values).tolist()
    f_price = ta.WCLPRICE(data['High'].values, data['Low'].values, data['Close'].values).tolist()
    f_natr = ta.NATR(data['High'].values, data['Low'].values, data['Close'].values, timeperiod=14).tolist()
    f_stddev = ta.STDDEV(data['Close'].values, timeperiod=5, nbdev=1).tolist()
    X = pd.DataFrame(
        pd.np.array([f_ema, f_ma, f_wma, f_momentum, f_roc, f_price, f_natr, f_stddev]).T[32:]
        ,columns=['f_ema','f_ma','f_wma','f_momentum','f_roc','f_price','f_natr','f_stddev'])
    # print X['f_ema'].size
    # print X
    data = data['Close'].tolist()
    finaldata = [[] for i in range(2)]
    for i in range(0, len(data) - 1):
        temp = (data[i + 1] - data[i]) / data[i]
        finaldata[0].append(temp)
        if (temp > 0):
            finaldata[1].append(1)
        else:
            finaldata[1].append(0)
    # print len(data)
    data = data[31:len(data) - 1]
    # print data
    Y = pd.DataFrame(pd.np.array(finaldata).T, columns=['change', 'label'])
    X = X.join(Y)
    return X

def classifier(x):
    trainSize = 0.8
    X_training = x.loc[:,['f_ema','f_ma','f_wma','f_momentum','f_roc','f_price','f_natr','f_stddev']].values[0:int(x['f_ema'].size*trainSize)]
    # print len(X_training)
    Y_training = x['label'].values[0:int(x['f_ema'].size*trainSize)]
    KNN_train = KNeighborsClassifier().fit(X_training,Y_training)
    return KNN_train

stocksData = []
classifiers = []
for code in nameList:
    # print code
    x = getData(code, start, end)
    stocksData.append(x)  # since 01-05
    classifiers.append(classifier(x))

moneyTo500 = 1000000
moneyToML = 1000000

data500 = pandasData.DataReader("SPY",'yahoo',start,end)
Size = data500['Open'].count()-32
testSize = 0.2
testCount = int(data500['Open'].count()*testSize)
#
# benefitsTo500 = []
# data500X = getData("SPY",start,end)
# for i in range(testCount):
#     # if data500X.loc[Size - i - 1, 'label'] == 1:
#     money = (data500X.loc[Size - i - 1, ['change']]+1)*moneyTo500
#     benefitsTo500.append(moneyTo500)
#
# print int(moneyTo500)


benefits = []
for i in range(testCount):
    count = 0
    benefit = 0
    changes = []
    for j in range(50):
        stock = stocksData[j]
        classifier = classifiers[j]
        if(classifier.predict(stock.loc[Size-i-1,['f_ema','f_ma','f_wma','f_momentum','f_roc','f_price','f_natr','f_stddev']].values.reshape(-1, 8))==1):
            changes.append(stock.loc[Size-i-1,['change']])
            # print changes[count]
            count += 1

    for j in changes:
        benefit = benefit + j

    moneyToML = (1+benefit/count)*moneyToML
    benefits.append(moneyToML)

print int(moneyToML)


plt.plot(benefits)
plt.ylabel('benefit')
plt.xlabel('date')
plt.title('KNN')
plt.show()