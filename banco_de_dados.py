import pandas as pd
import numpy as np
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from sklearn.metrics import precision_recall_curve
from keras import regularizers
import heapq as hq








def analisa_features_vs_papel(base):
    plt.rcParams['figure.figsize'] = (10, 8)
    base.plot(x = 'data', y = ['acao_close'])
    base.plot(x = 'data', y = ['p/l'])
    base.plot(x = 'data', y = ['roic'])
    base.plot(x = 'data', y = ['dolar_close','dolar_open','dolar_high','dolar_low'])
    base.plot(x = 'data', y = ['petroleo_close','petroleo_open','petroleo_high','petroleo_low'])
    base.plot(x = 'data', y = ['dji_close','dji_open','dji_high','dji_low'])
    base.plot(x = 'data', y = ['sp500_close','sp500_open','sp500_high','sp500_low'])
    base.plot(x = 'data', y = ['risco_brasil'])
    base.plot(x = 'data', y = ['ibov_fut_close','ibov_fut_open','ibov_fut_high','ibov_fut_low'])





def plot_variaveis(base):
    plt.rcParams['figure.figsize'] = (12, 16)
    f, axarr = plt.subplots(base.shape[1], sharex=True)
    f.suptitle('Variáveis')
    for group in range(base.shape[1]):
        axarr[group].plot(base.iloc[:, group])
        axarr[group].set_title(base.columns[group], loc = 'right')
        f.subplots_adjust(hspace=0.5)
    for ax in axarr:
        ax.label_outer()
    plt.show()

    
# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
	n_vars = 1 if type(data) is list else data.shape[1]
	df = DataFrame(data)
	cols, names = list(), list()
	# input sequence (t-n, ... t-1)
	for i in range(n_in, 0, -1):
		cols.append(df.shift(i))
		names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
	# forecast sequence (t, t+1, ... t+n)
	for i in range(0, n_out):
		cols.append(df.shift(-i))
		if i == 0:
			names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
		else:
			names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
	# put it all together
	agg = concat(cols, axis=1)
	agg.columns = names
	# drop rows with NaN values
	if dropnan:
		agg.dropna(inplace=True)
	return agg
    

def precision(y_true, y_pred):
    # Calculates the precision
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    # Calculates the recall
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall

def fbeta_score(y_true, y_pred, beta=1):
    # Calculates the F score, the weighted harmonic mean of precision and recall.

    if beta < 0:
        raise ValueError('The lowest choosable beta is zero (only precision).')
        
    # If there are no true positives, fix the F score at 0 like sklearn.
    if K.sum(K.round(K.clip(y_true, 0, 1))) == 0:
        return 0

    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    bb = beta ** 2
    fbeta_score = (1 + bb) * (p * r) / (bb * p + r + K.epsilon())
    return fbeta_score

def fmeasure(y_true, y_pred):
    # Calculates the f-measure, the harmonic mean of precision and recall.
    return fbeta_score(y_true, y_pred, beta=1)



def plot_prec_rec_thr(precisions_train, recalls_train, thresholds_train, precisions_trade, recalls_trade, thresholds_trade):

    plt.plot(thresholds_train, precisions_train[:-1], "b--", label = "Precision Train")
    plt.plot(thresholds_train, recalls_train[:-1], "g-", label = "Recall Train")
    plt.xlabel("Threshold")
    plt.legend(loc = "center left")
    plt.ylim([0,1])

    plt.plot(thresholds_trade, precisions_trade[:-1], "r--", label = "Precision Trade")
    plt.plot(thresholds_trade, recalls_trade[:-1], "m-", label = "Recall Trade")
    plt.xlabel("Threshold")
    plt.legend(loc = "botton left")
    plt.ylim([0,1])
    

def extrai_papeis(base_total,papeis):
    base = base_total.loc[base_total['codigo'].isin(papeis)]
    #base = base.dropna(axis = 0, how = 'any')
    return base



def dataprep(base_in,features,n_days,n_train_days,n_trade_days,day_init,ret_target):   

    base = base_in.set_index("data")    
    base['retorno'] = base['acao_close'].pct_change(1)
    base = base[features]
    base['y'] = ((base['retorno'] > ret_target)).astype(np.int)    
    base = base.dropna(axis = 0, how = 'any')
    base.shape
    base.head()
    plot_variaveis(base)
    values_x = base.iloc[:,:base.shape[1]-1].values
    values_y = base.iloc[:,-1].values.reshape(base.shape[0],1)
    n_features = values_x.shape[1]
    
    # normalize features
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    values_x_scaled = scaler_x.fit_transform(values_x)
    values = np.concatenate((values_x_scaled, values_y), axis = 1)
    plot_variaveis(DataFrame(values))    
    
    # frame as supervised learning
    values_reframed = series_to_supervised(values, n_days, 1)
    print(values_reframed.shape)
    
    # split into train and test sets
    values = values_reframed.values    
    train = values[day_init:day_init+n_train_days, :]
    trade = values[day_init+n_train_days:day_init+n_train_days+n_trade_days, :]
    
    # split into input and outputs
    n_obs = n_days * n_features
    train_X, train_y = train[:, :n_obs], train[:, -1]
    trade_X, trade_y = trade[:, :n_obs], trade[:, -1]
    print(train_X.shape, train_y.shape,trade_X.shape, trade_y.shape)
    
    # reshape input to be 3D [samples, timesteps, features]
    train_X = train_X.reshape((train_X.shape[0], n_days, n_features))
    trade_X = trade_X.reshape((trade_X.shape[0], n_days, n_features))
    print(train_X.shape, train_y.shape,trade_X.shape, trade_y.shape)

    return train_X, trade_X, train_y, trade_y








# dados históricos
base_total = carrega_dados()
# configure stock
papeis        = ['ITUB4','VALE5','PETR4','BBDC4']
#features    = ['p/l','roic','petroleo_close','dolar_close','dji_close','sp500_close','risco_brasil','ibov_fut_close','acao_close']
#features    = ['p/l','petroleo_close','dolar_close','dji_close','sp500_close','risco_brasil','ibov_fut_close','acao_close']
#features     = ['p/l','dolar_close','dji_close','sp500_close','risco_brasil','ibov_fut_close','acao_close','retorno']
features     = ['codigo','retorno','acao_close','roe', 'pl', 'irf','sharpe','dolar_close','dji_close','sp500_close','risco_brasil','ibov_fut_close']
base_in      = extrai_papeis(base_total,papeis)
#analisa_features_vs_papel(base)

# define dataset
n_days       = 21
day_init     = 800
n_train_days = 756
n_trade_days = 252
ret_target   = 0.0
train_X, trade_X, train_y, trade_y = dataprep(base_in,features,n_days,n_train_days,n_trade_days,day_init,ret_target)
print("% retornos > ", ret_target, " no treino:", train_y.mean())
print("% retornos > ", ret_target, " no trade:", trade_y.mean())

# configure network
n_batch      = 1
n_epoch      = 100
n_neurons    = 10
two_layers   = 0
n_neurons2   = 5
performance  = np.zeros((n_epoch,5))

# design network
model = Sequential()
model.add(LSTM(n_neurons, 
               batch_input_shape=(n_batch, train_X.shape[1], train_X.shape[2]), 
               stateful = True,
#               kernel_regularizer = regularizers.l2(0.01),
#               recurrent_regularizer = regularizers.l2(0.01), 
#               bias_regularizer = regularizers.l2(0.01), 
#               activity_regularizer = regularizers.l2(0.01),
               return_sequences = False))
#(lambda x: False if two_layers == 1 else True)
if two_layers == 1:
    model.add(LSTM(n_neurons2))
model.add(Dense(1))
model.add(Activation('sigmoid'))
model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy', precision, recall, fmeasure])
model.summary()

history = model.fit(train_X, train_y, epochs = n_epoch, batch_size = n_batch, verbose = 1, shuffle = False)
performance[:,0] = history.history["loss"]
performance[:,1] = history.history["acc"]
performance[:,2] = history.history["precision"]
performance[:,3] = history.history["recall"]
performance[:,4] = history.history["fmeasure"]


performance = DataFrame(performance, columns = ["loss","acc","precision","recall", "fmeasure"])
print(performance.iloc[-1,:])

# row and column sharing
plt.rcParams['figure.figsize'] = (10, 8)
f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, sharex='col')
ax1.plot(performance.loc[:,["loss"]])
ax2.plot(performance.loc[:,["acc"]])
ax3.plot(performance.loc[:,["precision"]])
ax4.plot(performance.loc[:,["recall"]])
ax1.set_title('Loss')
ax2.set_title('Accuracy')
ax3.set_title('Precision')
ax4.set_title('Recall')



train_y_pred = np.array(model.predict(train_X,batch_size=n_batch))
train_y_pred = DataFrame(np.append(np.reshape(train_y,(len(train_y),1)), np.reshape(train_y_pred,(len(train_y_pred),1)), axis = 1), columns = ["A","B"])
train_y_pred_ones = np.array(train_y_pred.query('A > 0').iloc[:,1])
train_y_pred_zeros = np.array(train_y_pred.query('A < 1').iloc[:,1])
n, bins, patches = plt.hist(train_y_pred_ones, 50, density = False, facecolor='g', alpha=0.50, label = ['retorno = 1'])
n, bins, patches = plt.hist(train_y_pred_zeros, 50, density = False, facecolor='b', alpha=0.50, label = ['retorno = 0'])
plt.xlabel('Probabilidade')
plt.ylabel('Frequência')
plt.title('Histograma da Probabilidade de Retorno - Período de Treino')
plt.grid(False)
plt.legend()
plt.show()


trade_y_pred = np.array(model.predict(trade_X,batch_size=n_batch))
trade_y_pred = DataFrame(np.append(np.reshape(trade_y,(len(trade_y),1)), np.reshape(trade_y_pred,(len(trade_y_pred),1)), axis = 1), columns = ["A","B"])
trade_y_pred_ones = np.array(trade_y_pred.query('A > 0').iloc[:,1])
trade_y_pred_zeros = np.array(trade_y_pred.query('A < 1').iloc[:,1])
n, bins, patches = plt.hist(trade_y_pred_ones, 50, density = False, facecolor='g', alpha=0.50, label = ['retorno = 1'])
n, bins, patches = plt.hist(trade_y_pred_zeros, 50, density = False, facecolor='b', alpha=0.50, label = ['retorno = 0'])
plt.xlabel('Probabilidade')
plt.ylabel('Frequência')
plt.title('Histograma da Probabilidade de Retorno - Período de Teste')
plt.grid(False)
plt.legend()
plt.show()


score_treino = model.evaluate(train_X, train_y, batch_size = n_batch, verbose = 1)
score_trade  = model.evaluate(trade_X, trade_y, batch_size = n_batch, verbose = 1)
print('Loss Treino..........:{:10.4f}'.format(score_treino[0]))
print('Loss Trade...........:{:10.4f}'.format(score_trade[0]))
print('Accuracy Treino......:{:10.4f}'.format(score_treino[1]))
print('Accuracy Trade.......:{:10.4f}'.format(score_trade[1]))
print('Precision Treino.....:{:10.4f}'.format(score_treino[2]))
print('Precision Trade......:{:10.4f}'.format(score_trade[2]))
print('Recall Treino........:{:10.4f}'.format(score_treino[3]))
print('Recall Trade.........:{:10.4f}'.format(score_trade[3]))


y_train_score = model.predict(train_X,batch_size = n_batch)
precisions_train, recalls_train, thresholds_train = precision_recall_curve(train_y,y_train_score)

y_trade_score = model.predict(trade_X,batch_size = n_batch)
precisions_trade, recalls_trade, thresholds_trade = precision_recall_curve(trade_y,y_trade_score)

plot_prec_rec_thr(precisions_train, recalls_train, thresholds_train,precisions_trade, recalls_trade, thresholds_trade)
plt.show()




# estratégia de compra de ações
def comprar_acoes(k, x_values, x_trade, ind_y_trade, model, tks, retornos_periodo_original):

    l0 = 0
    ln = x_values.shape[0]
    q = ln # qtde ações na base

    ret_medio_ibov = np.zeros(shape=(len(ind_y_trade)))
    ret_medio_ibov_acum = np.ones(shape=(len(ind_y_trade)+1))

    ret_medio_port = np.zeros(shape=(len(ind_y_trade),len(k)))
    ret_medio_port_acum = np.ones(shape=(len(ind_y_trade)+1,len(k)))    
    
    ret_medio_port_short = np.zeros(shape=(len(ind_y_trade),len(k)))
    ret_medio_port_short_acum = np.ones(shape=(len(ind_y_trade)+1,len(k)))

    ret_medio_port_long = np.zeros(shape=(len(ind_y_trade),len(k)))
    ret_medio_port_long_acum = np.ones(shape=(len(ind_y_trade)+1,len(k)))
    
    ret_versus_prob = np.zeros(shape=(1,2))
    
    for i in range(len(ind_y_trade)):
#        print("day-trade #", ind_y_trade[i], "i = ", i)
        
        trade_day = x_trade[l0:ln,:,:]
        l0+=q
        ln+=q
        prob = np.array(model.predict(trade_day))
        y = prob[:,0]
    
#        if (i % 50) == 0:        
#            x = np.arange(len(prob))
#            y = prob[:,0]
#            show_probs(x,y)
    
        ret_orig = retornos_periodo_original.iloc[:,ind_y_trade[i]].values
        ret_orig_y = DataFrame(np.stack((ret_orig, y), axis = 1))
        ret_orig_y = ret_orig_y.dropna(axis = 0, how = 'any')
        ret_orig_y = ret_orig_y.values
        ret_versus_prob = np.append(ret_versus_prob, ret_orig_y, axis = 0)
        ret_ibov = ret_orig_y[:,0]

        ret_medio_ibov[i] = np.array(np.mean(ret_ibov))
        ret_medio_ibov_acum[i+1] = ret_medio_ibov_acum[i] * (1 + ret_medio_ibov[i])

        for j in range(len(k)):
            y = ret_orig_y[:,1]
            ind_long = hq.nlargest(k[j], range(len(y)), y.take)
            ind_short = hq.nsmallest(k[j], range(len(y)), y.take)

            ret_port_long = ret_orig_y[ind_long,0]
            ret_port_short = -ret_orig_y[ind_short,0]

            ret_medio_port[i,j] = np.array(np.mean(ret_port_long) + np.mean(ret_port_short))
            ret_medio_port_long[i,j] = np.array(np.mean(ret_port_long))
            ret_medio_port_short[i,j] = np.array(np.mean(ret_port_short))

            ret_medio_port_acum[i+1,j] = ret_medio_port_acum[i,j] * (1 + ret_medio_port[i,j])
            ret_medio_port_long_acum[i+1,j] = ret_medio_port_long_acum[i,j] * (1 + ret_medio_port_long[i,j])
            ret_medio_port_short_acum[i+1,j] = ret_medio_port_short_acum[i,j] * (1 + ret_medio_port_short[i,j])
        
    for j in range(len(k)):
        print("% retorno diário médio portfolio k =", k[j], "...:", np.mean(ret_medio_port[:,j], axis = 0) * 100)
    print("% retorno diário médio ibovespa...............:", np.mean(ret_medio_ibov, axis = 0)*100)
    show_ret(ret_medio_port,ret_medio_ibov)
    show_ret_acum(ret_medio_port_acum,ret_medio_ibov_acum)
    
    return ret_medio_ibov, ret_medio_ibov_acum, ret_medio_port_long, ret_medio_port_long_acum, ret_medio_port_short, ret_medio_port_short_acum, ret_medio_port, ret_medio_port_acum, ret_versus_prob




























