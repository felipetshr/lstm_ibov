import numpy as np
import pandas as pd
from keras import regularizers
from sklearn import preprocessing
from keras import backend as K
from pandas import read_csv, DataFrame
from keras.models import Sequential, Model
from keras.layers import Dense, Activation
from keras.layers import LSTM
from keras.optimizers import RMSprop, SGD, Adam
from keras.layers.normalization import BatchNormalization
import matplotlib.pyplot as plt
from keras.callbacks import  LearningRateScheduler
import heapq as hq
#import tensorflow as tf





def ler_base_fechamento(path, tipo, nome):
    file = path + nome + tipo
    fechamento = read_csv(file,sep = ';')
    fechamento = fechamento.set_index('codigo')
    datas = fechamento.columns.values.tolist()
    tikers = fechamento.index.values.tolist()
    values = fechamento.values
    fechamento = DataFrame(values, index=tikers, columns=datas, dtype = 'float64')
    print(fechamento.shape)
    return datas, fechamento

#def ler_base_lpa(path, tipo, nome):
#    file = path + nome + tipo
#    lpa = read_csv(file,sep = ';')
#    lpa = fechamento.set_index('codigo')






# extração da base de dados para o perído de estudo 
# correspondente ao dia t >= 750 do histórico de preços
# O dia t refere-se ao último dia do período de treino, conforme (Krauss, 2018).
# Considera-se, então, 750 pra trás e 250 dias pra frente. Retorna o valor 
# fechamento e retorno das ações constituintes do Ibovespa no dia determinado
# período de estudo = período de treino + período de trade
#                     [t-750  ...    t] + [t+1  ... t+250]
    
def extrair_features(d, data_t, datas, compomentes, fechamento, usar_retornos, usar_fechamento,qtde_treino,qtde_trade):
    
    if data_t == None:
        dia = d
#        dia = 4750
    else:
        dia = datas.index(data_t)
    end = None
    mf = datas[dia][3:end] # mes final do periodo de treino
    tikers_periodo = compomentes.loc[:,mf][compomentes.loc[:,mf] > 0].index.values.tolist()
    fechamentos_periodo = fechamento.loc[tikers_periodo,:]
    fechamentos_periodo = fechamentos_periodo.iloc[:,dia-qtde_treino:dia+qtde_trade]
    fechamentos_periodo = fechamentos_periodo.dropna(axis = 0, how = 'all')
    tks = fechamentos_periodo.index.values
    fechamentos_periodo = fechamentos_periodo.fillna(0)
    
    retornos_periodo = (fechamentos_periodo / fechamentos_periodo.shift(1, axis=1)) - 1
    retornos_periodo_original = retornos_periodo
    retornos_periodo = retornos_periodo.fillna(0) 
    
    log_retornos_periodo = np.log(fechamentos_periodo) - np.log(fechamentos_periodo.shift(1, axis=1))
    log_retornos_periodo = log_retornos_periodo.fillna(0)
    
    print("Início período de treino...:", retornos_periodo.columns.values.tolist()[1])
    print("Final período de treino....:", retornos_periodo.columns.values.tolist()[qtde_treino])
    print("Final período de trade.....:", retornos_periodo.columns.values.tolist()[-1])
    
    # normaliza os retornos do período de estudo com base na
    # média e desvio padrão de todos os retornos do período de treino
#    mu = np.mean(retornos_periodo.values[:,:qtde_treino], axis=(0,1), dtype=np.float64)
#    sigma = np.std(retornos_periodo.values[:,:qtde_treino], axis=(0,1), dtype=np.float64)
#    retornos_periodo_norm = (retornos_periodo - mu) / sigma
    
    log_retornos_periodo_t = log_retornos_periodo.transpose()
    max_abs_scaler_ret = preprocessing.MaxAbsScaler()
    max_abs_scaler_ret.fit(log_retornos_periodo_t.iloc[:qtde_treino,:])
    retornos_periodo_std_t = max_abs_scaler_ret.transform(log_retornos_periodo_t)
    retornos_periodo_std = retornos_periodo_std_t.transpose()
    retornos_periodo_std = DataFrame(retornos_periodo_std)
        
    
    # scalling os valores de fechamento do período de estudo com base
    # no mín e máx dos retornos do período de treino
#    min_ = np.min(fechamentos_periodo.values[:,:qtde_treino], axis=(0,1))
#    max_ = np.max(fechamentos_periodo.values[:,:qtde_treino], axis=(0,1))
#    fechamentos_periodo_scaled = (fechamentos_periodo - min_) / (max_ - min_)
    log_fechamentos_periodo_t = np.log(fechamentos_periodo.transpose())
    max_abs_scaler_fec = preprocessing.MaxAbsScaler()
    max_abs_scaler_fec.fit(log_fechamentos_periodo_t.iloc[:qtde_treino,:])
    log_fechamentos_periodo_std_t = max_abs_scaler_fec.transform(log_fechamentos_periodo_t)
    log_fechamentos_periodo_std = log_fechamentos_periodo_std_t.transpose()
    log_fechamentos_periodo_std = DataFrame(log_fechamentos_periodo_std)
    
    
    
    # Agora temos:
    #
    #   1 - retornos_periodo             ----> calcular Y, retornos da k-carteira e do ibovespa
    #   2 - retornos_periodo_std         ----> feature 1
    #   3 - fechamentos_periodo_std      ----> feature 2
    #
    
    x_values_retornos = retornos_periodo_std.values.reshape((retornos_periodo_std.values.shape[0],
                                                             retornos_periodo_std.values.shape[1],
                                                             1))

    x_values_fechamentos = log_fechamentos_periodo_std.values.reshape((log_fechamentos_periodo_std.values.shape[0],
                                                                      log_fechamentos_periodo_std.values.shape[1],
                                                                      1))

    if usar_retornos == 1:
        x_values = x_values_retornos

    if usar_fechamento == 1:
        x_values = np.concatenate((x_values, x_values_fechamentos), axis = 2)

    y_values = retornos_periodo.values

    print(x_values.shape, y_values.shape)
    
    return x_values, y_values, retornos_periodo_original, tks











def base_x_y(x_values, y_values, definicao_y, r_wish, timesteps,qtde_treino,qtde_trade):
    
    # 1: y = 1 se retorno >= mediana do dia
    # 2: y = 1 se retorno >= r_wish
    
    sequencias = x_values.shape[1] - timesteps
    tn = timesteps
    t0 = 0
    for i in range(0,sequencias):
        
        x_aux = x_values[:,t0:tn,:]  
        y_aux = y_values[:,tn]    
        y_med = np.median(y_aux, axis = 0)
        
        if definicao_y == 1:
            y_aux = (y_aux >= y_med).astype(np.int)
        else:
            y_aux = (y_aux >= r_wish).astype(np.int)
        
        if i == 0:
            x_treino = x_aux
            y_treino = y_aux        
        elif i < qtde_treino-qtde_trade:
            x_treino = np.append(x_treino,x_aux, axis = 0)
            y_treino = np.append(y_treino,y_aux, axis = 0)    
        elif i == qtde_treino-qtde_trade:
            x_trade = x_aux
            y_trade = y_aux
            ind_y_trade = tn
        else:
            x_trade = np.append(x_trade,x_aux, axis = 0)
            y_trade = np.append(y_trade,y_aux, axis = 0)
            ind_y_trade = np.append(ind_y_trade, tn)
        
        t0+=1
        tn+=1
        
    shape1 = x_treino.shape[1]
    shape2 = x_treino.shape[2]
    
    print("-----------------------------------")
    print("Shape x no treino...:", x_treino.shape)
    print("Shape y no treino...:", y_treino.shape)
    print("-----------------------------------")
    print("Shape x no trade....:", x_trade.shape)
    print("Shape y no trade....:", y_trade.shape)
    print("-----------------------------------")
    print("Taxa de evento no treino...:", np.mean(y_treino))
    print("Taxa de evento no trade....:", np.mean(y_trade))
    print("-----------------------------------")

    return x_treino, y_treino, x_trade, y_trade, ind_y_trade, shape1, shape2








# definine the total number of epochs to train for along with the
# initial learning rate
def poly_decay(epoch):
	# initialize the maximum number of epochs, base learning rate,
	# and power of the polynomial
	maxEpochs = num_epochs
	baseLR = init_lr
	power = 3.0
 
	# compute the new learning rate based on polynomial decay
	alpha = baseLR * (1 - (epoch / float(maxEpochs))) ** power
 
	# return the new learning rate
	return alpha











# metrics functions 
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################
def as_keras_metric(method):
    import functools
    from keras import backend as K
    import tensorflow as tf
    @functools.wraps(method)
    def wrapper(self, args, **kwargs):
        """ Wrapper for turning tensorflow metrics into keras metrics """
        value, update_op = method(self, args, **kwargs)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([update_op]):
            value = tf.identity(value)
        return value
    return wrapper


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
###############################################################################
###############################################################################
###############################################################################
###############################################################################
###############################################################################



def lstm_model_3(num_epochs,batch_size,lstm_neurons,dense_neurons,shape1,shape2,optmiz) -> Model:
    
#    precision = as_keras_metric(tf.metrics.precision)
#    recall = as_keras_metric(tf.metrics.recall)
    
    # design network
    model = Sequential()
    model.add(LSTM(lstm_neurons, 
                   input_shape = (shape1, shape2),
                   dropout = 0.1,
                   recurrent_dropout = 0.1))
    model.add(BatchNormalization())
    model.add(Dense(dense_neurons))    
    model.add(Activation('sigmoid'))
    if optmiz == 1:
        opt = RMSprop(lr = 0.001, rho = 0.9, epsilon = None, decay = 0.0)
    elif optmiz == 2:
        opt = SGD(lr = 0.1, momentum = 0.9, decay = 0.05, nesterov = True)
    model.compile(loss='binary_crossentropy',
                  optimizer = opt, 
                  metrics = ['accuracy', 
                             precision,
                             recall,
                             fmeasure])
    return model
















def lstm_model_report(x_treino,y_treino,x_trade,y_trade,batch_size):
    
    # plot the training loss and accuracy
    H = history.history
    N = np.arange(0, len(H["loss"]))
    
    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title

  
    plt.figure()
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.plot(N, H["loss"], label="train_loss")
    plt.plot(N, H["val_loss"], label="valid_loss")
    plt.plot(N, H["acc"], label="train_acc")
    plt.plot(N, H["val_acc"], label="valid_acc")
    plt.title("LSTM on IBOV prediction task")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss / Accuracy")
    plt.legend(frameon=False)
    plt.show()
    
    
    plt.figure()
    plt.rcParams['figure.figsize'] = (16, 12)
    plt.plot(N, H["precision"], label="train_precision")
    plt.plot(N, H["val_precision"], label="valid_precision")
    plt.plot(N, H["recall"], label="train_recall")
    plt.plot(N, H["val_recall"], label="valid_recall")
    plt.title("LSTM on IBOV prediction task")
    plt.xlabel("Epoch #")
    plt.ylabel("precision / recall")
    plt.legend(frameon=False)
    plt.show()
    
    
    # evaluate performance on the trade period
    score_treino = model.evaluate(x_treino, y_treino, batch_size = batch_size,verbose = 1)
    score_trade  = model.evaluate(x_trade, y_trade, batch_size = batch_size, verbose = 1)
    print('Loss Treino.........:', score_treino[0])
    print('Loss Trade..........:', score_trade[0])
    print('Accuracy Treino.....:', score_treino[1])
    print('Accuracy Trade......:', score_trade[1])
    print('Precision Treino.....:', score_treino[2])
    print('Precision Trade......:', score_trade[2])
    print('Recall Treino.....:', score_treino[3])
    print('Recall Trade......:', score_trade[3])
    
    
    y_pred = model.predict(x_treino,batch_size=batch_size, verbose=1)

    from sklearn.metrics import precision_recall_curve
    
    precisions, recalls, thresholds = precision_recall_curve(y_treino,y_pred)
    
    def plot_precision_recall_vs_threshold(precisions, recalls, thresholds):
        plt.plot(thresholds, precisions[:-1], "b--", label = "Precision")
        plt.plot(thresholds, recalls[:-1], "g-", label = "Recall")
        plt.xlabel("Threshold")
        plt.legend(loc = "center left")
        plt.ylim([0,1])
        
    plt.figure()
    plot_precision_recall_vs_threshold(precisions, recalls, thresholds)
    plt.show()
        
    
    
    return score_treino, score_trade










def show_probs(x,y):

    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22
    
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
    
    plt.rcParams['figure.figsize'] = (12, 8)
    fig, ax = plt.subplots()
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.bar(x, y)
    plt.xticks(x, tks)
    plt.title("Probabilidade de Alta das Ações do IBOVESPA")
    plt.xlabel("Ações")
    plt.ylabel("Probabilidade")
    plt.xticks(rotation=90)
    plt.show()








def show_ret(ret_medio_port,ret_medio_ibov):

    # plot the return series of the trading period
    N = np.arange(0, len(ret_medio_port))

    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
    plt.figure()
    plt.rcParams['figure.figsize'] = (12, 8)
    for j in range(len(k)):
        label_ = "portfolio k =" + str(k[j])
        plt.plot(N, ret_medio_port[:,j], label=label_)
    plt.plot(N, ret_medio_ibov, label="ibovespa")
    plt.title("Retorno Diário")
    plt.xlabel("Dia")
    plt.ylabel("Retorno")
    plt.legend(frameon=False)









def show_ret_acum(ret_medio_port_acum,ret_medio_ibov_acum):
    # plot the acum return series of the trading period
    N = np.arange(0, len(ret_medio_port_acum))

    MEDIUM_SIZE = 16
    BIGGER_SIZE = 22
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
    plt.figure()
    plt.rcParams['figure.figsize'] = (12, 8)
    for j in range(len(k)):
        label_ = "portfolio k =" + str(k[j])
        plt.plot(N, ret_medio_port_acum[:,j], label=label_)
    plt.plot(N, ret_medio_ibov_acum, label="ibovespa")
    plt.title("Retorno Acumulado em 250 dias")
    plt.xlabel("Dia")
    plt.ylabel("Retorno Acumulado")
    plt.legend(frameon=False)











# predictions to the trade period
# para cada dia t no periodo de trade, faça:
#    - calcular a probabilidade de retorno >= mediana ou r_wish
#    - excluir as ações sem operação
#    - selecionar as k ações com maior probabilidade
#    - calcular o retorno médio das k ações no dia t+1
def predicao(k, x_values, x_trade, ind_y_trade, model, tks, retornos_periodo_original):

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










#################################################
#################################################
#################################################
#################################################
#################################################
#################################################
#################################################








#retornos_periodo_flat_dropedna.describe()
#count  506742.000000
#mean        0.000953
#std         0.031902
#min        -0.872197
#25%        -0.013072
#50%         0.000000
#75%         0.013699
#80%         0.018328
#85%         0.024096
#90%         0.032268
#95%         0.046988
#99%         0.089670
#max         1.502778



# inicialização
path             = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
tipo             = '.csv'
data_t           = None
qtde_treino      = 500
qtde_trade       = 100

#features
usar_retornos    = 1
usar_fechamento  = 0
#usar_volume      = 1
#usar_PL          = 1
#usar_dolar       = 1
#usar_juros       = 1

#target
definicao_y      = 1 
r_wish           = 0.032268
class_weight = {0: 1.0,
                1: 1.0}

# LSTM
timesteps        = 22
num_epochs       = 100
pacience         = 10
batch_size       = 512
lstm_neurons     = 200
dense_neurons    = 1
optmiz           = 2
init_lr          = 0.5
pw               = 3
val_split        = 0.2
salvar           = 1

# simulação
k                = [1,3,5,7,10,15,20]
D                = [3500]
#D                = [3000,3250,3500,3750,4000,4250,4500,4750,5000,5215]
#D                = [750,1000,1250,1500,1750,2000,2250,2500,2750,3000,3250,3500,3750,4000,4250,4500,4750,5000,5215]



score_treino_sim = np.zeros(shape=(len(D),5))
score_trade_sim = np.zeros(shape=(len(D),5))

# carrega a base de componentes do IBOVESPA
nome = '00_componentes'
compomentes = ler_base_componetes(path, tipo, nome)

# carrega todo o histórico de preços de fechamento de todas as ações disponíveis do IBOVESPA
nome = '01_fechamento'
datas, fechamento = ler_base_fechamento(path, tipo, nome)

#Para cada dia d:
l = 0
d = 3500
for d in D:
    
    print("Simulação", l, "referente ao dia", d)
    
    # extrai as features do período de estudo k
    x_values, y_values, retornos_periodo_original, tks = extrair_features(d, data_t, datas, compomentes, fechamento, usar_retornos, usar_fechamento,qtde_treino,qtde_trade)
    
    # define os períodos de treino e trade dentro do período de estudo k
    x_treino, y_treino, x_trade, y_trade, ind_y_trade, shape1, shape2 = base_x_y(x_values, y_values, definicao_y, r_wish, timesteps,qtde_treino,qtde_trade)
    


    # treina o modelo LSTM Deep Learning no período de treino
#    early_stop = EarlyStopping(monitor = 'val_acc', patience = pacience, mode = 'max') 
#    model = lstm_model_3(num_epochs,
#                         batch_size,
#                         lstm_neurons,
#                         dense_neurons,
#                         shape1,
#                         shape2,
#                         optmiz)
    
    # design network
    model = Sequential()
    model.add(LSTM(lstm_neurons,
                   input_shape = (shape1, shape2), 
                   activation='relu', 
                   recurrent_activation='linear', 
                   use_bias=True, 
                   kernel_initializer='glorot_uniform',
                   recurrent_initializer='orthogonal',
                   bias_initializer='zeros',
                   unit_forget_bias=True,
                   kernel_regularizer=regularizers.l2(0.01),
                   recurrent_regularizer=regularizers.l2(0.01), 
                   bias_regularizer=regularizers.l2(0.01), 
                   activity_regularizer=regularizers.l2(0.01),
                   kernel_constraint=None,
                   recurrent_constraint=None,
                   bias_constraint=None,
                   dropout = 0.2,
                   recurrent_dropout = 0.2,
                   implementation=1,
                   return_sequences=False,
                   return_state=False, 
                   go_backwards=False,
                   stateful=False, 
                   unroll=False))
    
    
#    model.add(BatchNormalization())
    model.add(Dense(dense_neurons))    
    model.add(Activation('sigmoid'))
#    opt = SGD(lr = init_lr, momentum = 0.9)
#    opt = SGD(lr = 0.99, momentum = 0.9, nesterov = True)
    opt = Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False)
    model.compile(loss='binary_crossentropy',
                  optimizer = opt, 
                  metrics = ['accuracy', precision, recall, fmeasure])

#    callbacks = [LearningRateScheduler(poly_decay)]
   
    model.summary()
    history = model.fit(x_treino, 
                        y_treino, 
                        validation_split = val_split, 
                        epochs = num_epochs, 
                        batch_size = batch_size, 
                        verbose = 1, 
                        shuffle = True,
                        class_weight = class_weight)
    
#                        callbacks = callbacks)

    score_treino, score_trade = lstm_model_report(x_treino,y_treino,x_trade,y_trade,batch_size)

    score_treino_sim[l,0] = score_treino[0]
    score_treino_sim[l,1] = score_treino[1]
    score_treino_sim[l,2] = score_treino[2]
    score_treino_sim[l,3] = score_treino[3]
    score_treino_sim[l,4] = score_treino[4]

    score_trade_sim[l,0]  = score_trade[0]
    score_trade_sim[l,1]  = score_trade[1]
    score_trade_sim[l,2]  = score_trade[2]
    score_trade_sim[l,3]  = score_trade[3]
    score_trade_sim[l,4]  = score_trade[4]

    # aplica o modelo no período de trade e calcula os resultados
    ret_medio_ibov, ret_medio_ibov_acum, ret_medio_port_long, ret_medio_port_long_acum, ret_medio_port_short, ret_medio_port_short_acum, ret_medio_port, ret_medio_port_acum, ret_versus_prob = predicao(k, x_values, x_trade, ind_y_trade, model, tks, retornos_periodo_original)
    if l == 0:
        ret_medio_ibov_sim = ret_medio_ibov
        ret_medio_port_sim = ret_medio_port
        ret_medio_port_long_sim = ret_medio_port_long
        ret_medio_port_short_sim = ret_medio_port_short
        corr_prob_ret_sim = ret_versus_prob
    else:
        ret_medio_ibov_sim = np.append(ret_medio_ibov_sim, ret_medio_ibov, axis = 0)
        ret_medio_port_sim = np.append(ret_medio_port_sim, ret_medio_port, axis = 0)
        ret_medio_port_long_sim = np.append(ret_medio_port_long_sim, ret_medio_port_long, axis = 0)
        ret_medio_port_short_sim = np.append(ret_medio_port_short_sim, ret_medio_port_short, axis = 0)
        corr_prob_ret_sim =  np.append(corr_prob_ret_sim, ret_versus_prob, axis = 0)
    l+=1




# resultados consolidados
score_treino_sim         = DataFrame(score_treino_sim)
score_trade_sim          = DataFrame(score_trade_sim)
ret_medio_ibov_sim       = DataFrame(ret_medio_ibov_sim)
ret_medio_port_sim       = DataFrame(ret_medio_port_sim)
ret_medio_port_long_sim  = DataFrame(ret_medio_port_long_sim)
ret_medio_port_short_sim = DataFrame(ret_medio_port_short_sim)
corr_prob_ret_sim        = DataFrame(corr_prob_ret_sim)






# plot the acum return series of the trading period
N = np.arange(0, len(ret_medio_ibov_sim))
MEDIUM_SIZE = 16
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
plt.figure()
plt.rcParams['figure.figsize'] = (12, 8)
for j in range(4,len(k)):
    label_ = "portfolio k = " + str(k[j])
    plt.plot(N, ret_acum(ret_medio_port_sim.iloc[:,j].values), label=label_)
plt.plot(N, ret_acum(ret_medio_ibov_sim.values), label="ibovespa")
plt.title("LSTM Deep Learning: Portfolio Long-Short Ibovespa")
plt.xlabel("Dia")
plt.ylabel("Retorno Acumulado")
plt.legend(frameon=False)






ini = 0
fim = -1
k_port = 0
# plot the acum return series of the trading periodo
N = np.arange(0, len(ret_medio_ibov_sim.iloc[ini:fim]))
MEDIUM_SIZE = 16
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
plt.figure()
plt.rcParams['figure.figsize'] = (12, 8)
for j in range(k_port,len(k)):
    label_ = "portfolio k = " + str(k[j])
    plt.plot(N, ret_acum(ret_medio_port_sim.iloc[ini:fim,j].values), label=label_)
plt.plot(N, ret_acum(ret_medio_ibov_sim.iloc[ini:fim].values), label="ibovespa")
plt.title("LSTM Deep Learning: Portfolio Long-Short Ibovespa")
plt.xlabel("Dia")
plt.ylabel("Retorno Acumulado")
plt.legend(frameon=False)





# plot the acum return series of the trading periodo
N = np.arange(0, len(ret_medio_ibov_sim.iloc[ini:fim]))
MEDIUM_SIZE = 16
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
plt.figure()
plt.rcParams['figure.figsize'] = (12, 8)
for j in range(k_port,len(k)):
    label_ = "portfolio k = " + str(k[j])
    plt.plot(N, ret_acum(ret_medio_port_long_sim.iloc[ini:fim,j].values), label=label_)
plt.plot(N, ret_acum(ret_medio_ibov_sim.iloc[ini:fim].values), label="ibovespa")
plt.title("LSTM Deep Learning: Portfolio Long Ibovespa")
plt.xlabel("Dia")
plt.ylabel("Retorno Acumulado")
plt.legend(frameon=False)






# plot the acum return series of the trading periodo
N = np.arange(0, len(ret_medio_ibov_sim.iloc[ini:fim]))
MEDIUM_SIZE = 16
BIGGER_SIZE = 22
plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
plt.figure()
plt.rcParams['figure.figsize'] = (12, 8)
for j in range(k_port,len(k)):
    label_ = "portfolio k = " + str(k[j])
    plt.plot(N, ret_acum(ret_medio_port_short_sim.iloc[ini:fim,j].values), label=label_)
plt.plot(N, ret_acum(ret_medio_ibov_sim.iloc[ini:fim].values), label="ibovespa")
plt.title("LSTM Deep Learning: Portfolio Short Ibovespa")
plt.xlabel("Dia")
plt.ylabel("Retorno Acumulado")
plt.legend(frameon=False)


# scatter plot prob vs. return
line = plt.figure()
x = corr_prob_ret_sim.iloc[:,0].values
y = corr_prob_ret_sim.iloc[:,1].values
plt.plot(x, y, "o")



# distribuição retornos e probabilidades
corr_prob_ret_sim.hist(bins = 100, grid=False, sharey = True)



#pd.set_option('display.width', 100)
#pd.set_option('precision', 6)
#correlations = ret_medio_port_sim.corr(method='pearson')
#correlations2 = corr_prob_ret_sim.corr(method='pearson')
#print(correlations)
#print(correlations2)





# statistics
model.summary()
score_treino_sim.describe()
score_trade_sim.describe()
ret_medio_ibov_sim.iloc[ini:fim].describe()
ret_medio_port_sim.iloc[ini:fim,:].describe()
ret_medio_port_long_sim.iloc[ini:fim,:].describe()
ret_medio_port_short_sim.iloc[ini:fim,:].describe()
corr_prob_ret_sim.describe()











#fechamentos_periodo = fechamento
#retornos_periodo = (fechamentos_periodo / fechamentos_periodo.shift(1, axis=1)) - 1
#retornos_periodo.shape
#retornos_periodo = retornos_periodo.values
#retornos_periodo_flat = retornos_periodo.flatten()
#retornos_periodo_flat.shape
#retornos_periodo_flat = DataFrame(retornos_periodo_flat)
#retornos_periodo_flat_dropedna = retornos_periodo_flat.dropna(axis = 0, how = 'all')
#retornos_periodo_flat_dropedna.describe()
#print(np.percentile(retornos_periodo_flat_dropedna,80))
#print(np.percentile(retornos_periodo_flat_dropedna,85))
#print(np.percentile(retornos_periodo_flat_dropedna,90))
#print(np.percentile(retornos_periodo_flat_dropedna,95))
#print(np.percentile(retornos_periodo_flat_dropedna,99))
