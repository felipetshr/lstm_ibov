import numpy as np

seed = 42
np.random.seed(seed)

import pandas as pd
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
import warnings
warnings.filterwarnings("ignore")





def lista_datas(base_total):
    datas = DataFrame(base_total['data'].drop_duplicates().values, columns = ['data'])
    datas = datas.set_index(['data'])
    datas = datas.sort_index(axis=0)
    limit_inf = '19990202 18:00:000'
    limit_sup = '20171230 18:00:000'
    datas = datas.loc[limit_inf:limit_sup]
    datas = datas.sort_index(axis=0)
    datas = datas.reset_index(['data'])
    return datas




def gera_graficos_e_sumario(ret_medio_ibov_sim,ret_medio_port_sim,ret_medio_port_long_sim,ret_medio_port_short_sim,corr_prob_ret_sim,datas_teste_sim,k):

    ini = 0
    fim = -1
    k_port = 0

    
    N = datas_teste_sim
#    N = np.arange(0, len(ret_medio_ibov_sim))
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
        plt.plot(N, ret_acum(ret_medio_port_sim.iloc[:,j].values), label=label_)
    plt.plot(N, ret_acum(ret_medio_ibov_sim.values), label="ibovespa")
    plt.title("LSTM Deep Learning: Portfolio Long-Short Ibovespa")
    plt.xlabel("Data")
    plt.ylabel("Retorno Acumulado")
    plt.legend(frameon=False)
    plt.show()
    
    
    
    N = datas_teste_sim
#    N = np.arange(0, len(ret_medio_ibov_sim))
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
        plt.plot(N, ret_acum(ret_medio_port_long_sim.iloc[:,j].values), label=label_)
    plt.plot(N, ret_acum(ret_medio_ibov_sim.values), label="ibovespa")
    plt.title("LSTM Deep Learning: Portfolio Long Ibovespa")
    plt.xlabel("Data")
    plt.ylabel("Retorno Acumulado")
    plt.legend(frameon=False)
    plt.show()
    
    N = datas_teste_sim
#    N = np.arange(0, len(ret_medio_ibov_sim))
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
        plt.plot(N, ret_acum(ret_medio_port_short_sim.iloc[:,j].values), label=label_)
    plt.plot(N, ret_acum(ret_medio_ibov_sim.values), label="ibovespa")
    plt.title("LSTM Deep Learning: Portfolio Short Ibovespa")
    plt.xlabel("Data")
    plt.ylabel("Retorno Acumulado")
    plt.legend(frameon=False)
    plt.show()
    
    
    # scatter plot prob vs. return
    x = corr_prob_ret_sim.iloc[:,0].values
    y = corr_prob_ret_sim.iloc[:,1].values
    plt.plot(x, y, "o")
    plt.xlabel("Retorno")
    plt.ylabel("Probabilidade")
    plt.show()
    
    # distribuição retornos e probabilidades
#    corr_prob_ret_sim.hist(bins = 100, grid=False, sharey = True)
    




# calcula o retorno acumulado a partir de uma série de retornos
def ret_acum(r):
    ret_acum = np.ones(len(r),)    
    for k in range(len(r)-1):
        ret_acum[k+1] = ret_acum[k] * (1 + r[k])
    return ret_acum 




def ler_base_componetes():
    nome = '00_componentes'
    path = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
    tipo = '.csv'
    file = path + nome + tipo
    compomentes = read_csv(file)
    compomentes = compomentes.set_index('codigo')
    print(compomentes.shape)
    return compomentes




def periodos_treino_teste(datas,inicio_treino,dias_treino,time_steps,dias_teste):
    dia_inicio_treino = pd.to_datetime(datas.iloc[inicio_treino].values)[0]
    dia_fim_treino = pd.to_datetime(datas.iloc[inicio_treino+dias_treino].values)[0]
    dia_inicio_teste = pd.to_datetime(datas.iloc[inicio_treino+dias_treino+1].values)[0]
    dia_inicio_teste_aux = pd.to_datetime(datas.iloc[inicio_treino+dias_treino-time_steps+1].values)[0]
    dia_fim_teste = pd.to_datetime(datas.iloc[inicio_treino+dias_treino+dias_teste].values)[0]
    
    print('Data início treino....: ',dia_inicio_treino)
    print('Data fim treino.......: ',dia_fim_treino)
    print('Data início teste aux.: ',dia_inicio_teste_aux)
    print('Data início teste.....: ',dia_inicio_teste)
    print('Data fim teste........: ',dia_fim_teste)
    
    month = "{0:02d}".format(dia_fim_treino.month)
    year =  "{0:04d}".format(dia_fim_treino.year)[2:]
    anomes = month + '/' + year
    papeis = compomentes.loc[:,anomes][compomentes.loc[:,anomes] > 0].index.values.tolist()
    return papeis,dia_inicio_treino,dia_fim_treino,dia_inicio_teste,dia_fim_teste,dia_inicio_teste_aux




def comprar_acoes(k, teste_retornos_total, papeis, datas_teste, teste_y_pred, qtde_treino):

    l0 = 0
    ln = qtde_treino
    q = ln                                                         # qtde ações na base
    dias_teste = int(teste_retornos_total.shape[0] / qtde_treino)  # qtde dias de teste
    
    ret_medio_ibov = np.zeros(shape=(dias_teste))
    ret_medio_ibov_acum = np.ones(shape=(dias_teste+1))

    ret_medio_port = np.zeros(shape=(dias_teste,len(k)))
    ret_medio_port_acum = np.ones(shape=(dias_teste+1,len(k)))    
    
    ret_medio_port_short = np.zeros(shape=(dias_teste,len(k)))
    ret_medio_port_short_acum = np.ones(shape=(dias_teste+1,len(k)))

    ret_medio_port_long = np.zeros(shape=(dias_teste,len(k)))
    ret_medio_port_long_acum = np.ones(shape=(dias_teste+1,len(k)))
    
    ret_versus_prob = np.zeros(shape=(1,2))
    
    for i in range(dias_teste):

        prob = teste_y_pred.iloc[l0:ln,1]      
        ret_orig = teste_retornos_total.iloc[l0:ln,:].values
        l0 += q
        ln += q
    
#        if i == 10:
#            x = np.arange(len(prob))
#            show_probs(x,prob,papeis)    
        
        probs = prob.values.reshape(prob.shape[0],1)
        ret_orig_y = DataFrame(np.concatenate((ret_orig, probs), axis = 1))
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
        if k[j] <= 9:
            print('Retorno Portfolio L k =', k[j], '..:{: 2.6f}'.format(np.mean(ret_medio_port_long[:,j], axis = 0) * 100))
        else:
            print('Retorno Portfolio L k =', k[j], '.:{: 2.6f}'.format(np.mean(ret_medio_port_long[:,j], axis = 0) * 100))
    print('Retorno Ibovespa............:{: 2.6f}'.format(np.mean(ret_medio_ibov, axis = 0) * 100))
    
    
    return ret_medio_ibov, ret_medio_ibov_acum, ret_medio_port_long, ret_medio_port_long_acum, ret_medio_port_short, ret_medio_port_short_acum, ret_medio_port, ret_medio_port_acum, ret_versus_prob









def prec_vs_rec(treino_y_pred,treino_y,teste_y_pred,teste_y):
    treino_y_pred = DataFrame(np.append(np.reshape(treino_y, (len(treino_y), 1)), np.reshape(treino_y_pred, (len(treino_y_pred), 1)), axis = 1), columns = ["A","B"])
    teste_y_pred = DataFrame(np.append(np.reshape(teste_y,(len(teste_y),1)), np.reshape(teste_y_pred,(len(teste_y_pred),1)), axis = 1), columns = ["A","B"])
    precisions_train, recalls_train, thresholds_train = precision_recall_curve(treino_y, treino_y_pred.iloc[:,1].values)
    precisions_trade, recalls_trade, thresholds_trade = precision_recall_curve(teste_y, teste_y_pred.iloc[:,1].values)
    plot_prec_rec_thr(precisions_train, recalls_train, thresholds_train,precisions_trade, recalls_trade, thresholds_trade)
    plt.show()







def histbonsmaus(treino_y_pred,treino_y,teste_y_pred,teste_y):
    treino_y_pred = DataFrame(np.append(np.reshape(treino_y, (len(treino_y), 1)), np.reshape(treino_y_pred, (len(treino_y_pred), 1)), axis = 1), columns = ["A","B"])
    treino_y_pred_uns = np.array(treino_y_pred.query('A > 0').iloc[:,1])
    treino_y_pred_zeros = np.array(treino_y_pred.query('A < 1').iloc[:,1])
    n, bins, patches = plt.hist(treino_y_pred_zeros, 50, density = False, facecolor='r', alpha=0.70, label = ['retorno = 0'])
    n, bins, patches = plt.hist(treino_y_pred_uns, 50, density = False, facecolor='g', alpha=0.80, label = ['retorno = 1'])
    plt.xlabel('Probabilidade')
    plt.ylabel('Frequência')
    plt.title('Histograma da Probabilidade de Retorno - Período de Treino')
    plt.grid(False)
    plt.legend()
    plt.show()

    teste_y_pred = DataFrame(np.append(np.reshape(teste_y,(len(teste_y),1)), np.reshape(teste_y_pred,(len(teste_y_pred),1)), axis = 1), columns = ["A","B"])
    teste_y_pred_uns = np.array(teste_y_pred.query('A > 0').iloc[:,1])
    teste_y_pred_zeros = np.array(teste_y_pred.query('A < 1').iloc[:,1])
    n, bins, patches = plt.hist(teste_y_pred_zeros, 50, density = False, facecolor='r', alpha=0.70, label = ['retorno = 0'])
    n, bins, patches = plt.hist(teste_y_pred_uns, 50, density = False, facecolor='g', alpha=0.80, label = ['retorno = 1'])
    plt.xlabel('Probabilidade')
    plt.ylabel('Frequência')
    plt.title('Histograma da Probabilidade de Retorno - Período de Teste')
    plt.grid(False)
    plt.legend()
    plt.show()






def loss_acc(H):
    N = np.arange(0, len(H["loss"]))
    MEDIUM_SIZE = 14
    BIGGER_SIZE = 18
    plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
    plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
    plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
    plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
    plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
    plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
    plt.figure()
    plt.rcParams['figure.figsize'] = (12, 8)
    plt.plot(N, H["loss"], label="train_loss")
    plt.plot(N, H["val_loss"], label="valid_loss")
    plt.plot(N, H["acc"], label="train_acc")
    plt.plot(N, H["val_acc"], label="valid_acc")
    plt.title("LSTM Model Performance")
    plt.xlabel("Epoch #")
    plt.ylabel("Loss / Accuracy")
    plt.legend(frameon=False)
    plt.show()






def show_performance_modelo(score_treino,score_teste,treino_y,teste_y):
    print('Loss Treino.................:{: 2.6f}'.format(score_treino[0]))
    print('Loss Teste..................:{: 2.6f}'.format(score_teste[0]))
    print('Gap de Generalização........:{: 2.6f}'.format(score_teste[0]-score_treino[0]))
    print('Acurácia Treino.............:{: 2.6f}'.format(score_treino[1]))
    print('Acurácia Teste..............:{: 2.6f}'.format(score_teste[1]))
    print('Taxa de Eventos Treino......:{: 2.6f}'.format(treino_y.mean()))
    print('Taxa de Eventos Teste.......:{: 2.6f}'.format(teste_y.mean()))
    
    


# confusion matrix
from sklearn.metrics import confusion_matrix

def thresh(treino_y_pred, teste_y_pred):
    thresholds = np.arange(0.5, 0.8, 0.05)
    for i, threshold in enumerate(thresholds):    
        print('threshold =',threshold)
        treino_y_pred['C'] = ((treino_y_pred['B'] > threshold)).astype(np.int)
        teste_y_pred['C'] = ((teste_y_pred['B'] > threshold)).astype(np.int)
        print('confusion matrix treino:\n',confusion_matrix(treino_y_pred.iloc[:,0], treino_y_pred.iloc[:,2]))
        print('confusion matrix teste:\n',confusion_matrix(teste_y_pred.iloc[:,0], teste_y_pred.iloc[:,2]))
        tn, fp, fn, tp = confusion_matrix(treino_y_pred.iloc[:,0], treino_y_pred.iloc[:,2]).ravel()
        print('acurácia no treino..:', (tp+tn)/(tp+tn+fp+fn))
        print('precision no treino.:', tp/(tp+fp))
        print('recall no treino....:', tp/(tp+fn))
        tn, fp, fn, tp = confusion_matrix(teste_y_pred.iloc[:,0], teste_y_pred.iloc[:,2]).ravel()
        print('acurácia no teste...:', (tp+tn)/(tp+tn+fp+fn))
        print('precision no teste..:',tp/(tp+fp))
        print('recall no teste.....:',tp/(tp+fn))
        print('-----------------------------')








def show_ret(ret_medio_port,ret_medio_ibov,k):

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









def show_ret_acum(ret_medio_port_acum,ret_medio_ibov_acum,k,datas_teste):
    # plot the acum return series of the trading period
    #N = np.arange(0, len(ret_medio_port_acum))
    N = datas_teste

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
        plt.plot(N, ret_medio_port_acum[1:,j], label = label_)
    plt.plot(N, ret_medio_ibov_acum[1:], label = "ibovespa")
    plt.title("Retorno Acumulado Período de Teste")
    plt.xlabel("Dia")
    plt.ylabel("Retorno Acumulado")
    plt.legend(frameon=False)






def show_probs(x,y,tks):

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





def plot_prec_rec_thr(precisions_train, recalls_train, thresholds_train, precisions_trade, recalls_trade, thresholds_trade):

    plt.plot(thresholds_train, precisions_train[:-1], "b--", label = "Precision Treino")
    plt.plot(thresholds_train, recalls_train[:-1], "g-", label = "Recall Treino")
    plt.xlabel("Threshold")
    plt.legend(loc = "center left")
    plt.ylim([0,1])

    plt.plot(thresholds_trade, precisions_trade[:-1], "r--", label = "Precision Teste")
    plt.plot(thresholds_trade, recalls_trade[:-1], "m-", label = "Recall Teste")
    plt.xlabel("Threshold")
    plt.legend(loc = "botton left")
    plt.ylabel("Precision / Recall")
    plt.ylim([0,1])
    

def extrai_papeis(base_total,papeis):
    base = base_total.loc[base_total['codigo'].isin(papeis)]
    #base = base.dropna(axis = 0, how = 'any')
    return base




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


