import numpy as np

seed = 42
np.random.seed(seed)

import pandas as pd
from pandas import read_csv
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler
from keras.callbacks import ModelCheckpoint
from keras import backend as K
from keras.models import Sequential
from keras.layers import Dense, Activation, LSTM
from sklearn.metrics import precision_recall_curve
from keras import regularizers
import heapq as hq
import warnings
import time
warnings.filterwarnings("ignore")



def simulacao(n_epoch, n_neurons, sim, ret_target, dias_treino, dias_teste, time_steps, D, k, datas, base_total, features, cols, modelfilepath, compomentes, p):

    start_time = time.time()

    # medidas de performance modelo
    performance_modelo = np.zeros(shape=(len(D),6))
    
    sim_ok = 0

    for d in D:
#        d = 1500
        print("\n\n*************************************")
        print("Simulação", sim+1, "/", len(D))
        print("*************************************")


        # define dataset
        inicio_treino = d
#        papeis,dia_inicio_treino,dia_fim_treino,dia_inicio_teste,dia_fim_teste,dia_inicio_teste_aux = periodos_treino_teste(datas,inicio_treino,dias_treino,time_steps,dias_teste)
#        base_in = base_total.loc[base_total['codigo'].isin(papeis)]

        
        # DataPrep da base para modelagem
        dataprepreturn, treino_x, treino_y, teste_x, teste_y, qtde_treino, treino_retornos_total, teste_retornos_total, treino_datas_total, teste_datas_total, qtde_acoes, papeis = dataprep_for_batches(datas, inicio_treino, dias_treino, time_steps, dias_teste, features, cols, ret_target, compomentes, base_total)
        
        
        
        if dataprepreturn == False:
            sim += 1
            continue
        sim_ok += 1


        # design da rede LSTM # 000
#        n_batch = 128
#        model = Sequential()
#        model.reset_states()
#        model.add(LSTM(n_neurons, stateful = False, return_sequences = False, input_shape = (treino_x.shape[1], treino_x.shape[2]), recurrent_dropout = 0.1, dropout = 0.1))
#        model.add(Dense(1))
#        model.add(Activation('sigmoid'))
#        model.compile(loss='binary_crossentropy', optimizer='RMSprop', metrics = ['accuracy'])
#        checkpoint = ModelCheckpoint(modelfilepath, monitor = 'val_acc', verbose = 0, save_best_only = True, mode = 'max')
#        callbacks_list = [checkpoint]
#        print(model.summary())
#        print('\nFitting model...')
#        history = model.fit(treino_x, treino_y, validation_split = 0.2, epochs = n_epoch, batch_size = n_batch, callbacks = callbacks_list, verbose = 0, shuffle = False)
      
        
        # design da rede LSTM # 001       
#        fracao_treino = int(np.floor(p * (treino_x.shape[0] / qtde_acoes)) * qtde_acoes)        
#        treino_x_tre = treino_x[:fracao_treino,:,:]
#        treino_y_tre = treino_y[:fracao_treino,:]
#        treino_x_val = treino_x[fracao_treino:,:,:] 
#        treino_y_val = treino_y[fracao_treino:,:] 
##        
#        n_batch = qtde_acoes
#        model = Sequential()
#        model.add(LSTM(n_neurons, stateful = True, return_sequences = False, batch_input_shape = (n_batch, treino_x.shape[1], treino_x.shape[2]), recurrent_dropout = 0.1, dropout = 0.1))
#        model.add(Dense(1))
#        model.add(Activation('sigmoid'))
#        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics = ['accuracy'])
#        checkpoint = ModelCheckpoint(modelfilepath, monitor = 'val_acc', verbose = 0, save_best_only = True, mode = 'max')
#        callbacks_list = [checkpoint]
#        print(model.summary())
#        print('\nTreinando o modelo com', n_epoch, 'epochs...')
#        history = model.fit(treino_x_tre, treino_y_tre, validation_data = (treino_x_val, treino_y_val), epochs = n_epoch, batch_size = n_batch, callbacks = callbacks_list, verbose = 0, shuffle = False)



        # design da rede LSTM # 002       
        fracao_treino = int(np.floor(p * (treino_x.shape[0] / qtde_acoes)) * qtde_acoes)        
        treino_x_tre = treino_x[:fracao_treino,:,:]
        treino_y_tre = treino_y[:fracao_treino,:]
        treino_x_val = treino_x[fracao_treino:,:,:] 
        treino_y_val = treino_y[fracao_treino:,:] 
#        
        n_batch = qtde_acoes
        model = Sequential()
        model.add(LSTM(n_neurons, stateful = False, return_sequences = False, batch_input_shape = (n_batch, treino_x.shape[1], treino_x.shape[2]), recurrent_dropout = 0.1, dropout = 0.1))
        model.add(Dense(1))
        model.add(Activation('sigmoid'))
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics = ['accuracy'])
        checkpoint = ModelCheckpoint(modelfilepath, monitor = 'val_acc', verbose = 0, save_best_only = True, mode = 'max')
        callbacks_list = [checkpoint]
        print(model.summary())
        print('\nTreinando o modelo com', n_epoch, 'epochs...')
        history = model.fit(treino_x_tre, treino_y_tre, validation_data = (treino_x_val, treino_y_val), epochs = n_epoch, batch_size = n_batch, callbacks = callbacks_list, verbose = 0, shuffle = False)

        
        # design da rede LSTM # 002
#        n_batch = qtde_acoes
#        model = Sequential()
#        model.add(LSTM(n_neurons, stateful = True, return_sequences = False, batch_input_shape = (n_batch, treino_x.shape[1], treino_x.shape[2])))
#        model.add(Dense(1))
#        model.add(Activation('sigmoid'))
#        model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        
        
        # design da rede LSTM # 003
    #    n_batch = qtde_acoes    
    #    model = Sequential()
    #    model.add(LSTM(n_neurons, stateful = True, return_sequences = True, batch_input_shape = (n_batch, treino_x.shape[1], treino_x.shape[2])))
    #    model.add(LSTM(n_neurons, stateful = True, return_sequences = True))
    #    model.add(LSTM(n_neurons, stateful = True))
    #    model.add(Dense(1))
    #    model.add(Activation('sigmoid'))
    #    model.compile(loss='binary_crossentropy', optimizer='adam', metrics = ['accuracy'])
        
        # treino do modelo    
        #history = model.fit(treino_x, treino_y, validation_data=(teste_x, teste_y), epochs = n_epoch, batch_size = n_batch, callbacks = callbacks_list, verbose = 0, shuffle = False)
        
        
                
        # performance do modelo no treino e no teste
        # carrega e compila os pesos do melhor modelo
        model.load_weights(modelfilepath)
        model.compile(loss='binary_crossentropy', optimizer='Adam', metrics = ['accuracy'])
        score_treino  = model.evaluate(treino_x, treino_y, batch_size = n_batch, verbose = 0)
        score_teste   = model.evaluate(teste_x, teste_y, batch_size = n_batch, verbose = 0)
        treino_y_pred = np.array(model.predict(treino_x, batch_size = n_batch, verbose = 0))
        teste_y_pred  = np.array(model.predict(teste_x, batch_size = n_batch, verbose = 0))            
        loss_acc(history.history)
        histbonsmaus(treino_y_pred,treino_y,teste_y_pred,teste_y)    
        prec_vs_rec(treino_y_pred,treino_y,teste_y_pred,teste_y)
        teste_y_pred = DataFrame(np.append(np.reshape(teste_y,(len(teste_y),1)), np.reshape(teste_y_pred,(len(teste_y_pred),1)), axis = 1), columns = ["A","B"])

        # Estratégia de compra de ações referentes ao perído treino-teste:
        #   Para cada dia no período de teste, toma-se a rentabilidade das 
        #   k = {1,3,5,7,10,15,20} ações com maior probabilidade de retorno
        #   positivo. Acumula-se o retorno diário das carteiras com k-ações.
        show_performance_modelo(score_treino,score_teste,treino_y,teste_y)

#        print(' ************ Dimensão das bases de datas, ações e probabilidades ***********')
#        print('# probs:',teste_y_pred.shape[0], '# datas:', teste_datas_total.shape[0], '# ações:', qtde_acoes, '# datas x # ações:', teste_datas_total.shape[0] * qtde_acoes)
            
        ret_medio_ibov, ret_medio_ibov_acum, ret_medio_port_long, ret_medio_port_long_acum, ret_medio_port_short, ret_medio_port_short_acum, ret_medio_port, ret_medio_port_acum, ret_versus_prob = comprar_acoes(k, teste_retornos_total, papeis, teste_datas_total, teste_y_pred, qtde_treino)
        
        # dados da simulação
        performance_modelo[sim,0] = score_treino[0]
        performance_modelo[sim,1] = score_teste[0]
        performance_modelo[sim,2] = score_treino[1]
        performance_modelo[sim,3] = score_teste[1]
        performance_modelo[sim,4] = treino_y.mean()
        performance_modelo[sim,5] = teste_y.mean()
    
        if sim_ok == 1:
            datas_teste_sim = teste_datas_total
            ret_medio_ibov_sim = ret_medio_ibov
            ret_medio_port_sim = ret_medio_port
            ret_medio_port_long_sim = ret_medio_port_long
            ret_medio_port_short_sim = ret_medio_port_short
            corr_prob_ret_sim = ret_versus_prob
        else:
            datas_teste_sim = np.append(datas_teste_sim, teste_datas_total, axis = 0)
            ret_medio_ibov_sim = np.append(ret_medio_ibov_sim, ret_medio_ibov, axis = 0)
            ret_medio_port_sim = np.append(ret_medio_port_sim, ret_medio_port, axis = 0)
            ret_medio_port_long_sim = np.append(ret_medio_port_long_sim, ret_medio_port_long, axis = 0)
            ret_medio_port_short_sim = np.append(ret_medio_port_short_sim, ret_medio_port_short, axis = 0)
            corr_prob_ret_sim =  np.append(corr_prob_ret_sim, ret_versus_prob, axis = 0)
        sim += 1
        elapsed_time = time.time() - start_time
        print('Tempo de simulação..........:{: 2.6f}'.format(elapsed_time/60), 'min')
        
    return performance_modelo, ret_medio_ibov_sim, ret_medio_port_sim, ret_medio_port_long_sim, ret_medio_port_short_sim, corr_prob_ret_sim, datas_teste_sim










# carrega do IBOVESPA e dados históricos
compomentes = ler_base_componetes()
base_total = carrega_dados()
datas = lista_datas(base_total)


# lista de variáveis para o modelo (features) e para aplicação de logaritmo (cols)
#features = ['data', 'codigo', 'retorno', 'acao_close', 'roe', 'pl', 'irf', 'sharpe', 'petroleo_close', 'dolar_close','dji_close', 'sp500_close', 'risco_brasil', 'ibov_fut_close']
#cols = ['acao_close', 'petroleo_close', 'dolar_close', 'dji_close', 'sp500_close', 'risco_brasil', 'ibov_fut_close']
features = ['data', 'codigo', 'retorno', 'acao_close', 'dolar_close', 'sp500_close', 'ibov_fut_close']
cols     = ['acao_close', 'dolar_close', 'sp500_close', 'ibov_fut_close']
modelfilepath = 'D:/Users/felip/Documents/07. FEA/Dissertacao/codigos/modelos/lstm-best-model.hdf5'


# inicializa atributos de simulação
n_epoch     = 1000
n_neurons   = 25
sim         = 0
ret_target  = 0.0
dias_treino = 252
dias_teste  = 126
time_steps  = 21
p           = 0.8
#D           = [3000, 3250, 3500, 3750]
D           = np.arange(0, datas.shape[0]-dias_treino-dias_teste, dias_teste)
k           = [1, 3, 5, 7, 10, 15, 20]

# simulação
performance_modelo, ret_medio_ibov_sim, ret_medio_port_sim, ret_medio_port_long_sim, ret_medio_port_short_sim, corr_prob_ret_sim, datas_teste_sim = simulacao(n_epoch,n_neurons,sim,ret_target,dias_treino,dias_teste,time_steps,D,k,datas,base_total,features,cols, modelfilepath,compomentes,p)

# resultados consolidados
performance_modelo       = DataFrame(performance_modelo, columns = ['loss_treino', 'loss_teste', 'acc_treino', 'acc_teste', 'taxa_treino', 'taxa_teste'])
ret_medio_ibov_sim       = DataFrame(ret_medio_ibov_sim)
ret_medio_port_sim       = DataFrame(ret_medio_port_sim)
ret_medio_port_long_sim  = DataFrame(ret_medio_port_long_sim)
ret_medio_port_short_sim = DataFrame(ret_medio_port_short_sim)
corr_prob_ret_sim        = DataFrame(corr_prob_ret_sim)
datas_teste_sim          = DataFrame(datas_teste_sim, columns = ['data'])

   
gera_graficos_e_sumario(ret_medio_ibov_sim,ret_medio_port_sim,ret_medio_port_long_sim,ret_medio_port_short_sim,corr_prob_ret_sim,datas_teste_sim,k)    
    

# estatísticas
print(performance_modelo.describe())
print(ret_medio_ibov_sim.describe())
print(ret_medio_port_sim.describe())
print(ret_medio_port_long_sim.describe())
print(ret_medio_port_short_sim.describe())


print(corr_prob_ret_sim.describe())



























#D           = [4284] #, 4347, 4410]
#D           = [4158, 4221, 4284, 4347, 4410]
#D           = [4158, 4221]





#
#
#datas_teste_sim.shape
#ret_medio_port_sim.shape
#
#
#
#datas_teste_sim.shape
#ret_medio_port_long_sim.shape
#aux0 = datas_teste_sim
#aux1 = ret_medio_port_long_sim
#aux2 = ret_medio_ibov_sim
#aux0.shape
#aux1.shape
#aux2.shape
#
#aux1.plot()
#aux2.plot()
#
#datas_teste_sim.shape
#ret_medio_port_long_sim.shape
#
#
#ini = 0
##max fim 4314
#fim = 500
#k_port = 1
#N = aux0.iloc[ini:fim,:].values    #4279
#
#
#N = np.arange(0, len(aux2.iloc[ini:fim,:]))
#MEDIUM_SIZE = 16
#BIGGER_SIZE = 22
#plt.rc('font', size=BIGGER_SIZE)         # controls default text sizes
#plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
#plt.rc('axes', labelsize=MEDIUM_SIZE)     # fontsize of the x and y labels
#plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
#plt.rc('legend', fontsize=MEDIUM_SIZE)    # legend fontsize
#plt.rc('figure', titlesize=BIGGER_SIZE)   # fontsize of the figure title
#plt.figure()
#plt.rcParams['figure.figsize'] = (12, 8)
#for j in range(k_port,len(k)):
#    label_ = "portfolio k = " + str(k[j])
#    plt.plot(N, ret_acum(aux1.iloc[ini:fim,j].values), label=label_)
#plt.plot(N, ret_acum(aux2.iloc[ini:fim,:].values), label="ibovespa")
#plt.title("LSTM Deep Learning: Portfolio Long Ibovespa")
#plt.xlabel("Dia")
#plt.ylabel("Retorno Acumulado")
#plt.legend(frameon=False)
#













