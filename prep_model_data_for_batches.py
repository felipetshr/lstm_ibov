import numpy as np

seed = 42
np.random.seed(seed)

import pandas as pd
import matplotlib.pyplot as plt
from pandas import DataFrame, concat
from sklearn.preprocessing import MinMaxScaler



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






def dataprep_for_batches(datas, inicio_treino, dias_treino, time_steps, dias_teste, features, cols, ret_target, compomentes, base_total):   

    dia_inicio_treino = pd.to_datetime(datas.iloc[inicio_treino].values)[0]
    dia_fim_treino = pd.to_datetime(datas.iloc[inicio_treino+dias_treino].values)[0]
    dia_inicio_teste = pd.to_datetime(datas.iloc[inicio_treino+dias_treino+1].values)[0]
    dia_inicio_teste_aux = pd.to_datetime(datas.iloc[inicio_treino+dias_treino-time_steps+1].values)[0]
    dia_fim_teste = pd.to_datetime(datas.iloc[inicio_treino+dias_treino+dias_teste].values)[0]
    
    month = "{0:02d}".format(dia_fim_treino.month)
    year =  "{0:04d}".format(dia_fim_treino.year)[2:]
    anomes = month + '/' + year
    papeis = compomentes.loc[:,anomes][compomentes.loc[:,anomes] > 0].index.values.tolist()
    
    base_in = base_total.loc[base_total['codigo'].isin(papeis)]
    base = base_in[features]
    base['y'] = ((base['retorno'] > ret_target)).astype(np.int) 
    base = base.dropna(axis = 0, how = 'any')
    new_features = list(base)
    n_features = base.shape[1]
    n_obs = time_steps * n_features





    # precisamos obter uma base onde existe a mesma quantidade de
    # ações para cada dia de treino e teste. Com isso, é possível
    # utilizar adequadamente o cell state de cada ação no tempo. 
    # Faremos isso com base no retorno diário.
    base_batch = base.iloc[:,0:3]
    base_batch = base_batch.set_index(['data','codigo'])
    base_batch = base_batch.sort_index(axis=0)
    base_batch = base_batch[~base_batch.index.duplicated(keep='first')]
    base_batch = base_batch.unstack()    
    base_batch = base_batch.loc[dia_inicio_treino:dia_fim_teste,:]    
    base_batch = base_batch.dropna(axis = 1, how = 'any')
    base_batch = base_batch.dropna(axis = 0, how = 'any')
    base_batch = base_batch.stack()
    base_batch = base_batch.reset_index(level=['data','codigo'])
    
    # Com a base de ações elegíveis, vamos guardar os retornos
    # do período de teste para a etapa de avaliação da 
    # performance das k-carteiras
#    base_retorno = base_batch.set_index(['data','codigo'])
#    base_retorno = base_retorno.unstack()
#    base_retorno_teste = base_retorno.loc[dia_inicio_teste:dia_fim_teste,:]    
    
    # Com a base de ações elegíveis em mãos, vamos trazer cada
    # feature com um "left join" (o retorno já está na base referencia)
    # aplica log nas variáveis definidas na lista "cols". Se as demais 
    # features forem missing, então serão substituídas por zero.    
    base_batch_total = base_batch
    for j in range(3,len(new_features)):
        feature = new_features[j]
        aux = base[['data','codigo',feature]]
        aux = aux.set_index(['data','codigo'])
        aux = aux[~aux.index.duplicated(keep='first')]
        if feature in cols:
            aux = np.log(aux)
        aux = aux.reset_index(level=['data','codigo'])
        base_batch_total = pd.merge(base_batch_total, aux, how = 'left', on = ['data', 'codigo'])    
        #print('shape da base_batch_total:', base_batch_total.shape)
    base_batch_total = base_batch_total.set_index(['data','codigo'])
    base_batch_total = base_batch_total.fillna(0)
    base_batch_total = base_batch_total.reset_index(level=['data','codigo'])
    base_batch_total = base_batch_total.set_index(['data'])
    
    if base_batch_total.shape[0] == 0:
        print('Dados indisponíveis para o período selecionado!!!')
        return False, [], [], [], [], [], [], [], [], [], [], []
    
    
    # seprada em teste e validação e escala as variáveis no período 
    # de treino e apenas aplicada no período de teste
    # pega os dados de uma certa ação
    aux = base_batch_total
    treino = aux.loc[dia_inicio_treino:dia_fim_treino, :]
    teste = aux.loc[dia_inicio_teste_aux:dia_fim_teste, :]
    if treino.shape[0] == 0:
        print('Dados de treino indisponíveis para o período selecionado!!!')
        return False, [], [], [], [], [], [], [], [], [], [], []
    if teste.shape[0] == 0:
        print('Dados de teste indisponíveis para o período selecionado!!!')
        return False, [], [], [], [], [], [], [], [], [], [], [] 
    treino = treino.reset_index(level=['data'])
    treino_x = treino.iloc[:,2:treino.shape[1]-1].values
    treino_y = treino.iloc[:,-1].values.reshape(treino.shape[0],1)
    treino_datas = treino.iloc[:,0].values.reshape(treino.shape[0],1)
    treino_papeis = treino.iloc[:,1].values.reshape(treino.shape[0],1)
    treino_retornos = treino.iloc[:,2].values.reshape(treino.shape[0],1)
    
    teste = teste.reset_index(level=['data'])
    teste_x = teste.iloc[:,2:teste.shape[1]-1].values
    teste_y = teste.iloc[:,-1].values.reshape(teste.shape[0],1)
    teste_datas = teste.iloc[:,0].values.reshape(teste.shape[0],1)
    teste_papeis = teste.iloc[:,1].values.reshape(teste.shape[0],1)
    teste_retornos = teste.iloc[:,2].values.reshape(teste.shape[0],1)
    
    scaler_x = MinMaxScaler(feature_range=(-1, 1))
    scaler_x.fit(treino_x)
    treino_x = scaler_x.transform(treino_x)    
    teste_x = scaler_x.transform(teste_x)
    
    treino_geral = DataFrame(np.concatenate((treino_papeis, treino_x, treino_datas, treino_retornos, treino_y), axis = 1))
    teste_geral = DataFrame(np.concatenate((teste_papeis, teste_x, teste_datas, teste_retornos, teste_y), axis = 1))
   
    
    qtde_treino = 0
    qtde_teste = 0    
    codigo_dist = base_batch_total.iloc[:,0].drop_duplicates()

    for i in range(len(codigo_dist)):
#        i = 1
        # pega os dados de uma certa ação
        treino = treino_geral.loc[treino_geral[0] == codigo_dist.iloc[i]]
        treino = treino.drop(columns=[0])

        treino_reframed = series_to_supervised(treino, time_steps, 1)
        if treino_reframed.shape[0] == 0:
            print('Dados de treino reframed indisponíveis para o período selecionado!!!')
            return False, [], [], [], [], [], [], [], [], [], [], []

        treino_x = treino_reframed.iloc[:, :n_obs]
        treino_y = treino_reframed.iloc[:, -1]
        treino_datas = DataFrame({'data': pd.to_datetime(treino_reframed.iloc[:, -3].values)},
                                  columns = ['data'])
        treino_retornos = DataFrame({'retorno': treino_reframed.iloc[:, -2].values},
                                  columns = ['retorno'])        
 
        if i == 0:
            treino_x_total = treino_x
            seq_len_treino = treino_x.shape[0]
            treino_y_total = treino_y
            treino_datas_total = treino_datas
            treino_retornos_total = treino_retornos
        else:
            treino_x_total = treino_x_total.append(treino_x)
            seq_len_treino = treino_x.shape[0]
            treino_y_total = treino_y_total.append(treino_y)
            treino_datas_total = treino_datas_total.append(treino_datas)
            treino_retornos_total = treino_retornos_total.append(treino_retornos)

        
        qtde_treino += 1

        
        # pega os dados de uma certa ação
        teste = teste_geral.loc[teste_geral[0] == codigo_dist.iloc[i]]
        teste = teste.drop(columns=[0])
            
        teste_reframed = series_to_supervised(teste, time_steps, 1)
        if teste_reframed.shape[0] == 0:
            print('Dados de teste reframed indisponíveis para o período selecionado!!!')
            return False, [], [], [], [], [], [], [], [], [], [], []
        teste_x = teste_reframed.iloc[:, :n_obs]
        teste_y = teste_reframed.iloc[:, -1]
        teste_datas = DataFrame({'data': pd.to_datetime(teste_reframed.iloc[:, -3].values)},
                                  columns = ['data'])
        teste_retornos = DataFrame({'retorno': teste_reframed.iloc[:, -2].values},
                                  columns = ['retorno'])   
        
        if i == 0:
            teste_x_total = teste_x
            seq_len_teste = teste_x.shape[0]
            teste_y_total = teste_y
            teste_datas_total = teste_datas
            teste_retornos_total = teste_retornos            
        else:
            teste_x_total = teste_x_total.append(teste_x)
            seq_len_teste = teste_x.shape[0]
            teste_y_total = teste_y_total.append(teste_y)
            teste_retornos_total = teste_retornos_total.append(teste_retornos)
                                                                 
        qtde_teste += 1
        



    # reshape input to be 3D [samples (blocks of stocks), timesteps, features]
    # considerando ações intercaladas para a formação dos batchs de treinamento:
    # n ações para cada dia de treinamento
    treino_x = treino_x_total.values.reshape((treino_x_total.shape[0], time_steps, n_features))
    treino_y = treino_y_total.values.reshape((treino_y_total.shape[0], 1))
    for k in range(0,seq_len_treino):
        ind_aux = np.arange(k, seq_len_treino*qtde_treino+k, seq_len_treino)
        treino_x_fatia = treino_x[ind_aux,:,:]
        treino_y_fatia = treino_y[ind_aux,:]
        if k == 0:
            treino_x_fatia_empilhado = treino_x_fatia
            treino_y_fatia_empilhado = treino_y_fatia
        else:
            treino_x_fatia_empilhado = np.append(treino_x_fatia_empilhado, treino_x_fatia, axis = 0)
            treino_y_fatia_empilhado = np.append(treino_y_fatia_empilhado, treino_y_fatia, axis = 0)
        #print(treino_x_fatia_empilhado.shape)
        
    teste_x = teste_x_total.values.reshape((teste_x_total.shape[0], time_steps, n_features))
    teste_y = teste_y_total.values.reshape((teste_y_total.shape[0], 1))
    for k in range(0,seq_len_teste):
        ind_aux = np.arange(k, seq_len_teste*qtde_teste+k, seq_len_teste)
        teste_x_fatia = teste_x[ind_aux,:,:]
        teste_y_fatia = teste_y[ind_aux,:]
        if k == 0:
            teste_x_fatia_empilhado = teste_x_fatia
            teste_y_fatia_empilhado = teste_y_fatia
        else:
            teste_x_fatia_empilhado = np.append(teste_x_fatia_empilhado, teste_x_fatia, axis = 0)
            teste_y_fatia_empilhado = np.append(teste_y_fatia_empilhado, teste_y_fatia, axis = 0)    
        #print(teste_x_fatia_empilhado.shape)
    
    treino_x = treino_x_fatia_empilhado[:,:,:n_features-3]
    treino_y = treino_y_fatia_empilhado
    teste_x = teste_x_fatia_empilhado[:,:,:n_features-3]
    teste_y = teste_y_fatia_empilhado
    
    treino_x = treino_x.astype(float).reshape(treino_x.shape[0],treino_x.shape[1],treino_x.shape[2])
    treino_y = treino_y.astype(float).reshape(treino_y.shape[0],treino_y.shape[1])
    teste_x = teste_x.astype(float).reshape(teste_x.shape[0],teste_x.shape[1],teste_x.shape[2])
    teste_y = teste_y.astype(float).reshape(teste_y.shape[0],teste_y.shape[1])
    
    print('\nData início treino....: ',treino_datas_total.iloc[0,0])
    print('Data fim treino.......: ',treino_datas_total.iloc[-1,0])
    print('Data início teste.....: ',teste_datas_total.iloc[0,0])
    print('Data fim teste........: ',teste_datas_total.iloc[-1,0])
    print('Shape final treino x..:', treino_x.shape)
    print('Shape final treino y..:', treino_y.shape)
    print('Shape final teste x...:', teste_x.shape)
    print('Shape final teste y...:', teste_y.shape)
    print('Qtde ações treino.....:', qtde_treino)
    print('Qtde ações teste......:', qtde_teste)
    print('Taxa Treino...........:{: 2.4f}'.format(treino_y.mean()))
    print('Taxa Teste............:{: 2.4f}'.format(teste_y.mean()))
    print('Qtde features.........:', n_features-2)
    print('\n')
    
    if qtde_treino != qtde_teste:
        print('Qtde de ações no treino e no teste incompatíveis no período selecionado!!!')
        return False, [], [], [], [], [], [], [], [], [], [], []
    
    return True, treino_x, treino_y, teste_x, teste_y, qtde_treino, treino_retornos_total, teste_retornos_total, treino_datas_total, teste_datas_total, qtde_treino, papeis










