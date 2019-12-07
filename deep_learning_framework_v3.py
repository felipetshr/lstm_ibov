#seed = 42
import numpy as np
#np.random.seed(seed)
from pandas import DataFrame
import warnings
warnings.filterwarnings("ignore")
import datetime
from load_model_data import carrega_dados
from funcoes_apoio import simulacao, ler_base_componetes, lista_datas, save_stats, save_results, save_graphics, timer
import time


# carrega do IBOVESPA e dados históricos
compomentes = ler_base_componetes()
base_total  = carrega_dados()
datas       = lista_datas(base_total)


# lista de variáveis para o modelo (features) e para aplicação de logaritmo (cols)
#features = ['data', 'codigo', 'retorno', 'acao_close', 'roe', 'pl', 'irf', 'sharpe', 'petroleo_close', 'dolar_close','dji_close', 'sp500_close', 'risco_brasil', 'ibov_fut_close']
#cols = ['acao_close', 'petroleo_close', 'dolar_close', 'dji_close', 'sp500_close', 'risco_brasil', 'ibov_fut_close']
#features = ['data', 'codigo', 'retorno', 'acao_close', 'dolar_close', 'sp500_close', 'ibov_fut_close', 'ibov_fut_close', 'ibov_close', 'ibov_ret']
features      = ['data', 'codigo', 'retorno', 'acao_close', 'dolar_close', 'risco_brasil', 'ibov_close']
cols          = ['acao_close', 'dolar_close', 'risco_brasil', 'ibov_close']
comm          = 0
stocks        = ['ITUB4']
modelfilepath = 'D:/Users/felip/Documents/07. FEA/Dissertacao/codigos/modelos/lstm-best-model.hdf5'
#modelfilepath = '/content/drive/My Drive/codigos/dados/'


# inicializa atributos de simulação
n_epoch     = 10                              # qtde epochs
n_neurons   = 12                              # qtde neurons
hidden      = 1                               # qtde de hidden layers
lr_value    = 0.05                            # learning rate
decay_value = 1e-3                            # decay value
n_batch_val = 0                               # 0: n_batch = qtde_acoes; 1:  n_batch = treino_x.shape[0]
ver         = 0                               # verbose mode: 0 (silent), 1, 2
sim         = 0                               # contador de simulações
ret_target  = 0.015                             # retorno target: se retorno ativo > ret_target, então y = 1; y = 0 c.c.
ret_ibov    = 0                               # retorno target = ret_ibov: se retorno ativo > ret_ibov, então y = 1; y = 0 c.c.
monitCen    = 0                               # 0 = val_loss; 1 = val_acc 
dias_treino = 900
dias_teste  = 60
time_steps  = 5
p           = 0.0                             # fracao treino vs validacao
#D           = [4000]                          # 2962, 3402, 3528, 3654, 3780, 3906, 4032]
D           = np.arange(0, datas.shape[0]-dias_treino-dias_teste, dias_teste)
k           = [1, 3, 5, 7, 10, 15, 20]
act_func    = 'sigmoid'                       # softmax, sigmoid
metr_func   = 'acc'                          # 'acc'


# simulação do cenário
s0 = time.time()
performance_modelo, ret_medio_ibov_sim, ret_medio_port_sim, ret_medio_port_long_sim, ret_medio_port_short_sim, corr_prob_ret_sim, datas_teste_sim = simulacao(monitCen, n_epoch,n_neurons,hidden,lr_value,decay_value,sim,ret_ibov,ret_target,dias_treino,dias_teste,time_steps,D,k,datas,base_total,features,cols,modelfilepath,compomentes,p,comm,stocks,n_batch_val,ver,act_func,metr_func)
s1 = time.time()
print('Tempo de total de simulação..........: ', timer(s0,s1))


# consolida resultados
performance_modelo       = DataFrame(performance_modelo, columns = ['loss_treino', 'loss_teste', 'acc_treino', 'acc_teste', 'taxa_treino', 'taxa_teste', 'std_treino', 'std_teste'])
ret_medio_ibov_sim       = DataFrame(ret_medio_ibov_sim)
ret_medio_port_sim       = DataFrame(ret_medio_port_sim)
ret_medio_port_long_sim  = DataFrame(ret_medio_port_long_sim)
ret_medio_port_short_sim = DataFrame(ret_medio_port_short_sim)
corr_prob_ret_sim        = DataFrame(corr_prob_ret_sim)
datas_teste_sim          = DataFrame(datas_teste_sim, columns = ['data'])


# get version and time
t = datetime.datetime.now()
hora = str(t.day) + "." + str(t.month) + "." + str(t.year) + "-" + str(t.hour) + "." + str(t.minute) + "." + str(t.second)
version = "epochs=" + str(n_epoch) + "_neurons=" + str(n_neurons) + "_hiddenlay=" + str(hidden) + "_monitcen=" + str(monitCen)  +  "_rettarget=" + str(ret_target) + "_retibov=" + str(ret_ibov) + "_diastreino=" + str(dias_treino) + "_diasteste=" + str(dias_teste) + "_timesteps=" + str(time_steps) + "_" + hora


# save stats
save_stats(version,performance_modelo,ret_medio_ibov_sim,ret_medio_port_sim,ret_medio_port_long_sim,ret_medio_port_short_sim)


# save results
save_results(version,performance_modelo,ret_medio_ibov_sim,ret_medio_port_sim,ret_medio_port_long_sim,ret_medio_port_short_sim,datas_teste_sim,corr_prob_ret_sim)


# save graphics
save_graphics(ret_medio_ibov_sim,ret_medio_port_sim,ret_medio_port_long_sim,ret_medio_port_short_sim,corr_prob_ret_sim,datas_teste_sim,k,version)