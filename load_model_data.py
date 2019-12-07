import pandas as pd
from pandas import read_csv

def carrega_dados():
    
    path = 'D:/Users/felip/Documents/07. FEA/Dissertacao/dados/BOVESPA/dataprep/'
#    path = '/content/drive/My Drive/codigos/dados/'
    
    print("loading Preço...")
    nome = 'fechamento_v2.txt'
    file = path + nome
    fechamento = read_csv(file,sep = ',')
    data = fechamento.iloc[:,1]
    codigo = fechamento.iloc[:,0]
    fechamento = fechamento.iloc[:,2]
    base_fechamento = pd.DataFrame({'data': pd.to_datetime(data),
                                    'codigo': codigo,
                                    'acao_close': fechamento}, columns = ['data', 'codigo', 'acao_close'])
    
    print("Calculating Retorno...")
    codigo_dist = codigo.drop_duplicates()
    for i in range(len(codigo_dist)):
        aux = base_fechamento.loc[base_fechamento['codigo'] == codigo_dist.iloc[i]]
        aux = aux.set_index(['data'])
        aux = aux.sort_index(axis = 0, level=['data'])
        aux = aux.reset_index(level=['data'])        
        aux['retorno'] = aux['acao_close'].pct_change(1)
        if i == 0:
            base_retorno = aux
        else:
            base_retorno = base_retorno.append(aux)
#        print(i, codigo_dist.iloc[i], aux.shape)  
    base_retorno = base_retorno.drop(columns=['acao_close'])
    
    print("loading LPA...")
    nome = 'lpa.txt'
    file = path + nome
    lpa = read_csv(file,sep = ',')
    data = lpa.iloc[:,1]
    codigo = lpa.iloc[:,0]
    lpa = lpa.iloc[:,2]
    base_lpa = pd.DataFrame({'data': pd.to_datetime(data),
                             'codigo': codigo,
                             'lpa': lpa}, columns = ['data', 'codigo', 'lpa'])
    
    print("loading Dolar...")
    nome = 'dolar_v2.txt'
    file = path + nome
    dolar = read_csv(file,sep = ',')
    data = dolar.iloc[:,0]
    dolar_close = dolar.iloc[:,1]
    dolar_open = dolar.iloc[:,2]
    dolar_high = dolar.iloc[:,3]
    dolar_low = dolar.iloc[:,4]
    base_dolar = pd.DataFrame({'data': pd.to_datetime(data),                           
                               'dolar_close': dolar_close,
                               'dolar_open': dolar_open,
                               'dolar_high': dolar_high,
                               'dolar_low': dolar_low}, 
                                columns = ['data',
                                           'dolar_close',
                                           'dolar_open',
                                           'dolar_high',
                                           'dolar_low'])

    print("loading Petroleo...")
    nome = 'petroleo.txt'
    file = path + nome
    petroleo = read_csv(file,sep = ',')
    data = petroleo.iloc[:,0]
    petroleo_close = petroleo.iloc[:,1]
    petroleo_open = petroleo.iloc[:,2]
    petroleo_high = petroleo.iloc[:,3]
    petroleo_low = petroleo.iloc[:,4]
    base_petroleo = pd.DataFrame({'data': pd.to_datetime(data),                           
                               'petroleo_close': petroleo_close,
                               'petroleo_open': petroleo_open,
                               'petroleo_high': petroleo_high,
                               'petroleo_low': petroleo_low}, 
                                columns = ['data',
                                           'petroleo_close',
                                           'petroleo_open',
                                           'petroleo_high',
                                           'petroleo_low'])
    print("loading DJI...")
    nome = 'dji.txt'
    file = path + nome
    dji = read_csv(file,sep = ',')
    data = dji.iloc[:,0]
    dji_close = dji.iloc[:,1]
    dji_open = dji.iloc[:,2]
    dji_high = dji.iloc[:,3]
    dji_low = dji.iloc[:,4]
    base_dji = pd.DataFrame({'data': pd.to_datetime(data),                           
                               'dji_close': dji_close,
                               'dji_open': dji_open,
                               'dji_high': dji_high,
                               'dji_low': dji_low}, 
                                columns = ['data',
                                           'dji_close',
                                           'dji_open',
                                           'dji_high',
                                           'dji_low'])
    
    print("loading S&P500...")
    nome = 'sp500_v2.txt'
    file = path + nome
    sp500 = read_csv(file,sep = ',')
    data = sp500.iloc[:,0]
    sp500_close = sp500.iloc[:,1]
    sp500_open = sp500.iloc[:,2]
    sp500_high = sp500.iloc[:,3]
    sp500_low = sp500.iloc[:,4]
    base_sp500 = pd.DataFrame({'data': pd.to_datetime(data),                           
                               'sp500_close': sp500_close,
                               'sp500_open': sp500_open,
                               'sp500_high': sp500_high,
                               'sp500_low': sp500_low}, 
                                columns = ['data',
                                           'sp500_close',
                                           'sp500_open',
                                           'sp500_high',
                                           'sp500_low'])
    
    print("loading Risco Brasil...")
    nome = 'risco_brasil_v2.txt'
    file = path + nome
    risco_brasil = read_csv(file,sep = ',')
    data = risco_brasil.iloc[:,0]
    risco = risco_brasil.iloc[:,1]
    base_risco_brasil = pd.DataFrame({'data': pd.to_datetime(data),                           
                               'risco_brasil': risco}, 
                                columns = ['data',
                                           'risco_brasil'])
   
    print("loading IBOV Futuro...")
    nome = 'ibov_futuro_v2.txt'
    file = path + nome
    ibov_fut = read_csv(file,sep = ',')
    data = ibov_fut.iloc[:,0]
    ibov_fut_close = ibov_fut.iloc[:,1]
    ibov_fut_open = ibov_fut.iloc[:,2]
    ibov_fut_high = ibov_fut.iloc[:,3]
    ibov_fut_low = ibov_fut.iloc[:,4]
    base_ibov_fut = pd.DataFrame({'data': pd.to_datetime(data),                           
                               'ibov_fut_close': ibov_fut_close,
                               'ibov_fut_open': ibov_fut_open,
                               'ibov_fut_high': ibov_fut_high,
                               'ibov_fut_low': ibov_fut_low}, 
                                columns = ['data',
                                           'ibov_fut_close',
                                           'ibov_fut_open',
                                           'ibov_fut_high',
                                           'ibov_fut_low'])


    print("loading IBOV...")
    nome = 'ibov_v2.txt'
    file = path + nome
    ibov = read_csv(file,sep = ',')
    data = ibov.iloc[:,0]
    ibov_close = ibov.iloc[:,1]
    ibov_open = ibov.iloc[:,2]
    ibov_high = ibov.iloc[:,3]
    ibov_low = ibov.iloc[:,4]
    ibov_ret = ibov.iloc[:,5]
    base_ibov = pd.DataFrame({'data': pd.to_datetime(data),                           
                               'ibov_close': ibov_close,
                               'ibov_open': ibov_open,
                               'ibov_high': ibov_high,
                               'ibov_low': ibov_low,
                               'ibov_ret': ibov_ret}, 
                                columns = ['data',
                                           'ibov_close',
                                           'ibov_open',
                                           'ibov_high',
                                           'ibov_low',
                                           'ibov_ret'])
    
        
    print("loading ROIC...")
    nome = 'roic.txt'
    file = path + nome
    roic = read_csv(file,sep = ',')
    data = roic.iloc[:,0]
    codigo = roic.iloc[:,1]
    roic = roic.iloc[:,2]
    base_roic = pd.DataFrame({'data': pd.to_datetime(data),
                             'codigo': codigo,
                             'roic': roic}, columns = ['data', 'codigo', 'roic'])

    print("loading ROE...")
    nome = 'roe.txt'
    file = path + nome
    roe = read_csv(file,sep = ',')
    data = roe.iloc[:,0]
    codigo = roe.iloc[:,1]
    roe = roe.iloc[:,2]
    base_roe = pd.DataFrame({'data': pd.to_datetime(data),
                             'codigo': codigo,
                             'roe': roe}, columns = ['data', 'codigo', 'roe'])
    
    print("loading P/L...")
    nome = 'pl.txt'
    file = path + nome
    pl = read_csv(file,sep = ',')
    data = pl.iloc[:,0]
    codigo = pl.iloc[:,1]
    pl = pl.iloc[:,2]
    base_pl = pd.DataFrame({'data': pd.to_datetime(data),
                             'codigo': codigo,
                             'pl': pl}, columns = ['data', 'codigo', 'pl'])
    
    print("loading Sharpe...")
    nome = 'sharpe.txt'
    file = path + nome
    sharpe = read_csv(file,sep = ',')
    data = sharpe.iloc[:,0]
    codigo = sharpe.iloc[:,1]
    sharpe = sharpe.iloc[:,2]
    base_sharpe = pd.DataFrame({'data': pd.to_datetime(data),
                             'codigo': codigo,
                             'sharpe': sharpe}, columns = ['data', 'codigo', 'sharpe'])    
    
    print("loading IRF...")
    nome = 'irf.txt'
    file = path + nome
    irf = read_csv(file,sep = ',')
    data = irf.iloc[:,0]
    codigo = irf.iloc[:,1]
    irf = irf.iloc[:,2]
    base_irf = pd.DataFrame({'data': pd.to_datetime(data),
                             'codigo': codigo,
                             'irf': irf}, columns = ['data', 'codigo', 'irf'])       
    
    print("Join Data Bases...")
    base_total = pd.merge(base_fechamento, base_retorno, how = 'outer', on = ['data', 'codigo'])
#    base_total = pd.merge(base_total, base_lpa, how = 'outer', on = ['data', 'codigo'])                          
#    base_total = pd.merge(base_total, base_pl, how = 'outer', on = ['data', 'codigo'])
#    base_total = pd.merge(base_total, base_roic, how = 'outer', on = ['data', 'codigo'])
#    base_total = pd.merge(base_total, base_roe, how = 'outer', on = ['data', 'codigo'])
#    base_total = pd.merge(base_total, base_sharpe, how = 'outer', on = ['data', 'codigo'])
#    base_total = pd.merge(base_total, base_irf, how = 'outer', on = ['data', 'codigo'])
    base_total = pd.merge(base_total, base_dolar, how = 'outer', on = ['data'])
#    base_total = pd.merge(base_total, base_petroleo, how = 'outer', on = ['data'])
#    base_total = pd.merge(base_total, base_dji, how = 'outer', on = ['data'])
    base_total = pd.merge(base_total, base_sp500, how = 'outer', on = ['data'])
    base_total = pd.merge(base_total, base_risco_brasil, how = 'outer', on = ['data'])
    base_total = pd.merge(base_total, base_ibov_fut, how = 'outer', on = ['data'])
    base_total = pd.merge(base_total, base_ibov, how = 'outer', on = ['data'])
    base_total = base_total.set_index(['data', 'codigo'])
    base_total = base_total.sort_index(axis=0, level=['data', 'codigo'])
    base_total = base_total.reset_index(level=['data','codigo'])
    
    
    return base_total