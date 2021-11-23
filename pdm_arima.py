# -*- coding: utf-8 -*-
"""
Created on Tue Mar  9 08:44:47 2021

@author: Jaime Gonzalez
"""

import pandas as pd
import numpy as np
from statsmodels.tsa.stattools import adfuller 
from pmdarima.arima import auto_arima

filtroFatorP = 0.05
filtroDataMin = '2021-01-01'
baseTreinoInicio = '2010-01-01'
baseTreinoFim = '2020-12-31'
horizontePrevisao = 4

lista_adf = []
forecasts=[]

df = pd.read_excel('C:/Users/jalberto/Desktop/Rotina de Planejamento/0.0_Dados/VENDAS_Demanda.xlsx',
                   squeeze=True)

df['Data']=df['Data'].astype(str)

df['Ano'] = df['Data'].astype(str).str[6:]
df['Mes'] = df['Data'].astype(str).str[3:5]
df['Dia'] = df['Data'].astype(str).str[:2]


df['Date'] = pd.to_datetime(df['Ano']+'-'+df['Mes']+'-'+df['Dia'],
                                       format='%Y-%m-%d')


df['Centro']=df['Filial_Cod']
df['Material']=df['Produto_Cod']
df['Qty']=df['Qtde_Faturada'].astype(int)

df01 = df.drop(columns = ['Produto_Categoria',
                        'Qtde Devolvida',
                        'Qtde Líquida',
                        'Qtde_Faturada',
                        'Ton Líquida',
                        'FAT Bruto',
                        'Desconto Total',
                        'DEV Bruto',
                        'Fat Líquido',
                        'Data',
                        'Ano',
                        'Mes',
                        'Dia',
                        'Filial_Cod',
                        'Produto_Cod']) 

df01 = df.set_index(['Date','Centro','Material']).groupby([pd.Grouper(
                        level='Centro'), 
                            pd.Grouper(level='Material'), 
                            pd.Grouper(level='Date', freq='W')]
                          )['Qty'].agg(
                              ['sum']
                              ).add_suffix('_of_Qty').reset_index()

df02 = df.set_index(['Centro','Material']).groupby([
                        pd.Grouper(level='Centro'), 
                        pd.Grouper(level='Material')]
                      )['Date'].agg(
                          ['min','max']
                          ).add_suffix('_Date').reset_index()

resumo = df01.groupby(by=['Centro','Material']
                      )['sum_of_Qty'].agg(['mean','std','count','min','max'])
                                               
resumo['cv']=resumo['std']/resumo['mean']

series = df01.set_index('Date')
centros = df[['Centro','Material']].drop_duplicates()

def analisar_DataFrame(dataset):
    dataset = dataset.astype('int32')
    try:
        adfuller(dataset)
        adfResult = adfuller(dataset)
        lista_adf.append((termo_filtrado_01,
                   termo_filtrado_02,
                   adfResult[0],adfResult[1],adfResult[4]['10%'],adfResult[4]['5%'],adfResult[4]['1%']))         
    except:
        pass

def prever_demanda(dataset):
    dataset = dataset.astype('int32')
    train = dataset.loc[baseTreinoInicio:baseTreinoFim]
    try:
        stepwise_model = auto_arima(dataset,m=12,trend=None,start_p=0,start_q=0,start_P=0,start_Q=0,
                                    error_action='ignore',max_p=5,max_d=2,max_q=5,max_P=5,max_D=1,max_Q=5,
                                    seasonal=True,trace=True,stepwise=True,suppress_warnings=True)
    except:
        try:
            stepwise_model = auto_arima(dataset,m=6,trend=None,start_p=0,start_q=0,start_P=0,start_Q=0,
                                    error_action='ignore',max_p=5,max_d=2,max_q=5,max_P=5,max_D=1,max_Q=5,
                                    seasonal=True,trace=True,stepwise=True,suppress_warnings=True)
        except:
                try:
                    stepwise_model = auto_arima(dataset,m=1,trend=None,start_p=0,start_q=0,start_P=0,start_Q=0,
                                    error_action='ignore',max_p=5,max_d=2,max_q=5,max_P=5,max_D=1,max_Q=5,
                                    seasonal=True,trace=True,stepwise=True,suppress_warnings=True)
                except:
                    try:
                        stepwise_model = auto_arima(dataset,m=12,trend=None,start_p=0,start_q=0,seasonal=False,
                                        error_action='ignore',max_p=5,max_d=2,max_q=5,
                                        trace=True,stepwise=True,suppress_warnings=True)
                    except:
                       pass
    stepwise_model.fit(train)
    future_forecast = stepwise_model.predict(n_periods=horizontePrevisao)
    forecasts.append((centro,
                      material,
                      future_forecast[0],
                      future_forecast[1],
                      future_forecast[2],
                      future_forecast[3]
                      )
                     )

for(i,j) in centros.iterrows():
  termo_filtrado_01 = j.loc['Centro']
  termo_filtrado_02 = j.loc['Material']
  df_filtrada = series[(series['Centro']==termo_filtrado_01)&
                   (series['Material']==termo_filtrado_02)].drop(columns=['Centro','Material'])
  analisar_DataFrame(df_filtrada)
df_ADF = pd.DataFrame(data=lista_adf,
                      columns=['Centro','Material','ADF',
                               'P_value','10%','5%','1%'])  

df_statistics = pd.merge(pd.merge(df_ADF,resumo,how = 'left',on=['Centro','Material']),
                                         df02,how='left',on=['Centro','Material'])

estacionarios=df_statistics[(df_statistics['P_value']<filtroFatorP)&
                                (df_statistics['P_value']>0)&
                                (df_statistics['max_Date']>filtroDataMin)
                            ]

estacionarios = estacionarios[(estacionarios['count']>=50)]

estacionarios = estacionarios[(estacionarios['mean']>0)]

estacionarios = estacionarios.dropna()
    
centros_previsao = estacionarios[['Centro','Material']].drop_duplicates()

for(i,j) in centros_previsao.iterrows():
  centro = j.loc['Centro']
  material = j.loc['Material']
  df_filtrada = series[(series['Centro']==centro)&
                       (series['Material']==material)
                       ].drop(
                           columns=[
                               'Centro','Material'
                               ])
  prever_demanda(df_filtrada)
    
resultSet = pd.merge(estacionarios,
                     pd.DataFrame(data=forecasts,
                                  columns=['Centro',
                                           'Material',
                                           'X1(t+1)',
                                           'X2(t+2)',
                                           'X3(t+3)',
                                           'X4(t+4)']
                                  ),
                                     how='left',
                                     on=['Centro','Material']
                        )
