import pandas as pd
import warnings

from sklearn.metrics import mean_squared_error
from statsmodels.tsa.arima_model import ARIMA

df = pd.read_csv('filepath/filename.csv',header=0, sep=',')

series = df.set_index('Date')

materialsBranches = df[['Material_Code','Branch/Filial']].drop_duplicates()

mseResults = []

# evaluate an ARIMA model for a given order (p,d,q)
def evaluate_arima_model(X, arima_order):
	# prepare training dataset
	train_size = int(len(X) * 0.66)
	train, test = X[0:train_size], X[train_size:]
	history = [x for x in train]
	# make predictions
	predictions = list()
	for t in range(len(test)):
		model = ARIMA(history, order=arima_order)
		model_fit = model.fit(disp=0)
		yhat = model_fit.forecast()[0]
		predictions.append(yhat)
		history.append(test[t])
	# calculate out of sample error
	error = mean_squared_error(test, predictions)
	return error
 
# evaluate combinations of p, d and q values for an ARIMA model
def evaluate_models(dataset, p_values, d_values, q_values):
	dataset = dataset.astype('int32')
	best_score, best_cfg = float("inf"), None
	for p in p_values:
		for d in d_values:
			for q in q_values:
				order = (p,d,q)
				try:
					mse = evaluate_arima_model(dataset, order)
					if mse < best_score:
						best_score, best_cfg = mse, order
					print('ARIMA%s MSE=%.3f' % (order,mse))
				except:
					continue
	mseResults.append((branch_filter,material_filter,best_cfg, best_score,frequency,mean,stdev,maximum,minimum))
    
# evaluate parameters
p_values = [0, 1, 2, 3, 4, 5]
d_values = range(0, 3)
q_values = range(0, 5)
warnings.filterwarnings("ignore")

# iterate over your data for each demand
for (i,j) in materialsBranches.iterrows():
   material_filter =  j.loc['Material']
   branch_filter =  j.loc['Branch/Filial']
   dataset = series[(series['Material']==material_filter)&
                            (series['Branch/Filial']==branch_filter)].drop(columns=['Branch/Filial','Material'])
   stdev = dataset['demand.'].std()
   mean = dataset['demand.'].mean()
   maximum = dataset['demand.'].max()
   minimum = dataset['demand.'].min()
   frequency = dataset['demand.'].count()
   evaluate_models(dataset.values, p_values, d_values, q_values)

resultado = pd.DataFrame(mseResults, columns=['Branch/Filial','Material','Parameters','MSE','Freq','Mean','Stdev','Max','Min'])
resultado.to_excel('filepath/resultBestARIMA.xlsx',header=True,columns=(['Branch/Filial','Material','Parameters','MSE','Freq','Mean','Stdev','Max','Min']))
