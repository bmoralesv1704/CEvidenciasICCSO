import pandas as pd	#importamos pandas
from sklearn.cross_validation import train_test_split	# del repositorio ed scikit-learn importamos la libreria 
from sklearn.linear_model import LinearRegression	# del repositorio ed scikit-learn importamos la libreria para la regresion lineal 
import matplotlib.pyplot as plt		# importamos la libreria matplot para la graficacion

dataset = pd.read_csv('Salary_Data.csv')	# obtenemos con panda el archivo con los datos a graficar
X = dataset.iloc[:,:-1].values		# acomodamos en las columnas correspondientes x
y = dataset.iloc[:,1].values		# acomodamos en las columnas correspondientes y

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
regressor = LinearRegression()		# la variable regressor se iguala al metodo 
regressor.fit(X_train, y_train)	

y_pred = regressor.predict(X_test)
plt.scatter(X_train, y_train, color = 'red')
plt.plot(X_train, regressor.predict(X_train), color = 'blue')
plt.show()

