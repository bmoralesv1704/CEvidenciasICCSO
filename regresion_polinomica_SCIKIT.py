from sklearn.linear_model import LinearRegression	# se importa la libreria de la regresion
from sklearn.preprocessing import PolynomialFeatures	# se importa la libreria de las caracteristicas polinomicas
import matplotlib.pyplot as plt			
import pandas as pd

dataset = pd.read_csv('Salary_Data.csv')		# obtenemos el archivo con pandas
X = dataset.iloc[:,:-1].values				# ordenamos las columnas de x, y
y = dataset.iloc[:,1].values				# ordenamos las columnas de x, y

lin_reg = LinearRegression()
poly_reg = PolynomialFeatures(degree=18)		# agregamos a la variable el grado de ajuste 

X_poly = poly_reg.fit_transform(X)		
poly_reg.fit(X_poly,y)
lin_reg.fit(X_poly,y)

plt.scatter(X,y)			
plt.plot(X,lin_reg.predict(poly_reg.fit_transform(X)))
plt.show()						# mostramos la grafica
