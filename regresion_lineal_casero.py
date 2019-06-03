x = [1.0, 1.6, 3.4, 4.0, 5.2] # Declaramos la variable x y le asignamos los valores del arreglo
y = [1.2, 2.0, 2.4, 3.5, 3.5] # Declaramos la variable y y le asignamos los valores del arreglo

def suma (w): # definimos el metodo suma con la variable w
    sum = 0 
    for i in range (len(w)): # ciclo for con la variable i en el rango de longitud de la variable w
        sum = w[i] + sum	# la variable suma es igualada al valor del arreglo w del numero i + la variable sum
    return sum

print (suma(x))
print (suma(y))

import pandas as pd	# impotamos la libreria pandas con nombre pd

x1 = pd.array(x)   # la variable x1 es igualada al espacio array del valor x obtenida por pandas 
print (x1)
y1 = pd.array(y)
print (y1)

xy = x1*y1
print (xy)


x2 = x1*x1
print (x2)

n = len(x)
print (n)

def lin_reg(m,n):	# se define el metodo lin_reg que requiere de dos valores (m y n)
    x = pd.array(m)	# x se iguala al valor del espacio m del array obtenido por pandas	
    y = pd.array(n)
    n = len(m)
    x2 = x*x
    sumx2 = (suma(x) * suma(x))	#la variable sumx2 se iguala a la multiplicacion de los valores obtenidos por el metodo suma
    xy=x*y
    arriba1 = (n*(suma(xy))) - (suma(x) * suma(y)) #  el valor de la operacion obtenida 
    abajo1 = (n*(suma (x2))) - sumx2

    a = arriba1 / abajo1

    arriba2 = (suma(y) * suma(x2)) - (suma(xy) * suma(x))
    abajo2 = (n*(suma(x2))) - sumx2

    b = arriba2 / abajo2

    return a , b

print (lin_reg (x,y))











