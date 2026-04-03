## **Perceptron**

#### How many parameters does the neural network introduced in the perceptron have?

Descripción de la red

2 inputs → 1 output

Es la red neuronal más sencilla posible: el clasificador binario con una única neurona

Definición matemática de la red:
y = sigmoid(w1·x1 + w2·x2 + b)

x1, x2: Entradas, correspondientes a coordenadas (x, y) de un punto
w1, w2: Pesos, cuyo valor se determina (aprende) durante el entrenamiento
b     : Bias, el 3er parámetro
y     : Salida (output), valor entre 0 y 1 que corresponde a la probabilidad de pertenecer a una de las clases


Por tanto la respuesta es: la red tiene 3 parámetros: w1, w2 y b



