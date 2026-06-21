# **Los Modelos como grafos**

Grafo:
- Es un mapa que permite organizar y visualizar una serie de operaciones matemáticas.
- Elementos:
    - Datos: Son números o matrices (tensores en el caso de matrices n-dimensionales)
    - Nodos: Representados como círculos. Operaciones matemáticas: suma, multiplicación o más complejas.
    - Aristas: Representados como flechas. Los caminos por donde viajan los datos de una operación a otra

Un grafo es como una receta de cocina detallada paso a paso, o como una línea de montaje de una fábrica.
Ejemplo: calcular el precio de una cena precio = (comida + bebida) * IVA

Grafos en Deep Learning:
El grafo describe la secuencia de cálculos para obtener la salida de una red neuronal.
Estos calculos son los necesarios para determinar los gradientes y la retropropagación, que es el principal algoritmo usado en entrenamiento de redes neuronales.

![Alt text](grafo.png)

Autograd:
- módulo de diferenciación (cálculo numérico de diferenciales) de PyTorch
- Funciones para calcular gradientes en grafos

Autograd es la librería de diferenciación (cálculo numérico de diferenciales) de PyTorch, Permite calcular gradientes en grafos computacionales dinámicos de forma automática. El término correcto es diferenciación y no derivada ya que es un cálculo multivariable, usando la regla de la cadena

Derivada: función que mide la tasa de cambio instantánea de una variable respecto a otra. Geométricamente es la pendiente de la recta tangente a una curva en un punto dado. Notación: $f'(x)$ o también $\frac{dy}{dx}$

Derivada / Diferencial: 

Si f(x) = y
Derivada de la función f'(x) = dy/dx
Diferencial de y: dy
Diferencial de x: dx
