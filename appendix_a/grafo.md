**Los Modelos como grafos**

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
- módulo de difernciación automática de PyTorch
- Funciones para calcular gradientes en grafos


Ahora veamos el motor de diferenciación automática de PyTorch, también conocido como autograd. El sistema autograd de PyTorch proporciona funciones para calcular gradientes en grafos computacionales dinámicos de forma automática.
