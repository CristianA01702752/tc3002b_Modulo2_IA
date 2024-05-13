# tc3002b_Modulo2_IA
En este repositorio se alojarán todos los archivos del Módulo 2 de la materia de Desarrollo de aplicaciones avanzadas de ciencias computacionales

## Food Classification

### Dataset

El conjunto de datos utilizado en este proyecto fue creado por Harish Kumar y se obtuvo del conjunto de datos de Kaggle conocido como [Food Image Classification Dataset](https://www.https://www.kaggle.com/datasets/harishkumardatalab/food-image-classification-dataset)

El objetivo principal de este modelo es clasificar 13 categorías diferentes de platillos de comida.

Originalmente, el conjunto de datos de Kaggle incluía 35 clases de platillos, pero para este modelo se redujo a 13, que son las siguientes:
* Apple Pie
* Baked Potato
* Cheesecake
* Chicken Curry
* Crispy Chicken
* Donut
* Fries
* Hot Dog
* Ice Cream
* Omelette
* Sandwich
* Sushi
* Taco

### Estructura del dataset
Las clases tienen una diferencia relevante en la cantidad de imágenes, las siguientes clases poseen 1500 imágenes:
* Baked Potato
* Crispy Chicken
* Donut
* Fries
* Hot Dog
* Sandwich
* Taco
Mientras que el resto de las clases poseen 1000 imágenes, las cuales son:
* Apple Pie
* Cheesecake
* Chicken Curry
* Ice Cream
* Omelette
* Sushi

En el conjunto de datos original de Kaggle, cada clase contiene todas las imágenes sin una estructura definida. Con el fin de mejorar la eficiencia del desarrollo del modelo y reducir posibles desequilibrios, las imágenes de las clases se dividirán en tres categorías principales:

* **Train (Entrenamiento):** Esta categoría se utilizará para entrenar el modelo y contendrá el 70% de las imágenes de cada clase. Se estima que habrá aproximadamente 11,550 imágenes en esta categoría.

* **Validation (Validación):** Aquí se evaluará el rendimiento del modelo durante el entrenamiento, lo que permitirá ajustar los hiperparámetros del modelo antes de pasar a la fase de prueba. Esto nos ayuda a evitar problemas de overfitting (sobreajuste) o underfitting (subajuste) del modelo. La categoría de validación contendrá el 15% de las imágenes de cada clase, aproximadamente 2,475 imágenes.

  **Overfitting (sobreajuste):** Ocurre cuando un modelo se ajusta demasiado bien a los datos de entrenamiento, lo que resulta en un bajo rendimiento con nuevos datos no vistos previamente.

  **Underfitting (subajuste):** Ocurre cuando un modelo es demasiado simple para capturar la estructura subyacente de los datos. 

* **Test (Prueba):** Esta categoría permitirá evaluar el rendimiento del modelo frente a imágenes que nunca ha visto durante la fase de entrenamiento. Contendrá también el 15% de las imágenes de cada clase, aproximadamente 2,475 imágenes.







