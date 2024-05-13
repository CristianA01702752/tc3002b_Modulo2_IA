import os
import shutil
from sklearn.model_selection import train_test_split

# Ruta de la carpeta de imágenes
carpeta_imagenes = 'Food Classification dataset/Taco'

# Definir las rutas para train, test y validation
ruta_train = 'train/Taco'
ruta_test = 'test/Taco'
ruta_validation = 'validation/Taco'

# Obtener la lista de imágenes
lista_imagenes = os.listdir(carpeta_imagenes)

# Dividir las imágenes en conjunto de entrenamiento y conjunto de prueba + validación
lista_train, lista_test_validation = train_test_split(lista_imagenes, test_size=0.3, random_state=42)
lista_test, lista_validation = train_test_split(lista_test_validation, test_size=0.5, random_state=42)

# Función para copiar imágenes de una lista a una carpeta de destino
def copiar_imagenes(lista, origen, destino):
    for imagen in lista:
        shutil.copy(os.path.join(origen, imagen), os.path.join(destino, imagen))

# Copiar las imágenes a los conjuntos correspondientes
copiar_imagenes(lista_train, carpeta_imagenes, ruta_train)
copiar_imagenes(lista_test, carpeta_imagenes, ruta_test)
copiar_imagenes(lista_validation, carpeta_imagenes, ruta_validation)

print("División completa.")