Predicción de Series Temporales con LSTM + Atención

Microservicio analítico en Python/TensorFlow que entrena modelos LSTM y BiLSTM con un mecanismo de Atención personalizado para predecir precios diarios por producto. Proyecto aplicado de Deep Learning para predicción de series temporales.

Características principales

Entrenamiento secuencial por ID de producto.

Normalización MinMax, ventanas temporales y separación train/val/test.

Modelos:

LSTM clásico

LSTM Bidireccional

Capa de Atención personalizada (TensorFlow/Keras)

EarlyStopping + guardado automático del mejor modelo por producto.

Métricas MAE / RMSE por serie.

Ejecución reproducible: entorno virtual + requirements; opción Docker y Docker Compose con JupyterLab incluido.

Notebook principal (codigo_completo_LSTM_ATTENTION.ipynb) para análisis y experimentos.

Estructura del Proyecto
DEEP/
  codigo_completo.ipynb
  codigo_completo_LSTM_ATTENTION.ipynb   # Notebook principal
  src/
    entrenar_lstm_attention_loop.py      # Entrenamiento masivo (loop por producto)
    attention_layer.py                   # Capa de Atención personalizada
    utils.py                             # Preprocesos y funciones auxiliares
  lstm_prueba3/                          # Modelos entrenados (no incluido en Git)
  data/                                  # CSV (ignorados por .gitignore)
  requirements.txt
  docker-compose.yml
  Dockerfile
  README.md

Instalación y ejecución
1) OPCIÓN A — Ejecutar SIN Docker (entorno local)

Clonar el repositorio

git clone git@github.com:Camilocs7/Machine-Learning.git
cd Machine-Learning


Crear y activar entorno virtual

Windows:
python -m venv venv && venv\Scripts\activate

Linux/macOS:

python3 -m venv venv
source venv/bin/activate


Instalar dependencias

pip install --upgrade pip
pip install -r requirements.txt


Verificar TensorFlow

import tensorflow as tf
tf.__version__


Ejecutar entrenamiento

python src/entrenar_lstm_attention_loop.py

2) OPCIÓN B — Ejecutar CON Docker (recomendado)

Construir la imagen

docker build -t lstm-attention .


Levantar el entorno con JupyterLab

docker-compose up


Acceder a:
👉 http://localhost:8888
(sin token, listo para usar)

Abrir:

codigo_completo_LSTM_ATTENTION.ipynb

Datos de entrada esperados

Los datasets no se incluyen en el repositorio por su tamaño (ignorados por .gitignore).

Se pueden descargar desde Kaggle:

👉 https://www.kaggle.com/datasets/paulohernandezt/price-product-chile/data

El pipeline espera archivos como:

precio_2023.csv

precio_2024.csv

precio_2025.csv

precios_combinados2.csv

precios_estandarizados*.csv

precios_final_features2.csv

Los datos se unen, limpian y normalizan antes del entrenamiento.

Arquitectura del Modelo
Componentes principales

Capa de entrada

LSTM o BiLSTM

AttentionLayer personalizada, que:

Calcula pesos por timestamp

Genera un contexto ponderado

Capa Dense final para regresión

Diagrama simplificado
Input → LSTM/BiLSTM → AttentionLayer → Dense → Output

Métricas de Evaluación

El sistema calcula:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

El mejor modelo por ID se guarda automáticamente en:

lstm_prueba3/models/prod_<ID>/best_model.h5

Ejecución del Pipeline Completo

Instalar dependencias

Abrir el notebook principal

Ejecutar el loop:

for product_id in unique_ids:
    train_lstm_attention(product_id)


El sistema entrena, evalúa y guarda un modelo por cada producto.

Resultados

Se genera un modelo óptimo por serie.

Se obtiene MAE y RMSE por producto.

El modelo con atención mejora especialmente en series con patrones complejos y señales ruidosas.

Resultados reproducibles con Docker.

Autor

Camilo Cerda Sarabia
Proyecto aplicado a predicción de series temporales — 2025.
