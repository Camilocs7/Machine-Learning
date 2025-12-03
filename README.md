Predicción de Series Temporales con LSTM + Mecanismo de Atención
Proyecto — Modelos LSTM con Atención aplicados a datos de precios

Este repositorio contiene el código, datos y notebooks utilizados para entrenar y evaluar un modelo de LSTM con mecanismo de Atención para la predicción de series temporales de precios. El foco principal del proyecto es mejorar la capacidad del modelo para aprender dependencias a largo plazo incorporando un módulo de atención personalizado en TensorFlow/Keras.

🧠 Descripción del Proyecto

El objetivo es predecir el valor futuro de un producto usando su historial de precios diarios. Se implementa:

Normalización MinMaxScaler

Generación de ventanas temporales

Modelo LSTM clásico

Modelo LSTM Bidireccional

Capa de Atención personalizada (AttentionLayer)

Entrenamiento con EarlyStopping

Evaluación mediante MAE y RMSE

Guardado automático de los mejores modelos por producto

Ejecución en bucle sobre múltiples IDs de productos

El pipeline está diseñado para procesar múltiples series temporales y generar un modelo óptimo para cada una.

📂 Estructura del Repositorio
📁 DEEP/
│── codigo_completo.ipynb
│── codigo_completo_LSTM_ATTENTION.ipynb   ← Notebook principal del proyecto
│── lstm_prueba3/                           ← Modelos entrenados (uno por producto)
│── mapeo_productos.csv
│── mapeo_productos2.csv
│── mis_datos_finales.csv
│── precio_2023.csv
│── precio_2024.csv
│── precio_2025.csv
│── precios_combinados2.csv
│── precios_estandarizados.csv
│── precios_estandarizados2.csv
│── precios_final_features2.csv
│── requirements.txt
│── README.md   ← (este archivo)



🔧 OPCIÓN A — Ejecutar SIN Docker (entorno local)
1. Clonar el repositorio
git clone git@github.com:Camilocs7/Machine-Learning.git
cd Machine-Learning

2. Crear entorno virtual
python3 -m venv venv
source venv/bin/activate

3. Instalar dependencias
pip install -r requirements.txt

4. Verificar TensorFlow
import tensorflow as tf
tf.__version__





🐳 OPCIÓN B — Ejecutar CON Docker (recomendado)
1. Construir la imagen
docker build -t lstm-attention .

2. Ejecutar el contenedor con JupyterLab
docker-compose up


Esto levanta JupyterLab en:
👉 http://localhost:8888

Sin token, todo listo para usar.




⚙️ Tecnologías Utilizadas

Python 3.x

TensorFlow / Keras

NumPy

Pandas

Scikit-learn

Matplotlib

Joblib

🏗️ Arquitectura del Modelo

El modelo central está compuesto por:

Capa de entrada

LSTM o BiLSTM

Capa de Atención personalizada, que:

Calcula pesos de importancia para cada timestamp

Genera un contexto ponderado

Densas finales para regresión

🔍 Diagrama simplificado
Input → LSTM/BiLSTM → AttentionLayer → Dense → Output

📊 Métricas de Evaluación

El modelo evalúa:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

Automáticamente guarda el modelo con menor pérdida de validación.

🚀 Cómo Ejecutar el Proyecto
1. Instalar dependencias
pip install -r requirements.txt

2. Ejecutar el notebook principal

Abre en Jupyter o VS Code:

codigo_completo_LSTM_ATTENTION.ipynb

3. Entrenar el modelo para todos los productos

El notebook incluye un loop como este:

for product_id in unique_ids:
    train_lstm_attention(product_id)

📁 Datos

Debido al tamaño, los datasets no están incluidos en este repositorio.  
Puedes descargarlos desde Kaggle:

https://www.kaggle.com/datasets/paulohernandezt/price-product-chile/data

El pipeline une y limpia estos datos para generar series de entrenamiento.

🏆 Resultados

Se genera un modelo por producto

Se obtiene MAE y RMSE por cada serie

Los mejores modelos quedan guardados en:

/lstm_prueba3/models/prod_X/best_model.h5


El proyecto permite comparar rendimiento entre LSTM clásico y LSTM con Atención, mostrando mejoras en series con patrones complejos.

📌 Autor

Camilo Cerda Sarabia
Proyecto aplicado a series temporales.# Machine-Learning
