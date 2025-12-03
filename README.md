# Predicción de Series Temporales con LSTM + Atención
Microservicio analítico en Python/TensorFlow que entrena modelos **LSTM** y **BiLSTM** con un mecanismo de **Atención** personalizado para predecir precios diarios por producto. Proyecto aplicado de Deep Learning para predicción de series temporales.

## Caracteristicas principales
- **Modelos:** Implementación de LSTM clásico, LSTM Bidireccional y una Capa de Atención personalizada (TensorFlow/Keras).
- **Entrenamiento:** Secuencial por ID de producto con normalización MinMax, ventanas temporales y separación train/val/test.
- **Optimización:** **EarlyStopping** y guardado automático del mejor modelo (en formato `.h5`) por cada producto, basado en la métrica de validación.
- **Métricas:** Cálculo y reporte de **MAE** (Mean Absolute Error) y **RMSE** (Root Mean Squared Error) por serie temporal.
- **Reproducibilidad:** Entorno virtual (`venv`) + `requirements.txt`; opción de **Docker** y **Docker Compose** con JupyterLab para un entorno de desarrollo integrado y reproducible.
- **Análisis:** Notebook principal (`codigo_completo_LSTM_ATTENTION.ipynb`) para análisis, visualización y experimentos detallados.


## Instalacion y ejecucion

### Opción A — Ejecutar SIN Docker (entorno local)
1) **Clonar el repositorio:**
    ```bash
    git clone git@github.com:Camilocs7/Machine-Learning.git
    cd Machine-Learning
    ```
2) **Crear y activar entorno virtual:**
    ```bash
    # Linux/macOS
    python3 -m venv venv
    source venv/bin/activate
    # Windows (ejecutar en PowerShell)
    python -m venv venv
    .\venv\Scripts\activate
    ```
3) **Instalar dependencias:**
    ```bash
    pip install --upgrade pip
    pip install -r requirements.txt
    ```
4) **Verificar TensorFlow y Ejecutar entrenamiento:**
    ```bash
    # Verifica la instalación (opcional)
    python -c "import tensorflow as tf; print(tf.__version__)"
    # Inicia el loop de entrenamiento por producto
    python src/entrenar_lstm_attention_loop.py
    ```

### Opción B — Ejecutar CON Docker (recomendado)
1) **Construir la imagen:**
    ```bash
    docker build -t lstm-attention .
    ```
2) **Levantar el entorno con JupyterLab:**
    ```bash
    docker-compose up
    ```
3) **Acceder y ejecutar:**
    - Abrir el navegador en `http://localhost:8888` (acceso directo, sin token).
    - Abrir el notebook `codigo_completo_LSTM_ATTENTION.ipynb` y ejecutar las celdas.

---

## Datos de entrada esperados
Los archivos de datos **no se incluyen** en el repositorio (ignorados por `.gitignore`) debido a su tamaño.

- **Fuente:** Se pueden descargar desde la fuente en Kaggle:
  👉 [https://www.kaggle.com/datasets/paulohernandezt/price-product-chile/data](https://www.kaggle.com/datasets/paulohernandezt/price-product-chile/data)
- **Archivos Esperados en `data/`:**
  - `precio_2023.csv`, `precio_2024.csv`, `precio_2025.csv`
  - `precios_combinados2.csv`, `precios_estandarizados*.csv`, `precios_final_features2.csv`
- **Preprocesamiento:** Los datos son automáticamente unidos, limpiados y normalizados (`MinMaxScaler`) en el *pipeline* antes de ser usados en la creación de las ventanas temporales para el entrenamiento.

## Arquitectura del Modelo (src/attention_layer.py)


El modelo de predicción de series temporales utiliza una arquitectura secuencial con un mecanismo de atención para mejorar la precisión y la interpretabilidad.

**Componentes principales:**
1.  **Capa de entrada:** Formato de ventana temporal.
2.  **LSTM o BiLSTM:** Extrae características secuenciales de los datos.
3.  **AttentionLayer personalizada:**
    * Calcula dinámicamente **pesos** por cada *timestamp* (paso de tiempo) de la secuencia.
    * Genera un vector de **contexto ponderado** que enfatiza la información relevante para la predicción.
4.  **Capa Dense final:** Capa de regresión para generar el valor de predicción.

**Diagrama simplificado:**

Input (Secuencia) → LSTM/BiLSTM → AttentionLayer → Dense → Output (Predicción)


## Ejecución del Pipeline Completo
El script `entrenar_lstm_attention_loop.py` orquesta el proceso:

1.  Carga y prepara los datos de entrada.
2.  Itera sobre la lista de IDs de productos únicos.
3.  Para cada `product_id`:
    * Genera las ventanas temporales.
    * Construye y compila el modelo (LSTM + Atención).
    * Entrena, aplicando *EarlyStopping*.
    * Evalúa el modelo y registra las métricas.
    * Guarda el modelo óptimo en `lstm_prueba3/models/prod_<ID>/best_model.h5`.

## Resultados y Rendimiento
- **Salida:** Un modelo óptimo (`best_model.h5`) y las métricas **MAE** y **RMSE** se generan por cada serie temporal procesada.
- **Rendimiento:** El uso de la Capa de Atención mejora la capacidad del modelo para identificar patrones cruciales, especialmente en series con alta **variabilidad** o **ruido**, llevando a predicciones más precisas que un LSTM sin atención.
- **Reproducibilidad:** Los resultados son completamente reproducibles utilizando el entorno Docker.

## Autor
Proyecto elaborado por Camilo Cerda Sarabia.
*Proyecto aplicado a predicción de series temporales — 2025.*
