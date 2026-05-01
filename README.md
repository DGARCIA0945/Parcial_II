# Parcial_II


## Integrantes
- Nombre 1
- Nombre 2
- Nombre 3
- Nombre 4
- Nombre 5

## Requisitos
- Python 3.10+
- PyTorch
- Google Colab (recomendado)

## Cómo ejecutar

### 1. Instalar dependencias
pip install imbalanced-learn

### 2. Descargar el dataset
Descargar UNSW-NB15 desde:
https://www.kaggle.com/datasets/mrwellsdavid/unsw-nb15

### 3. Correr el notebook
Abrir notebook.ipynb en Google Colab y ejecutar todas las celdas en orden.

### 4. Archivos
- model.py      → Arquitectura del modelo (SelectiveSSM, SpectralBlock, etc.)
- data.py       → Carga y preprocesamiento del dataset
- kalman.py     → Filtro de Kalman para post-procesamiento
- train.py      → Loop de entrenamiento
- evaluate.py   → Cálculo de métricas
- notebook.ipynb → Ejecución completa paso a paso
