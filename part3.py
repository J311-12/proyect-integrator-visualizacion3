import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go

# Cargando los datos
df = pd.read_csv("datos_limpios.csv")

# Eliminando la columna que contiene la información si la persona murió o no
df = df.drop(columns=["is_dead"])

# Eliminando la columna categoria_edad
df = df.drop(columns=["edad_categoria"])

# Convirtiendo el DataFrame a un NumPy array
X = df.values

# Obteneniendo el vector objetivo
y = df["is_dead"].values

# Aplicando la reducción de dimensionalidad
X_embedded = TSNE(
    n_components=3,
    learning_rate='auto',
    init='random',
    perplexity=3
).fit_transform(X)

# Creando un gráfico de dispersión 3D con Plotly
fig = go.Figure(data=[
    go.Scatter3D(
        x=X_embedded[:, 0],
        y=X_embedded[:, 1],
        z=X_embedded[:, 2],
        mode="markers",
        marker=dict(
            color=y,
            size=10,
            line=dict(
                width=0.5
            )
        )
    )
])

# Mostrando el gráfico
fig.show()
