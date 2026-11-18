import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# 1. Crear un diccionario con diez listas y seis elementos por lista.
inventario_medicamentos = {
    'id_medicamento': [501, 502, 503, 504, 505, 506],
    'nombre_comercial': ['Aspirina', 'Amoxicilina', 'Omeprazol', 'Ibuprofeno', 'Loratadina', 'Metformina'],
    'principio_activo': ['Acido acetilsalicilico', 'Amoxicilina', 'Omeprazol', 'Ibuprofeno', 'Loratadina', 'Metformina'],
    'presentacion': ['Tabletas', 'Capsulas', 'Capsulas', 'Tabletas', 'Jarabe', 'Tabletas'],
    'stock_actual': [150, 80, 200, 120, 90, 110],
    'precio_unitario': [45.50, 120.00, 85.00, 65.00, 95.00, 150.00],
    'fecha_caducidad': ['2025-06', '2024-12', '2026-01', '2025-03', '2024-08', '2025-11'],
    'proveedor': ['Farmax', 'SaludTotal', 'MediCorp', 'Farmax', 'BioGen', 'MediCorp'],
    'requiere_receta': [False, True, False, False, False, True],
    'ubicacion_estante': ['A1', 'B3', 'A2', 'C1', 'B2', 'C3']
}
print("Analisís exploratorio de data EDA")
print("Analisís exploratorio de datos (EDA, Exploratory DataAnalysis) por Meredith Aguirre NC = 0013")
print("5. Crear el diccionario con 10 listas y 6 elementos")
# 2. Convertir el diccionario en un DataFrame.
df_medicamentos = pd.DataFrame(inventario_medicamentos)
print("\n--- DATAFRAME DEL INVENTARIO 0013---")
print(df_medicamentos)

#3. Explorar la estructura de los datos.
print("\n--- DIMENSIONES DEL DATAFRAME 0013---")
print("Filas y columnas:", df_medicamentos.shape)
print("\n--- PRIMEROS REGISTROS ---")
print(df_medicamentos.head())
print("\n--- INFORMACIÓN GENERAL ---")
df_medicamentos.info()
# 4. Guardar la información en un archivo CSV.
df_medicamentos.to_csv(
    "inventario_medicamentos_0013.csv",
    index=False,
    encoding="utf-8-sig"
)
print("\nArchivo CSV guardado correctamente.")
# 5. Leer nuevamente los datos desde el CSV.
df_medicamentos = pd.read_csv("inventario_medicamentos_0013.csv")
print("\n--- DATOS LEÍDOS DESDE EL CSV 0013---")
print(df_medicamentos)
# 6. Identificar tipos de datos, valores faltantes y estadísticas.
# Convertir la fecha de caducidad a formato fecha
df_medicamentos["fecha_caducidad"] = pd.to_datetime(
    df_medicamentos["fecha_caducidad"],
    format="%Y-%m"
)
print("\n--- TIPOS DE DATOS 0013---")
print(df_medicamentos.dtypes)
print("\n--- VALORES FALTANTES 0013---")
print(df_medicamentos.isnull().sum())
print("\n--- ESTADÍSTICAS DESCRIPTIVAS 0013---")
print(df_medicamentos.describe())

# 7. Ordenar, filtrar y agrupar información.
# Ordenar medicamentos por stock de mayor a menor
print("\n--- MEDICAMENTOS ORDENADOS POR STOCK 0013---")
ordenados = df_medicamentos.sort_values(
    by="stock_actual",
    ascending=False
)
print(ordenados)
# Filtrar medicamentos con menos de 100 unidades
print("\n--- MEDICAMENTOS CON STOCK MENOR A 100 ---")
filtro = df_medicamentos[
    df_medicamentos["stock_actual"] < 100
]
print(filtro)
# Agrupar medicamentos por proveedor
print("\n--- AGRUPACIÓN POR PROVEEDOR ---")
agrupacion = df_medicamentos.groupby("proveedor").agg(
    cantidad=("id_medicamento", "count"),
    stock_total=("stock_actual", "sum"),
    precio_promedio=("precio_unitario", "mean")
)
print(agrupacion)

# 8. Crear gráficas para interpretar los datos.
# Gráfica de stock por medicamento
plt.figure(figsize=(10, 5))
plt.bar(
    df_medicamentos["nombre_comercial"],
    df_medicamentos["stock_actual"],
    color="skyblue"
)
plt.title("Gráfica de barras: Medicamentos por stock")
plt.xlabel("Nombre del medicamento")
plt.ylabel("Stock actual")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 5))
plt.bar(
    df_medicamentos["nombre_comercial"],
    df_medicamentos["precio_unitario"],
    color="orange"
)
plt.title("Gráfica de barras: Precios por medicamento")
plt.xlabel("Nombre del medicamento")
plt.ylabel("Precio unitario ($)")
plt.xticks(rotation=35)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.hist(
    df_medicamentos["stock_actual"],
    bins=5,
    color="lightgreen",
    edgecolor="black"
)
plt.title("Histograma: Distribución del stock")
plt.xlabel("Cantidad de unidades")
plt.ylabel("Frecuencia")
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.scatter(
    df_medicamentos["precio_unitario"],
    df_medicamentos["stock_actual"],
    color="red",
    s=100
)
plt.title("Gráfica de dispersión: Precio vs stock")
plt.xlabel("Precio unitario ($)")
plt.ylabel("Stock actual")
plt.grid(True)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 5))
plt.boxplot(
    df_medicamentos["stock_actual"],
    patch_artist=True
)
plt.title("Diagrama de caja: Distribución del stock")
plt.ylabel("Cantidad de unidades")
plt.tight_layout()
plt.show()
# Seleccionar las variables numéricas de interés
datos_numericos = df_medicamentos[
    ["stock_actual", "precio_unitario"]
]
# Calcular correlación
correlacion = datos_numericos.corr()
plt.figure(figsize=(7, 5))
plt.imshow(
    correlacion,
    cmap="coolwarm",
    vmin=-1,
    vmax=1
)
plt.colorbar(label="Nivel de correlación")
plt.xticks(
    range(len(correlacion.columns)),
    correlacion.columns
)
plt.yticks(
    range(len(correlacion.columns)),
    correlacion.columns
)
# Mostrar valores dentro de la matriz
for i in range(len(correlacion.columns)):
    for j in range(len(correlacion.columns)):
        plt.text(
            j,
            i,
            round(correlacion.iloc[i, j], 2),
            ha="center",
            va="center",
            color="black"
        )
plt.title("Matriz de correlación: Inventario de medicamentos")
plt.tight_layout()
plt.show()