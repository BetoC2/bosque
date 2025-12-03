import pandas as pd
import os
from pathlib import Path

# Cambiar al directorio del script
os.chdir(Path(__file__).parent)

# 1. Cargar la base de datos
df = pd.read_excel('base_de_datos.xlsx')

print("=" * 80)
print("CÁLCULO DE DENSIDADES PONDERADAS")
print("=" * 80)

# 2. Calcular densidad por sitio (UXRY)
# Agrupar por Sitio_ID y calcular la suma de flores y arbustos, y tomar el área
df_sitio = df.groupby('Sitio_ID').agg({
    'Num_Flores': 'sum',
    'Num_Arbustos': 'sum',
    'Area_m2': 'first'  # Asumimos que todas las filas del mismo sitio tienen la misma área
}).reset_index()

# Calcular densidades ponderadas por sitio
df_sitio['Densidad_Flores_Sitio'] = df_sitio['Num_Flores'] / df_sitio['Area_m2']
df_sitio['Densidad_Arbustos_Sitio'] = df_sitio['Num_Arbustos'] / df_sitio['Area_m2']

# Extraer la unidad del Sitio_ID
def extraer_unidad(sitio_id):
    """Extrae la unidad del Sitio_ID (ej: U5R3 -> U5, U4-8 -> U4)"""
    if 'R' in sitio_id:
        return sitio_id.split('R')[0]
    elif '-' in sitio_id:
        return sitio_id.split('-')[0]
    else:
        return sitio_id

df_sitio['Unidad'] = df_sitio['Sitio_ID'].apply(extraer_unidad)

print("\nDensidades por sitio (primeros 20):")
print(df_sitio.head(20)[['Sitio_ID', 'Unidad', 'Num_Flores', 'Num_Arbustos', 'Area_m2', 
                          'Densidad_Flores_Sitio', 'Densidad_Arbustos_Sitio']].to_string())

# 3. Calcular densidad promedio por unidad
# Contar cuántos R (sitios) tiene cada unidad
num_sitios_por_unidad = df_sitio.groupby('Unidad')['Sitio_ID'].count().reset_index()
num_sitios_por_unidad.columns = ['Unidad', 'Num_Sitios']

# Sumar las densidades por unidad
df_unidad = df_sitio.groupby('Unidad').agg({
    'Densidad_Flores_Sitio': 'sum',
    'Densidad_Arbustos_Sitio': 'sum'
}).reset_index()

# Unir con el número de sitios
df_unidad = df_unidad.merge(num_sitios_por_unidad, on='Unidad')

# Calcular el promedio dividiendo entre el número de sitios
df_unidad['Densidad_Flores_Unidad'] = df_unidad['Densidad_Flores_Sitio'] / df_unidad['Num_Sitios']
df_unidad['Densidad_Arbustos_Unidad'] = df_unidad['Densidad_Arbustos_Sitio'] / df_unidad['Num_Sitios']

print("\n" + "=" * 80)
print("Densidades promedio por unidad:")
print("=" * 80)
print(df_unidad[['Unidad', 'Num_Sitios', 'Densidad_Flores_Unidad', 'Densidad_Arbustos_Unidad']].to_string())

# 4. Guardar resultados
# Guardar densidades por sitio
df_sitio_export = df_sitio[['Sitio_ID', 'Unidad', 'Num_Flores', 'Num_Arbustos', 'Area_m2', 
                             'Densidad_Flores_Sitio', 'Densidad_Arbustos_Sitio']]
df_sitio_export.to_excel('densidades_por_sitio.xlsx', index=False)
print("\n✓ Archivo 'densidades_por_sitio.xlsx' creado")

# Guardar densidades por unidad
df_unidad_export = df_unidad[['Unidad', 'Num_Sitios', 'Densidad_Flores_Unidad', 'Densidad_Arbustos_Unidad']]
df_unidad_export.to_excel('densidades_por_unidad.xlsx', index=False)
print("✓ Archivo 'densidades_por_unidad.xlsx' creado")

print("\n" + "=" * 80)
print("RESUMEN:")
print(f"Total de sitios procesados: {len(df_sitio)}")
print(f"Total de unidades procesadas: {len(df_unidad)}")
print("=" * 80)
