import pandas as pd
import os
from enum import Enum
from pathlib import Path
import sys

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from campos import Campos

class CamposOriginales(str, Enum):
    """Nombres de las columnas del archivo Excel original (carlos.xlsx)."""
    
    CODIGO = "Codigo"
    NUM_FOTOS = "Num_fotos"
    DESCRIPCION_HERB = "Descripcion_herb"
    NUM_FLORES = "Num_flores"
    NUM_ESPECIES_HERBACEAS = "Num_especies_herbaceas"
    DESCRIPCION_ARBUSTOS = "Descripcion_arbustos"
    NUM_ARBUSTOS = "Num_arbustos"
    NUM_ESPECIES_ARBUSTOS = "Num_especies_arbustos"


# 1. Cargar el archivo Excel
df = pd.read_excel('carlos.xlsx')

# 2. Eliminar filas completamente vacías
df = df.dropna(how='all')

# 3. Lista para guardar los datos limpios
datos_limpios = []

# 4. Procesar fila por fila
for index, row in df.iterrows():
    # Saltar filas sin código
    if pd.isna(row[CamposOriginales.CODIGO]):
        continue
        
    # Datos generales del sitio
    codigo_completo = row[CamposOriginales.CODIGO]
    
    # Extraer sitio_id (hasta antes del guion, ej: U1R7-1 -> U1R7)
    sitio_id = codigo_completo.split('-')[0] if '-' in codigo_completo else codigo_completo
    
    # Extraer unidad (ej: U1R7 -> U1)
    unidad = sitio_id.split('R')[0]
    
    # Carlos no tiene fecha/hora en sus datos, usar None
    fecha = None
    hora = None
    area = 1  # 1 metro cuadrado para datos con guion
    
    # Manejo de flores
    flores = row[CamposOriginales.NUM_FLORES] if pd.notna(row[CamposOriginales.NUM_FLORES]) else 0
    
    # A. Revisar si hay Herbácea en esta fila
    herbacea = row[CamposOriginales.DESCRIPCION_HERB]
    if pd.notna(herbacea) and str(herbacea).strip() != '':
        datos_limpios.append({
            Campos.UNIDAD.value: unidad,
            Campos.SITIO_ID.value: sitio_id,
            Campos.FECHA.value: fecha,
            Campos.HORA.value: hora,
            Campos.ESPECIE_GENERO_FAMILIA.value: herbacea,
            Campos.HABITO.value: 'Herbacea',
            Campos.NUM_FLORES.value: flores,
            Campos.NUM_ARBUSTOS.value: 0,
            Campos.AREA_M2.value: area
        })
        
    # B. Revisar si hay Arbusto en esta fila
    arbusto = row[CamposOriginales.DESCRIPCION_ARBUSTOS]
    num_arbustos = row[CamposOriginales.NUM_ARBUSTOS] if pd.notna(row[CamposOriginales.NUM_ARBUSTOS]) else 0
    if pd.notna(arbusto) and str(arbusto).strip() != '':
        datos_limpios.append({
            Campos.UNIDAD.value: unidad,
            Campos.SITIO_ID.value: sitio_id,
            Campos.FECHA.value: fecha,
            Campos.HORA.value: hora,
            Campos.ESPECIE_GENERO_FAMILIA.value: arbusto,
            Campos.HABITO.value: 'Arbusto',
            Campos.NUM_FLORES.value: flores,
            Campos.NUM_ARBUSTOS.value: num_arbustos,
            Campos.AREA_M2.value: area
        })

# 5. Crear la tabla final
df_final = pd.DataFrame(datos_limpios)

# Mostrar las primeras filas
print(df_final.head(10))

# 6. Guardar o agregar a base_de_datos.xlsx
archivo_base = "base_de_datos.xlsx"

if os.path.exists(archivo_base):
    # Si el archivo existe, cargar datos existentes y hacer append
    print(f"\nArchivo '{archivo_base}' encontrado. Agregando nuevos datos...")
    df_existente = pd.read_excel(archivo_base)
    df_combinado = pd.concat([df_existente, df_final], ignore_index=True)
    df_combinado.to_excel(archivo_base, index=False)
    print(f"Se agregaron {len(df_final)} registros. Total: {len(df_combinado)} registros")
else:
    # Si no existe, crear nuevo archivo
    print(f"\nCreando nuevo archivo '{archivo_base}'...")
    df_final.to_excel(archivo_base, index=False)
    print(f"Archivo creado con {len(df_final)} registros")
