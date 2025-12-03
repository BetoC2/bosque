import pandas as pd
import os
from enum import Enum
from pathlib import Path
import sys

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from campos import Campos

class CamposOriginales(str, Enum):
    """Nombres de las columnas del archivo Excel original (sofi.xlsx)."""
    
    CODIGO = "Código"
    NUM_FOTOS = "num_fotos"
    DESCRIPCION_HERB = "Descripcion_herb"
    NUM_FLORES = "num_flores"
    NUM_ESPECIES_HERBACEAS = "num especies herbaceas"
    FECHA = "fecha"
    HORA = "hora"


# 1. Cargar el archivo Excel
df = pd.read_excel('sofi.xlsx')

# 2. Rellenar los metadatos hacia abajo (Técnica "Fill Down")
df[CamposOriginales.CODIGO] = df[CamposOriginales.CODIGO].ffill()
df[CamposOriginales.FECHA] = df[CamposOriginales.FECHA].ffill()
df[CamposOriginales.HORA] = df[CamposOriginales.HORA].ffill()

# 3. Lista para guardar los datos limpios
datos_limpios = []

# 4. Procesar fila por fila
for index, row in df.iterrows():
    # Datos generales del sitio
    codigo_completo = row[CamposOriginales.CODIGO]
    
    # Extraer sitio_id (hasta antes del guion, ej: U5R1-1 -> U5R1)
    sitio_id = codigo_completo.split('-')[0] if '-' in codigo_completo else codigo_completo
    
    # Extraer unidad (ej: U5R1 -> U5)
    unidad = sitio_id.split('R')[0]
    
    fecha = row[CamposOriginales.FECHA]
    hora = row[CamposOriginales.HORA]
    area = 1  # 1 metro cuadrado para datos con guion
    
    # Revisar si hay Herbácea en esta fila
    herbacea = row[CamposOriginales.DESCRIPCION_HERB]
    if pd.notna(herbacea) and str(herbacea).strip() != '':
        # Manejo de flores (si está vacío, es /, o NaN es 0)
        flores_raw = row[CamposOriginales.NUM_FLORES]
        if pd.notna(flores_raw) and str(flores_raw).strip() not in ['/', 'NaN', '']:
            try:
                flores = float(flores_raw)
            except:
                flores = 0
        else:
            flores = 0
            
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
