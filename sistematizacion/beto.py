import pandas as pd
import os
from enum import Enum
from pathlib import Path
import sys

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from campos import Campos

class CamposOriginales(str, Enum):
    """Nombres de las columnas del archivo Excel original (beto.xlsx)."""
    
    CODIGO = "Código"
    NUM_FOTOS = "Num_fotos"
    DESCRIPCION_HERB = "Descripción_herb"
    NUM_FLORES = "Num_flores"
    NUM_ESPECIES_HERBACEAS = "Num especies herbaceas"
    DESCRIPCION_ARBUSTOS = "Descripción_arbustos"
    NUM_ARBUSTOS = "Num_arbustos"
    NUM_ESPECIES_ARBUSTOS = "Num especies arbustos"
    FECHA = "Fecha"
    HORA = "Hora"


# 1. Cargar el archivo Excel
df = pd.read_excel('beto.xlsx')

# 2. Rellenar los metadatos hacia abajo (Técnica "Fill Down")
# Esto soluciona que U5R3 solo aparezca en la primera fila
df[CamposOriginales.CODIGO] = df[CamposOriginales.CODIGO].ffill()
df[CamposOriginales.FECHA] = df[CamposOriginales.FECHA].ffill()
df[CamposOriginales.HORA] = df[CamposOriginales.HORA].ffill()

# 3. Lista para guardar los datos limpios
datos_limpios = []

# 4. Procesar fila por fila
for index, row in df.iterrows():
    # Datos generales del sitio
    sitio_id = row[CamposOriginales.CODIGO]
    unidad = sitio_id.split('R')[0] # Extrae "U5" de "U5R3" o "U12" de "U12R3"
    fecha = row[CamposOriginales.FECHA]
    hora = row[CamposOriginales.HORA]
    area = 4 # Valor fijo según tu descripción
    
    # Manejo de flores (si está vacío es 0)
    flores = row[CamposOriginales.NUM_FLORES] if pd.notna(row[CamposOriginales.NUM_FLORES]) else 0
    
    # A. Revisar si hay Herbácea en esta fila
    herbacea = row[CamposOriginales.DESCRIPCION_HERB]
    if pd.notna(herbacea):
        datos_limpios.append({
            Campos.UNIDAD: unidad,
            Campos.SITIO_ID: sitio_id,
            Campos.FECHA: fecha,
            Campos.HORA: hora,
            Campos.ESPECIE_GENERO_FAMILIA: herbacea,
            Campos.HABITO: 'Herbacea',
            Campos.NUM_FLORES: flores,
            Campos.AREA_M2: area
        })
        
    # B. Revisar si hay Arbusto en esta fila
    arbusto = row[CamposOriginales.DESCRIPCION_ARBUSTOS]
    if pd.notna(arbusto):
        # Nota: Asumimos que si hay arbusto y herbácea en la misma fila, 
        # las flores pertenecen al registro principal. Si están separadas, mejor.
        datos_limpios.append({
            Campos.UNIDAD: unidad,
            Campos.SITIO_ID: sitio_id,
            Campos.FECHA: fecha,
            Campos.HORA: hora,
            Campos.ESPECIE_GENERO_FAMILIA: arbusto,
            Campos.HABITO: 'Arbusto',
            Campos.NUM_FLORES: flores, # Asignamos las flores de la fila
            Campos.AREA_M2: area
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
