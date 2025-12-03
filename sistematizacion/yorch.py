import pandas as pd
import os
from enum import Enum
from pathlib import Path
import sys

# Agregar el directorio padre al path
sys.path.append(str(Path(__file__).parent.parent))

from campos import Campos

class CamposOriginales(str, Enum):
    """Nombres de las columnas del archivo Excel original (yorch.xlsx)."""
    
    CODIGO = "Código"
    DESCRIPCION_HERB = "Descripción herb"
    NUM_FLORES = "Num flores"
    NUM_ESPECIES_HERB = "Num especies herb"
    DESCRIPCION_ARB = "Descripción arb"
    NUM_ESPECIES_ARB = "Num especies arb"
    NUM_ARBUSTOS = "Num de arbustos"
    FECHA = "Fecha"
    HORA = "Hora"
    FOTO = "Foto"
    ESPECIES = "Especies"


# 1. Cargar el archivo Excel
df = pd.read_excel('yorch.xlsx')

# 2. Lista para guardar los datos limpios
datos_limpios = []

# 3. Procesar fila por fila
for index, row in df.iterrows():
    # Datos generales del sitio
    codigo_completo = row[CamposOriginales.CODIGO]
    
    # Extraer sitio_id y unidad
    # El formato de Yorch: U4-8-1 -> sitio_id = U4R8, unidad = U4
    # O formato U3-R4-1 -> sitio_id = U3R4, unidad = U3
    partes = codigo_completo.split('-')
    if len(partes) >= 3:
        # Formato U4-8-1: unidad es U4, sitio_id es U4R8 (sin guion)
        unidad = partes[0]
        sitio_id = f"{partes[0]}R{partes[1]}"
    elif len(partes) == 2:
        # Formato U3-R4-1: extraer de manera similar
        if 'R' in partes[1]:
            # Ya tiene R en el nombre (U3-R4-1)
            sitio_id = f"{partes[0]}{partes[1]}"  # U3R4
            unidad = partes[0]  # U3
        else:
            unidad = partes[0]
            sitio_id = f"{partes[0]}R{partes[1]}"
    else:
        # Fallback
        unidad = partes[0]
        sitio_id = codigo_completo
    
    fecha = row[CamposOriginales.FECHA] if pd.notna(row[CamposOriginales.FECHA]) else None
    hora = row[CamposOriginales.HORA] if pd.notna(row[CamposOriginales.HORA]) else None
    area = 1  # 1 metro cuadrado
    
    # Obtener el número de flores (si existe)
    flores = row[CamposOriginales.NUM_FLORES] if pd.notna(row[CamposOriginales.NUM_FLORES]) else 0
    
    # Obtener especies de la columna "Especies"
    especies_texto = row[CamposOriginales.ESPECIES]
    
    # Si hay especies en la columna "Especies", procesarlas
    if pd.notna(especies_texto) and str(especies_texto).strip() != '':
        # Dividir por saltos de línea para obtener cada especie
        especies_lista = [esp.strip() for esp in str(especies_texto).split('\n') if esp.strip()]
        
        # Determinar qué especies son herbáceas y cuáles son arbustos
        # Usamos las descripciones para ayudar a clasificar
        desc_herb = row[CamposOriginales.DESCRIPCION_HERB] if pd.notna(row[CamposOriginales.DESCRIPCION_HERB]) else ''
        desc_arb = row[CamposOriginales.DESCRIPCION_ARB] if pd.notna(row[CamposOriginales.DESCRIPCION_ARB]) else ''
        num_especies_herb = row[CamposOriginales.NUM_ESPECIES_HERB] if pd.notna(row[CamposOriginales.NUM_ESPECIES_HERB]) else 0
        num_especies_arb = row[CamposOriginales.NUM_ESPECIES_ARB] if pd.notna(row[CamposOriginales.NUM_ESPECIES_ARB]) else 0
        num_arbustos = row[CamposOriginales.NUM_ARBUSTOS] if pd.notna(row[CamposOriginales.NUM_ARBUSTOS]) else 0
        
        # Identificar especies herbáceas y arbustos basándonos en palabras clave
        especies_herbaceas = []
        especies_arbustos = []
        
        for especie in especies_lista:
            especie_lower = especie.lower()
            # Clasificar como arbusto si contiene palabras clave de arbustos
            if any(keyword in especie_lower for keyword in ['arbusto', 'cabello', 'melinis', 'muhlenbergia', 'mimosa', 'wigandia', 'verbesina']):
                especies_arbustos.append(especie)
            else:
                # Por defecto, clasificar como herbácea (flores)
                especies_herbaceas.append(especie)
        
        # Crear registros para especies herbáceas
        # IMPORTANTE: Solo la primera especie herbácea lleva el conteo de flores
        # para evitar duplicar el conteo total
        for idx, especie in enumerate(especies_herbaceas):
            datos_limpios.append({
                Campos.UNIDAD.value: unidad,
                Campos.SITIO_ID.value: sitio_id,
                Campos.FECHA.value: fecha,
                Campos.HORA.value: hora,
                Campos.ESPECIE_GENERO_FAMILIA.value: especie,
                Campos.HABITO.value: 'Herbacea',
                Campos.NUM_FLORES.value: flores if idx == 0 else 0,  # Solo la primera lleva las flores
                Campos.NUM_ARBUSTOS.value: 0,
                Campos.AREA_M2.value: area
            })
        
        # Crear registros para especies arbustos
        # IMPORTANTE: Solo el primer arbusto lleva el conteo de arbustos
        # para evitar duplicar el conteo total
        for idx, especie in enumerate(especies_arbustos):
            datos_limpios.append({
                Campos.UNIDAD.value: unidad,
                Campos.SITIO_ID.value: sitio_id,
                Campos.FECHA.value: fecha,
                Campos.HORA.value: hora,
                Campos.ESPECIE_GENERO_FAMILIA.value: especie,
                Campos.HABITO.value: 'Arbusto',
                Campos.NUM_FLORES.value: 0,
                Campos.NUM_ARBUSTOS.value: num_arbustos if idx == 0 else 0,  # Solo el primero lleva los arbustos
                Campos.AREA_M2.value: area
            })
    else:
        # Si no hay especies en la columna "Especies", intentar usar las descripciones
        # A. Revisar si hay Herbácea en la descripción
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
        
        # B. Revisar si hay Arbusto en la descripción
        arbusto = row[CamposOriginales.DESCRIPCION_ARB]
        num_arbustos = row[CamposOriginales.NUM_ARBUSTOS] if pd.notna(row[CamposOriginales.NUM_ARBUSTOS]) else 0
        if pd.notna(arbusto) and str(arbusto).strip() != '':
            datos_limpios.append({
                Campos.UNIDAD.value: unidad,
                Campos.SITIO_ID.value: sitio_id,
                Campos.FECHA.value: fecha,
                Campos.HORA.value: hora,
                Campos.ESPECIE_GENERO_FAMILIA.value: arbusto,
                Campos.HABITO.value: 'Arbusto',
                Campos.NUM_FLORES.value: 0,
                Campos.NUM_ARBUSTOS.value: num_arbustos,
                Campos.AREA_M2.value: area
            })

# 4. Crear la tabla final
df_final = pd.DataFrame(datos_limpios)

# Mostrar las primeras filas
print(df_final.head(20))
print(f"\nTotal de registros creados: {len(df_final)}")

# 5. Guardar o agregar a base_de_datos.xlsx
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
