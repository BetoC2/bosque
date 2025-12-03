import pandas as pd
import os

# 1. Cargar el archivo Excel
df = pd.read_excel('beto.xlsx')

# 2. Rellenar los metadatos hacia abajo (Técnica "Fill Down")
# Esto soluciona que U5R3 solo aparezca en la primera fila
df['Código'] = df['Código'].ffill()
df['Fecha'] = df['Fecha'].ffill()
df['Hora'] = df['Hora'].ffill()

# 3. Lista para guardar los datos limpios
datos_limpios = []

# 4. Procesar fila por fila
for index, row in df.iterrows():
    # Datos generales del sitio
    sitio_id = row['Código']
    unidad = sitio_id.split('R')[0] # Extrae "U5" de "U5R3" o "U12" de "U12R3"
    fecha = row['Fecha']
    hora = row['Hora']
    area = 4 # Valor fijo según tu descripción
    
    # Manejo de flores (si está vacío es 0)
    flores = row['Num_flores'] if pd.notna(row['Num_flores']) else 0
    
    # A. Revisar si hay Herbácea en esta fila
    herbacea = row['Descripción_herb']
    if pd.notna(herbacea):
        datos_limpios.append({
            'Unidad': unidad,
            'Sitio_ID': sitio_id,
            'Fecha': fecha,
            'Hora': hora,
            'Especie_Genero_Familia': herbacea,
            'Habito': 'Herbacea',
            'Num_Flores': flores,
            'Area_m2': area
        })
        
    # B. Revisar si hay Arbusto en esta fila
    arbusto = row['Descripción_arbustos']
    if pd.notna(arbusto):
        # Nota: Asumimos que si hay arbusto y herbácea en la misma fila, 
        # las flores pertenecen al registro principal. Si están separadas, mejor.
        datos_limpios.append({
            'Unidad': unidad,
            'Sitio_ID': sitio_id,
            'Fecha': fecha,
            'Hora': hora,
            'Especie_Genero_Familia': arbusto,
            'Habito': 'Arbusto',
            'Num_Flores': flores, # Asignamos las flores de la fila
            'Area_m2': area
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
