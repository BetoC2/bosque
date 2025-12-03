"""
Definición de campos (columnas) de la base de datos normalizada.
Este módulo centraliza los nombres de las columnas para evitar errores de tipeo
y facilitar el mantenimiento del código.
"""

from enum import Enum


class Campos(str, Enum):
    """Nombres de las columnas de la base de datos normalizada."""
    
    # Identificadores
    UNIDAD = "Unidad"
    SITIO_ID = "Sitio_ID"
    
    # Temporales
    FECHA = "Fecha"
    HORA = "Hora"
    
    # Taxonomía
    ESPECIE_GENERO_FAMILIA = "Clasificacion"
    HABITO = "Habito"
    
    # Métricas
    NUM_FLORES = "Num_Flores"
    NUM_ARBUSTOS = "Num_Arbustos"
    AREA_M2 = "Area_m2"
    
    # Calculados
    DENSIDAD_PONDERADA = "Densidad_Ponderada"
