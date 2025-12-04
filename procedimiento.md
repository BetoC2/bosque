checa este camino
al principio lo que es apertura a partir de la tabla de Excel que les pasé

(La columna expuesta debe ser una proporcion normalizada, porque hay un area en donde faltó una pequeña parte y por eso la unidad 5 es menor)

se hace ahora mismo con una sola variable explicatoria, exposición, porque el área en Dosel forma una variable composicional con el de apertura (sumadas dan el área total, en proporciones, suman 1), es decir, no son varibales independientes. por eso toca elegir una, en este caso exposición. Tendremos que ajustar la secciónde hipótesis, objetivos y predicciones al final.


En este análisis, apertura es la proporción real de área abierta dentro del buffer (un valor entre 0 y 1 que representa directamente la disponibilidad de microhábitats luminosos), mientras que apertura_s es simplemente la versión estandarizada de esa proporción, obtenida al restar la media y dividir entre la desviación estándar. Es decir, apertura sí es una proporción, pero apertura_s ya no lo es: se convierte en un valor continuo centrado en cero, útil para que los modelos estadísticos (como GLM Gamma o Binomial Negativa) estimen mejor los coeficientes y eviten problemas de escala entre predictores. Ambas variables son válidas, pero apertura_s se usa típicamente en modelado porque mejora la estabilidad y la interpretación estadística, mientras que apertura como proporción expresa directamente el significado ecológico del gradiente de luz.


📘 **SCRIPT COMPLETO EN PYTHON
Modelo Gamma GLM para densidad_ponderada vs apertura_s**
(con validación estilo DHARMa)

✅ 1. Importar librerías necesarias
# 1. Librerías básicas
import pandas as pd
import numpy as np

# 2. Graficación
import matplotlib.pyplot as plt
import seaborn as sns

# 3. Modelos estadísticos
import statsmodels.api as sm
import statsmodels.formula.api as smf

# 4. Pruebas estadísticas
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as st

✅ 2. Cargar la base de datos
Asegúrate de que tu archivo .csv tenga al menos las columnas:
densidad_ponderada, apertura
df = pd.read_csv("datos_vegetacion.csv")

✅ 3. Preparación y estandarización de la variable predictora
La variable de apertura se estandariza para comparabilidad:
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
df["apertura_s"] = scaler.fit_transform(df[["apertura"]])

✅ 4. Ajuste del modelo Gamma GLM
(el mejor para una variable positiva continua como densidad_ponderada)
model = smf.glm(
    formula="densidad_ponderada ~ apertura_s",
    data=df,
    family=sm.families.Gamma(sm.families.links.log())
).fit()

print(model.summary())

📌 Interpretación rápida

Interpretación del coeficiente
beta = model.params["apertura_s"]
print("β apertura =", beta)
print("exp(β) =", np.exp(beta))
Si por ejemplo exp(β)=1.28, significa:
“Por cada aumento de 1 desviación estándar en apertura del dosel,
la densidad aumenta un 28%”.

✅ 5. Predicción y visualización del efecto
# Crear un rango de valores de apertura estandarizada
new = pd.DataFrame({
    "apertura_s": np.linspace(df.apertura_s.min(),
                              df.apertura_s.max(), 100)
})

# Predicciones con intervalo de confianza
pred = model.get_prediction(new)
pred_summary = pred.summary_frame()

# Gráfica
plt.figure(figsize=(8,5))
plt.plot(new.apertura_s, pred_summary["mean"], label="Predicción media")
plt.fill_between(
    new.apertura_s,
    pred_summary["mean_ci_lower"],
    pred_summary["mean_ci_upper"],
    alpha=0.3, label="IC 95%"
)

plt.xlabel("Apertura del dosel (estandarizada)")
plt.ylabel("Densidad ponderada de inflorescencias")
plt.title("Efecto de la apertura del dosel en la densidad de inflorescencias")
plt.legend()
plt.show()

✅ 6. Validación del modelo (estilo DHARMa para Python)
6.1 Residuos y valores ajustados
resid_dev = model.resid_deviance.copy()
fitted = model.fittedvalues.copy()

6.2 QQ-Plot de residuos
st.probplot(resid_dev, dist="norm", plot=plt)
plt.title("QQ-plot de residuos (Gamma GLM)")
plt.show()
Interpretación:
	Línea recta → modelo bien especificado
	Curvatura → revisar familia / transformaciones

6.3 Residuos vs valores ajustados
plt.scatter(fitted, resid_dev)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Valores ajustados")
plt.ylabel("Residuos de desviación")
plt.title("Residuos vs Ajustados")
plt.show()
Interpretación:
	Sin patrón → buena homogeneidad
	Forma de U o ∧ → falta predictor o mala distribución

6.4 Test de heterocedasticidad
bp = het_breuschpagan(resid_dev, sm.add_constant(fitted))
print("Breusch–Pagan p-value =", bp[1])
Interpretación:
	p > 0.05 → no hay heterocedasticidad fuerte
	p < 0.05 → revisar modelo

6.5 Identificación de puntos influyentes
influence = model.get_influence()
cooks = influence.cooks_distance[0]

plt.stem(cooks, use_line_collection=True)
plt.xlabel("Observación")
plt.ylabel("Distancia de Cook")
plt.title("Observaciones influyentes")
plt.show()
Interpretación:
	Valores > 1 → posible punto influyente
	Revisar si corresponde a error o outlier ecológico real

✅ 7. Interpretación final del modelo
Incluye:
	Signo del coeficiente ✔️
	Magnitud del efecto (exp(beta)) ✔️
	Significancia estadística ✔️
	Concordancia con hipótesis ecológicas
	Cumplimiento de supuestos (validación)
Ejemplo:
“El modelo Gamma con enlace log mostró un efecto positivo y significativo de la apertura del dosel sobre la densidad ponderada de inflorescencias (β = 0.25, p = 0.041). Esto implica que un incremento de 1 DE en apertura se traduce en un aumento del 28% en la densidad promedio de inflorescencias por m². Los residuos no presentaron patrones sistemáticos, y el test de Breusch–Pagan no indicó heterocedasticidad (p = 0.32), por lo que consideramos que el modelo está bien especificado.”
SCRIPT:

# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as st


# ============================================================
# 2. CARGAR BASE DE DATOS
# ============================================================
# Debe contener al menos:
# densidad_ponderada, apertura

df = pd.read_csv("datos_vegetacion.csv")

print("Columnas disponibles:", df.columns)
print(df.head())


# ============================================================
# 3. PREPARAR Y ESTANDARIZAR VARIABLE DE APERTURA
# ============================================================

scaler = StandardScaler()
df["apertura_s"] = scaler.fit_transform(df[["apertura"]])


# ============================================================
# 4. AJUSTE DEL MODELO GAMMA GLM (ENLACE LOG)
# ============================================================

model = smf.glm(
    formula="densidad_ponderada ~ apertura_s",
    data=df,
    family=sm.families.Gamma(sm.families.links.log())
).fit()

print("\n\n=== RESUMEN DEL MODELO GAMMA ===")
print(model.summary())


# ============================================================
# 5. INTERPRETACIÓN DEL EFECTO
# ============================================================

beta = model.params["apertura_s"]
exp_beta = np.exp(beta)

print("\nCoeficiente β apertura_s =", beta)
print("Interpretación multiplicativa exp(β) =", exp_beta)
print("→ Un incremento de 1 DE en apertura cambia la densidad en un factor de", round(exp_beta, 3))


# ============================================================
# 6. PREDICCIÓN Y GRÁFICA DEL EFECTO
# ============================================================

# Crear rango de apertura estandarizada
new = pd.DataFrame({
    "apertura_s": np.linspace(df.apertura_s.min(),
                              df.apertura_s.max(), 100)
})

# Obtener predicciones con IC95
pred = model.get_prediction(new)
pred_summary = pred.summary_frame()

# Graficar
plt.figure(figsize=(8,5))
plt.plot(new.apertura_s, pred_summary["mean"], label="Predicción media")
plt.fill_between(
    new.apertura_s,
    pred_summary["mean_ci_lower"],
    pred_summary["mean_ci_upper"],
    alpha=0.3,
    label="IC 95%"
)

plt.xlabel("Apertura del dosel (estandarizada)")
plt.ylabel("Densidad ponderada de inflorescencias")
plt.title("Efecto de la apertura del dosel en la densidad de inflorescencias")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. VALIDACIÓN DEL MODELO (ESTILO DHARMa)
# ============================================================

# ----------------------------
# 7.1 Obtener residuos y ajustados
# ----------------------------
resid_dev = model.resid_deviance.copy()
fitted = model.fittedvalues.copy()

# ----------------------------
# 7.2 QQ-PLOT DE RESIDUOS
# ----------------------------
plt.figure(figsize=(6,6))
st.probplot(resid_dev, dist="norm", plot=plt)
plt.title("QQ-plot de residuos (Gamma GLM)")
plt.tight_layout()
plt.show()

# ----------------------------
# 7.3 RESIDUOS VS AJUSTADOS
# ----------------------------
plt.figure(figsize=(7,5))
plt.scatter(fitted, resid_dev)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Valores ajustados")
plt.ylabel("Residuos de desviación")
plt.title("Residuos vs Ajustados (homogeneidad)")
plt.tight_layout()
plt.show()

# ----------------------------
# 7.4 TEST DE HETEROCEDASTICIDAD (Breusch–Pagan)
# ----------------------------
bp = het_breuschpagan(resid_dev, sm.add_constant(fitted))
print("\nBreusch–Pagan p-value:", bp[1])
if bp[1] > 0.05:
    print("✔ No se detecta heterocedasticidad fuerte.")
else:
    print("⚠ Posible heterocedasticidad, revisar modelo.")

# ----------------------------
# 7.5 PUNTOS INFLUYENTES (Cook's Distance)
# ----------------------------
influence = model.get_influence()
cooks = influence.cooks_distance[0]

plt.figure(figsize=(8,4))
plt.stem(cooks, use_line_collection=True)
plt.xlabel("Observación")
plt.ylabel("Distancia de Cook")
plt.title("Observaciones influyentes")
plt.tight_layout()
plt.show()

high_influence = np.where(cooks > 1)[0]
print("\nObservaciones influyentes (Cook > 1):", high_influence)


# ============================================================
# 8. RESUMEN DE INTERPRETACIÓN FINAL
# ============================================================

print("\n\n=== INTERPRETACIÓN DEL MODELO ===\n")

if model.pvalues["apertura_s"] < 0.05:
    print("✔ La apertura del dosel es un predictor significativo de la densidad ponderada.")
else:
    print("⚠ La apertura del dosel NO es significativa (p > 0.05). Interpretación con cautela.")

print("""
Interpretación ecológica sugerida:
- Si β > 0 → la apertura del dosel incrementa la densidad de inflorescencias.
- Si β < 0 → las zonas más sombreadas presentan mayor densidad.
- El valor exp(β) indica el cambio proporcional por 1 desviación estándar de apertura.

Revisar:
• QQ-plot para normalidad de residuos.
• Residuos vs Ajustados para homogeneidad.
• Breusch–Pagan para heterocedasticidad.
• Cook’s distance para puntos extremos o errores de muestreo.

""")


# ============================================================
# 9. MENSAJE FINAL
# ============================================================

print(">>> Script completado correctamente.")





# Este ahora es otro camino que podamos tomar:

Párrafo para incluir en Métodos (Análisis estadístico)
Para evaluar si las diferencias estructurales entre los dos sitios de muestreo (Planillas y Planillas Sur) modifican la relación entre la apertura del dosel y la densidad ponderada de inflorescencias, se incorporó la variable perturbación como un factor categórico dentro del modelo Gamma GLM. Dado que la perturbación representa condiciones discretas del ecosistema (alta perturbación vs. menor perturbación) y no un gradiente continuo, se codificó explícitamente como variable categórica. En el ajuste del GLM, el software genera automáticamente una comparación entre niveles mediante un coeficiente estimado para el nivel no-referente (Planillas Sur), lo que permite interpretar si la densidad ponderada difiere entre sitios incluso después de controlar por la apertura del dosel. La inclusión de esta variable no funciona como un offset (pues no corrige esfuerzo de muestreo), sino como un predictor fijo que incorpora diferencias estructurales del paisaje relevantes para el comportamiento del sotobosque. Este enfoque permite comparar directamente el efecto marginal de la apertura del dosel y distinguir si la perturbación modifica la densidad esperada.


# ============================================================
# 1. IMPORTAR LIBRERÍAS
# ============================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm
import statsmodels.formula.api as smf

from sklearn.preprocessing import StandardScaler
from statsmodels.stats.diagnostic import het_breuschpagan
import scipy.stats as st


# ============================================================
# 2. CARGAR BASE DE DATOS
# ============================================================
# Debe contener:
# densidad_ponderada, apertura, perturbacion

df = pd.read_csv("datos_vegetacion.csv")

print("Columnas disponibles:", df.columns)
print(df.head())


# ============================================================
# 3. PREPARAR VARIABLES
# ============================================================

# --- Estandarizar apertura ---
scaler = StandardScaler()
df["apertura_s"] = scaler.fit_transform(df[["apertura"]])

# --- Convertir perturbación en variable categórica ---
# Se espera que tenga valores como: "Planillas" y "PlanillasSur"
df["perturbacion"] = df["perturbacion"].astype("category")

print("\nNiveles de perturbación:", df["perturbacion"].cat.categories)


# ============================================================
# 4. AJUSTE DEL MODELO GAMMA GLM (ENLACE LOG)
# ============================================================

model = smf.glm(
    formula="densidad_ponderada ~ apertura_s + perturbacion",
    data=df,
    family=sm.families.Gamma(sm.families.links.log())
).fit()

print("\n\n=== RESUMEN DEL MODELO GAMMA (con perturbación) ===")
print(model.summary())


# ============================================================
# 5. INTERPRETACIÓN DE EFECTOS DEL MODELO
# ============================================================

# Coeficiente apertura
beta_ap = model.params["apertura_s"]
exp_beta_ap = np.exp(beta_ap)

print("\n--- EFECTO DE APERTURA ---")
print("β (apertura_s) =", beta_ap)
print("exp(β) =", exp_beta_ap)
print("Interpretación: un aumento de 1 DE en apertura cambia la densidad en un factor de",
      round(exp_beta_ap, 3))

# Coeficiente perturbación (nivel no base)
pert_name = model.params.index[2]   # nombre automático del factor
beta_pert = model.params[pert_name]
exp_beta_pert = np.exp(beta_pert)

print("\n--- EFECTO DE PERTURBACIÓN ---")
print(f"Coeficiente para {pert_name} =", beta_pert)
print("exp(β) =", exp_beta_pert)
print(f"Interpretación: la unidad categórica '{pert_name}' tiene una densidad",
      round(exp_beta_pert, 3),
      "veces la densidad del nivel de referencia.")


# ============================================================
# 6. PREDICCIÓN Y GRÁFICA DEL EFECTO (MANTENIENDO PERTURBACIÓN FIJA)
# ============================================================

# Nivel base de perturbación
base = df["perturbacion"].cat.categories[0]

new = pd.DataFrame({
    "apertura_s": np.linspace(df.apertura_s.min(), df.apertura_s.max(), 100),
    "perturbacion": base  # fijamos el nivel base
})

pred = model.get_prediction(new)
pred_summary = pred.summary_frame()

plt.figure(figsize=(8,5))
plt.plot(new.apertura_s, pred_summary["mean"], label=f"Predicción ({base})")

plt.fill_between(
    new.apertura_s,
    pred_summary["mean_ci_lower"],
    pred_summary["mean_ci_upper"],
    alpha=0.3,
    label="IC 95%"
)

plt.xlabel("Apertura del dosel (estandarizada)")
plt.ylabel("Densidad ponderada de inflorescencias")
plt.title("Efecto de la apertura del dosel (controlando perturbación)")
plt.legend()
plt.tight_layout()
plt.show()


# ============================================================
# 7. VALIDACIÓN DEL MODELO (ESTILO DHARMa)
# ============================================================

resid_dev = model.resid_deviance.copy()
fitted = model.fittedvalues.copy()

# ---- QQ-plot ----
plt.figure(figsize=(6,6))
st.probplot(resid_dev, dist="norm", plot=plt)
plt.title("QQ-plot de residuos (Gamma GLM)")
plt.tight_layout()
plt.show()

# ---- Residuos vs Ajustados ----
plt.figure(figsize=(7,5))
plt.scatter(fitted, resid_dev)
plt.axhline(0, color="black", linestyle="--")
plt.xlabel("Valores ajustados")
plt.ylabel("Residuos de desviación")
plt.title("Residuos vs Ajustados")
plt.tight_layout()
plt.show()

# ---- Breusch–Pagan ----
bp = het_breuschpagan(resid_dev, sm.add_constant(fitted))
print("\nBreusch–Pagan p-value:", bp[1])
if bp[1] > 0.05:
    print("✔ No se detecta heterocedasticidad fuerte.")
else:
    print("⚠ Posible heterocedasticidad, revisar modelo.")

# ---- Cook's Distance ----
influence = model.get_influence()
cooks = influence.cooks_distance[0]

plt.figure(figsize=(8,4))
plt.stem(cooks, use_line_collection=True)
plt.xlabel("Observación")
plt.ylabel("Distancia de Cook")
plt.title("Observaciones influyentes")
plt.tight_layout()
plt.show()

high_influence = np.where(cooks > 1)[0]
print("\nObservaciones influyentes (Cook > 1):", high_influence)


# ============================================================
# 8. RESUMEN FINAL
# ============================================================

print("\n\n=== INTERPRETACIÓN DEL MODELO (RESUMEN) ===\n")

if model.pvalues["apertura_s"] < 0.05:
    print("✔ Apertura del dosel es significativa.")
else:
    print("⚠ Apertura NO es significativa.")

if model.pvalues[pert_name] < 0.05:
    print(f"✔ Perturbación ({pert_name}) es significativa.")
else:
    print(f"⚠ Perturbación ({pert_name}) NO es significativa.")

print("""
Interpretación ecológica:
- El modelo Gamma permite capturar la asimetría típica de densidades ecológicas.
- exp(β) describe cambios proporcionales, más realistas que diferencias lineales.
- Perturbación se interpreta como diferencia estructural entre Planillas y Planillas Sur.
""")

print(">>> Script completado correctamente.")

Ejemplo de tabla resultado:

Variable respuesta,Modelo (familia),Variables explicativas,Coeficiente (β),Error estándar,z,p,AIC
Densidad ponderada de inflorescencias,GLM Gamma (log),Intercepto,-0.742,0.31,-2.39,0.017,112.4
,,Apertura_s,0.523,0.192,2.72,0.006,
,,PC1_clima,-0.214,0.144,-1.48,0.138,
,,Perturbación,-0.331,0.201,-1.64,0.101,
Densidad ponderada de arbustos,GLMM Binomial Negativa,Intercepto,1.212,0.28,4.33,<0.001,124.8
,,Elevación_s,0.487,0.165,2.95,0.003,
,,PC1_clima,-0.302,0.171,-1.76,0.078,
,,Perturbación,-0.611,0.204,-3,0.003,
Riqueza observada de herbáceas,GLMM Binomial Negativa,Intercepto,0.984,0.25,3.94,<0.001,118.2
,,Apertura_s,0.441,0.15,2.94,0.003,
,,PC1_clima,-0.265,0.13,-2.04,0.041,
,,Perturbación,-0.223,0.18,-1.24,0.215,
Riqueza observada de arbustos,GLMM Binomial Negativa,Intercepto,1.601,0.29,5.51,<0.001,131.7
,,Elevación_s,0.592,0.18,3.29,0.001,
,,PC1_clima,-0.154,0.142,-1.08,0.28,
,,Perturbación,-0.701,0.229,-3.06,0.002,