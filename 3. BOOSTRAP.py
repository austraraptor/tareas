# =============================================================================
# REGRESIÓN LINEAL + BOOTSTRAPPING (DATASET REAL: datossalarios.xlsx)
# =============================================================================

# ── LIBRERÍAS ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

np.random.seed(42)
sns.set(style="whitegrid")

# =============================================================================
# 1. CARGA DE DATOS
# =============================================================================
df = pd.read_excel("D:/SEMESTRE 7 2026/APRENDIZAJE SUPERVISADO/regresion/datossalarios.xlsx")

print("Dimensiones:", df.shape)
print(df.head())

# =============================================================================
# 2. PREPROCESAMIENTO
# =============================================================================
target = "salario"

# Convertir variables categóricas (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=[target])
y = df[target]

# División train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# 3. MODELO BASE
# =============================================================================
model_base = LinearRegression()
model_base.fit(X_train_sc, y_train)

y_pred_base = model_base.predict(X_test_sc)

print("\n=== MODELO BASE ===")
print(f"R2   : {r2_score(y_test, y_pred_base):.4f}")
print(f"RMSE : {np.sqrt(mean_squared_error(y_test, y_pred_base)):.4f}")

# =============================================================================
# 4. BOOTSTRAPPING
# =============================================================================
N_BOOTSTRAP = 500   # puedes subir a 1000 si tu PC lo soporta

coefs_boot = []
intercepts_boot = []
r2_boot = []
rmse_boot = []

for _ in range(N_BOOTSTRAP):

    # Remuestreo SOLO en train (correcto)
    X_res, y_res = resample(X_train_sc, y_train)

    model = LinearRegression()
    model.fit(X_res, y_res)

    # Evaluación en TEST (clave para evitar sobreajuste)
    y_pred = model.predict(X_test_sc)

    coefs_boot.append(model.coef_)
    intercepts_boot.append(model.intercept_)
    r2_boot.append(r2_score(y_test, y_pred))
    rmse_boot.append(np.sqrt(mean_squared_error(y_test, y_pred)))

# Convertir a arrays
coefs_boot = np.array(coefs_boot)
intercepts_boot = np.array(intercepts_boot)
r2_boot = np.array(r2_boot)
rmse_boot = np.array(rmse_boot)

# =============================================================================
# 5. INTERVALOS DE CONFIANZA
# =============================================================================
print("\n=== INTERVALOS DE CONFIANZA (95%) ===")

nombres = X.columns

for i, nombre in enumerate(nombres):
    ic_inf = np.percentile(coefs_boot[:, i], 2.5)
    ic_sup = np.percentile(coefs_boot[:, i], 97.5)

    print(f"{nombre}: [{ic_inf:.4f}, {ic_sup:.4f}]")

# =============================================================================
# 6. MÉTRICAS BOOTSTRAP
# =============================================================================
print("\n=== MÉTRICAS BOOTSTRAP ===")
print(f"R2 promedio   : {r2_boot.mean():.4f} ± {r2_boot.std():.4f}")
print(f"RMSE promedio : {rmse_boot.mean():.4f} ± {rmse_boot.std():.4f}")

# =============================================================================
# 7. VISUALIZACIÓN DE COEFICIENTES (TOP 4)
# =============================================================================
plt.figure(figsize=(10,6))

for i in range(min(4, coefs_boot.shape[1])):  # solo primeras 4 variables
    sns.kdeplot(coefs_boot[:, i], label=X.columns[i])

plt.title("Distribución Bootstrap de Coeficientes")
plt.legend()
plt.show()

# =============================================================================
# 8. DISTRIBUCIÓN DE MÉTRICAS
# =============================================================================
plt.figure()
sns.histplot(r2_boot, kde=True)
plt.title("Distribución Bootstrap de R2")
plt.show()

plt.figure()
sns.histplot(rmse_boot, kde=True)
plt.title("Distribución Bootstrap de RMSE")
plt.show()
