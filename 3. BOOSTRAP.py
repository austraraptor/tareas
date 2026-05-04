# =============================================================================
# REGRESIÓN LINEAL + BOOTSTRAPPING (FORMATO INFORME FINAL)
# =============================================================================

# ── LIBRERÍAS ────────────────────────────────────────────────────────────────
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.linear_model import LinearRegression
from sklearn.utils import resample
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler

import warnings
warnings.filterwarnings("ignore")

sns.set(style="whitegrid")
np.random.seed(42)

# =============================================================================
# 🔹 PARÁMETROS NUMÉRICOS
# =============================================================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
N_BOOTSTRAP = 500

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

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=[target])
y = df[target]

# =============================================================================
# 3. TRAIN / TEST
# =============================================================================
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

# =============================================================================
# 4. ESCALADO
# =============================================================================
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)

# =============================================================================
# 5. MODELO BASE
# =============================================================================
model = LinearRegression()
model.fit(X_train_sc, y_train)

y_pred = model.predict(X_test_sc)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))

print("\n=== MODELO BASE ===")
print(f"R2   : {r2:.4f}")
print(f"RMSE : {rmse:.4f}")

print("\nParámetros numéricos usados:")
print(f"test_size={TEST_SIZE}, random_state={RANDOM_STATE}, cv={CV_FOLDS}, n_bootstrap={N_BOOTSTRAP}")

# =============================================================================
# 6. BOOTSTRAPPING
# =============================================================================
coefs_boot = []
r2_boot = []
rmse_boot = []

for _ in range(N_BOOTSTRAP):

    X_res, y_res = resample(X_train_sc, y_train)

    model_boot = LinearRegression()
    model_boot.fit(X_res, y_res)

    y_pred_boot = model_boot.predict(X_test_sc)

    coefs_boot.append(model_boot.coef_)
    r2_boot.append(r2_score(y_test, y_pred_boot))
    rmse_boot.append(np.sqrt(mean_squared_error(y_test, y_pred_boot)))

coefs_boot = np.array(coefs_boot)
r2_boot = np.array(r2_boot)
rmse_boot = np.array(rmse_boot)

# =============================================================================
# 7. INTERVALOS DE CONFIANZA
# =============================================================================
print("\n=== INTERVALOS DE CONFIANZA (95%) ===")

for i, col in enumerate(X.columns):
    ic_inf = np.percentile(coefs_boot[:, i], 2.5)
    ic_sup = np.percentile(coefs_boot[:, i], 97.5)

    print(f"{col}: [{ic_inf:.4f}, {ic_sup:.4f}]")

# =============================================================================
# 8. MÉTRICAS BOOTSTRAP
# =============================================================================
print("\n=== MÉTRICAS BOOTSTRAP ===")
print(f"R2 promedio   : {r2_boot.mean():.4f} ± {r2_boot.std():.4f}")
print(f"RMSE promedio : {rmse_boot.mean():.4f} ± {rmse_boot.std():.4f}")

# =============================================================================
# 9. GRÁFICOS
# =============================================================================

# Distribución coeficientes
plt.figure(figsize=(10,6))
for i in range(min(4, coefs_boot.shape[1])):
    sns.kdeplot(coefs_boot[:, i], label=X.columns[i])

plt.title("Distribución Bootstrap de Coeficientes")
plt.legend()
plt.savefig("coeficientes_bootstrap.png", dpi=300)
plt.show()

# R2
plt.figure()
sns.histplot(r2_boot, kde=True)
plt.title("Distribución Bootstrap de R2")
plt.savefig("r2_bootstrap.png", dpi=300)
plt.show()

# RMSE
plt.figure()
sns.histplot(rmse_boot, kde=True)
plt.title("Distribución Bootstrap de RMSE")
plt.savefig("rmse_bootstrap.png", dpi=300)
plt.show()
