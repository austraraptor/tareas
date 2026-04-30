# =========================================
# REGRESIÓN LINEAL MÚLTIPLE (FIXED)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error
from sklearn.linear_model import LinearRegression

import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
np.random.seed(42)


# =========================================
# 1. CARGA DE DATOS
# =========================================
df = pd.read_excel("D:/SEMESTRE 7 2026/APRENDIZAJE SUPERVISADO/regresion/datossalarios.xlsx")

target = "salario"

# =========================================
# 2. PREPROCESAMIENTO
# =========================================
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=[target])
y = df[target]


# =========================================
# 3. TRAIN / TEST
# =========================================
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# =========================================
# 4. ESCALADO
# =========================================
scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# =========================================
# 5. MODELO (SKLEARN → para CV)
# =========================================
model_sk = LinearRegression()
model_sk.fit(X_train_sc, y_train)

y_pred_test = model_sk.predict(X_test_sc)

r2_test = r2_score(y_test, y_pred_test)
rmse_test = np.sqrt(mean_squared_error(y_test, y_pred_test))

print("\n=== RESULTADOS (TEST) ===")
print(f"R2: {r2_test:.4f}")
print(f"RMSE: {rmse_test:.2f}")


# =========================================
# 6. CROSS VALIDATION (CORRECTO)
# =========================================
cv_scores = cross_val_score(
    model_sk,
    X_train_sc,
    y_train,
    cv=5,
    scoring='r2'
)

print(f"\nCV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# =========================================
# 7. MODELO STATS (INTERPRETACIÓN)
# =========================================
X_train_sm = sm.add_constant(X_train_sc)
modelo_sm = sm.OLS(y_train, X_train_sm).fit()

print("\n=== RESUMEN (STATS) ===")
print(modelo_sm.summary())


# =========================================
# 8. IMPORTANCIA DE VARIABLES
# =========================================
coef_df = pd.DataFrame({
    'Variable': ['Intercepto'] + list(X.columns),
    'Coeficiente': modelo_sm.params
}).sort_values(by='Coeficiente', key=abs, ascending=False)

print("\n=== VARIABLES IMPORTANTES ===")
print(coef_df.head(10))


# =========================================
# 9. GRÁFICO
# =========================================
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_test, alpha=0.5)
plt.plot([y_test.min(), y_test.max()],
         [y_test.min(), y_test.max()],
         linestyle='--')
plt.xlabel("Real")
plt.ylabel("Predicción")
plt.title("Real vs Predicho")
plt.show()


# =========================================
# 10. PREDICCIÓN NUEVA
# =========================================
nuevo = X.iloc[[0]]

nuevo_sc = scaler.transform(nuevo)
pred = model_sk.predict(nuevo_sc)

print("\nPredicción ejemplo:", pred[0])
