# =========================================
# 1. LIBRERÍAS
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

import warnings
warnings.filterwarnings("ignore")

plt.style.use("ggplot")


# =========================================
# 2. CARGA DE DATOS
# =========================================
df = pd.read_excel("D:/SEMESTRE 7 2026/APRENDIZAJE SUPERVISADO/regresion/datossalarios.xlsx")

print("Dimensiones:", df.shape)
print(df.head())


# =========================================
# 3. PREPROCESAMIENTO
# =========================================
target = "salario"

# One-Hot Encoding
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


# =========================================
# 4. MODELO BASE (OLS)
# =========================================
ols = LinearRegression()
ols.fit(X_train_sc, y_train)

y_pred_ols = ols.predict(X_test_sc)

rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
r2_ols = r2_score(y_test, y_pred_ols)

print("\nOLS → R2:", round(r2_ols,4), "| RMSE:", round(rmse_ols,2))


# =========================================
# 5. RIDGE (REGULARIZACIÓN L2)
# =========================================
ridge = RidgeCV(alphas=np.logspace(-3, 3, 100), cv=5)
ridge.fit(X_train_sc, y_train)

y_pred_ridge = ridge.predict(X_test_sc)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

print("\nRIDGE → R2:", round(r2_ridge,4), "| RMSE:", round(rmse_ridge,2))
print("Alpha óptimo Ridge:", ridge.alpha_)


# =========================================
# 6. LASSO (REGULARIZACIÓN L1)
# =========================================
lasso = LassoCV(alphas=np.logspace(-3, 3, 100), cv=5, max_iter=10000)
lasso.fit(X_train_sc, y_train)

y_pred_lasso = lasso.predict(X_test_sc)

rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

print("\nLASSO → R2:", round(r2_lasso,4), "| RMSE:", round(rmse_lasso,2))
print("Alpha óptimo Lasso:", lasso.alpha_)

# Variables seleccionadas (coef ≠ 0)
coef_lasso = pd.Series(lasso.coef_, index=X.columns)
vars_lasso = coef_lasso[coef_lasso != 0]

print("\nVariables seleccionadas por LASSO:")
print(vars_lasso)


# =========================================
# 7. ELASTIC NET (L1 + L2)
# =========================================
elastic = ElasticNetCV(
    l1_ratio=[0.1, 0.5, 0.7, 0.9, 0.95, 1],
    alphas=np.logspace(-3, 3, 100),
    cv=5
)

elastic.fit(X_train_sc, y_train)

y_pred_elastic = elastic.predict(X_test_sc)

rmse_elastic = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
r2_elastic = r2_score(y_test, y_pred_elastic)

print("\nELASTIC NET → R2:", round(r2_elastic,4), "| RMSE:", round(rmse_elastic,2))
print("Alpha óptimo:", elastic.alpha_)
print("l1_ratio óptimo:", elastic.l1_ratio_)


# =========================================
# 8. COMPARACIÓN FINAL
# =========================================
resultados = pd.DataFrame({
    "Modelo": ["OLS", "Ridge", "Lasso", "ElasticNet"],
    "R2": [r2_ols, r2_ridge, r2_lasso, r2_elastic],
    "RMSE": [rmse_ols, rmse_ridge, rmse_lasso, rmse_elastic]
})

resultados = resultados.sort_values("R2", ascending=False)

print("\nCOMPARACIÓN FINAL:")
print(resultados)


# =========================================
# 9. GRÁFICO
# =========================================
plt.figure()
plt.bar(resultados["Modelo"], resultados["R2"])
plt.title("Comparación de Modelos (R2)")
plt.xlabel("Modelo")
plt.ylabel("R2")
plt.show()


# =========================================
# 10. MEJOR MODELO
# =========================================
best = resultados.iloc[0]

print("\n🏆 MEJOR MODELO:")
print(best)
