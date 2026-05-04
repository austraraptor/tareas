# =========================================
# 1. LIBRERÍAS
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, r2_score

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import RidgeCV, LassoCV, ElasticNetCV

import warnings
warnings.filterwarnings("ignore")

plt.style.use("ggplot")


# =========================================
# PARÁMETROS GENERALES
# =========================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV = 5


# =========================================
# 2. CARGA DE DATOS
# =========================================
df = pd.read_excel("D:/SEMESTRE 7 2026/APRENDIZAJE SUPERVISADO/regresion/datossalarios.xlsx")

print("Dimensiones:", df.shape)


# =========================================
# 3. PREPROCESAMIENTO
# =========================================
target = "salario"

df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=[target])
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE
)

scaler = StandardScaler()
X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# =========================================
# FUNCIÓN PARA MOSTRAR RESULTADOS
# =========================================
def mostrar_resultados(nombre, r2, rmse, params):
    print(f"\n{nombre.upper()} → R2={r2:.3f} | RMSE={rmse:.3f}")
    print(f"   Parámetros usados: {params}")


# =========================================
# 4. MODELO BASE (OLS)
# =========================================
ols = LinearRegression(fit_intercept=True)

ols.fit(X_train_sc, y_train)

y_pred_ols = ols.predict(X_test_sc)

rmse_ols = np.sqrt(mean_squared_error(y_test, y_pred_ols))
r2_ols = r2_score(y_test, y_pred_ols)

params_ols = f"fit_intercept=True, test_size={TEST_SIZE}, random_state={RANDOM_STATE}"

mostrar_resultados("OLS", r2_ols, rmse_ols, params_ols)


# =========================================
# 5. RIDGE (L2)
# =========================================
alphas_ridge = np.logspace(-3, 3, 100)

ridge = RidgeCV(alphas=alphas_ridge, cv=CV)
ridge.fit(X_train_sc, y_train)

y_pred_ridge = ridge.predict(X_test_sc)

rmse_ridge = np.sqrt(mean_squared_error(y_test, y_pred_ridge))
r2_ridge = r2_score(y_test, y_pred_ridge)

params_ridge = f"alpha_optimo={round(ridge.alpha_,5)}, cv={CV}, alphas=logspace(-3,3,100)"

mostrar_resultados("Ridge", r2_ridge, rmse_ridge, params_ridge)


# =========================================
# 6. LASSO (L1)
# =========================================
alphas_lasso = np.logspace(-3, 3, 100)

lasso = LassoCV(alphas=alphas_lasso, cv=CV, max_iter=10000)
lasso.fit(X_train_sc, y_train)

y_pred_lasso = lasso.predict(X_test_sc)

rmse_lasso = np.sqrt(mean_squared_error(y_test, y_pred_lasso))
r2_lasso = r2_score(y_test, y_pred_lasso)

params_lasso = f"alpha_optimo={round(lasso.alpha_,5)}, cv={CV}, max_iter=10000"

mostrar_resultados("LASSO", r2_lasso, rmse_lasso, params_lasso)


# Variables seleccionadas
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
    cv=CV
)

elastic.fit(X_train_sc, y_train)

y_pred_elastic = elastic.predict(X_test_sc)

rmse_elastic = np.sqrt(mean_squared_error(y_test, y_pred_elastic))
r2_elastic = r2_score(y_test, y_pred_elastic)

params_elastic = f"alpha_optimo={round(elastic.alpha_,5)}, l1_ratio={elastic.l1_ratio_}, cv={CV}"

mostrar_resultados("ElasticNet", r2_elastic, rmse_elastic, params_elastic)


# =========================================
# 8. COMPARACIÓN FINAL
# =========================================
resultados = pd.DataFrame({
    "Modelo": ["OLS", "Ridge", "Lasso", "ElasticNet"],
    "R2": [r2_ols, r2_ridge, r2_lasso, r2_elastic],
    "RMSE": [rmse_ols, rmse_ridge, rmse_lasso, rmse_elastic]
}).sort_values("R2", ascending=False)

print("\nCOMPARACIÓN FINAL:")
print(resultados.to_string(index=False))


# =========================================
# 9. GRÁFICO
# =========================================
plt.figure()
plt.bar(resultados["Modelo"], resultados["R2"])
plt.title("Comparación de Modelos (R2)")
plt.xlabel("Modelo")
plt.ylabel("R2")

plt.savefig("comparacion_modelos_regularizados.png", dpi=300, bbox_inches='tight')
plt.show()


# =========================================
# 10. MEJOR MODELO
# =========================================
best = resultados.iloc[0]

print("\n🏆 MEJOR MODELO:")
print(best)
