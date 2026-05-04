# =========================================
# REGRESIÓN LINEAL MÚLTIPLE (FULL PARÁMETROS)
# =========================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

import statsmodels.api as sm

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Ridge, Lasso

import warnings
warnings.filterwarnings('ignore')

sns.set(style="whitegrid")
np.random.seed(42)


# =========================================
# 🔹 PARÁMETROS GENERALES
# =========================================
TEST_SIZE = 0.2
TRAIN_SIZE = None
RANDOM_STATE = 42
SHUFFLE = True

CV_FOLDS = 5
SCORING = 'r2'
N_JOBS = None   # -1 para usar todos los núcleos

# =========================================
# 🔹 PARÁMETROS DEL MODELO
# =========================================
MODELO_TIPO = "linear"   # "linear", "ridge", "lasso"

# LinearRegression
FIT_INTERCEPT = True
COPY_X = True
POSITIVE = False

# Ridge / Lasso
ALPHA = 1.0
MAX_ITER = 1000
TOL = 0.0001
SOLVER = 'auto'
SELECTION = 'cyclic'   # solo Lasso

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
    X,
    y,
    test_size=TEST_SIZE,
    train_size=TRAIN_SIZE,
    random_state=RANDOM_STATE,
    shuffle=SHUFFLE
)


# =========================================
# 4. ESCALADO
# =========================================
scaler = StandardScaler(
    copy=True,
    with_mean=True,
    with_std=True
)

X_train_sc = scaler.fit_transform(X_train)
X_test_sc  = scaler.transform(X_test)


# =========================================
# 5. MODELO (TODOS LOS PARÁMETROS)
# =========================================
if MODELO_TIPO == "linear":
    model = LinearRegression(
        fit_intercept=FIT_INTERCEPT,
        copy_X=COPY_X,
        n_jobs=N_JOBS,
        positive=POSITIVE
    )

elif MODELO_TIPO == "ridge":
    model = Ridge(
        alpha=ALPHA,
        fit_intercept=FIT_INTERCEPT,
        copy_X=COPY_X,
        max_iter=MAX_ITER,
        tol=TOL,
        solver=SOLVER,
        random_state=RANDOM_STATE
    )

elif MODELO_TIPO == "lasso":
    model = Lasso(
        alpha=ALPHA,
        fit_intercept=FIT_INTERCEPT,
        max_iter=MAX_ITER,
        tol=TOL,
        selection=SELECTION,
        random_state=RANDOM_STATE
    )

model.fit(X_train_sc, y_train)

y_pred_test = model.predict(X_test_sc)


# =========================================
# 6. MÉTRICAS
# =========================================
r2_test = r2_score(y_test, y_pred_test)
mse_test = mean_squared_error(y_test, y_pred_test)
rmse_test = np.sqrt(mse_test)
mae_test = mean_absolute_error(y_test, y_pred_test)

print("\n=== RESULTADOS (TEST) ===")
print(f"R2: {r2_test:.4f}")
print(f"MSE: {mse_test:.4f}")
print(f"RMSE: {rmse_test:.4f}")
print(f"MAE: {mae_test:.4f}")


# =========================================
# 7. CROSS VALIDATION
# =========================================
cv_scores = cross_val_score(
    model,
    X_train_sc,
    y_train,
    cv=CV_FOLDS,
    scoring=SCORING,
    n_jobs=N_JOBS
)

print(f"\nCV R2: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")


# =========================================
# 8. MODELO STATS (INTERPRETACIÓN)
# =========================================
X_train_sm = sm.add_constant(X_train_sc)
modelo_sm = sm.OLS(y_train, X_train_sm).fit()

print("\n=== RESUMEN (STATS) ===")
print(modelo_sm.summary())


# =========================================
# 9. IMPORTANCIA DE VARIABLES
# =========================================
coef_df = pd.DataFrame({
    'Variable': ['Intercepto'] + list(X.columns),
    'Coeficiente': modelo_sm.params
}).sort_values(by='Coeficiente', key=abs, ascending=False)

print("\n=== VARIABLES IMPORTANTES ===")
print(coef_df.head(10))


# =========================================
# 10. GRÁFICO
# =========================================
plt.figure(figsize=(6,5))
plt.scatter(y_test, y_pred_test, alpha=0.5)

plt.plot(
    [y_test.min(), y_test.max()],
    [y_test.min(), y_test.max()],
    linestyle='--'
)

plt.xlabel("Real")
plt.ylabel("Predicción")
plt.title("Real vs Predicho")
plt.show()


# =========================================
# 11. PREDICCIÓN NUEVA
# =========================================
nuevo = X.iloc[[0]]

nuevo_sc = scaler.transform(nuevo)
pred = model.predict(nuevo_sc)

print("\nPredicción ejemplo:", pred[0])
