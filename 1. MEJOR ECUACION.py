# =========================================
# 1. LIBRERÍAS
# =========================================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, LassoCV, RidgeCV
from sklearn.feature_selection import SelectKBest, f_regression, mutual_info_regression, RFE, RFECV
from sklearn.metrics import mean_squared_error, r2_score

import warnings
warnings.filterwarnings("ignore")

plt.style.use("ggplot")


# =========================================
# PARÁMETROS
# =========================================
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_FOLDS = 5
K_FEATURES = 10
CORR_THRESHOLD = 0.1


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

# 🔥 IMPORTANTE: mantener como DataFrame
scaler = StandardScaler()

X_train_sc = pd.DataFrame(
    scaler.fit_transform(X_train),
    columns=X.columns,
    index=X_train.index
)

X_test_sc = pd.DataFrame(
    scaler.transform(X_test),
    columns=X.columns,
    index=X_test.index
)


# =========================================
# FUNCIÓN EVALUAR
# =========================================
def evaluar(X_tr, X_te, y_tr, y_te, nombre, params_txt):

    model = LinearRegression(fit_intercept=True)
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    cv = cross_val_score(model, X_tr, y_tr, cv=CV_FOLDS).mean()

    print(f"{nombre:25} | R2={r2:.3f} | RMSE={rmse:.3f} | CV={cv:.3f}")
    print(f"   Parámetros usados: {params_txt}")

    return {
        "Metodo": nombre,
        "R2": round(r2, 3),
        "RMSE": round(rmse, 3),
        "CV": round(cv, 3),
        "Parametros": params_txt
    }


resultados = []


# =========================================
# MODELO BASE
# =========================================
print("\nMODELO BASE")

params = f"fit_intercept=True, test_size={TEST_SIZE}, random_state={RANDOM_STATE}, cv={CV_FOLDS}"

res = evaluar(X_train_sc, X_test_sc, y_train, y_test, "Todas", params)
resultados.append(res)


# =========================================
# FILTER METHODS
# =========================================

# Pearson
corr = X_train_sc.corrwith(y_train).abs()
vars_pearson = corr[corr > CORR_THRESHOLD].index.tolist()

if len(vars_pearson) == 0:
    vars_pearson = corr.sort_values(ascending=False).head(K_FEATURES).index.tolist()

params = f"corr_threshold={CORR_THRESHOLD}, n_vars={len(vars_pearson)}, cv={CV_FOLDS}"

res = evaluar(
    X_train_sc[vars_pearson],
    X_test_sc[vars_pearson],
    y_train, y_test,
    "Pearson",
    params
)
resultados.append(res)


# F-test
selector = SelectKBest(f_regression, k=K_FEATURES)
X_train_f = selector.fit_transform(X_train_sc, y_train)
X_test_f = selector.transform(X_test_sc)

params = f"k={K_FEATURES} (F-test), cv={CV_FOLDS}"

res = evaluar(X_train_f, X_test_f, y_train, y_test, "F-test", params)
resultados.append(res)


# Mutual Info
selector = SelectKBest(mutual_info_regression, k=K_FEATURES)
X_train_mi = selector.fit_transform(X_train_sc, y_train)
X_test_mi = selector.transform(X_test_sc)

params = f"k={K_FEATURES} (Mutual Info), cv={CV_FOLDS}"

res = evaluar(X_train_mi, X_test_mi, y_train, y_test, "Mutual Info", params)
resultados.append(res)


# =========================================
# WRAPPER METHODS
# =========================================

# RFE
rfe = RFE(LinearRegression(), n_features_to_select=K_FEATURES)
X_train_rfe = rfe.fit_transform(X_train_sc, y_train)
X_test_rfe = rfe.transform(X_test_sc)

params = f"n_features={K_FEATURES} (RFE)"

res = evaluar(X_train_rfe, X_test_rfe, y_train, y_test, "RFE", params)
resultados.append(res)


# RFECV
rfecv = RFECV(LinearRegression(), cv=CV_FOLDS)
X_train_rfecv = rfecv.fit_transform(X_train_sc, y_train)
X_test_rfecv = rfecv.transform(X_test_sc)

params = f"cv={CV_FOLDS} (RFECV automático)"

res = evaluar(X_train_rfecv, X_test_rfecv, y_train, y_test, "RFECV", params)
resultados.append(res)


# =========================================
# EMBEDDED METHODS
# =========================================

# LASSO
lasso = LassoCV(cv=CV_FOLDS, random_state=RANDOM_STATE)
lasso.fit(X_train_sc, y_train)

y_pred = lasso.predict(X_test_sc)

alpha_lasso = round(lasso.alpha_, 5)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cv = cross_val_score(lasso, X_train_sc, y_train, cv=CV_FOLDS).mean()

params = f"alpha={alpha_lasso}, cv={CV_FOLDS}"

print(f"LASSO                    | R2={r2:.3f} | RMSE={rmse:.3f} | CV={cv:.3f}")
print(f"   Parámetros usados: {params}")

resultados.append({
    "Metodo": "LASSO",
    "R2": round(r2, 3),
    "RMSE": round(rmse, 3),
    "CV": round(cv, 3),
    "Parametros": params
})


# Ridge
ridge = RidgeCV(cv=CV_FOLDS)
ridge.fit(X_train_sc, y_train)

y_pred = ridge.predict(X_test_sc)

alpha_ridge = round(ridge.alpha_, 5)

r2 = r2_score(y_test, y_pred)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
cv = cross_val_score(ridge, X_train_sc, y_train, cv=CV_FOLDS).mean()

params = f"alpha={alpha_ridge}, cv={CV_FOLDS}"

print(f"Ridge                    | R2={r2:.3f} | RMSE={rmse:.3f} | CV={cv:.3f}")
print(f"   Parámetros usados: {params}")

resultados.append({
    "Metodo": "Ridge",
    "R2": round(r2, 3),
    "RMSE": round(rmse, 3),
    "CV": round(cv, 3),
    "Parametros": params
})


# =========================================
# RESULTADOS FINALES
# =========================================
res_df = pd.DataFrame(resultados).sort_values("R2", ascending=False)

print("\nRESULTADOS FINALES")
print(res_df.to_string(index=False))


# =========================================
# MEJOR MODELO
# =========================================
best = res_df.iloc[0]

print("\n🏆 MEJOR MODELO:")
for k, v in best.items():
    print(f"{k}: {v}")


# =========================================
# GRÁFICO
# =========================================
res_df.plot(x="Metodo", y="R2", kind="bar", legend=False)
plt.title("Comparación de Modelos (R2)")
plt.xticks(rotation=45)

plt.savefig("comparacion_modelos.png", dpi=300, bbox_inches='tight')
plt.show()
