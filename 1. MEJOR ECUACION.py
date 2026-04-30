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
# 2. CARGA DE DATOS
# =========================================
df = pd.read_excel("D:/SEMESTRE 7 2026/APRENDIZAJE SUPERVISADO/regresion/datossalarios.xlsx")

print("Dimensiones:", df.shape)
print(df.head())


# =========================================
# 3. PREPROCESAMIENTO
# =========================================
target = "salario"

# Convertir variables categóricas correctamente (One-Hot Encoding)
df = pd.get_dummies(df, drop_first=True)

X = df.drop(columns=[target])
y = df[target]

# Train/Test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Escalado
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
# 4. FUNCIÓN DE EVALUACIÓN
# =========================================
def evaluar(X_tr, X_te, y_tr, y_te, nombre):

    if X_tr.shape[1] == 0:
        print(f"{nombre:20} ❌ SIN VARIABLES")
        return None

    model = LinearRegression()
    model.fit(X_tr, y_tr)

    y_pred = model.predict(X_te)

    r2 = r2_score(y_te, y_pred)
    rmse = np.sqrt(mean_squared_error(y_te, y_pred))
    cv = cross_val_score(model, X_tr, y_tr, cv=5, scoring="r2").mean()

    print(f"{nombre:25} | R2={r2:.4f} | RMSE={rmse:.2f} | CV={cv:.4f}")

    return {"Metodo": nombre, "R2": r2, "RMSE": rmse, "CV": cv}


resultados = []


# =========================================
# 5. MODELO BASE
# =========================================
print("\nMODELO BASE")
res = evaluar(X_train_sc, X_test_sc, y_train, y_test, "Todas")
if res: resultados.append(res)


# =========================================
# 6. FILTER METHODS
# =========================================

# Pearson
corr = X_train_sc.corrwith(y_train).abs()

vars_pearson = corr[corr > 0.1].index.tolist()

if len(vars_pearson) == 0:
    vars_pearson = corr.sort_values(ascending=False).head(5).index.tolist()

res = evaluar(
    X_train_sc[vars_pearson],
    X_test_sc[vars_pearson],
    y_train, y_test, "Pearson"
)
if res: resultados.append(res)


# F-test
k = min(10, X_train_sc.shape[1])
selector = SelectKBest(f_regression, k=k)
X_train_f = selector.fit_transform(X_train_sc, y_train)
X_test_f = selector.transform(X_test_sc)

res = evaluar(X_train_f, X_test_f, y_train, y_test, "F-test")
if res: resultados.append(res)


# Mutual Info
selector = SelectKBest(mutual_info_regression, k=k)
X_train_mi = selector.fit_transform(X_train_sc, y_train)
X_test_mi = selector.transform(X_test_sc)

res = evaluar(X_train_mi, X_test_mi, y_train, y_test, "Mutual Info")
if res: resultados.append(res)


# =========================================
# 7. WRAPPER METHODS
# =========================================

# RFE
rfe = RFE(LinearRegression(), n_features_to_select=min(10, X_train_sc.shape[1]))
X_train_rfe = rfe.fit_transform(X_train_sc, y_train)
X_test_rfe = rfe.transform(X_test_sc)

res = evaluar(X_train_rfe, X_test_rfe, y_train, y_test, "RFE")
if res: resultados.append(res)


# RFECV
rfecv = RFECV(LinearRegression(), cv=5)
X_train_rfecv = rfecv.fit_transform(X_train_sc, y_train)
X_test_rfecv = rfecv.transform(X_test_sc)

res = evaluar(X_train_rfecv, X_test_rfecv, y_train, y_test, "RFECV")
if res: resultados.append(res)


# =========================================
# 8. EMBEDDED METHODS
# =========================================

# LASSO
lasso = LassoCV(cv=5, max_iter=10000)
lasso.fit(X_train_sc, y_train)

y_pred = lasso.predict(X_test_sc)

res = {
    "Metodo": "LASSO",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "CV": cross_val_score(lasso, X_train_sc, y_train, cv=5).mean()
}
print(f"LASSO                    | R2={res['R2']:.4f}")
resultados.append(res)


# Ridge
ridge = RidgeCV(cv=5)
ridge.fit(X_train_sc, y_train)

y_pred = ridge.predict(X_test_sc)

res = {
    "Metodo": "Ridge",
    "R2": r2_score(y_test, y_pred),
    "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
    "CV": cross_val_score(ridge, X_train_sc, y_train, cv=5).mean()
}
print(f"Ridge                    | R2={res['R2']:.4f}")
resultados.append(res)


# =========================================
# 9. RESULTADOS FINALES
# =========================================
res_df = pd.DataFrame(resultados).sort_values("R2", ascending=False)

print("\nRESULTADOS FINALES")
print(res_df)


# =========================================
# 10. MEJOR MODELO
# =========================================
best = res_df.iloc[0]

print("\n🏆 MEJOR MODELO:")
print(best)


# =========================================
# 11. GRÁFICO COMPARATIVO
# =========================================
res_df.plot(x="Metodo", y="R2", kind="bar", legend=False)
plt.title("Comparación de Modelos (R2)")
plt.xticks(rotation=45)
plt.show()
