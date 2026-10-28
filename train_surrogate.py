"""
train_surrogate.py  -  reads your OpenRocket observation sheet, cleans it,
trains models to predict APOGEE and STABILITY from the fin shape, tests them
on designs they have never seen, and saves the models + a chart.

HOW TO USE:  put this file next to your .xlsx file, change EXCEL_FILE below, run:
    python train_surrogate.py
""" 
import numpy as np, pandas as pd, joblib, time
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, confusion_matrix

# ---------- 1. SETTINGS ----------
EXCEL_FILE = "OpenRocket_Observation_Record_Sheet_1_1_FILLED.xlsx"   # <<< change to your file name
SEED = 42

# ---------- 2. LOAD AND CLEAN ----------
df = pd.read_excel(EXCEL_FILE, sheet_name="Observation Log").iloc[:, :27]
df = df.dropna(subset=["Fin_Shape", "Apogee_m", "Stability"])       # drop empty rows
print(f"Rows with data: {len(df)}")

df["tumbled"] = df["Notes"].astype(str).str.contains("tumble", case=False)
print(f"Rows where the rocket tumbled (apogee not reliable): {df['tumbled'].sum()}")

# ---------- 3. INPUTS (only things that change between rows) ----------
raw_inputs = ["Fin_Count", "Fin_Thickness_mm", "Fin_Span_mm", "Root_Chord_mm", "Tip_Chord_mm"]
def add_features(d):
    d = d.copy()
    d["fin_area"]   = 0.5 * (d["Root_Chord_mm"] + d["Tip_Chord_mm"]) * d["Fin_Span_mm"]
    d["taper"]      = d["Tip_Chord_mm"] / d["Root_Chord_mm"]
    d["aspect"]     = d["Fin_Span_mm"] ** 2 / d["fin_area"]
    d["total_area"] = d["fin_area"] * d["Fin_Count"]
    return d
df = add_features(df)
FEATURES = raw_inputs + ["fin_area", "taper", "aspect", "total_area"]

def models():
    return {
        "Ridge (simple baseline)": make_pipeline(StandardScaler(), Ridge(alpha=1.0)),
        "Random Forest":           RandomForestRegressor(n_estimators=400, random_state=SEED, n_jobs=-1),
        "Gradient Boosting":       HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=SEED),
    }

def evaluate(name, data, target):
    X, y = data[FEATURES], data[target]
    Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.2, random_state=SEED)
    print(f"\n=== {name}  ({len(data)} rows: {len(Xtr)} train / {len(Xte)} test) ===")
    best, best_r2, rows = None, -9, []
    for mname, m in models().items():
        m.fit(Xtr, ytr); p = m.predict(Xte)
        r2 = r2_score(yte, p); mae = mean_absolute_error(yte, p); rmse = mean_squared_error(yte, p) ** 0.5
        cv = cross_val_score(models()[mname], X, y, cv=5, scoring="r2").mean()
        rows.append((mname, r2, mae, rmse, cv))
        if r2 > best_r2: best, best_r2, best_p, best_name = m, r2, p, mname
    print(pd.DataFrame(rows, columns=["Model", "R2 (test)", "MAE", "RMSE", "R2 (5-fold CV)"]).round(3).to_string(index=False))
    return best, best_name, yte, best_p

# ---------- 4. TRAIN ----------
stab_model, stab_name, stab_y, stab_p = evaluate("STABILITY (calibers)", df, "Stability")
good = df[~df["tumbled"]]
apo_model, apo_name, apo_y, apo_p = evaluate("APOGEE (meters), tumbled flights removed", good, "Apogee_m")

joblib.dump(stab_model, "model_stability.pkl"); joblib.dump(apo_model, "model_apogee.pkl")

# ---------- 5. IS THE ROCKET STABLE? (yes/no) ----------
df["is_stable"] = (df["Stability"] >= 1).astype(int)
Xtr, Xte, ytr, yte = train_test_split(df[FEATURES], df["is_stable"], test_size=0.2, random_state=SEED, stratify=df["is_stable"])
clf = RandomForestClassifier(n_estimators=400, random_state=SEED, n_jobs=-1).fit(Xtr, ytr)
pc = clf.predict(Xte)
print(f"\n=== STABLE (stability >= 1 cal) yes/no ===\nAccuracy: {accuracy_score(yte, pc):.3f}")
print("Confusion matrix [rows=actual no/yes, cols=predicted no/yes]:\n", confusion_matrix(yte, pc))

# ---------- 6. HARDER TEST: the model has never seen the longest fins ----------
print("\n=== HARD TEST: train on span 2-6, test on span 7 (extrapolation) ===")
tr, te = df[df.Fin_Span_mm < 7], df[df.Fin_Span_mm == 7]
m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=SEED).fit(tr[FEATURES], tr["Stability"])
print(f"Stability R2 on unseen span: {r2_score(te['Stability'], m.predict(te[FEATURES])):.3f}  MAE: {mean_absolute_error(te['Stability'], m.predict(te[FEATURES])):.3f}")
tr, te = tr[~tr.tumbled], te[~te.tumbled]
m = HistGradientBoostingRegressor(max_iter=500, learning_rate=0.05, random_state=SEED).fit(tr[FEATURES], tr["Apogee_m"])
print(f"Apogee    R2 on unseen span: {r2_score(te['Apogee_m'], m.predict(te[FEATURES])):.3f}  MAE: {mean_absolute_error(te['Apogee_m'], m.predict(te[FEATURES])):.3f} m")

# ---------- 7. WHICH INPUT MATTERS MOST ----------
from sklearn.inspection import permutation_importance
Xs = df[FEATURES]; r = permutation_importance(stab_model, Xs, df["Stability"], n_repeats=5, random_state=SEED)
print("\nWhat drives STABILITY (bigger = more important):")
print(pd.Series(r.importances_mean, FEATURES).sort_values(ascending=False).round(3).to_string())
Xa = good[FEATURES]; r = permutation_importance(apo_model, Xa, good["Apogee_m"], n_repeats=5, random_state=SEED)
print("\nWhat drives APOGEE:")
print(pd.Series(r.importances_mean, FEATURES).sort_values(ascending=False).round(3).to_string())

# ---------- 8. SPEED TEST ----------
t = time.time(); stab_model.predict(df[FEATURES]); dt = time.time() - t
print(f"\nPredicted {len(df)} designs in {dt:.3f} s  ({dt/len(df)*1000:.3f} ms each). OpenRocket takes seconds per design.")

# ---------- 9. CHART: predicted vs actual ----------
BLUE, INK, GRID = "#2a6fdb", "#1f2937", "#e5e7eb"
fig, ax = plt.subplots(1, 2, figsize=(11, 4.8), facecolor="white")
for a, (y, p, ttl, unit) in zip(ax, [(apo_y, apo_p, f"Apogee ({apo_name})", "m"), (stab_y, stab_p, f"Stability ({stab_name})", "calibers")]):
    a.scatter(y, p, s=14, color=BLUE, alpha=0.55, edgecolor="white", linewidth=0.4)
    lo, hi = min(y.min(), p.min()), max(y.max(), p.max()); a.plot([lo, hi], [lo, hi], color=INK, lw=1.2, ls="--")
    a.set_title(f"{ttl}\nR² = {r2_score(y, p):.3f}   average error = {mean_absolute_error(y, p):.2f} {unit}", fontsize=11, color=INK, loc="left")
    a.set_xlabel(f"OpenRocket result ({unit})", color=INK); a.set_ylabel(f"Model prediction ({unit})", color=INK)
    a.grid(color=GRID, lw=0.8); a.set_axisbelow(True)
    for s in ("top", "right"): a.spines[s].set_visible(False)
    a.set_aspect("equal", adjustable="datalim")
fig.suptitle("Unseen test designs: dots close to the dashed line mean the model matches OpenRocket", fontsize=10, color="#4b5563", y=0.03)
plt.tight_layout(rect=(0, 0.08, 1, 1)); plt.savefig("predicted_vs_actual.png", dpi=150)
print("\nSaved: model_stability.pkl, model_apogee.pkl, predicted_vs_actual.png")
