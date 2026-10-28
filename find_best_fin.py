"""find_best_fin.py - tries every fin design in a fine grid using the trained models and
shows the ones with the HIGHEST apogee while stability stays in the safe range."""
import joblib, itertools, numpy as np, pandas as pd

MIN_STAB, MAX_STAB = 1.5, 2.5          # <<< safe stability range you want
steps = np.arange(2, 7.01, 0.5)        # stay INSIDE your sampled range (2 to 7)
rows = [dict(Fin_Count=n, Fin_Thickness_mm=t, Fin_Span_mm=s, Root_Chord_mm=r, Tip_Chord_mm=p)
        for n, t, s, r, p in itertools.product([3, 4, 5], [2, 3, 4, 5], steps, steps, steps)]
d = pd.DataFrame(rows)
d["fin_area"] = 0.5 * (d.Root_Chord_mm + d.Tip_Chord_mm) * d.Fin_Span_mm
d["taper"] = d.Tip_Chord_mm / d.Root_Chord_mm
d["aspect"] = d.Fin_Span_mm ** 2 / d.fin_area
d["total_area"] = d.fin_area * d.Fin_Count
F = ["Fin_Count","Fin_Thickness_mm","Fin_Span_mm","Root_Chord_mm","Tip_Chord_mm","fin_area","taper","aspect","total_area"]
d["pred_stability"] = joblib.load("model_stability.pkl").predict(d[F])
d["pred_apogee_m"]  = joblib.load("model_apogee.pkl").predict(d[F])
ok = d[(d.pred_stability >= MIN_STAB) & (d.pred_stability <= MAX_STAB) & (d.Tip_Chord_mm <= d.Root_Chord_mm)]
best = ok.sort_values("pred_apogee_m", ascending=False).head(10)
print(f"Tested {len(d)} designs, {len(ok)} are safe. TOP 10:")
print(best[["Fin_Count","Fin_Thickness_mm","Fin_Span_mm","Root_Chord_mm","Tip_Chord_mm","pred_stability","pred_apogee_m"]].round(2).to_string(index=False))
best.to_csv("best_fins.csv", index=False)
