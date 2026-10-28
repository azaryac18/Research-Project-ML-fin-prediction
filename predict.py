"""predict.py - type ONE fin design, get predicted stability and apogee instantly.
Needs model_stability.pkl and model_apogee.pkl (made by train_surrogate.py) in the same folder."""
import joblib, pandas as pd

# >>> TYPE YOUR FIN DESIGN HERE (use the SAME units as your Excel sheet) <<<
fin_count = 4      # 3, 4 or 5
thickness = 3      # between 2 and 5
span      = 5      # between 2 and 7
root      = 4      # between 2 and 7
tip       = 3      # between 2 and 7

area = 0.5 * (root + tip) * span
x = pd.DataFrame([dict(Fin_Count=fin_count, Fin_Thickness_mm=thickness, Fin_Span_mm=span,
                       Root_Chord_mm=root, Tip_Chord_mm=tip, fin_area=area, taper=tip/root,
                       aspect=span**2/area, total_area=area*fin_count)])
stab = joblib.load("model_stability.pkl").predict(x)[0]
apo  = joblib.load("model_apogee.pkl").predict(x)[0]
print(f"Predicted stability: {stab:.2f} calibers")
print(f"Predicted apogee   : {apo:.1f} m")
if stab < 1:
    print("WARNING: stability below 1 - the rocket may tumble, so the apogee number is NOT reliable.")
elif stab > 2.5:
    print("Note: very stable (over 2.5) - it may weathercock into the wind.")
else:
    print("Stability is in the good range.")
