"""
Tahap 7: Regresi Linear - prediksi volume sampah per kecamatan

Jalankan:  python model_regresi.py <folder_output>
Membaca data_final_2024.csv.
Menghasilkan folder <folder_output>/model_regresi/ berisi grafik, perbandingan_model.csv,
koefisien_model_penuh.csv, prediksi_cv.csv, dan ringkasan_regresi.txt.

Catatan metode:
- Standardisasi (StandardScaler) dilakukan DI DALAM pipeline cross-validation (tidak ada data leakage).
- n hanya 42, jadi dipakai Repeated K-Fold (5 lipatan x 20 ulangan).
- Metrik dihitung dari SEMUA prediksi out-of-fold dalam satu putaran (pooled), lalu dirata-rata
  atas 20 putaran. Ini lebih stabil daripada R2 per lipatan yang hanya berisi +-8 kecamatan.
- Setiap model dibandingkan dengan baseline "tebak rata-rata".
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.compose import TransformedTargetRegressor
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
MD = OUT / "model_regresi"
MD.mkdir(exist_ok=True)

df = pd.read_csv(OUT / "data_final_2024.csv")
TARGET = "volume_sampah_rata2"
y = df[TARGET]
NAMA = {"jumlah_penduduk": "Jumlah penduduk", "kepadatan": "Kepadatan",
        "rth_per_kapita": "RTH per kapita", "ip_air_rata2": "IP air sumur"}
F4 = list(NAMA)
log = []


def linreg(log_target=False):
    m = make_pipeline(StandardScaler(), LinearRegression())
    return TransformedTargetRegressor(regressor=m, func=np.log1p, inverse_func=np.expm1) if log_target else m


KANDIDAT = {  # nama: (model, fitur)
    "Baseline (tebak rata-rata)": (DummyRegressor(strategy="mean"), ["jumlah_penduduk"]),
    "Linear berganda: 4 fitur": (linreg(), F4),
    "Linear berganda: 4 fitur, target di-log": (linreg(True), F4),
    "Linear berganda: penduduk + IP air": (linreg(), ["jumlah_penduduk", "ip_air_rata2"]),
    "Linear sederhana: penduduk saja": (linreg(), ["jumlah_penduduk"]),
}

rows = []
for nama, (model, fitur) in KANDIDAT.items():
    mae, rmse, r2 = [], [], []
    for rep in range(20):
        p = cross_val_predict(model, df[fitur], y, cv=KFold(5, shuffle=True, random_state=rep))
        mae.append(mean_absolute_error(y, p)); rmse.append(np.sqrt(mean_squared_error(y, p))); r2.append(r2_score(y, p))
    rows.append({"model": nama, "jumlah_fitur": len(fitur), "MAE": np.mean(mae), "RMSE": np.mean(rmse), "R2": np.mean(r2)})
hasil = pd.DataFrame(rows).round(3)
hasil.to_csv(MD / "perbandingan_model.csv", index=False)
log.append("Perbandingan model (cross-validation 5-fold x 20 ulangan, metrik dari prediksi out-of-fold):\n"
           + hasil.to_string(index=False))

# model terbaik = R2 tertinggi di antara model linear tanpa transformasi target (mudah ditafsirkan)
kand = hasil[~hasil["model"].str.contains("Baseline|di-log")]
terbaik = kand.loc[kand["R2"].idxmax(), "model"]
fitur_t = KANDIDAT[terbaik][1]
log.append(f"Model terpilih: {terbaik} (fitur: {', '.join(NAMA[c] for c in fitur_t)})")

# koefisien model penuh (4 fitur) untuk melihat arah pengaruh
penuh = linreg().fit(df[F4], y)
lr = penuh.named_steps["linearregression"]
koef = pd.DataFrame({"fitur": [NAMA[c] for c in F4], "koefisien_terstandar": lr.coef_.round(2)})
koef.to_csv(MD / "koefisien_model_penuh.csv", index=False)
log.append("Koefisien model 4 fitur (terstandar = perubahan volume sampah, m3, bila fitur naik 1 simpangan baku):\n"
           + koef.to_string(index=False))

# model terbaik dilatih pada seluruh data -> persamaan
final = linreg().fit(df[fitur_t], y)
lf = final.named_steps["linearregression"]
sd = df[fitur_t].std(ddof=0).values; mu = df[fitur_t].mean().values
slope = lf.coef_ / sd
icpt = lf.intercept_ - float(np.sum(lf.coef_ * mu / sd))
persamaan = f"volume_sampah = {icpt:.1f} " + " ".join(f"{'+' if s >= 0 else '-'} {abs(s):.6f} x {c}" for s, c in zip(slope, fitur_t))
log.append("Persamaan model terpilih (nilai asli): " + persamaan)
log.append(f"R2 pada seluruh data (in-sample, hanya pembanding): {final.score(df[fitur_t], y):.3f}")

# prediksi out-of-fold (satu putaran) untuk grafik
pred = cross_val_predict(linreg(), df[fitur_t], y, cv=KFold(5, shuffle=True, random_state=0))
res = df[["wilayah", "kecamatan"]].copy()
res["aktual"], res["prediksi_cv"] = y.round(1), pred.round(1)
res["selisih"] = (res["aktual"] - res["prediksi_cv"]).round(1)
res.to_csv(MD / "prediksi_cv.csv", index=False)
log.append("Selisih terbesar (aktual - prediksi):\n"
           + res.reindex(res["selisih"].abs().sort_values(ascending=False).index).head(5).to_string(index=False))

# uji kepekaan tanpa 3 kecamatan sampah tertinggi
ek = df.nlargest(3, TARGET).index; m2 = ~df.index.isin(ek)
r2k, maek = [], []
for rep in range(20):
    p = cross_val_predict(linreg(), df.loc[m2, fitur_t], y[m2], cv=KFold(5, shuffle=True, random_state=rep))
    r2k.append(r2_score(y[m2], p)); maek.append(mean_absolute_error(y[m2], p))
log.append("Uji kepekaan tanpa " + ", ".join(df.loc[ek, "kecamatan"].str.title())
           + f": R2={np.mean(r2k):.3f}, MAE={np.mean(maek):.1f}")

# grafik 1: aktual vs prediksi
fig, ax = plt.subplots(figsize=(6.2, 5.6))
ax.scatter(y, pred, color="#4a6fa5", s=40)
lim = [0, max(y.max(), pred.max()) * 1.05]
ax.plot(lim, lim, color="#c0504d", lw=1.3, ls="--")
for i in res["selisih"].abs().nlargest(3).index:
    ax.annotate(df.loc[i, "kecamatan"].title(), (y[i], pred[i]), xytext=(5, 4), textcoords="offset points", fontsize=8)
ax.set_xlim(lim); ax.set_ylim(lim)
ax.set_xlabel("Volume sampah aktual (m³/bulan)"); ax.set_ylabel("Prediksi (cross-validation)")
ax.set_title("Aktual vs prediksi (garis putus = prediksi sempurna)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(MD / "1_aktual_vs_prediksi.png", dpi=140); plt.close(fig)

# grafik 2: koefisien model penuh
k = koef.sort_values("koefisien_terstandar")
fig, ax = plt.subplots(figsize=(6.8, 3.8))
ax.barh(k["fitur"], k["koefisien_terstandar"], color=["#c0504d" if v < 0 else "#4a6fa5" for v in k["koefisien_terstandar"]])
ax.axvline(0, color="gray", lw=0.8)
ax.set_xlabel("Koefisien terstandar (m³ per +1 simpangan baku)")
ax.set_title("Pengaruh tiap fitur (model 4 fitur)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(MD / "2_koefisien.png", dpi=140); plt.close(fig)

# grafik 3: perbandingan model (R2 dan MAE)
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
nm = [n.replace(": ", ":\n") for n in hasil["model"]]
ax[0].barh(nm, hasil["R2"], color="#4a6fa5"); ax[0].set_title("R² (makin tinggi makin baik)")
ax[1].barh(nm, hasil["MAE"], color="#c0504d"); ax[1].set_title("MAE, m³ (makin rendah makin baik)")
ax[1].set_yticklabels([])
for a in ax:
    a.tick_params(labelsize=7); a.spines[["top", "right"]].set_visible(False)
ax[0].axvline(0, color="gray", lw=0.8)
fig.tight_layout(); fig.savefig(MD / "3_perbandingan_model.png", dpi=140); plt.close(fig)

# grafik 4: residual
fig, ax = plt.subplots(figsize=(6.4, 4.4))
ax.scatter(pred, res["selisih"], color="#4a6fa5", s=40)
ax.axhline(0, color="#c0504d", lw=1.2, ls="--")
ax.set_xlabel("Prediksi (m³/bulan)"); ax.set_ylabel("Selisih aktual - prediksi")
ax.set_title("Plot residual (model terpilih)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(MD / "4_residual.png", dpi=140); plt.close(fig)

(MD / "ringkasan_regresi.txt").write_text("\n\n".join(log), encoding="utf-8")
print("\n\n".join(log))
