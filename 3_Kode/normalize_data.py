"""
Tahap 4: Normalisasi data final 2024

Jalankan:  python normalize_data.py <folder_output>
Membaca data_final_2024.csv (hasil clean_data.py).

Langkah normalisasi:
  1. Ukur kemiringan (skewness) tiap variabel.
  2. Variabel yang miring ke kanan (|skewness| > 1) ditransformasi log(1 + x)
     supaya kecamatan besar (outlier alami) tidak mendominasi.
  3. Semua variabel distandardisasi dengan Z-score: z = (x - rata-rata) / simpangan baku
     sehingga rata-rata 0 dan simpangan baku 1 (skala disamakan untuk PCA dan K-Means).
"""
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import skew

OUT = Path(sys.argv[1])
df = pd.read_csv(OUT / "data_final_2024.csv")

VAR = ["volume_sampah_rata2", "jumlah_penduduk", "luas_km2", "kepadatan", "total_luas_rth",
       "jumlah_rth", "rth_per_kapita", "ip_air_rata2", "jumlah_titik_air"]
# variabel yang dipakai untuk model (sisanya redundan: luas_km2 ~ penduduk & kepadatan,
# total_luas_rth ~ rth_per_kapita, jumlah_rth dan jumlah_titik_air bersifat pendukung)
FITUR_MODEL = ["volume_sampah_rata2", "jumlah_penduduk", "kepadatan", "rth_per_kapita", "ip_air_rata2"]
BATAS_SKEW = 1.0

ringkas, transform, z = [], pd.DataFrame(), pd.DataFrame()
for c in VAR:
    x = df[c].astype(float)
    sk0 = float(skew(x))
    pakai_log = abs(sk0) > BATAS_SKEW
    t = np.log1p(x) if pakai_log else x
    sk1 = float(skew(t))
    mu, sd = t.mean(), t.std(ddof=0)  # ddof=0 sama seperti StandardScaler dan STDEVP di Excel
    transform[c] = t
    z[c] = (t - mu) / sd
    ringkas.append({
        "variabel": c,
        "min_asli": x.min(), "maks_asli": x.max(),
        "rata2_asli": x.mean(), "simpangan_baku_asli": x.std(ddof=0),
        "skewness_asli": sk0,
        "transformasi": "log(1+x) lalu Z-score" if pakai_log else "Z-score",
        "skewness_setelah_transformasi": sk1,
        "fitur_model": "ya" if c in FITUR_MODEL else "tidak",
    })

ringkas = pd.DataFrame(ringkas).round(4)
out = pd.concat([df[["wilayah", "kecamatan"]], z.add_prefix("z_")], axis=1).round(6)
out.to_csv(OUT / "data_normalisasi_2024.csv", index=False)
ringkas.to_csv(OUT / "ringkasan_normalisasi.csv", index=False)

# cek: rata-rata ~0, simpangan baku ~1
print(out.drop(columns=["wilayah", "kecamatan"]).agg(["mean", "std"]).round(3).to_string())
print(ringkas[["variabel", "skewness_asli", "transformasi", "skewness_setelah_transformasi"]].to_string())

# grafik sebelum / sesudah untuk variabel yang di-log
log_vars = [r["variabel"] for r in ringkas.to_dict("records") if r["transformasi"].startswith("log")]
fig, ax = plt.subplots(2, len(log_vars), figsize=(3 * len(log_vars), 5.2))
for j, c in enumerate(log_vars):
    ax[0, j].hist(df[c], bins=12, color="#9bb1d3", edgecolor="white")
    ax[0, j].set_title(c, fontsize=9)
    ax[1, j].hist(z[c], bins=12, color="#1f3864", edgecolor="white")
    ax[0, j].tick_params(labelsize=7)
    ax[1, j].tick_params(labelsize=7)
ax[0, 0].set_ylabel("Sebelum\n(nilai asli)", fontsize=9)
ax[1, 0].set_ylabel("Sesudah\n(log + Z-score)", fontsize=9)
fig.suptitle("Sebaran variabel sebelum dan sesudah normalisasi (42 kecamatan)", fontsize=11)
fig.tight_layout()
fig.savefig(OUT / "normalisasi_sebelum_sesudah.png", dpi=140)
print("selesai")
