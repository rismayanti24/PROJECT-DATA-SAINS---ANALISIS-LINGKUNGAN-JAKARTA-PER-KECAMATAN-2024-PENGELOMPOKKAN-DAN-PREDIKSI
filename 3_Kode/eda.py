"""
Tahap 5: EDA (Exploratory Data Analysis)

Jalankan:  python eda.py <folder_output>
Membaca data_final_2024.csv dan sampah_bulanan_2024_bersih.csv (hasil clean_data.py).
Menghasilkan folder <folder_output>/eda/ berisi 5 grafik (.png),
korelasi_spearman.csv, dan ringkasan_eda.txt.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

OUT = Path(sys.argv[1])
EDA = OUT / "eda"
EDA.mkdir(exist_ok=True)

df = pd.read_csv(OUT / "data_final_2024.csv")
bln = pd.read_csv(OUT / "sampah_bulanan_2024_bersih.csv")
df["kec"] = df["kecamatan"].str.title()

BIRU, GELAP, MERAH = "#4a6fa5", "#1f3864", "#c0504d"
plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False})

# 1. Bar chart: 10 kecamatan dengan sampah terbanyak
top = df.nlargest(10, "volume_sampah_rata2").sort_values("volume_sampah_rata2")
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(top["kec"], top["volume_sampah_rata2"], color=BIRU)
for i, v in enumerate(top["volume_sampah_rata2"]):
    ax.text(v + 20, i, f"{v:,.0f}", va="center", fontsize=9)
ax.set_xlabel("Rata-rata volume sampah per bulan (m³)")
ax.set_title("10 kecamatan dengan volume sampah tertinggi (2024)")
fig.tight_layout(); fig.savefig(EDA / "1_bar_top10_sampah.png", dpi=140); plt.close(fig)

# 2. Line chart: total sampah Jakarta per bulan, per kota administrasi
bln["kota"] = bln["wilayah"].str.replace("KOTA ADM. JAKARTA ", "", regex=False).str.title()
pv = bln.pivot_table(index="bulan", columns="kota", values="volume_sampah", aggfunc="sum")
fig, ax = plt.subplots(figsize=(8, 4.8))
for k in pv.columns:
    ax.plot(pv.index, pv[k], marker="o", ms=3, label=k)
ax.set_xticks(range(1, 13))
ax.set_xticklabels(["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"])
ax.set_ylabel("Total volume sampah (m³)")
ax.set_title("Volume sampah per bulan menurut kota administrasi (2024)")
ax.legend(frameon=False, fontsize=8)
fig.tight_layout(); fig.savefig(EDA / "2_line_sampah_bulanan.png", dpi=140); plt.close(fig)

# 3. Boxplot: sebaran sampah per kota administrasi
df["kota"] = df["wilayah"].str.replace("KOTA ADM. JAKARTA ", "", regex=False).str.title()
urut = df.groupby("kota")["volume_sampah_rata2"].median().sort_values().index
fig, ax = plt.subplots(figsize=(8, 4.8))
ax.boxplot([df.loc[df.kota == k, "volume_sampah_rata2"] for k in urut], vert=True,
           patch_artist=True, boxprops=dict(facecolor="#cdd9ec"), medianprops=dict(color=GELAP))
ax.set_xticklabels(urut)
ax.set_ylabel("Rata-rata volume sampah (m³/bulan)")
ax.set_title("Sebaran volume sampah antar kecamatan, per kota administrasi")
fig.tight_layout(); fig.savefig(EDA / "3_boxplot_sampah_per_kota.png", dpi=140); plt.close(fig)

# 4. Scatter: penduduk vs sampah
fig, ax = plt.subplots(figsize=(7, 5))
ax.scatter(df["jumlah_penduduk"], df["volume_sampah_rata2"], color=BIRU, s=35)
m, b = np.polyfit(df["jumlah_penduduk"], df["volume_sampah_rata2"], 1)
xs = np.linspace(df["jumlah_penduduk"].min(), df["jumlah_penduduk"].max(), 50)
ax.plot(xs, m * xs + b, color=MERAH, lw=1.5)
for _, r in df.nlargest(3, "volume_sampah_rata2").iterrows():
    ax.annotate(r["kec"], (r["jumlah_penduduk"], r["volume_sampah_rata2"]), xytext=(5, 4),
                textcoords="offset points", fontsize=8)
ax.set_xlabel("Jumlah penduduk"); ax.set_ylabel("Rata-rata volume sampah (m³/bulan)")
ax.set_title("Jumlah penduduk vs volume sampah (tiap titik = 1 kecamatan)")
fig.tight_layout(); fig.savefig(EDA / "4_scatter_penduduk_sampah.png", dpi=140); plt.close(fig)

# 5. Heatmap korelasi (Spearman, tahan terhadap data miring)
kolom = {"volume_sampah_rata2": "Volume sampah", "jumlah_penduduk": "Penduduk", "kepadatan": "Kepadatan",
         "rth_per_kapita": "RTH per kapita", "ip_air_rata2": "IP air sumur"}
cor = df[list(kolom)].rename(columns=kolom).corr(method="spearman")
cor.round(3).to_csv(EDA / "korelasi_spearman.csv")
fig, ax = plt.subplots(figsize=(6.4, 5.4))
im = ax.imshow(cor, cmap="RdBu_r", vmin=-1, vmax=1)
ax.set_xticks(range(len(cor))); ax.set_xticklabels(cor.columns, rotation=35, ha="right")
ax.set_yticks(range(len(cor))); ax.set_yticklabels(cor.columns)
for i in range(len(cor)):
    for j in range(len(cor)):
        ax.text(j, i, f"{cor.iloc[i, j]:.2f}", ha="center", va="center", fontsize=9,
                color="white" if abs(cor.iloc[i, j]) > 0.6 else "black")
fig.colorbar(im, shrink=0.8, label="Korelasi Spearman")
ax.set_title("Korelasi antar variabel")
fig.tight_layout(); fig.savefig(EDA / "5_heatmap_korelasi.png", dpi=140); plt.close(fig)

# ringkasan angka untuk bahan laporan
with open(EDA / "ringkasan_eda.txt", "w", encoding="utf-8") as f:
    f.write("Statistik deskriptif (42 kecamatan)\n")
    f.write(df[list(kolom)].rename(columns=kolom).describe().round(2).T.to_string() + "\n\n")
    f.write("Korelasi Spearman dengan volume sampah\n")
    f.write(cor["Volume sampah"].drop("Volume sampah").sort_values(ascending=False).round(3).to_string() + "\n\n")
    bul = bln.groupby("bulan")["volume_sampah"].sum()
    f.write(f"Bulan sampah tertinggi: {bul.idxmax()} ({bul.max():,.0f} m3); terendah: {bul.idxmin()} ({bul.min():,.0f} m3)\n")
    f.write("Top 5 kecamatan sampah: " + ", ".join(df.nlargest(5, "volume_sampah_rata2")["kec"]) + "\n")
    f.write("Top 5 RTH per kapita: " + ", ".join(df.nlargest(5, "rth_per_kapita")["kec"]) + "\n")
    f.write("Terendah RTH per kapita: " + ", ".join(df.nsmallest(5, "rth_per_kapita")["kec"]) + "\n")
    f.write("Status mutu air dominan:\n" + df["status_mutu_dominan"].value_counts().to_string() + "\n")
print(open(EDA / "ringkasan_eda.txt", encoding="utf-8").read())
