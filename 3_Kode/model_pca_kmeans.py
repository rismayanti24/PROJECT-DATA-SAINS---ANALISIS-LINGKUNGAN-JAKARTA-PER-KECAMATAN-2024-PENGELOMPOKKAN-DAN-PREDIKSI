"""
Tahap 6: PCA + K-Means (pengelompokan 42 kecamatan)

Jalankan:  python model_pca_kmeans.py <folder_output>
Membaca data_final_2024.csv dan data_normalisasi_2024.csv.
Menghasilkan folder <folder_output>/model_kmeans/ berisi grafik, hasil_cluster.csv,
profil_cluster.csv, dan ringkasan_kmeans.txt.
"""
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(".")
MD = OUT / "model_kmeans"
MD.mkdir(exist_ok=True)
SEED = 42

asli = pd.read_csv(OUT / "data_final_2024.csv")
z = pd.read_csv(OUT / "data_normalisasi_2024.csv")
FITUR = ["volume_sampah_rata2", "jumlah_penduduk", "kepadatan", "rth_per_kapita", "ip_air_rata2"]
NAMA = {"volume_sampah_rata2": "Volume sampah", "jumlah_penduduk": "Penduduk", "kepadatan": "Kepadatan",
        "rth_per_kapita": "RTH per kapita", "ip_air_rata2": "IP air sumur"}
X = z[["z_" + c for c in FITUR]].values
log = []

# ---- 1. PCA ----
pca = PCA(random_state=SEED).fit(X)
ev = pca.explained_variance_ratio_
log.append("PCA: varians yang dijelaskan tiap komponen (PC1..PC5): " + ", ".join(f"{v:.1%}" for v in ev))
log.append(f"PC1+PC2 = {ev[:2].sum():.1%}")
load = pd.DataFrame(pca.components_[:2].T, index=[NAMA[c] for c in FITUR], columns=["PC1", "PC2"]).round(3)
log.append("Loading (bobot variabel pada PC1 dan PC2):\n" + load.to_string())
load.to_csv(MD / "loading_pca.csv")
P = PCA(n_components=2, random_state=SEED).fit_transform(X)

fig, ax = plt.subplots(figsize=(6.5, 4.2))
ax.bar(range(1, 6), ev * 100, color="#4a6fa5")
ax.plot(range(1, 6), np.cumsum(ev) * 100, color="#c0504d", marker="o")
ax.set_xlabel("Komponen utama"); ax.set_ylabel("Varians dijelaskan (%)")
ax.set_title("PCA: varians per komponen (biru) dan kumulatif (merah)")
ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(MD / "1_pca_varians.png", dpi=140); plt.close(fig)

# ---- 2. Elbow + Silhouette (dihitung pada data ternormalisasi lengkap) ----
ks = range(2, 9)
inertia, sil = [], []
for k in ks:
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    inertia.append(km.inertia_); sil.append(silhouette_score(X, km.labels_))
log.append("k, inertia, silhouette:\n" + "\n".join(f"  k={k}: {i:.1f}, {s:.3f}" for k, i, s in zip(ks, inertia, sil)))
fig, ax = plt.subplots(1, 2, figsize=(10, 4))
ax[0].plot(list(ks), inertia, marker="o", color="#1f3864"); ax[0].set_title("Elbow (inertia)")
ax[1].plot(list(ks), sil, marker="o", color="#c0504d"); ax[1].set_title("Silhouette score")
for a in ax:
    a.set_xlabel("Jumlah cluster (k)"); a.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(MD / "2_elbow_silhouette.png", dpi=140); plt.close(fig)

# ---- 3. K-Means dengan k terpilih ----
K = int(sys.argv[2]) if len(sys.argv) > 2 else int(list(ks)[int(np.argmax(sil))])
km = KMeans(n_clusters=K, n_init=20, random_state=SEED).fit(X)
log.append(f"k dipakai = {K} (silhouette = {silhouette_score(X, km.labels_):.3f})")
res = asli[["wilayah", "kecamatan"]].copy()
res["cluster"] = km.labels_ + 1
res["PC1"], res["PC2"] = P[:, 0].round(3), P[:, 1].round(3)
res.to_csv(MD / "hasil_cluster.csv", index=False)

prof = asli[FITUR].copy(); prof["cluster"] = res["cluster"]
prof = prof.groupby("cluster").agg(["mean"]).droplevel(1, axis=1)
prof.insert(0, "jumlah_kecamatan", res["cluster"].value_counts().sort_index())
prof.round(2).to_csv(MD / "profil_cluster.csv")
log.append("Profil rata-rata tiap cluster (nilai asli):\n" + prof.round(2).to_string())
for c in sorted(res["cluster"].unique()):
    log.append(f"Cluster {c}: " + ", ".join(res.loc[res.cluster == c, "kecamatan"].str.title()))

fig, ax = plt.subplots(figsize=(8, 5.6))
warna = ["#1f3864", "#c0504d", "#4a8f5a", "#d9a21b", "#7b5ea7", "#2a9d9d"]
for c in sorted(res["cluster"].unique()):
    m = res["cluster"] == c
    ax.scatter(res.loc[m, "PC1"], res.loc[m, "PC2"], s=45, color=warna[c - 1], label=f"Cluster {c} (n={m.sum()})")
for _, r in res.iterrows():
    ax.annotate(r["kecamatan"].title(), (r["PC1"], r["PC2"]), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
ax.set_xlabel(f"PC1 ({ev[0]:.0%})"); ax.set_ylabel(f"PC2 ({ev[1]:.0%})")
ax.set_title("Cluster kecamatan pada ruang PCA")
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(MD / "3_cluster_pca.png", dpi=140); plt.close(fig)

# profil dalam z-score (supaya variabel sebanding) -> grafik batang berkelompok
zp = pd.DataFrame(X, columns=[NAMA[c] for c in FITUR]); zp["cluster"] = res["cluster"]
zm = zp.groupby("cluster").mean()
fig, ax = plt.subplots(figsize=(8, 4.5))
w = 0.8 / len(zm)
for i, c in enumerate(zm.index):
    ax.bar(np.arange(5) + i * w, zm.loc[c], w, color=warna[c - 1], label=f"Cluster {c}")
ax.set_xticks(np.arange(5) + 0.4 - w / 2); ax.set_xticklabels(zm.columns)
ax.axhline(0, color="gray", lw=0.8); ax.set_ylabel("Rata-rata Z-score")
ax.set_title("Karakter tiap cluster (di atas 0 = di atas rata-rata Jakarta)")
ax.legend(frameon=False, fontsize=8); ax.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(MD / "4_profil_cluster.png", dpi=140); plt.close(fig)

(MD / "ringkasan_kmeans.txt").write_text("\n\n".join(log), encoding="utf-8")
print("\n\n".join(log))
