"""
Tahap 2-3: Cleaning + Penggabungan data (tahun 2024)
Proyek Data Sains: Analisis Lingkungan Jakarta per Kecamatan

Jalankan:  python clean_data.py <folder_data_mentah> <folder_output>

Sumber (Satu Data Jakarta):
  1. Rata-rata volume sampah sungai   (Dinas Lingkungan Hidup)
  2. Kepadatan penduduk               (Kemendagri / Dukcapil)
  3. Luas RTH publik                  (Dinas Pertamanan dan Hutan Kota)
  4. Kualitas air sumur               (Dinas Lingkungan Hidup)

Catatan: file .xls dari portal sebenarnya berisi tabel HTML, jadi dibuka
dengan pd.read_html(), bukan pd.read_excel().
"""
import re
import sys
from pathlib import Path

import pandas as pd

RAW = Path(sys.argv[1])
OUT = Path(sys.argv[2])
OUT.mkdir(parents=True, exist_ok=True)

FILES = {
    "sampah": "fbd3391d-rata-rata-volume-sampah-sungai-di-provinsi-dki-jakarta-menurut-kota-administrasi-komponen-data.xls",
    "kepadatan": "369590d1-data-kepadatan-penduduk-provinsi-dki-jakarta-komponen-data.xls",
    "rth": "a09b27e7-data-luas-ruang-terbuka-hijau-rth-publik-komponen-data.xls",
    "air": "6b5b13ea-data-kualitas-air-sumur-komponen-data.xls",
}
TAHUN = 2024

LOG = []  # catatan cleaning: dipakai untuk laporan Bab 3


def catat(dataset, masalah, jumlah, tindakan, alasan):
    LOG.append(
        {
            "dataset": dataset,
            "masalah": masalah,
            "jumlah_terdampak": jumlah,
            "tindakan": tindakan,
            "alasan": alasan,
        }
    )


def baca(nama):
    return pd.read_html(RAW / FILES[nama])[0]


# ---------------------------------------------------------------- helper nama
def wilayah_baku(w):
    """Samakan penulisan wilayah (ada banyak variasi di file mentah)."""
    w = str(w).upper().strip()
    for kunci, baku in [
        ("SERIBU", "KAB. ADM. KEP. SERIBU"),
        ("PUSAT", "KOTA ADM. JAKARTA PUSAT"),
        ("UTARA", "KOTA ADM. JAKARTA UTARA"),
        ("BARAT", "KOTA ADM. JAKARTA BARAT"),
        ("SELATAN", "KOTA ADM. JAKARTA SELATAN"),
        ("TIMUR", "KOTA ADM. JAKARTA TIMUR"),
    ]:
        if kunci in w:
            return baku
    return w


def kunci_kec(k):
    """KRAMATJATI, KRAMAT JATI, Kramat Jati -> 'KRAMATJATI'."""
    return re.sub(r"[^A-Z]", "", str(k).upper())


# Daftar kecamatan baku diambil dari data kepadatan (44 kecamatan, ejaan rapi)
kep_raw = baca("kepadatan")
kep_raw["wilayah"] = kep_raw["wilayah"].map(wilayah_baku)
kep_raw["kecamatan"] = kep_raw["kecamatan"].astype(str).str.strip().str.upper()
BAKU = {}  # (wilayah, kunci) -> nama kecamatan baku
for (w, k), _ in kep_raw.groupby(["wilayah", "kecamatan"]):
    BAKU[(w, kunci_kec(k))] = k
KEC_KE_WILAYAH = {}  # kunci kecamatan -> wilayah, hanya jika tidak ambigu
for (w, kk), _ in BAKU.items():
    KEC_KE_WILAYAH.setdefault(kk, set()).add(w)


def samakan(df, dataset):
    """Samakan wilayah + kecamatan ke ejaan baku. Perbaiki wilayah yang salah."""
    df = df.copy()
    df["wilayah"] = df["wilayah"].map(wilayah_baku)
    hasil_w, hasil_k, salah_wilayah, diubah_ejaan = [], [], 0, 0
    for w, k in zip(df["wilayah"], df["kecamatan"]):
        kk = kunci_kec(k)
        if (w, kk) in BAKU:
            baku = BAKU[(w, kk)]
            if baku != str(k).strip().upper():
                diubah_ejaan += 1
            hasil_w.append(w)
            hasil_k.append(baku)
        elif len(KEC_KE_WILAYAH.get(kk, [])) == 1:
            w2 = next(iter(KEC_KE_WILAYAH[kk]))
            salah_wilayah += 1
            hasil_w.append(w2)
            hasil_k.append(BAKU[(w2, kk)])
        else:
            hasil_w.append(w)
            hasil_k.append(str(k).strip().upper())
    df["wilayah"], df["kecamatan"] = hasil_w, hasil_k
    if diubah_ejaan:
        catat(dataset, "Ejaan nama kecamatan berbeda antar file (contoh KRAMATJATI vs KRAMAT JATI)",
              diubah_ejaan, "Disamakan ke ejaan baku dari data kepadatan",
              "Supaya kunci gabung wilayah + kecamatan cocok")
    if salah_wilayah:
        catat(dataset, "Kecamatan tercatat di wilayah (kota) yang salah", salah_wilayah,
              "Wilayah diperbaiki sesuai kecamatan sebenarnya",
              "Kecamatan hanya ada di satu kota, jadi wilayah yang benar bisa dipastikan")
    return df


# ------------------------------------------------------------------- 1 SAMPAH
s = baca("sampah")
n_awal = len(s)
n_spasi = int((s["wilayah"].astype(str) != s["wilayah"].astype(str).str.strip()).sum())
s["wilayah"] = s["wilayah"].astype(str).str.strip()  # ada tab di akhir teks
if n_spasi:  # pd.read_html biasanya sudah memangkas spasi/tab, jadi hanya dicatat bila ada
    catat("Sampah", "Kolom wilayah mengandung spasi/tab di akhir teks", n_spasi,
          "Dihapus dengan strip()", "Supaya nama wilayah konsisten")
s = s[(s["periode_data"] >= TAHUN * 100 + 1) & (s["periode_data"] <= TAHUN * 100 + 12)].copy()
catat("Sampah", f"Data mencakup Jan 2024 - Jul 2026 ({n_awal} baris)", int(n_awal - len(s)),
      "Diambil periode 202401-202412 saja", "Tahun disamakan dengan dataset lain (2024)")
s = samakan(s, "Sampah")

s["rasio_volume_residu"] = s["volume_sampah"] / s["total_sampah_residu"]
outlier = s["rasio_volume_residu"] > 5
for _, r in s[outlier].iterrows():
    catat("Sampah",
          f"Outlier: {r['kecamatan']} periode {int(r['periode_data'])}, volume {r['volume_sampah']:.0f} "
          f"padahal residu {r['total_sampah_residu']:.0f} (rasio {r['rasio_volume_residu']:.1f}x)",
          1, "Nilai volume diganti median volume kecamatan itu pada bulan lain",
          "Rasio volume/residu normalnya 1-3x, rasio >5x kemungkinan salah input")
    med = s[(s["kecamatan"] == r["kecamatan"]) & (~outlier)]["volume_sampah"].median()
    s.loc[_, "volume_sampah"] = med
s = s.drop(columns="rasio_volume_residu")
# data bulanan yang sudah bersih (untuk grafik tren per bulan)
sampah_bulanan = s[["wilayah", "kecamatan", "periode_data", "volume_sampah", "total_sampah_residu"]].copy()
sampah_bulanan["bulan"] = sampah_bulanan["periode_data"] % 100
sampah_bulanan = (sampah_bulanan[["wilayah", "kecamatan", "periode_data", "bulan", "volume_sampah", "total_sampah_residu"]]
                  .sort_values(["wilayah", "kecamatan", "periode_data"]).reset_index(drop=True))
sampah = (s.groupby(["wilayah", "kecamatan"], as_index=False)
            .agg(volume_sampah_rata2=("volume_sampah", "mean"),
                 residu_rata2=("total_sampah_residu", "mean"),
                 jumlah_bulan=("periode_data", "nunique")))
sampah[["volume_sampah_rata2", "residu_rata2"]] = sampah[["volume_sampah_rata2", "residu_rata2"]].round(2)
catat("Sampah", "Data bulanan per kecamatan (12 bulan)", len(s),
      "Dirata-ratakan per kecamatan jadi volume_sampah_rata2 (meter kubik)",
      "Satuan analisis adalah kecamatan")

# ---------------------------------------------------------------- 2 KEPADATAN
k = kep_raw[kep_raw["periode_data"] == TAHUN].copy()
catat("Kepadatan", "Data memuat 2024 dan 2025 (534 baris)", int(len(kep_raw) - len(k)),
      "Diambil 2024 saja (267 kelurahan)", "Tahun disamakan dengan dataset lain")
k = samakan(k, "Kepadatan")
k["luas_km2"] = k["jumlah_penduduk"] / k["jumlah_kepadatan"]
catat("Kepadatan", "Kepadatan per kelurahan tidak boleh dirata-ratakan biasa", len(k),
      "Luas kelurahan dihitung = penduduk / kepadatan, dijumlahkan per kecamatan, "
      "kepadatan kecamatan = total penduduk / total luas",
      "Rata-rata biasa akan salah karena luas tiap kelurahan berbeda")
kepadatan = (k.groupby(["wilayah", "kecamatan"], as_index=False)
               .agg(jumlah_penduduk=("jumlah_penduduk", "sum"), luas_km2=("luas_km2", "sum")))
kepadatan["kepadatan"] = (kepadatan["jumlah_penduduk"] / kepadatan["luas_km2"]).round(2)
kepadatan["luas_km2"] = kepadatan["luas_km2"].round(6)

# ------------------------------------------------------------------------ 3 RTH
r = baca("rth")
n_rth = len(r)
r = r[r["periode_data"] == TAHUN].copy()
catat("RTH", f"Data mencakup 2022-2025 ({n_rth} baris)", int(n_rth - len(r)),
      "Diambil 2024 saja", "Tahun disamakan dengan dataset lain")
r["luas"] = pd.to_numeric(r["luas"], errors="coerce")  # luas terbaca sebagai teks
catat("RTH", "Kolom luas terbaca sebagai teks", int(len(r)), "Diubah ke angka (to_numeric)",
      "Supaya bisa dijumlahkan")
r = samakan(r, "RTH")
dup = r.duplicated(["kecamatan", "nama_rth", "alamat", "luas"])
catat("RTH", "Baris dobel (kecamatan, nama, alamat, luas sama)", int(dup.sum()),
      "Baris dobel dihapus", "Satu RTH tercatat dua kali akan menggandakan luas")
r = r[~dup]
catat("RTH", "Kolom kelurahan kosong di banyak baris", int(r["kelurahan"].isna().sum()),
      "Analisis dilakukan di level kecamatan, kelurahan tidak dipakai",
      "Kecamatan terisi lengkap, kelurahan tidak")
r = r.dropna(subset=["luas"])
rth = (r.groupby(["wilayah", "kecamatan"], as_index=False)
         .agg(total_luas_rth=("luas", "sum"), jumlah_rth=("nama_rth", "count")))
rth["total_luas_rth"] = rth["total_luas_rth"].round(2)

# ------------------------------------------------------------------- 4 AIR SUMUR
a = baca("air")
n_air = len(a)
a = a[a["periode_data"] == TAHUN].copy()
catat("Air sumur", f"Data memuat 2024 dan 2025 ({n_air} baris); 2025 hanya 8 parameter tanpa IP/STATUS",
      int(n_air - len(a)), "Diambil 2024 saja", "Hanya 2024 yang punya Indeks Pencemaran (IP) dan STATUS")
a = samakan(a, "Air sumur")
ip = a[a["parameter"] == "IP"].copy()
catat("Air sumur", "Data berbentuk panjang (banyak parameter per lokasi)", int(len(a)),
      "Diambil baris parameter = IP (Indeks Pencemaran) saja",
      "IP sudah merangkum kualitas air per lokasi dan musim")
ip["ip"] = pd.to_numeric(ip["nilai"], errors="coerce")
catat("Air sumur", "Kolom nilai terbaca sebagai teks", int(len(ip)), "Diubah ke angka",
      "Semua nilai IP berhasil diubah (tidak ada yang gagal)")
ip = ip.dropna(subset=["ip"])
status = a[(a["parameter"] == "STATUS") & a["nilai"].isin(["BAIK", "RINGAN", "SEDANG", "BERAT"])]
catat("Air sumur", "Ada 1 baris STATUS bernilai angka (0.0050) bukan kategori", 1,
      "Baris itu diabaikan", "Nilai bukan salah satu dari BAIK/RINGAN/SEDANG/BERAT")
dom = (status.groupby(["wilayah", "kecamatan"])["nilai"]
             .agg(lambda x: x.value_counts().idxmax()).rename("status_mutu_dominan").reset_index())
air = (ip.groupby(["wilayah", "kecamatan"], as_index=False)
         .agg(ip_air_rata2=("ip", "mean"), jumlah_titik_air=("nama_lokasi", "nunique")))
air["ip_air_rata2"] = air["ip_air_rata2"].round(3)
air = air.merge(dom, on=["wilayah", "kecamatan"], how="left")
catat("Air sumur", "Jumlah titik pantau berbeda tiap kecamatan (1 sampai 11 titik)", len(air),
      "Dicatat di kolom jumlah_titik_air", "Kecamatan dengan titik sedikit kurang mewakili (keterbatasan)")

# --------------------------------------------------------------------- GABUNG
kunci = ["wilayah", "kecamatan"]
final = (sampah[kunci + ["volume_sampah_rata2"]]
         .merge(kepadatan, on=kunci, how="left")
         .merge(rth, on=kunci, how="left")
         .merge(air, on=kunci, how="left"))
final["rth_per_kapita"] = (final["total_luas_rth"] / final["jumlah_penduduk"]).round(6)
catat("Gabungan", "Kepulauan Seribu (2 kecamatan) hanya ada di data kepadatan dan air sumur, tidak ada di data sampah dan RTH",
      int((~kepadatan["wilayah"].isin(final["wilayah"])).sum()),
      "Dikeluarkan dari data final (basis data sampah 42 kecamatan)",
      "Data sampah dan RTH tidak punya Kepulauan Seribu")
catat("Gabungan", "Penggabungan 4 dataset dengan kunci wilayah + kecamatan", len(final),
      "Left join dari data sampah", "Kunci wilayah + kecamatan lebih aman daripada nama kecamatan saja")
catat("Gabungan", "RTH per kapita belum ada", len(final),
      "Dihitung = total_luas_rth / jumlah_penduduk", "Membandingkan RTH antar kecamatan dengan adil (satuan luas RTH tidak dicantumkan portal)")

kolom = ["wilayah", "kecamatan", "volume_sampah_rata2", "jumlah_penduduk", "luas_km2", "kepadatan",
         "total_luas_rth", "jumlah_rth", "rth_per_kapita", "ip_air_rata2", "jumlah_titik_air",
         "status_mutu_dominan"]
final = final[kolom].sort_values(kunci).reset_index(drop=True)

# ---------------------------------------------------------------------- SIMPAN
sampah.to_csv(OUT / "sampah_2024_bersih.csv", index=False)
sampah_bulanan.to_csv(OUT / "sampah_bulanan_2024_bersih.csv", index=False)
kepadatan.to_csv(OUT / "kepadatan_2024_bersih.csv", index=False)
rth.to_csv(OUT / "rth_2024_bersih.csv", index=False)
air.to_csv(OUT / "air_sumur_2024_bersih.csv", index=False)
final.to_csv(OUT / "data_final_2024.csv", index=False)
pd.DataFrame(LOG).to_csv(OUT / "catatan_cleaning.csv", index=False)

print("Sampah", sampah.shape, "| Kepadatan", kepadatan.shape, "| RTH", rth.shape, "| Air", air.shape)
print("FINAL", final.shape)
print("missing per kolom:\n", final.isna().sum().to_string())
print(final.describe().T.to_string())
