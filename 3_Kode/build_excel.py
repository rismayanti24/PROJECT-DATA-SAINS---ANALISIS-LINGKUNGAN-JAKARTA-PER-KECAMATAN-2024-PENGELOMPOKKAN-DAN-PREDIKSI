"""Susun hasil cleaning jadi satu workbook Excel yang rapi untuk dibaca tim."""
import sys
from pathlib import Path

import pandas as pd
from openpyxl import Workbook
from openpyxl.styles import Alignment, Border, Font, PatternFill, Side
from openpyxl.utils import get_column_letter

OUT = Path(sys.argv[1])
FONT = "Arial"
HEAD_FILL = PatternFill("solid", start_color="1F3864")
ALT_FILL = PatternFill("solid", start_color="F2F5FA")
SIDE = Side(style="thin", color="BFC7D5")
BORDER = Border(left=SIDE, right=SIDE, top=SIDE, bottom=SIDE)

wb = Workbook()


def judul(ws, teks, sub=None):
    ws["A1"] = teks
    ws["A1"].font = Font(name=FONT, size=14, bold=True, color="1F3864")
    if sub:
        ws["A2"] = sub
        ws["A2"].font = Font(name=FONT, size=10, italic=True, color="555555")


def tabel(ws, df, mulai=4, lebar=None, fmt=None, wrap=()):
    """Tulis DataFrame ke sheet dengan header berwarna, garis tipis, baris selang-seling."""
    fmt = fmt or {}
    for j, c in enumerate(df.columns, 1):
        sel = ws.cell(row=mulai, column=j, value=c)
        sel.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
        sel.fill = HEAD_FILL
        sel.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
        sel.border = BORDER
    for i, baris in enumerate(df.itertuples(index=False), 1):
        for j, v in enumerate(baris, 1):
            if pd.isna(v):
                v = None
            sel = ws.cell(row=mulai + i, column=j, value=v)
            sel.font = Font(name=FONT, size=10)
            sel.border = BORDER
            sel.alignment = Alignment(vertical="top", wrap_text=df.columns[j - 1] in wrap)
            if df.columns[j - 1] in fmt:
                sel.number_format = fmt[df.columns[j - 1]]
            if i % 2 == 0:
                sel.fill = ALT_FILL
    ws.row_dimensions[mulai].height = 32
    for j, c in enumerate(df.columns, 1):
        w = (lebar or {}).get(c)
        if w is None:
            isi = [len(str(c))] + [len(str(x)) for x in df[c].head(200)]
            w = min(max(isi) + 3, 45)
        ws.column_dimensions[get_column_letter(j)].width = w
    ws.freeze_panes = ws.cell(row=mulai + 1, column=3 if "kecamatan" in df.columns else 1)
    ws.auto_filter.ref = f"A{mulai}:{get_column_letter(len(df.columns))}{mulai + len(df)}"
    ws.sheet_view.showGridLines = False


# ------------------------------------------------------------ Ringkasan
ws = wb.active
ws.title = "Ringkasan"
judul(ws, "Data Bersih Proyek Data Sains: Analisis Lingkungan Jakarta (2024)",
      "Hasil cleaning dan penggabungan 4 dataset Satu Data Jakarta, satuan analisis: kecamatan")
baris = [
    ("Sheet", "Isi"),
    ("Metadata", "Data collection: nama, instansi, tahun, jumlah baris, variabel, format, satuan, URL tiap dataset"),
    ("Catatan_Cleaning", "Daftar masalah di data mentah, jumlah yang terdampak, tindakan, dan alasannya (bahan Bab 3 laporan)"),
    ("Data_Final_2024", "Hasil gabungan 4 dataset, 42 kecamatan. Dipakai untuk EDA, PCA + K-Means, dan Linear Regression"),
    ("Normalisasi_2024", "Hasil normalisasi (log lalu Z-score) dengan rumus Excel, siap dipakai PCA dan K-Means"),
    ("Ringkasan_Normalisasi", "Alasan, skewness sebelum dan sesudah, dan grafik sebaran untuk tiap variabel"),
    ("Sampah_2024", "Volume sampah rata-rata per kecamatan (setelah outlier ditangani)"),
    ("Sampah_Bulanan_2024", "Volume sampah per bulan per kecamatan (504 baris), untuk grafik tren per bulan"),
    ("Kepadatan_2024", "Penduduk, luas, dan kepadatan per kecamatan (44 kecamatan, termasuk Kep. Seribu)"),
    ("RTH_2024", "Total luas dan jumlah RTH publik per kecamatan"),
    ("AirSumur_2024", "Indeks Pencemaran (IP) air sumur dan jumlah titik pantau per kecamatan (44 kecamatan)"),
]
for i, (a, b) in enumerate(baris, 4):
    for j, v in enumerate((a, b), 1):
        sel = ws.cell(row=i, column=j, value=v)
        sel.border = BORDER
        sel.alignment = Alignment(vertical="top", wrap_text=True)
        if i == 4:
            sel.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
            sel.fill = HEAD_FILL
        else:
            sel.font = Font(name=FONT, size=10, bold=(j == 1))
r0 = 4 + len(baris) + 1
ws.cell(row=r0, column=1, value="Kolom di Data_Final_2024").font = Font(name=FONT, bold=True, size=11, color="1F3864")
kolom = [
    ("Kolom", "Arti dan satuan"),
    ("wilayah, kecamatan", "Kunci gabung (huruf kapital, ejaan disamakan dengan data kepadatan)"),
    ("volume_sampah_rata2", "Rata-rata volume sampah per bulan selama 2024, satuan meter kubik (sesuai metadata)"),
    ("jumlah_penduduk", "Jumlah penduduk 2024 (jumlah dari semua kelurahan di kecamatan)"),
    ("luas_km2", "Luas kecamatan = jumlah (penduduk / kepadatan) per kelurahan, km2 (dihitung)"),
    ("kepadatan", "Penduduk / luas, orang per km2 (rumus, bukan rata-rata biasa)"),
    ("total_luas_rth", "Jumlah luas RTH publik di kecamatan. Satuan tidak dicantumkan portal"),
    ("jumlah_rth", "Banyaknya RTH publik yang tercatat di kecamatan"),
    ("rth_per_kapita", "total_luas_rth / jumlah_penduduk (rumus). Satuan mengikuti satuan luas RTH"),
    ("ip_air_rata2", "Rata-rata Indeks Pencemaran air sumur (semua titik dan 2 musim). Makin besar makin tercemar"),
    ("jumlah_titik_air", "Banyaknya titik pantau air sumur di kecamatan (3 sampai 11), dicatat sebagai keterbatasan"),
    ("status_mutu_dominan", "Status mutu air yang paling sering muncul (BAIK, RINGAN, SEDANG, BERAT)"),
]
for i, (a, b) in enumerate(kolom, r0 + 1):
    for j, v in enumerate((a, b), 1):
        sel = ws.cell(row=i, column=j, value=v)
        sel.border = BORDER
        sel.alignment = Alignment(vertical="top", wrap_text=True)
        if i == r0 + 1:
            sel.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
            sel.fill = HEAD_FILL
        else:
            sel.font = Font(name=FONT, size=10, bold=(j == 1))
r1 = r0 + len(kolom) + 2
catatan = [
    "Catatan: tahun 2024 dipilih karena hanya data air sumur 2024 yang punya Indeks Pencemaran (2025 hanya 8 parameter).",
    "Kepulauan Seribu dikeluarkan dari data final karena tidak ada di data sampah dan RTH.",
    "Kolom kepadatan dan rth_per_kapita di Data_Final_2024 berisi rumus Excel, jadi ikut berubah jika angka sumbernya diubah.",
]
for k, t in enumerate(catatan):
    c = ws.cell(row=r1 + k, column=1, value=t)
    c.font = Font(name=FONT, size=10, italic=True, color="555555")
ws.column_dimensions["A"].width = 26
ws.column_dimensions["B"].width = 100
ws.sheet_view.showGridLines = False

# ------------------------------------------------------------- Metadata
meta = pd.DataFrame([
    ["1", "Rata-rata Volume Sampah Sungai di DKI Jakarta menurut Kota Administrasi", "Dinas Lingkungan Hidup",
     "Januari 2024 - Juli 2026 (bulanan), dipakai 2024", 1302, 504, 6, ".xls (isi tabel HTML)", "meter kubik",
     "1 bulan sekali", "Metadata dibuat 4 Feb 2025, diperbarui 12 Okt 2026", "Terbuka", "dinaslh@jakarta.go.id",
     "https://satudata.jakarta.go.id/open-data/detail?kategori=dataset&page_url=rata-rata-volume-sampah-sungai-di-dki-jakarta-menurut-kota-administrasi&data_no=1"],
    ["2", "Data Kepadatan Penduduk Provinsi DKI Jakarta", "Kementerian Dalam Negeri, Dinas Kependudukan dan Pencatatan Sipil",
     "2024 dan 2025, dipakai 2024", 534, 267, 6, ".xls (isi tabel HTML)", "orang per kilometer persegi",
     "1 tahun sekali", "Metadata dibuat 14 Feb 2025, diperbarui 3 Sep 2026", "Terbuka", "bdi.dukcapil@jakarta.go.id",
     "https://satudata.jakarta.go.id/open-data/detail?kategori=dataset&page_url=data-kepadatan-penduduk-provinsi-dki-jakarta&data_no=1"],
    ["3", "Data Luas Ruang Terbuka Hijau (RTH) Publik", "Dinas Pertamanan dan Hutan Kota",
     "2022 - 2025, dipakai 2024", 10565, 2545, 8, ".xls (isi tabel HTML)", "tidak dicantumkan",
     "1 tahun sekali", "Metadata dibuat 29 Des 2023, diperbarui 18 Mei 2026, file data diperbarui 4 Mei 2026", "Terbuka",
     "distama@jakarta.go.id",
     "https://satudata.jakarta.go.id/open-data/detail?kategori=dataset&page_url=data-luas-ruang-terbuka-hijau-rth-publik&data_no=1"],
    ["4", "Data Kualitas Air Sumur", "Dinas Lingkungan Hidup",
     "2024 dan 2025, dipakai 2024", 16740, 12420, 10, ".xls (isi tabel HTML)", "tidak dicantumkan",
     "1 tahun sekali", "Metadata dibuat 9 Apr 2025, diperbarui 17 Apr 2026, file data diperbarui 17 Apr 2026", "Terbuka",
     "dinaslh@jakarta.go.id",
     "https://satudata.jakarta.go.id/open-data/detail?kategori=dataset&page_url=data-kualitas-air-sumur&data_no=1"],
], columns=["No", "Nama dataset", "Instansi", "Tahun data", "Jumlah baris (file)", "Jumlah baris 2024",
            "Jumlah variabel", "Format", "Satuan", "Pemutakhiran", "Tanggal metadata", "Sifat data", "Kontak", "URL"])
ws = wb.create_sheet("Metadata")
judul(ws, "Metadata Dataset (Data Collection)", "Portal: Satu Data Jakarta. Metadata disalin dari halaman dataset masing-masing.")
tabel(ws, meta, lebar={"No": 5, "Nama dataset": 40, "Instansi": 34, "Tahun data": 26, "Jumlah baris (file)": 12,
                       "Jumlah baris 2024": 12, "Jumlah variabel": 10, "Format": 18, "Satuan": 20, "Pemutakhiran": 14,
                       "Tanggal metadata": 38, "Sifat data": 10, "Kontak": 28, "URL": 80},
      fmt={"Jumlah baris (file)": "#,##0", "Jumlah baris 2024": "#,##0"},
      wrap=("Nama dataset", "Instansi", "Tahun data", "Tanggal metadata", "URL"))
ws.freeze_panes = "C5"
n = 4 + len(meta) + 2
ws.cell(row=n, column=2, value="Catatan: kolom 'Sumber Data' di halaman dataset air sumur kosong; instansi diambil dari alamat kontak (dinaslh@jakarta.go.id) dan dikonfirmasi pengguna.").font = Font(name=FONT, size=10, italic=True, color="555555")
ws.cell(row=n + 1, column=2, value="Catatan: judul dataset sampah menyebut 'menurut kota administrasi', tetapi isi filenya per kecamatan.").font = Font(name=FONT, size=10, italic=True, color="555555")

# ------------------------------------------------------ Catatan cleaning
log = pd.read_csv(OUT / "catatan_cleaning.csv")
log.insert(0, "No", range(1, len(log) + 1))
log.columns = ["No", "Dataset", "Masalah di data mentah", "Jumlah terdampak (baris/kecamatan)", "Tindakan", "Alasan"]
ws = wb.create_sheet("Catatan_Cleaning")
judul(ws, "Catatan Cleaning", "Setiap keputusan dicatat: masalah, jumlah terdampak, tindakan, dan alasannya (bahan Bab 3)")
tabel(ws, log, lebar={"No": 5, "Dataset": 13, "Masalah di data mentah": 62, "Jumlah terdampak (baris/kecamatan)": 16,
                      "Tindakan": 58, "Alasan": 52},
      fmt={"Jumlah terdampak (baris/kecamatan)": "#,##0"},
      wrap=("Masalah di data mentah", "Tindakan", "Alasan"))
ws.freeze_panes = "A5"

# ----------------------------------------------------------- Data final
fin = pd.read_csv(OUT / "data_final_2024.csv")
ws = wb.create_sheet("Data_Final_2024")
judul(ws, "Data Final 2024 (42 kecamatan)",
      "Gabungan sampah + kepadatan + RTH + air sumur. Kolom kepadatan dan rth_per_kapita berisi rumus.")
cols = list(fin.columns)
tabel(ws, fin, lebar={"wilayah": 28, "kecamatan": 22, "volume_sampah_rata2": 20, "jumlah_penduduk": 17,
                      "luas_km2": 11, "kepadatan": 14, "total_luas_rth": 15, "jumlah_rth": 11,
                      "rth_per_kapita": 14, "ip_air_rata2": 12, "jumlah_titik_air": 17, "status_mutu_dominan": 20},
      fmt={"volume_sampah_rata2": "#,##0.00", "jumlah_penduduk": "#,##0", "luas_km2": "#,##0.00",
           "kepadatan": "#,##0.00", "total_luas_rth": "#,##0.00", "jumlah_rth": "#,##0",
           "rth_per_kapita": "0.0000", "ip_air_rata2": "0.000", "jumlah_titik_air": "0"})
D, E, F = (get_column_letter(cols.index(c) + 1) for c in ("jumlah_penduduk", "luas_km2", "kepadatan"))
G, I = (get_column_letter(cols.index(c) + 1) for c in ("total_luas_rth", "rth_per_kapita"))
for r in range(5, 5 + len(fin)):
    ws[f"{F}{r}"] = f"={D}{r}/{E}{r}"
    ws[f"{I}{r}"] = f"={G}{r}/{D}{r}"
a, b = 5, 4 + len(fin)
ws.cell(row=b + 2, column=2, value="Total / rata-rata").font = Font(name=FONT, bold=True, size=10)
tot = {"jumlah_penduduk": "SUM", "luas_km2": "SUM", "total_luas_rth": "SUM", "jumlah_rth": "SUM",
       "volume_sampah_rata2": "AVERAGE", "ip_air_rata2": "AVERAGE"}
for c, fn in tot.items():
    L = get_column_letter(cols.index(c) + 1)
    sel = ws[f"{L}{b + 2}"]
    sel.value = f"={fn}({L}{a}:{L}{b})"
    sel.font = Font(name=FONT, bold=True, size=10)
    sel.number_format = ws[f"{L}{a}"].number_format
ws.cell(row=b + 3, column=2, value="Baris total: SUM untuk penduduk, luas, RTH; AVERAGE untuk volume sampah dan IP.").font = Font(name=FONT, size=9, italic=True, color="555555")

# --------------------------------------------------------- Normalisasi
VAR = ["volume_sampah_rata2", "jumlah_penduduk", "luas_km2", "kepadatan", "total_luas_rth",
       "jumlah_rth", "rth_per_kapita", "ip_air_rata2", "jumlah_titik_air"]
SRC = dict(zip(VAR, "CDEFGHIJK"))  # kolom variabel di sheet Data_Final_2024
rn = pd.read_csv(OUT / "ringkasan_normalisasi.csv")
pakai_log = {r["variabel"]: r["transformasi"].startswith("log") for r in rn.to_dict("records")}
ws = wb.create_sheet("Normalisasi_2024", index=wb.sheetnames.index("Data_Final_2024") + 1)
judul(ws, "Normalisasi 2024 (rumus Excel)",
      "Langkah 1: log(1+x) untuk variabel yang miring (|skewness| > 1). Langkah 2: Z-score = (nilai - rata-rata) / simpangan baku.")
hdr = ["wilayah", "kecamatan"] + [f"t_{v}" for v in VAR] + [f"z_{v}" for v in VAR]
for j, h in enumerate(hdr, 1):
    sel = ws.cell(row=4, column=j, value=h)
    sel.font = Font(name=FONT, bold=True, color="FFFFFF", size=10)
    sel.fill = HEAD_FILL
    sel.alignment = Alignment(horizontal="center", vertical="center", wrap_text=True)
    sel.border = BORDER
ws.row_dimensions[4].height = 44
ws.cell(row=3, column=3, value="Langkah 1: transformasi (log untuk variabel miring, lainnya tetap)").font = Font(name=FONT, bold=True, size=10, color="1F3864")
ws.cell(row=3, column=12, value="Langkah 2: Z-score (rata-rata 0, simpangan baku 1)").font = Font(name=FONT, bold=True, size=10, color="1F3864")
n_fin = len(fin)
r_a, r_b = 5, 4 + n_fin
for i in range(n_fin):
    r = r_a + i
    ws.cell(row=r, column=1, value=f"=Data_Final_2024!A{r}")
    ws.cell(row=r, column=2, value=f"=Data_Final_2024!B{r}")
    for k, v in enumerate(VAR):
        ct, cz = 3 + k, 12 + k
        s = f"Data_Final_2024!{SRC[v]}{r}"
        ws.cell(row=r, column=ct, value=f"=LN(1+{s})" if pakai_log[v] else f"={s}")
        L = get_column_letter(ct)
        ws.cell(row=r, column=cz, value=f"=({L}{r}-AVERAGE({L}${r_a}:{L}${r_b}))/STDEVP({L}${r_a}:{L}${r_b})")
    for j in range(1, 21):
        sel = ws.cell(row=r, column=j)
        sel.font = Font(name=FONT, size=10)
        sel.border = BORDER
        if j >= 3:
            sel.number_format = "0.0000"
        if i % 2 == 1:
            sel.fill = ALT_FILL
for lbl, fn, row in (("Rata-rata (Z harus 0)", "AVERAGE", r_b + 2), ("Simpangan baku (Z harus 1)", "STDEVP", r_b + 3)):
    ws.cell(row=row, column=2, value=lbl).font = Font(name=FONT, bold=True, size=10)
    for cz in range(12, 21):
        L = get_column_letter(cz)
        sel = ws.cell(row=row, column=cz, value=f"={fn}({L}{r_a}:{L}{r_b})")
        sel.font = Font(name=FONT, bold=True, size=10)
        sel.number_format = "0.0000"
for j in range(1, 21):
    ws.column_dimensions[get_column_letter(j)].width = 28 if j == 1 else (22 if j == 2 else 15)
ws.freeze_panes = "C5"
ws.sheet_view.showGridLines = False

rn_t = rn.copy()
rn_t.columns = ["Variabel", "Min asli", "Maks asli", "Rata-rata asli", "Simpangan baku asli", "Skewness asli",
                "Transformasi", "Skewness setelah transformasi", "Dipakai untuk model"]
ws = wb.create_sheet("Ringkasan_Normalisasi", index=wb.sheetnames.index("Normalisasi_2024") + 1)
judul(ws, "Ringkasan Normalisasi", "Alasan dan hasil normalisasi tiap variabel (bahan Bab 3 dan presentasi pertemuan 6)")
tabel(ws, rn_t, lebar={"Variabel": 28, "Min asli": 14, "Maks asli": 14, "Rata-rata asli": 14, "Simpangan baku asli": 14,
                       "Skewness asli": 12, "Transformasi": 24, "Skewness setelah transformasi": 16, "Dipakai untuk model": 12},
      fmt={"Min asli": "#,##0.00", "Maks asli": "#,##0.00", "Rata-rata asli": "#,##0.00",
           "Simpangan baku asli": "#,##0.00", "Skewness asli": "0.00", "Skewness setelah transformasi": "0.00"})
ws.freeze_panes = "B5"
penjelasan = [
    "Kenapa dinormalisasi: variabel punya skala sangat berbeda (sampah ratusan, penduduk ratusan ribu, luas RTH jutaan). Tanpa disamakan, variabel bernilai besar mendominasi jarak di K-Means dan PCA.",
    "Langkah 1 (log): variabel miring ke kanan (skewness > 1) diubah dengan log(1+x) agar kecamatan sangat besar tidak mendominasi.",
    "Langkah 2 (Z-score): z = (x - rata-rata) / simpangan baku (populasi), sehingga rata-rata 0 dan simpangan baku 1.",
    "Variabel model: volume sampah, jumlah penduduk, kepadatan, RTH per kapita, IP air. luas_km2 dan total_luas_rth dikecualikan karena berkorelasi tinggi dengan variabel lain (redundan).",
    "Catatan: untuk Linear Regression, standardisasi sebaiknya dilakukan di dalam pipeline cross-validation agar tidak terjadi kebocoran data. Normalisasi ini terutama untuk EDA, PCA, dan K-Means.",
]
r0 = 4 + len(rn_t) + 2
for k, t in enumerate(penjelasan):
    c = ws.cell(row=r0 + k, column=1, value=t)
    c.font = Font(name=FONT, size=10, italic=True, color="555555")
try:
    from openpyxl.drawing.image import Image as XLImage
    img = XLImage(str(OUT / "normalisasi_sebelum_sesudah.png"))
    img.width, img.height = 1000, 289
    ws.add_image(img, f"A{r0 + len(penjelasan) + 2}")
except Exception as e:  # gambar opsional
    print("gambar dilewati:", e)

# ---------------------------------------------------- Sheet data bersih
def sheet_bersih(nama, csv, sub, fmt, lebar=None):
    d = pd.read_csv(OUT / csv)
    ws = wb.create_sheet(nama)
    judul(ws, nama.replace("_", " "), sub)
    tabel(ws, d, fmt=fmt, lebar=lebar)


sheet_bersih("Sampah_2024", "sampah_2024_bersih.csv",
             "Rata-rata bulanan 2024 per kecamatan. Outlier Kramat Jati (Feb) dan Jatinegara (Mei) diganti median kecamatan.",
             {"volume_sampah_rata2": "#,##0.00", "residu_rata2": "#,##0.00", "jumlah_bulan": "0"},
             {"wilayah": 28, "kecamatan": 24})
sheet_bersih("Sampah_Bulanan_2024", "sampah_bulanan_2024_bersih.csv",
             "Volume sampah per bulan per kecamatan (12 bulan x 42 kecamatan, meter kubik). Untuk grafik tren per bulan.",
             {"volume_sampah": "#,##0.00", "total_sampah_residu": "#,##0.00", "periode_data": "0", "bulan": "0"},
             {"wilayah": 28, "kecamatan": 24})
sheet_bersih("Kepadatan_2024", "kepadatan_2024_bersih.csv",
             "Penduduk dan luas dijumlahkan dari kelurahan, kepadatan = penduduk / luas (orang per km2).",
             {"jumlah_penduduk": "#,##0", "luas_km2": "#,##0.00", "kepadatan": "#,##0.00"},
             {"wilayah": 28, "kecamatan": 28})
sheet_bersih("RTH_2024", "rth_2024_bersih.csv",
             "Total luas dan jumlah RTH publik per kecamatan (satuan luas tidak dicantumkan portal).",
             {"total_luas_rth": "#,##0.00", "jumlah_rth": "#,##0"},
             {"wilayah": 28, "kecamatan": 24})
sheet_bersih("AirSumur_2024", "air_sumur_2024_bersih.csv",
             "IP air sumur rata-rata per kecamatan (2 musim). Kecamatan dengan titik sedikit kurang mewakili.",
             {"ip_air_rata2": "0.000", "jumlah_titik_air": "0"},
             {"wilayah": 28, "kecamatan": 28})

for w in wb.worksheets:  # cetak rapi: landscape, muat satu halaman lebar
    w.page_setup.orientation = "landscape"
    w.page_setup.fitToWidth = 1
    w.page_setup.fitToHeight = 0
    w.sheet_properties.pageSetUpPr.fitToPage = True
wb.save(OUT / "Data_Bersih_Proyek_Lingkungan_2024.xlsx")
print("selesai")
