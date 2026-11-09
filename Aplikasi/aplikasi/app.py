"""Aplikasi Analisis Lingkungan Jakarta per Kecamatan 2024 (Project Data Sains).
Jalankan:  streamlit run app.py
"""
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from scipy.stats import skew, spearmanr
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score, silhouette_score
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

st.set_page_config(page_title="Lingkungan Jakarta 2024", layout="wide")

WARNA = ["#1b7a4b", "#e0a526", "#2f6f9f", "#b5473a", "#6b5b95", "#7a9a3a"]
import plotly.io as pio
pio.templates["jkt"] = go.layout.Template(layout=dict(
    font=dict(family="Segoe UI, Helvetica Neue, Arial, sans-serif", color="#1d3b2f", size=13),
    colorway=WARNA, paper_bgcolor="white", plot_bgcolor="white",
    xaxis=dict(gridcolor="#e8f0ea", zeroline=False, linecolor="#c9d9ce", ticks="outside", tickcolor="#c9d9ce"),
    yaxis=dict(gridcolor="#e8f0ea", zeroline=False, linecolor="#c9d9ce"),
    title=dict(font=dict(size=15, color="#0d3b2a")), margin=dict(t=55, l=40, r=20, b=45),
    legend=dict(bgcolor="rgba(255,255,255,0)"), hoverlabel=dict(font_family="Segoe UI")))
px.defaults.template = "jkt"


def tampil_grafik(fig, **kw):
    fig.update_layout(plot_bgcolor="#ffffff", paper_bgcolor="#ffffff", margin=dict(t=50, l=40, r=20, b=85))
    fig.update_xaxes(automargin=False, title_standoff=10)
    fig.update_yaxes(automargin=True)
    st.plotly_chart(fig, theme=None, **kw)


px.defaults.color_discrete_sequence = WARNA
px.defaults.color_continuous_scale = ["#eaf5ee", "#9fd3b2", "#3f9d6a", "#1b6b45", "#0d3b2a"]
st.markdown("""
<style>
html, body, [class*="css"] {font-family: "Segoe UI", "Helvetica Neue", Arial, sans-serif;}
.stApp {background: #f3f8f4;}
.block-container {padding-top: 4.2rem; max-width: 1180px;}
h1, h2, h3 {color:#0d3b2a; font-weight:650; letter-spacing:-0.01em;}
h2 {font-size:1.45rem; margin-top:1.6rem; padding-bottom:4px;}
h2::after, h3::after {content:""; display:block; width:46px; height:4px; border-radius:2px; background:#1b7a4b; margin-top:6px;}
h3 {font-size:1.15rem; margin-top:1.3rem;}
h3::after {width:34px; height:3px; background:#5cc08a;}
p, li {color:#243d33;}

/* kartu angka */
[data-testid="stMetric"] {background:#ffffff; border:1px solid #d7e6dc; border-radius:14px; padding:16px 18px;
    box-shadow:0 2px 8px rgba(13,59,42,0.06); position:relative; overflow:hidden;}
[data-testid="stMetric"]::before {content:""; position:absolute; left:0; top:0; bottom:0; width:5px; background:linear-gradient(#1b7a4b,#5cc08a);}
[data-testid="stMetricLabel"] {color:#55705f;}
[data-testid="stMetricValue"] {color:#0d3b2a; font-weight:700;}

/* grafik dan tabel sebagai kartu */
[data-testid="stPlotlyChart"] {overflow:visible !important; padding-bottom:22px !important;}
[data-testid="stPlotlyChart"], [data-testid="stDataFrame"] {background:#ffffff; border:1px solid #d7e6dc; border-radius:14px;
    padding:6px; box-shadow:0 2px 8px rgba(13,59,42,0.05);}
[data-testid="stAlert"] {border-radius:12px; border:1px solid #d7e6dc;}
[data-testid="stExpander"] {background:#fff; border:1px solid #d7e6dc; border-radius:12px;}

/* menu samping */
[data-testid="stSidebar"] {background:linear-gradient(180deg,#0a2e21 0%,#0d3b2a 55%,#14513a 100%);}
[data-testid="stSidebar"] * {color:#e8f3ec !important;}
[data-testid="stSidebar"] h3::after {display:none;}
[data-testid="stSidebar"] h3 {border:none; padding-left:0; color:#ffffff !important; font-size:1.1rem; line-height:1.45;}
[data-testid="stSidebar"] [role="radiogroup"] {gap:4px;}
[data-testid="stSidebar"] [role="radiogroup"] label {padding:9px 12px; border-radius:10px; transition:background .15s; width:100%;}
[data-testid="stSidebar"] [role="radiogroup"] label:hover {background:rgba(255,255,255,0.09);}
[data-testid="stSidebar"] [role="radiogroup"] label:has(input:checked) {background:rgba(92,192,138,0.28); box-shadow: inset 3px 0 0 #5cc08a;}
[data-testid="stSidebar"] [role="radiogroup"] label > div:first-child {display:none;}

/* header halaman */
.hero {display:flex; align-items:center; gap:24px; padding:26px 30px; border-radius:18px; margin-bottom:22px; color:white;
    background: radial-gradient(circle at 92% 18%, rgba(255,255,255,0.16), transparent 38%),
                linear-gradient(120deg,#0a2e21 0%,#0d3b2a 40%,#1b6b45 100%);
    box-shadow:0 8px 24px rgba(13,59,42,0.25);}
.hero-logo {width:82px; height:82px; flex-shrink:0; background:#fff; border-radius:50%; padding:4px; box-shadow:0 0 0 4px rgba(92,192,138,0.45);}
.hero h1 {color:white; margin:0; font-size:1.75rem; font-weight:700; line-height:1.25;}
.hero p {color:#c5e3d1; margin:8px 0 0 0; font-size:0.98rem;}

/* tab, tombol, input */
div[data-baseweb="tab-list"] {gap:6px;}
div[data-baseweb="tab-list"] button {border-radius:999px; padding:6px 16px; background:#e3efe7;}
div[data-baseweb="tab-list"] button[aria-selected="true"] {background:#1b7a4b; color:#fff !important; font-weight:600;}
div[data-baseweb="tab-highlight"], div[data-baseweb="tab-border"] {display:none;}
.stButton > button, .stDownloadButton > button {border-radius:10px; font-weight:600; border:1px solid #1b7a4b; transition: all .15s;}
.stButton > button[kind="primary"] {background:#1b7a4b; color:#fff;}
.stButton > button:hover, .stDownloadButton > button:hover {box-shadow:0 4px 12px rgba(27,122,75,0.3); transform: translateY(-1px);}
[data-baseweb="select"] > div, [data-baseweb="input"] {border-radius:10px;}
</style>
""", unsafe_allow_html=True)


import base64
_LOGO = Path(__file__).parent / "assets" / "logo.png"
_LOGO64 = base64.b64encode(_LOGO.read_bytes()).decode() if _LOGO.exists() else ""


def judul(teks, sub=""):
    img = f'<img src="data:image/png;base64,{_LOGO64}" class="hero-logo">' if _LOGO64 else ""
    st.markdown(f'<div class="hero">{img}<div><h1>{teks}</h1>{"<p>"+sub+"</p>" if sub else ""}</div></div>', unsafe_allow_html=True)

D = Path(__file__).parent / "data"
SEED = 42

NAMA = {"volume_sampah_rata2": "Volume sampah (m³/bulan)", "jumlah_penduduk": "Jumlah penduduk (jiwa)",
        "luas_km2": "Luas wilayah (km²)", "kepadatan": "Kepadatan (jiwa/km²)",
        "total_luas_rth": "Total luas RTH", "jumlah_rth": "Jumlah RTH",
        "rth_per_kapita": "RTH per kapita", "ip_air_rata2": "IP air sumur (rata-rata)",
        "jumlah_titik_air": "Jumlah titik pantau air"}
FITUR = ["volume_sampah_rata2", "jumlah_penduduk", "kepadatan", "rth_per_kapita", "ip_air_rata2"]
BULAN = ["Jan", "Feb", "Mar", "Apr", "Mei", "Jun", "Jul", "Agu", "Sep", "Okt", "Nov", "Des"]


@st.cache_data
def muat():
    return (pd.read_csv(D / "data_final_2024.csv"), pd.read_csv(D / "data_normalisasi_2024.csv"),
            pd.read_csv(D / "sampah_bulanan_2024_bersih.csv"), pd.read_csv(D / "ringkasan_normalisasi.csv"),
            pd.read_csv(D / "catatan_cleaning.csv"))


df, zn, bulanan, ringkas, catatan = muat()
df["kota"] = df["wilayah"].str.replace("KOTA ADM. JAKARTA ", "", regex=False).str.title()
df["kecamatan_t"] = df["kecamatan"].str.title()
bulanan["kota"] = bulanan["wilayah"].str.replace("KOTA ADM. JAKARTA ", "", regex=False).str.title()

st.sidebar.markdown("### Analisis Lingkungan\nDKI Jakarta · 2024")
if _LOGO.exists():
    st.logo(str(_LOGO), size="large")
halaman = st.sidebar.radio("Menu", ["Beranda dan Data", "Eksplorasi Data (EDA)", "Normalisasi",
                                    "Pengelompokan (PCA + K-Means)", "Prediksi Sampah (Regresi)",
                                    "Kesimpulan dan Rekomendasi", "Tentang Proyek"])
st.sidebar.caption("Project Data Sains · Universitas Maritim Raja Ali Haji")

# ------------------------------------------------------------------ 1
if halaman == "Beranda dan Data":
    judul("Analisis Kondisi Lingkungan Jakarta per Kecamatan (2024)", "Sampah · Kepadatan Penduduk · RTH · Kualitas Air Sumur — 42 kecamatan DKI Jakarta")
    st.write("Aplikasi ini menampilkan hasil analisis 4 dataset terbuka Satu Data Jakarta tahun 2024 yang digabung "
             "menjadi satu tabel berisi **42 kecamatan**. Metode: PCA + K-Means (pengelompokan) dan Regresi Linear (prediksi volume sampah).")
    c = st.columns(4)
    c[0].metric("Kecamatan", len(df)); c[1].metric("Variabel", df.shape[1] - 2)
    c[2].metric("Rata-rata sampah (m³/bln)", f"{df.volume_sampah_rata2.mean():,.0f}")
    c[3].metric("Rata-rata IP air sumur", f"{df.ip_air_rata2.mean():.2f}")

    st.subheader("Empat dataset yang dipakai")
    st.dataframe(pd.DataFrame({
        "Dataset": ["Volume sampah sungai", "Kepadatan penduduk", "RTH publik", "Kualitas air sumur"],
        "Instansi": ["Dinas Lingkungan Hidup", "Kemendagri / Dukcapil", "Dinas Pertamanan dan Hutan Kota", "Dinas Lingkungan Hidup"],
        "Tahun": [2024] * 4, "Baris mentah": [1302, 534, 10565, 16740], "Baris dipakai (2024)": [504, 267, 2545, 12420],
        "Format": [".xls (tabel HTML)"] * 4, "Jenis": ["Sekunder"] * 4}), hide_index=True, width="stretch")

    st.subheader("Tabel data final")
    kota = st.multiselect("Filter kota administrasi", sorted(df.kota.unique()), default=sorted(df.kota.unique()))
    tampil = df[df.kota.isin(kota)].drop(columns=["wilayah", "kecamatan"]).rename(columns={"kecamatan_t": "kecamatan"})
    st.dataframe(tampil.set_index("kecamatan"), width="stretch")
    st.download_button("Unduh data final (CSV)", df.drop(columns=["kota", "kecamatan_t"]).to_csv(index=False), "data_final_2024.csv")

    with st.expander("Catatan cleaning (23 keputusan)"):
        st.dataframe(catatan, hide_index=True, width="stretch")
    st.info("**Keterbatasan:** data sampah adalah sampah sungai (bukan sampah rumah tangga); satuan RTH dan air sumur tidak "
            "dicantumkan portal; hanya satu tahun dan 42 kecamatan; titik pantau air sumur hanya 3–11 per kecamatan; "
            "Kepulauan Seribu tidak ikut karena tidak ada di data sampah dan RTH.")

# ------------------------------------------------------------------ 2
elif halaman == "Eksplorasi Data (EDA)":
    judul("Eksplorasi Data (EDA)", "Mengintip pola data sebelum membuat model")
    t = st.tabs(["Top 10 sampah", "Tren bulanan", "Per kota", "Penduduk vs sampah", "Korelasi"])
    with t[0]:
        top = df.nlargest(10, "volume_sampah_rata2").sort_values("volume_sampah_rata2")
        f = px.bar(top, x="volume_sampah_rata2", y="kecamatan_t", orientation="h", color="kota",
                   labels={"volume_sampah_rata2": "Volume sampah rata-rata (m³/bulan)", "kecamatan_t": "", "kota": "Kota"})
        tampil_grafik(f, width="stretch")
        st.caption("Jagakarsa dan Penjaringan tertinggi. Catatan: volume bulanan Jagakarsa sangat berfluktuasi sepanjang 2024.")
    with t[1]:
        tot = bulanan.groupby("bulan").volume_sampah.sum().reset_index()
        tot["bulan_nama"] = tot.bulan.map(lambda b: BULAN[b - 1])
        f = px.line(tot, x="bulan_nama", y="volume_sampah", markers=True,
                    labels={"bulan_nama": "Bulan", "volume_sampah": "Total volume 42 kecamatan (m³)"})
        tampil_grafik(f, width="stretch")
        st.caption("Total tertinggi bulan November, terendah bulan Maret.")
    with t[2]:
        f = px.box(df, x="kota", y="volume_sampah_rata2", points="all", hover_name="kecamatan_t",
                   labels={"kota": "Kota administrasi", "volume_sampah_rata2": "Volume sampah (m³/bulan)"})
        tampil_grafik(f, width="stretch")
    with t[3]:
        f = px.scatter(df, x="jumlah_penduduk", y="volume_sampah_rata2", color="kota", hover_name="kecamatan_t",
                       labels={"jumlah_penduduk": "Jumlah penduduk", "volume_sampah_rata2": "Volume sampah (m³/bulan)"})
        b, a = np.polyfit(df.jumlah_penduduk, df.volume_sampah_rata2, 1)
        xs = np.array([df.jumlah_penduduk.min(), df.jumlah_penduduk.max()])
        f.add_trace(go.Scatter(x=xs, y=a + b * xs, mode="lines", name="Garis tren", line=dict(color="gray", dash="dash")))
        tampil_grafik(f, width="stretch")
        st.caption("Makin banyak penduduk, cenderung makin besar volume sampah, tetapi sebarannya lebar.")
    with t[4]:
        cols = FITUR
        corr = df[cols].corr(method="spearman").round(2)
        lab = [NAMA[c].split(" (")[0] for c in cols]
        f = px.imshow(corr.values, x=lab, y=lab, text_auto=True, zmin=-1, zmax=1, color_continuous_scale="RdBu_r")
        tampil_grafik(f, width="stretch")
        r = corr["volume_sampah_rata2"].drop("volume_sampah_rata2")
        st.caption("Korelasi Spearman dengan volume sampah: " + ", ".join(f"{NAMA[k].split(' (')[0]} {v:+.2f}" for k, v in r.items())
                   + ". Korelasi bukan berarti sebab-akibat.")

# ------------------------------------------------------------------ 3
elif halaman == "Normalisasi":
    judul("Normalisasi Data", "Menyamakan skala dan bentuk sebaran variabel")
    st.write("Dua langkah: (1) **log(1 + x)** untuk variabel dengan |skewness| > 1, (2) **Z-score** untuk semua variabel "
             "(rata-rata 0, simpangan baku 1). Z-score hanya menyamakan skala; bentuk sebaran berubah karena log.")
    pilih = st.selectbox("Pilih variabel", list(ringkas.variabel), format_func=lambda v: NAMA[v])
    x = df[pilih].astype(float); r = ringkas.set_index("variabel").loc[pilih]
    t = np.log1p(x) if r.transformasi.startswith("log") else x
    z = (t - t.mean()) / t.std(ddof=0)
    c = st.columns(2)
    c[0].plotly_chart(px.histogram(x, nbins=10, title=f"Sebelum (skewness {skew(x):.2f})").update_layout(showlegend=False, xaxis_title=NAMA[pilih]), width="stretch")
    c[1].plotly_chart(px.histogram(z, nbins=10, title=f"Sesudah: {r.transformasi} (skewness {skew(z):.2f})", color_discrete_sequence=["#1f3864"]).update_layout(showlegend=False, xaxis_title="Nilai ternormalisasi"), width="stretch")
    st.subheader("Ringkasan skewness semua variabel")
    tab = ringkas[["variabel", "skewness_asli", "transformasi", "skewness_setelah_transformasi", "fitur_model"]].copy()
    tab["variabel"] = tab.variabel.map(NAMA)
    tab.columns = ["Variabel", "Skewness sebelum", "Transformasi", "Skewness sesudah", "Dipakai model"]
    st.dataframe(tab, hide_index=True, width="stretch")

# ------------------------------------------------------------------ 4
elif halaman == "Pengelompokan (PCA + K-Means)":
    judul("Pengelompokan Kecamatan (PCA + K-Means)", "Kecamatan mana yang kondisinya mirip?")
    X = zn[["z_" + c for c in FITUR]].values

    @st.cache_data
    def elbow():
        out = []
        for k in range(2, 9):
            km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
            out.append((k, km.inertia_, silhouette_score(X, km.labels_)))
        return pd.DataFrame(out, columns=["k", "inertia", "silhouette"])

    e = elbow()
    st.subheader("Memilih jumlah kelompok (k)")
    c = st.columns(2)
    c[0].plotly_chart(px.line(e, x="k", y="inertia", markers=True, title="Elbow (inertia)"), width="stretch")
    c[1].plotly_chart(px.line(e, x="k", y="silhouette", markers=True, title="Silhouette score"), width="stretch")
    kbest = int(e.loc[e.silhouette.idxmax(), "k"])
    st.caption(f"Silhouette tertinggi pada k = {kbest} ({e.silhouette.max():.3f}). Nilai 0,26–0,50 tergolong lemah, di bawah 0,25 berarti "
               "tidak ada struktur yang jelas. Jadi kelompok di bawah ini bersifat indikatif, bukan pemisahan yang tegas.")
    k = st.slider("Jumlah kelompok (k)", 2, 6, kbest)
    km = KMeans(n_clusters=k, n_init=20, random_state=SEED).fit(X)
    pca = PCA(n_components=2, random_state=SEED).fit(X); P = pca.transform(X)
    hasil = df[["kota", "kecamatan_t"] + FITUR].copy()
    hasil["Kelompok"] = (km.labels_ + 1).astype(str); hasil["PC1"], hasil["PC2"] = P[:, 0], P[:, 1]
    ev = pca.explained_variance_ratio_
    st.write(f"Silhouette untuk k = {k}: **{silhouette_score(X, km.labels_):.3f}**")
    f = px.scatter(hasil, x="PC1", y="PC2", color="Kelompok", hover_name="kecamatan_t", text="kecamatan_t",
                   labels={"PC1": f"PC1 O({ev[0]:.0%})", "PC2": f"PC2 ({ev[1]:.0%})"},
                   category_orders={"Kelompok": sorted(hasil.Kelompok.unique())})
    f.update_traces(textposition="top center", textfont_size=9)
    tampil_grafik(f, width="stretch")
    st.caption(f"Dua komponen utama menjelaskan {ev[:2].sum():.0%} variasi data.")
    st.subheader("Karakter tiap kelompok (rata-rata nilai asli)")
    prof = hasil.groupby("Kelompok")[FITUR].mean().round(2)
    prof.insert(0, "Jumlah kecamatan", hasil.Kelompok.value_counts().sort_index())
    prof.columns = ["Jumlah kecamatan"] + [NAMA[c] for c in FITUR]
    st.dataframe(prof, width="stretch")
    st.download_button("Unduh hasil kelompok (CSV)", hasil[["kota", "kecamatan_t", "Kelompok"]].rename(columns={"kota": "Kota", "kecamatan_t": "Kecamatan"}).to_csv(index=False),
                       f"kelompok_kecamatan_k{k}.csv", mime="text/csv")
    st.subheader("Anggota kelompok")
    for g in sorted(hasil.Kelompok.unique()):
        st.write(f"**Kelompok {g}:** " + ", ".join(hasil.loc[hasil.Kelompok == g, "kecamatan_t"]))


    st.subheader("Tentukan kelompok untuk wilayah baru")
    st.write("Isi data satu wilayah, lalu aplikasi menentukan wilayah itu masuk **kelompok** yang mana "
             f"(memakai model K-Means dengan k = {k} di atas).")
    nilai = {}
    cols = st.columns(len(FITUR))
    for col, c_ in zip(cols, FITUR):
        nilai[c_] = col.number_input(NAMA[c_], min_value=0.0, value=float(df[c_].median()), key="km_" + c_)
    if st.button("Tentukan kelompok", type="primary"):
        z_baru = []
        for c_ in FITUR:
            logik = ringkas.set_index("variabel").loc[c_, "transformasi"].startswith("log")
            t_all = np.log1p(df[c_].astype(float)) if logik else df[c_].astype(float)
            t_new = np.log1p(nilai[c_]) if logik else nilai[c_]
            z_baru.append((t_new - t_all.mean()) / t_all.std(ddof=0))
        z_baru = np.array(z_baru).reshape(1, -1)
        g = int(km.predict(z_baru)[0]) + 1
        jarak = np.linalg.norm(km.cluster_centers_ - z_baru, axis=1)
        urut = np.argsort(jarak)
        st.success(f"Wilayah ini masuk **Kelompok {g}**.")
        if len(jarak) > 1 and (jarak[urut[1]] - jarak[urut[0]]) < 0.3:
            st.warning(f"Posisinya berada dekat batas dengan Kelompok {urut[1] + 1}, jadi penentuannya kurang tegas "
                       "(sejalan dengan nilai Silhouette yang rendah).")
        rata = hasil[hasil.Kelompok == str(g)][FITUR].mean()
        banding = pd.DataFrame({"Wilayah baru": [nilai[c_] for c_ in FITUR],
                                f"Rata-rata Kelompok {g}": [rata[c_] for c_ in FITUR],
                                "Rata-rata semua kecamatan": [df[c_].mean() for c_ in FITUR]},
                               index=[NAMA[c_] for c_ in FITUR]).round(2)
        st.dataframe(banding, width="stretch")
        p_baru = pca.transform(z_baru)[0]
        f2 = px.scatter(hasil, x="PC1", y="PC2", color="Kelompok", hover_name="kecamatan_t",
                        category_orders={"Kelompok": sorted(hasil.Kelompok.unique())},
                        labels={"PC1": f"PC1 ({ev[0]:.0%})", "PC2": f"PC2 ({ev[1]:.0%})"})
        f2.add_trace(go.Scatter(x=[p_baru[0]], y=[p_baru[1]], mode="markers+text", text=["Wilayah baru"],
                                textposition="top center", name="Wilayah baru",
                                marker=dict(symbol="star", size=18, color="black")))
        tampil_grafik(f2, width="stretch")

# ------------------------------------------------------------------ 5
elif halaman == "Prediksi Sampah (Regresi)":
    judul("Prediksi Volume Sampah (Regresi Linear)", "Seberapa akurat volume sampah bisa diprediksi?")
    y = df["volume_sampah_rata2"]
    PILIHAN = {"Penduduk saja (sederhana)": ["jumlah_penduduk"],
               "Penduduk + IP air": ["jumlah_penduduk", "ip_air_rata2"],
               "4 fitur (penduduk, kepadatan, RTH per kapita, IP air)": ["jumlah_penduduk", "kepadatan", "rth_per_kapita", "ip_air_rata2"]}
    nama = st.selectbox("Pilih model", list(PILIHAN))
    fitur = PILIHAN[nama]
    mk = lambda: make_pipeline(StandardScaler(), LinearRegression())

    @st.cache_data
    def evaluasi(fitur_t):
        res = {}
        for lab, model in (("Model", mk()), ("Baseline (tebak rata-rata)", DummyRegressor())):
            m = []
            for rep in range(20):
                p = cross_val_predict(model, df[list(fitur_t)], y, cv=KFold(5, shuffle=True, random_state=rep))
                m.append((mean_absolute_error(y, p), np.sqrt(mean_squared_error(y, p)), r2_score(y, p)))
            res[lab] = np.mean(m, axis=0)
        p0 = cross_val_predict(mk(), df[list(fitur_t)], y, cv=KFold(5, shuffle=True, random_state=0))
        return res, p0

    res, p0 = evaluasi(tuple(fitur))
    st.subheader("Seberapa akurat? (cross-validation 5-fold × 20 ulangan)")
    tab = pd.DataFrame(res, index=["MAE (m³)", "RMSE (m³)", "R²"]).T.round(3)
    st.dataframe(tab, width="stretch")
    st.markdown("**Cara membaca tabel:** MAE dan RMSE adalah rata-rata meleset prediksi (**makin kecil makin baik**). "
                "R² adalah porsi variasi yang bisa dijelaskan (**makin dekat 1 makin baik**, di bawah 0 berarti lebih buruk daripada sekadar menebak rata-rata). "
                "Baris *Baseline* adalah pembanding: model yang selalu menebak rata-rata. Model dikatakan berguna kalau lebih baik dari baseline.")
    r2 = res["Model"][2]
    st.caption(f"R² = {r2:.2f}: model menjelaskan sekitar {max(r2, 0):.0%} variasi volume sampah antarkecamatan. "
               f"MAE = {res['Model'][0]:.0f} m³ artinya prediksi rata-rata meleset sekitar {res['Model'][0]:.0f} m³/bulan. "
               "Ini model sederhana pada data kecil (42 kecamatan), jadi akurasinya terbatas. Standardisasi dilakukan di dalam "
               "cross-validation (tanpa data leakage).")
    lim = [0, float(max(y.max(), p0.max()) * 1.05)]
    f = px.scatter(x=y, y=p0, hover_name=df.kecamatan_t, labels={"x": "Aktual (m³/bulan)", "y": "Prediksi cross-validation"},
                   title="Aktual vs prediksi")
    f.add_trace(go.Scatter(x=lim, y=lim, mode="lines", line=dict(dash="dash", color="red"), name="Prediksi sempurna"))
    tampil_grafik(f, width="stretch")

    st.subheader("Cek per kecamatan")
    st.write("Pilih satu kecamatan untuk membandingkan volume sampah **sebenarnya** dengan **prediksi** model. "
             "Prediksi ini dibuat tanpa melihat kecamatan tersebut (cross-validation), jadi adil.")
    pilih = st.selectbox("Pilih kecamatan", sorted(df.kecamatan_t))
    i = df.index[df.kecamatan_t == pilih][0]
    aktual, prediksi = float(y[i]), float(p0[i]); selisih = aktual - prediksi
    c = st.columns(3)
    c[0].metric("Sebenarnya (m³/bulan)", f"{aktual:,.0f}")
    c[1].metric("Prediksi model (m³/bulan)", f"{prediksi:,.0f}")
    c[2].metric("Selisih", f"{selisih:+,.0f}")
    arah = "lebih banyak" if selisih > 0 else "lebih sedikit"
    persen = abs(selisih) / aktual
    if persen <= 0.15:
        st.success(f"Prediksi cukup dekat: meleset sekitar {persen:.0%}.")
    elif persen <= 0.4:
        st.warning(f"Prediksi meleset sekitar {persen:.0%}. Sampah sebenarnya {arah} dari perkiraan model.")
    else:
        st.error(f"Prediksi meleset jauh (sekitar {persen:.0%}). Sampah sebenarnya {arah} dari perkiraan model, "
                 "artinya ada faktor lain di kecamatan ini yang belum tertangkap data.")
    st.caption("Data kecamatan ini: " + ", ".join(f"{NAMA[c_].split(' (')[0]} {df.loc[i, c_]:,.2f}" for c_ in FITUR[1:]))
    with st.expander("Lihat semua kecamatan (urut dari yang paling meleset)"):
        semua = pd.DataFrame({"Kecamatan": df.kecamatan_t, "Sebenarnya": y.round(0), "Prediksi": p0.round(0)})
        semua["Selisih"] = (semua["Sebenarnya"] - semua["Prediksi"])
        st.dataframe(semua.reindex(semua["Selisih"].abs().sort_values(ascending=False).index), hide_index=True, width="stretch")

    final = mk().fit(df[fitur], y); lr = final.named_steps["linearregression"]
    sd = df[fitur].std(ddof=0).values; mu = df[fitur].mean().values
    slope = lr.coef_ / sd; icpt = lr.intercept_ - float(np.sum(lr.coef_ * mu / sd))
    st.subheader("Persamaan model")
    st.code(f"volume_sampah = {icpt:.1f} " + " ".join(f"{'+' if s >= 0 else '-'} {abs(s):.6f} × {c}" for s, c in zip(slope, fitur)))

    st.subheader("Coba prediksi")
    st.write("Isi nilai variabel, aplikasi menghitung perkiraan volume sampah per bulan.")
    nilai = {}
    cols = st.columns(len(fitur))
    for col, c in zip(cols, fitur):
        lo, hi = float(df[c].min()), float(df[c].max())
        nilai[c] = col.number_input(NAMA[c], min_value=0.0, value=float(df[c].median()), help=f"Rentang data latihan: {lo:,.2f} sampai {hi:,.2f}. Tekan Enter setelah mengetik.")
    pred = float(final.predict(pd.DataFrame([nilai]))[0])
    mae = res["Model"][0]
    st.success(f"Perkiraan volume sampah: **{pred:,.0f} m³/bulan** (kira-kira {pred - mae:,.0f} sampai {pred + mae:,.0f}, "
               f"berdasarkan MAE ±{mae:,.0f}).")
    if any(not (df[c].min() <= v <= df[c].max()) for c, v in nilai.items()):
        st.warning("Nilai di luar rentang data latihan, prediksi kurang bisa dipercaya.")

elif halaman == "Kesimpulan dan Rekomendasi":
    judul("Kesimpulan dan Rekomendasi", "Apa yang dapat dipelajari dari data, dan apa implikasinya")
    Xk = zn[["z_" + c for c in FITUR]].values
    kmk = KMeans(n_clusters=3, n_init=20, random_state=SEED).fit(Xk)
    kk = df.copy(); kk["g"] = kmk.labels_ + 1
    prof = kk.groupby("g")[FITUR].mean()
    g_tinggi = int(prof.volume_sampah_rata2.idxmax()); g_padat = int(prof.kepadatan.idxmax()); g_hijau = int(prof.rth_per_kapita.idxmax())
    anggota = lambda g: ", ".join(kk.loc[kk.g == g, "kecamatan_t"])
    rho = {c: spearmanr(df[c], df.volume_sampah_rata2)[0] for c in FITUR[1:]}
    berat = df[df.status_mutu_dominan == "BERAT"].sort_values("ip_air_rata2", ascending=False)
    st.subheader("Temuan utama")
    st.markdown(f"""
1. **Jumlah penduduk adalah variabel yang paling berhubungan dengan volume sampah** (korelasi Spearman {rho['jumlah_penduduk']:+.2f}). Kepadatan hampir tidak berhubungan ({rho['kepadatan']:+.2f}), dan RTH per kapita berhubungan lemah dan negatif ({rho['rth_per_kapita']:+.2f}).
2. **Kecamatan dapat dikelompokkan menjadi 3 kelompok**, tetapi pemisahannya lemah (Silhouette 0,258), jadi kelompok bersifat indikatif.
3. **Volume sampah dapat diprediksi secara kasar dari jumlah penduduk**: R² sekitar 0,35 dan MAE sekitar 270 m³/bulan. Menambah variabel lain tidak memperbaiki model (model 4 fitur R² 0,18).
4. **Kualitas air sumur Jakarta secara umum mengkhawatirkan**: dari 42 kecamatan, {int((df.status_mutu_dominan == 'SEDANG').sum())} berstatus SEDANG, {len(berat)} BERAT, {int((df.status_mutu_dominan == 'RINGAN').sum())} RINGAN, dan {int((df.status_mutu_dominan == 'BAIK').sum())} BAIK.
""")
    st.subheader("Karakter kelompok")
    st.markdown(f"""
- **Kelompok {g_tinggi} (sampah tinggi, {int((kk.g == g_tinggi).sum())} kecamatan):** rata-rata {prof.loc[g_tinggi, 'volume_sampah_rata2']:,.0f} m³/bulan dan {prof.loc[g_tinggi, 'jumlah_penduduk']:,.0f} penduduk. Anggota: {anggota(g_tinggi)}.
- **Kelompok {g_padat} (padat, RTH rendah, {int((kk.g == g_padat).sum())} kecamatan):** kepadatan rata-rata {prof.loc[g_padat, 'kepadatan']:,.0f} jiwa/km² dengan RTH per kapita hanya {prof.loc[g_padat, 'rth_per_kapita']:.2f}. Anggota: {anggota(g_padat)}.
- **Kelompok {g_hijau} (sampah rendah, RTH lebih tinggi, {int((kk.g == g_hijau).sum())} kecamatan):** RTH per kapita rata-rata {prof.loc[g_hijau, 'rth_per_kapita']:.2f}. Anggota: {anggota(g_hijau)}.
""")
    st.subheader("Rekomendasi")
    st.markdown(f"""
1. **Prioritaskan pengelolaan sampah sungai di Kelompok {g_tinggi}**, karena volumenya jauh di atas kelompok lain dan penduduknya terbanyak.
2. **Tambah atau perbaiki RTH di Kelompok {g_padat}**, karena kepadatannya tertinggi sedangkan RTH per kapitanya paling rendah.
3. **Perkuat pemantauan air sumur di {len(berat)} kecamatan berstatus BERAT** (tertinggi: {", ".join(berat.kecamatan_t.head(3))}), sekaligus menambah titik pantau karena jumlahnya hanya 3 sampai 11 per kecamatan.
4. **Gunakan hasil ini sebagai penyaring awal, bukan keputusan final.** Perlu data tambahan sebelum kebijakan diambil.
""")
    st.subheader("Keterbatasan")
    st.markdown("""
- Data sampah adalah sampah sungai, bukan sampah rumah tangga.
- Hanya satu tahun (2024) dan 42 kecamatan, sehingga model mudah tidak stabil.
- Satuan RTH dan air sumur tidak dicantumkan portal.
- Pengelompokan lemah dan regresi hanya menjelaskan sekitar sepertiga variasi.
- Korelasi tidak berarti sebab-akibat.
""")

else:
    judul("Tentang Proyek", "Project Data Sains · Teknik Informatika · Universitas Maritim Raja Ali Haji")
    st.subheader("Tujuan")
    st.write("Menganalisis kondisi lingkungan DKI Jakarta per kecamatan (sampah, kepadatan penduduk, ruang terbuka hijau, dan kualitas air sumur) "
             "menggunakan PCA + K-Means untuk pengelompokan dan Regresi Linear untuk prediksi volume sampah.")
    st.subheader("Alur kerja")
    st.markdown("""
1. **Data collection:** 4 dataset terbuka dari Satu Data Jakarta (data sekunder, tahun 2024).
2. **Data cleaning:** penyeragaman nama kecamatan, perbaikan outlier, penghapusan duplikat, konversi tipe data, penggabungan.
3. **Normalisasi:** transformasi log untuk variabel miring, lalu Z-score.
4. **Eksplorasi data:** grafik dan korelasi.
5. **Pemodelan:** K-Means dengan PCA (dipilih dengan Elbow dan Silhouette) dan Regresi Linear (dinilai dengan MAE, RMSE, R² lewat cross-validation).
6. **Kesimpulan dan rekomendasi.**
""")
    st.subheader("Anggota kelompok")
    st.dataframe(pd.DataFrame({"No": range(1, 7), "Nama": ["Dhini Khairunnisa", "Erlinda Amira Putri S.", "Niki Arzetty Siregar", "Rismayanti", "Rizka Sufianty", "Jian Safitri Wibowo"],
                               "NIM": ["2401020016", "2401020021", "2401020030", "2401020041", "2401020053", "2401020056"]}),
                 hide_index=True, width="stretch")
    st.write("Dosen pengampu: Nerfita Nikentari, S.T., M.Cs., Ph.D.")
    st.subheader("Sumber data")
    st.write("Satu Data Jakarta (https://satudata.jakarta.go.id).")
