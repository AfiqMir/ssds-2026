# Prediksi TMA — Bengawan Solo (Pipeline Lokal)

## 1. Struktur folder yang dibutuhkan

Taruh 4 script ini **di luar** folder dataset kamu (atau di mana saja), lalu jalankan
dengan `--data-dir` menunjuk ke folder dataset. Struktur dataset kamu:

```
sebelas-maret-statistics-data-.../
├── data_pendukung/
│   ├── HydroRIVERS_v10_au_shp/
│   ├── data_lingkungan.csv
│   ├── HydroRIVERS_TechDoc_v10.pdf
│   └── koordinat_pos.csv
├── sample_submission.csv
├── test.csv
└── train.csv
```

Ini sudah sesuai default script — tidak perlu diubah.

## 2. Setup environment

```bash
python3 -m venv venv
source venv/bin/activate      # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## 3. Jalankan pipeline (urut, dari folder tempat script berada)

```bash
DATA_DIR="/path/ke/sebelas-maret-statistics-data-..."

python 01_build_exog_features.py     --data-dir "$DATA_DIR" --output-dir ./output
python 02_build_lag_features.py      --data-dir "$DATA_DIR" --output-dir ./output
python 03_train_and_predict.py       --output-dir ./output
python 04_final_train_and_forecast.py --data-dir "$DATA_DIR" --output-dir ./output
```

Kalau kamu jalankan dari **dalam** folder dataset itu sendiri, cukup:

```bash
python 01_build_exog_features.py --output-dir ./output
python 02_build_lag_features.py --output-dir ./output
python 03_train_and_predict.py --output-dir ./output
python 04_final_train_and_forecast.py --output-dir ./output
```
(karena `--data-dir` default-nya `.`)

## 4. Output

Semua file perantara (`.parquet`, model `.txt`, `feature_cols.pkl`) dan
`submission.csv` final akan tersimpan di folder `./output/`.

## 5. Estimasi waktu & RAM

| Script | Yang dilakukan | Perkiraan waktu | Catatan RAM |
|---|---|---|---|
| `01` | Baca `data_lingkungan.csv` (155 MB), agregasi rolling per pos | 2–5 menit | Butuh ~2–4 GB RAM |
| `02` | Bangun grid waktu penuh + capping outlier | <1 menit | Ringan |
| `03` | Training LightGBM + validasi | 1–3 menit | Ringan |
| `04` | Retrain + forecast rekursif 726 timestamp | 3–8 menit | Ringan |

Kalau RAM laptop kamu terbatas (<8GB) dan script `01` lambat/crash, kabari saya —
bisa dioptimasi jadi chunked reading.

## 6. Parameter yang bisa diubah

- `03_train_and_predict.py --val-cutoff 2025-05-19` — ubah tanggal potong validasi
- `03_train_and_predict.py --num-boost-round 3000 --early-stopping 100` — tuning training
- `04_final_train_and_forecast.py --num-boost-round N` — override jumlah round training final (default: pakai `best_iteration` dari script 03)

## 7. Hasil baseline saat ini

- Validation RMSE (holdout Mei–Sep 2025): **~0.27** (skala campuran 30 pos)
- Pos dengan error terbesar: `Bojonegoro - Kali Kethek`, `Brangkal`, `Kedungupit`
