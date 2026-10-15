# Eksperimen_SML_M-Zulfan-Afandi

Repository ini berisi eksperimen pra-pemrosesan data dan pembangunan pipeline otomatis untuk Proyek Akhir Machine Learning Dicoding: **Membangun Sistem Machine Learning**.

## Struktur Direktori
```
Eksperimen_SML_M-Zulfan-Afandi
├── .github/workflows/ci.yml
├── .workflow/ci.yml
├── namadataset_raw/
│   └── Full Match Results Data.csv
└── preprocessing/
    ├── Eksperimen_M-Zulfan-Afandi.ipynb
    ├── automate_M-Zulfan-Afandi.py
    ├── namadataset_preprocessing.csv
    └── encoder_mappings.pkl
```

## Menjalankan Otomasi Preprocessing
```bash
cd preprocessing
python automate_M-Zulfan-Afandi.py
```
Output dataset siap latih akan otomatis dibuat di `namadataset_preprocessing.csv`.
