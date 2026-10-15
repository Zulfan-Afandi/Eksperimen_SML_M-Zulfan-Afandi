import os
import argparse
import joblib
import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_data(file_path: str) -> pd.DataFrame:
    """
    Memuat dataset mentah dari file CSV.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File dataset tidak ditemukan pada path: {file_path}")
    print(f"[INFO] Memuat data mentah dari: {file_path}")
    df = pd.read_csv(file_path)
    print(f"[INFO] Dimensi data mentah: {df.shape}")
    return df

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Membersihkan missing value dan menduplikasi data yang tidak valid.
    """
    print("[INFO] Membersihkan missing values...")
    initial_rows = len(df)
    df_cleaned = df.dropna().copy()
    cleaned_rows = len(df_cleaned)
    print(f"[INFO] Baris sebelum dibersihkan: {initial_rows}, setelah: {cleaned_rows}")
    return df_cleaned

def transform_data(df: pd.DataFrame, 
                   feature_cols=['Team', 'Ban', 'Role Ban', 'Pick', 'Role Pick'], 
                   target_col='MVP'):
    """
    Memilih kolom fitur dan target, kemudian menerapkan Label Encoding pada kolom kategorikal.
    Menghasilkan DataFrame dengan nama kolom standar huruf kecil (snake_case) dan dictionary encoders.
    """
    print(f"[INFO] Mengambil fitur: {feature_cols} dan target: {target_col}")
    selected_cols = feature_cols + [target_col]
    df_subset = df[selected_cols].copy()
    
    encoders = {}
    for col in feature_cols:
        le = LabelEncoder()
        df_subset[col] = le.fit_transform(df_subset[col].astype(str))
        encoders[col.lower().replace(' ', '_')] = le
        print(f"[INFO] Kolom '{col}' berhasil di-encode ({len(le.classes_)} kategori unik).")
    
    # Standarisasi nama kolom ke lowercase snake_case
    rename_mapping = {
        'Team': 'team',
        'Ban': 'ban',
        'Role Ban': 'role_ban',
        'Pick': 'pick',
        'Role Pick': 'role_pick',
        'MVP': 'mvp'
    }
    df_transformed = df_subset.rename(columns=rename_mapping)
    return df_transformed, encoders

def save_data(df: pd.DataFrame, output_path: str, encoders: dict = None, encoder_path: str = None):
    """
    Menyimpan hasil preprocessing ke file CSV dan menyimpan encoder ke file pkl.
    """
    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir, exist_ok=True)
        
    df.to_csv(output_path, index=False)
    print(f"[SUCCESS] Dataset preprocessing berhasil disimpan di: {output_path}")
    print(f"[INFO] Ringkasan data hasil preprocessing:")
    print(df.head())
    print(f"[INFO] Distribusi target:\n{df['mvp'].value_counts()}")
    
    if encoders and encoder_path:
        joblib.dump(encoders, encoder_path)
        print(f"[SUCCESS] Encoders berhasil disimpan di: {encoder_path}")

def main():
    parser = argparse.ArgumentParser(description="Automate Data Preprocessing for ML M5 Match Results")
    parser.add_argument(
        "--raw_path", 
        type=str, 
        default="../namadataset_raw/Full Match Results Data.csv", 
        help="Path ke file dataset mentah"
    )
    parser.add_argument(
        "--output_path", 
        type=str, 
        default="namadataset_preprocessing.csv", 
        help="Path tujuan file hasil preprocessing"
    )
    parser.add_argument(
        "--encoder_path", 
        type=str, 
        default="encoder_mappings.pkl", 
        help="Path tujuan file label encoder"
    )
    args = parser.parse_args()

    # Fallback path jika dijalankan dari root direktori
    raw_path = args.raw_path
    if not os.path.exists(raw_path):
        alternative_paths = [
            "namadataset_raw/Full Match Results Data.csv",
            "../namadataset_raw/Full Match Results Data.csv",
            "Dataset_ML_TugasDicoding/Full Match Results Data.csv",
            "Eksperimen_SML_M-Zulfan-Afandi/namadataset_raw/Full Match Results Data.csv"
        ]
        for p in alternative_paths:
            if os.path.exists(p):
                raw_path = p
                break

    df_raw = load_data(raw_path)
    df_clean = clean_data(df_raw)
    df_transformed, encoders = transform_data(df_clean)
    save_data(df_transformed, args.output_path, encoders=encoders, encoder_path=args.encoder_path)

if __name__ == "__main__":
    main()
