import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler

def preprocessing_diabetes(file_path, save_path=None):
    """ 
    Fungsi ini melakukan preprocessing pada dataset diabetes-prediction.
    Args:
        file_path : Path ke file CSV yang berisi dataset diabetes-prediction.
        save_path : Path untuk menyimpan DataFrame yang telah diproses. 
                                   Jika None, tidak akan disimpan.
    Returns:
        pd.DataFrame: DataFrame yang telah diproses.
    """

    # Load dataset
    try:
        df = pd.read_csv(file_path)
    except Exception as e:
        print(f"Gagal membaca file: {e}")
        return None

    # 1. Menghapus atau Menangani Data Kosong (Missing Values)
    if df.isnull().values.any():
        print("Menangani missing values...")
        for col in df.columns:
            if df[col].isnull().any():
                if pd.api.types.is_numeric_dtype(df[col]):
                    df[col].fillna(df[col].mean(), inplace=True)
                else:
                    df[col].fillna(df[col].mode()[0], inplace=True)
    else:
        print("Tidak ditemukan missing values.")

    # 2. Menghapus Data Duplikat
    if df.duplicated().any():
        print("Menghapus data duplikat...")
        df = df.drop_duplicates()
    else:
        print("Tidak ditemukan data duplikat.")

    # 3. Deteksi dan Penanganan Outlier
    outlier_cols = ['bmi', 'HbA1c_level', 'blood_glucose_level']
    try:
        Q1 = df[outlier_cols].quantile(0.25)
        Q3 = df[outlier_cols].quantile(0.75)
        IQR = Q3 - Q1
        mask = ~((df[outlier_cols] < (Q1 - 1.5 * IQR)) | (df[outlier_cols] > (Q3 + 1.5 * IQR))).any(axis=1)
        df = df[mask]
        print("Outlier telah ditangani.")
    except KeyError:
        print("Kolom untuk deteksi outlier tidak lengkap, melewati langkah ini.")

    # 4. Binning 'age'
    if 'age' in df.columns:
        bins = [-np.inf, 18.0, 35.0, 50.0, 65.0, np.inf]
        labels = ['<18', '18-35', '35-50', '50-65', '>65']
        df['age_group'] = pd.cut(df['age'], bins=bins, labels=labels)
        print("Binning kolom 'age' selesai.")
    else:
        print("Kolom 'age' tidak ditemukan atau bukan numerik, melewati binning.")

    # 5. Encoding Data Kategorikal
    categorical_cols = ['gender', 'smoking_history', 'age_group']
    for col in categorical_cols:
        if col in df.columns:
            le = LabelEncoder()
            df[col] = le.fit_transform(df[col].astype(str))
    print("Encoding kategori selesai.")

    # 6. Normalisasi atau Standarisasi Fitur
    num_cols = ['age', 'bmi', 'HbA1c_level', 'blood_glucose_level']
    scaler = StandardScaler()
    df[num_cols] = scaler.fit_transform(df[num_cols])
    print("Standarisasi numerik selesai.")

    if save_path:
        try:
            df.to_csv(save_path, index=False)
            print(f"Data berhasil disimpan di: {save_path}")
        except Exception as e:
            print(f"Gagal menyimpan data: {e}")

    print("Preprocessing selesai tanpa error.")
    return df

def main():
    """
    Fungsi utama untuk menjalankan preprocessing dataset diabetes.
    """
    file_path = r"diabetes-prediction_raw/diabetes-prediction.csv"
    save_path = r"preprocessing/diabetes-prediction_preprocessing.csv"
    df_processed = preprocessing_diabetes(file_path, save_path)
    if df_processed is not None:
        print("Data preprocessing berhasil.")
        print(df_processed.head())
    else:
        print("Terjadi kesalahan dalam preprocessing data.")

if __name__ == "__main__":
    main() # Menjalankan fungsi utama