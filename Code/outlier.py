import pandas as pd

# Membaca data dari file CSV ke dalam dataframe
df = pd.read_csv('video_games_effect.csv')

# Memilih kolom-kolom yang ingin diambil untuk analisis outlier
selected_columns = ['waktu_bermain_perhari', 'waktu_tidur_perhari', 'sering_lalai_kerjakan_tugas',
                    'sering_melewatkan_waktu_tidur', 'kesulitan_mengurangi_waktu', 'merasa_kekurangan_waktu',
                    'pengaruh_tingkat_konsentrasi', 'pengaruh_game_terhadap_tingkat_motivasi_belajar',
                    'kesulitan_saat_ujian', 'nilai_menurun']

# Mengonversi kolom-kolom terpilih ke tipe data numerik
df[selected_columns] = df[selected_columns].apply(pd.to_numeric, errors='coerce')

# Menghapus baris yang memiliki nilai NaN setelah konversi
df = df.dropna()

# Mencari outlier menggunakan metode IQR
def find_outliers_iqr(data):
    outliers_dict = {}

    # Looping untuk setiap kolom
    for col in selected_columns:
        # Menghitung kuartil pertama (Q1) dan kuartil ketiga (Q3)
        Q1 = data[col].quantile(0.25)
        Q3 = data[col].quantile(0.75)

        # Menghitung Interquartile Range (IQR)
        IQR = Q3 - Q1

        # Menentukan batas bawah dan batas atas untuk mendeteksi outlier
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        # Mengidentifikasi outlier
        outliers = data[(data[col] < lower_bound) | (data[col] > upper_bound)][col].tolist()

        # Menyimpan outlier ke dalam dictionary
        outliers_dict[col] = outliers

    return outliers_dict

# Menggunakan fungsi untuk mencari outlier pada kolom-kolom yang dipilih
outliers_dict = find_outliers_iqr(df)

# Menampilkan informasi tentang outlier
print("Dataframe Info:")
print(df.info())

# Menampilkan outliers untuk setiap kolom
for col, outliers in outliers_dict.items():
    print("\nOutliers in '{}' column:".format(col))
    print(outliers)
