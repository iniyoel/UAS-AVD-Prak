import pandas as pd
import numpy as np
from scipy import stats
from sklearn.preprocessing import MinMaxScaler
import seaborn as sns
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.stats import zscore
from io import BytesIO

#1. Data Video Games
data = pd.read_csv('video_games_effect.csv')
df = pd.DataFrame(data)

#2. Create data
print ('Data Keseluruhan = ')

#3. data record
data = pd.read_csv ('video_games_effect.csv')

# Fungsi untuk membuat kolom klasifikasi
def classify(row):
    if row['kesulitan_saat_ujian'] == 'Tidak' and row['nilai_menurun'] == 'Tidak':
        return 1
    elif row['kesulitan_saat_ujian'] == 'Iya' and row['nilai_menurun'] == 'Tidak':
        return 2
    elif row['kesulitan_saat_ujian'] == 'Tidak' and row['nilai_menurun'] == 'Ya':
        return 3
    elif row['kesulitan_saat_ujian'] == 'Iya' and row['nilai_menurun'] == 'Ya':
        return 4

# Buat kolom klasifikasi baru dalam dataset
data['klasifikasi'] = data.apply(classify, axis=1)

# Menyimpan data yang sudah diperbaharui
data.to_csv('video_games_effect.csv', index=False) 

# Menampilkan dataset hasil
print(data)

# 1 ====================================================================================================
#Deteksi Missing Values
missing_values = df.isna().sum()
print("Missing Values:")
print(df)

# Menangani Missing Values
df_cleaned = df.dropna()
print("\nDataFrame setelah Menghapus Missing Values:")
print(df_cleaned)

# 2 ====================================================================================================
# Kolom numerik yang akan diperiksa untuk outliers
kolom_numerik = ['waktu_bermain_perhari', 'waktu_tidur_perhari', 'dampak_buruk_kecanduan_game', 
                 'tingkatan_mengatur_waktu', 'sering_lalai_kerjakan_tugas', 
                 'sering_melewatkan_waktu_tidur', 'kesulitan_mengurangi_waktu', 
                 'merasa_kekurangan_waktu', 'pengaruh_tingkat_konsentrasi', 
                 'pengaruh_game_terhadap_tingkat_motivasi_belajar']

# Deteksi dan Penanganan Outliers menggunakan IQR
for kolom in kolom_numerik:
    Q1 = df[kolom].quantile(0.25)
    Q3 = df[kolom].quantile(0.75)
    IQR = Q3 - Q1
    
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    # Mendeteksi outliers
    outliers = df[(df[kolom] < lower_bound) | (df[kolom] > upper_bound)]
    print(f"\nOutliers pada kolom '{kolom}':")
    print(outliers)

    # Menangani outliers (opsional, tergantung pada keputusan Anda)
    df = df[(df[kolom] >= lower_bound) & (df[kolom] <= upper_bound)]
    print(f"Jumlah baris setelah menghandle outliers pada kolom '{kolom}': {len(df)}")

# Menghitung jumlah baris setelah menangani outliers (jika diterapkan)
print(f"Jumlah total baris setelah penanganan outliers: {len(df)}")


# 3 ==================================================================================================
# Scatter Plot untuk melihat hubungan antar variabel input dengan output
plt.figure(figsize=(12, 10))
for i, kolom in enumerate(kolom_numerik, start=1):
    plt.subplot(3, 4, i)
    plt.scatter(df[kolom], df['klasifikasi'], alpha=0.5)
    plt.xlabel(kolom)
    plt.ylabel('klasifikasi')
    plt.title(f'Scatter Plot: {kolom} vs klasifikasi')

plt.tight_layout()
plt.show()

# Heatmap untuk melihat korelasi antar variabel input
corr = df[kolom_numerik + ['klasifikasi']].corr()
plt.figure(figsize=(10, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', linewidths=.5)
plt.title("Heatmap: Korelasi antar Variabel Input")
plt.show()

# 4 ==================================================================================================
# Mengambil hanya kolom-kolom numerik dari dataframe
numerical_columns = data.select_dtypes(include=[np.number])

# Menghitung korelasi antar variabel numerik dengan output 'klasifikasi'
correlation_with_output = numerical_columns.corr()['klasifikasi'].drop(['klasifikasi'])

# Menentukan nilai ambang batas (threshold) untuk korelasi yang signifikan
threshold = 0.2

# Mengidentifikasi variabel yang memiliki korelasi signifikan dengan output
significant_variables = correlation_with_output[abs(correlation_with_output) >= threshold].index.tolist()

# Menghapus variabel non-numerik yang tidak memiliki korelasi signifikan dengan output
filtered_variables = [col for col in data.columns if col in significant_variables or data[col].dtype in [np.float64, np.int64]]

# Membuat dataframe baru hanya dengan variabel yang signifikan
data_filtered = data[filtered_variables]

# Menampilkan variabel-variabel yang dihapus
irrelevant_variables = set(data.columns) - set(filtered_variables)
print("Variabel yang dihapus karena tidak memiliki korelasi signifikan dengan output:")
print(irrelevant_variables)

# Menampilkan dataframe yang sudah difilter
print("\nDataframe setelah menghapus variabel yang tidak memiliki korelasi signifikan:")
print(data_filtered)

# 5 ===================================================================================================
# Diagram Batang untuk satu variabel kategorikal
plt.figure(figsize=(8, 6))
data_to_plot_bar = df['klasifikasi'].value_counts()
data_to_plot_bar.plot(kind='bar')
plt.xlabel('Kategori')
plt.ylabel('Jumlah')
plt.title('Diagram Batang: Distribusi Kelasifikasi')
plt.show()

# Diagram Garis untuk dua variabel numerik terhadap waktu
plt.figure(figsize=(10, 6))
sns.lineplot(x='waktu_bermain_perhari', y='waktu_tidur_perhari', hue='klasifikasi', data=df)
plt.title('Diagram Garis: Waktu Bermain per Hari vs Waktu Tidur per Hari')
plt.show()

# Diagram Pie untuk melihat proporsi klasifikasi
plt.figure(figsize=(6, 6))
data_to_plot_pie = df['klasifikasi'].value_counts()
data_to_plot_pie.plot(kind='pie', autopct='%1.1f%%', startangle=90)
plt.title('Diagram Pie: Proporsi Kelasifikasi')
plt.ylabel('')
plt.show()

# Diagram Area untuk melihat perubahan waktu tidur terhadap waktu bermain
plt.figure(figsize=(16, 6))
data_to_plot_area = df.groupby('klasifikasi')[['waktu_bermain_perhari', 'waktu_tidur_perhari']].sum()
data_to_plot_area.plot(kind='area', stacked=True)
plt.xlabel('Kategori')
plt.ylabel('Jumlah (dalam ribuan)')
plt.title('Total (Area Plot)')
plt.show()

# 7 ================================================================================================
# Memilih kolom-kolom yang akan dianalisis
columns_to_analyze = ['waktu_bermain_perhari', 'waktu_tidur_perhari', 'dampak_buruk_kecanduan_game', 'klasifikasi']

# Memeriksa dan menghapus baris dengan nilai kosong dalam kolom yang dipilih
data_clean = data.dropna(subset=columns_to_analyze)

# Membuat diagram batang 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y']

width = 0.2 

for i, column in enumerate(columns_to_analyze):
    xs = np.arange(len(data_clean))
    ys = data_clean[column]  # Gunakan nilai kolom langsung sebagai ys
    zs = i  # Gunakan nilai indeks loop sebagai zs (posisi di sumbu z)

    ax.bar(xs, ys, zs, zdir='y', color=colors[i], alpha=0.8, width=width, label=column)

# Menyusun label pada sumbu-sumbu
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')  # Set z-axis as y values

# Menambahkan legenda
ax.legend()

# Menampilkan plot
plt.show()

# Memilih kolom-kolom yang akan dianalisis
columns_to_analyze = ['waktu_bermain_perhari', 'waktu_tidur_perhari', 'dampak_buruk_kecanduan_game', 'klasifikasi']

# Memeriksa dan menghapus baris dengan nilai kosong dalam kolom yang dipilih
data_clean = data.dropna(subset=columns_to_analyze)

# Membuat diagram batang 3D
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
colors = ['r', 'g', 'b', 'y']

width = 0.2 

for i, column in enumerate(columns_to_analyze):
    xs = np.arange(len(data_clean))
    ys = data_clean[column]  # Gunakan nilai kolom langsung sebagai ys
    zs = i  # Gunakan nilai indeks loop sebagai zs (posisi di sumbu z)

    ax.bar(xs, ys, zs, zdir='y', color=colors[i], alpha=0.8, width=width, label=column)

# Menambahkan dimensi waktu sebagai variabel yang diplot
time_dimension = np.random.randint(1, 5, len(data_clean))  # Contoh data waktu (dalam bulan)
ax.bar(xs, time_dimension, zs=len(columns_to_analyze), zdir='y', color='orange', alpha=0.8, width=width, label='Waktu')

# Menyusun label pada sumbu-sumbu
ax.set_xlabel('X')
ax.set_ylabel('Y')
ax.set_zlabel('Z')  # Set z-axis as y values

# Menambahkan legenda
ax.legend()

# Menampilkan plot
plt.show()


# Membuat plot 3D untuk diagram garis
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Menampilkan diagram garis 3D
ax.plot(df['waktu_bermain_perhari'], df['waktu_tidur_perhari'], df['dampak_buruk_kecanduan_game'], label='Klasifikasi')

# Menambahkan label sumbu
ax.set_xlabel('Waktu Bermain per Hari')
ax.set_ylabel('Waktu Tidur per Hari')
ax.set_zlabel('Dampak Buruk Kecanduan Game')

plt.title('Diagram Garis 3D: Waktu Bermain, Waktu Tidur, dan Dampak Buruk Kecanduan Game')
plt.legend()
plt.show()

# Membuat plot 3D untuk diagram area
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Menampilkan diagram area 3D
ax.plot(df['waktu_bermain_perhari'], df['waktu_tidur_perhari'], df['dampak_buruk_kecanduan_game'], label='Klasifikasi', alpha=0.5)

# Menambahkan label sumbu
ax.set_xlabel('Waktu Bermain per Hari')
ax.set_ylabel('Waktu Tidur per Hari')
ax.set_zlabel('Dampak Buruk Kecanduan Game')

plt.title('Diagram Area 3D: Waktu Bermain, Waktu Tidur, dan Dampak Buruk Kecanduan Game')
plt.legend()
plt.show()

# 9 =========================================================================================
#Visualisasi high dimensional dan multivariate menggunakan Scatter Plot

# Menentukan variabel yang akan digunakan
var_x = 'waktu_bermain_perhari'
var_y = 'waktu_tidur_perhari'
var_z = 'dampak_buruk_kecanduan_game'
var_color = 'klasifikasi'  # Variabel untuk menentukan warna

# Membuat scatter plot 3D dengan warna berdasarkan variabel klasifikasi
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Menampilkan scatter plot 3D dengan warna berdasarkan variabel klasifikasi
scatter = ax.scatter(df[var_x], df[var_y], df[var_z], c=df[var_color], cmap='viridis', marker='o')

# Menentukan label pada sumbu x, y, dan z
ax.set_xlabel(var_x)
ax.set_ylabel(var_y)
ax.set_zlabel(var_z)

# Menambahkan colorbar untuk menunjukkan klasifikasi
plt.colorbar(scatter, label='Klasifikasi')

plt.title('Scatter Plot 3D dengan Perbandingan Klasifikasi')
plt.show()


# Memilih variabel-variabel yang akan digunakan
var_x = 'waktu_bermain_perhari'
var_y = 'waktu_tidur_perhari'
var_z = 'dampak_buruk_kecanduan_game'
var_klasifikasi = 'klasifikasi'

# Membuat scatter plot 3D
fig = plt.figure(figsize=(10, 8))
ax = fig.add_subplot(111, projection='3d')

# Data untuk sumbu x, y, z, dan klasifikasi
x = df_cleaned[var_x]
y = df_cleaned[var_y]
z = df_cleaned[var_z]
klasifikasi = df_cleaned[var_klasifikasi]

# Menampilkan scatter plot 3D dengan warna berbeda untuk setiap klasifikasi
for klasifikasi_value in klasifikasi.unique():
    ax.scatter(x[klasifikasi == klasifikasi_value], 
               y[klasifikasi == klasifikasi_value], 
               z[klasifikasi == klasifikasi_value], 
               label=f'Klasifikasi {klasifikasi_value}')

# Menambahkan label sumbu
ax.set_xlabel(var_x)
ax.set_ylabel(var_y)
ax.set_zlabel(var_z)

# Menambahkan legenda
ax.legend()

plt.title('Multivariate Scatter Plot 3D dengan Klasifikasi')
plt.show()
