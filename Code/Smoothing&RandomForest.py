import pandas as pd
from statsmodels.tsa.api import ExponentialSmoothing
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler  # Tambahkan impor ini
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import matplotlib.pyplot as plt

# Membaca data dari file CSV
df = pd.read_csv('video_games_effect.csv')

#2.A =======================================================================================================================
# Menerapkan eksponensial smoothing pada waktu bermain per hari
model_bermain = ExponentialSmoothing(df['waktu_bermain_perhari'], trend='add').fit()
prediksi_bermain = model_bermain.predict(start=df.index.min(), end=df.index.max())

# Menerapkan eksponensial smoothing pada waktu tidur per hari
model_tidur = ExponentialSmoothing(df['waktu_tidur_perhari'], trend='add').fit()
prediksi_tidur = model_tidur.predict(start=df.index.min(), end=df.index.max())

# Plot data asli dan prediksi waktu bermain
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['waktu_bermain_perhari'], label='Waktu Bermain Asli')
plt.plot(df.index, prediksi_bermain, label='Prediksi Waktu Bermain')
plt.title('Eksponensial Smoothing - Waktu Bermain per Hari')
plt.xlabel('Tanggal')
plt.ylabel('Jam')
plt.legend()
plt.show()

# Plot data asli dan prediksi waktu tidur
plt.figure(figsize=(10, 6))
plt.plot(df.index, df['waktu_tidur_perhari'], label='Waktu Tidur Asli')
plt.plot(df.index, prediksi_tidur, label='Prediksi Waktu Tidur')
plt.title('Eksponensial Smoothing - Waktu Tidur per Hari')
plt.xlabel('Tanggal')
plt.ylabel('Jam')
plt.legend()
plt.show()

# Contoh menggantikan NaN dengan nilai median
imputer_y = SimpleImputer(strategy='median')
df['klasifikasi'] = imputer_y.fit_transform(df['klasifikasi'].values.reshape(-1, 1))

# Memisahkan variabel independen (X) dan dependen (y)
X = df[['waktu_bermain_perhari', 'sering_lalai_kerjakan_tugas', 'sering_melewatkan_waktu_tidur',
 'kesulitan_mengurangi_waktu', 'merasa_kekurangan_waktu', 'pengaruh_tingkat_konsentrasi']]
y = df['klasifikasi']  # Gantilah 'label_target' dengan nama kolom target yang sesuai

# Membagi dataset menjadi data training dan data testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Menampilkan jumlah data training dan testing dalam persen
percentage_train = len(X_train) / len(df) * 100
percentage_test = len(X_test) / len(df) * 100

print(f"Persentase Data Training: {percentage_train:.2f}%")
print(f"Persentase Data Testing: {percentage_test:.2f}%")

# Menggunakan SimpleImputer untuk menggantikan NaN dengan mean
imputer = SimpleImputer(strategy='mean')

X_train_imputed = imputer.fit_transform(X_train)
X_test_imputed = imputer.transform(X_test)

# Menerapkan K-Means untuk membuat klaster
kmeans = KMeans(n_clusters=3, random_state=42)
X_train_clusters = kmeans.fit_predict(X_train_imputed)

# Menampilkan Silhouette Score
silhouette_avg = silhouette_score(X_train_imputed, X_train_clusters)
print(f"Silhouette Score: {silhouette_avg}")

#3.A =======================================================================================================================
# Membuat model Random Forest
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)

# Melatih model menggunakan data training yang telah diimputasi
rf_model.fit(X_train_imputed, y_train)

# Melakukan prediksi menggunakan data testing yang telah diimputasi
y_pred = rf_model.predict(X_test_imputed)

#5.A =======================================================================================================================
# Evaluasi performa model
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred)

# Akurasi dalam persen
accuracy_percent = accuracy * 100
print(f'Accuracy: {accuracy_percent:.2f}%')

print(f'Confusion Matrix:\n{conf_matrix}')
print(f'Classification Report:\n{class_report}')