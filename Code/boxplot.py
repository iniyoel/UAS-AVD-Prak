import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file CSV ke dalam dataframe
df = pd.read_csv('video_games_effect.csv')

# Memilih kolom-kolom yang ingin diambil untuk analisis boxplot
selected_column = ['waktu_bermain_perhari', 'waktu_tidur_perhari', 'sering_lalai_kerjakan_tugas',
                    'sering_melewatkan_waktu_tidur', 'kesulitan_mengurangi_waktu', 'merasa_kekurangan_waktu',
                    'pengaruh_tingkat_konsentrasi', 'pengaruh_game_terhadap_tingkat_motivasi_belajar',
                    'kesulitan_saat_ujian', 'nilai_menurun']

# Membuat boxplot untuk satu kolom
plt.figure(figsize=(8, 6))
df[selected_column].plot(kind='box', vert=False)
plt.title('Boxplot untuk Kolom: {}'.format(selected_column))
plt.xlabel('Nilai')
plt.show()
