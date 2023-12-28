import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file CSV dengan koma sebagai pemisah desimal
df = pd.read_csv('video_games_effect.csv')

# Memilih kolom-kolom yang ingin diambil untuk analisis histogram
selected_columns = ['waktu_bermain_perhari', 'waktu_tidur_perhari', 'sering_lalai_kerjakan_tugas',
                    'sering_melewatkan_waktu_tidur', 'kesulitan_mengurangi_waktu', 'merasa_kekurangan_waktu',
                    'pengaruh_tingkat_konsentrasi', 'pengaruh_game_terhadap_tingkat_motivasi_belajar',
                    'kesulitan_saat_ujian', 'nilai_menurun']

# Membuat histogram untuk setiap kolom
plt.figure(figsize=(15, 12))
df[selected_columns].hist(bins=20, edgecolor='black', grid=False)
plt.suptitle('Histogram untuk Setiap Kolom', y=1.02)
plt.tight_layout()
plt.show()
