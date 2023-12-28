import pandas as pd
import matplotlib.pyplot as plt

# Membaca data dari file CSV dengan koma sebagai pemisah desimal
df = pd.read_csv('video_games_effect.csv', decimal=',')

# Memilih dua kolom yang ingin diambil untuk analisis scatter plot
column_x = 'waktu_bermain_perhari'
column_y = 'nilai_menurun'

# Membuat scatter plot
plt.figure(figsize=(8, 6))
plt.scatter(df[column_x], df[column_y], alpha=0.5)
plt.title('Scatter Plot antara {} dan {}'.format(column_x, column_y))
plt.xlabel(column_x)
plt.ylabel(column_y)
plt.grid(True)
plt.show()
