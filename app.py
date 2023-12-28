from flask import Flask, request, jsonify, render_template
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Load data atau sesuaikan dengan kebutuhan
df = pd.read_csv('video_games_effect.csv')

# Transformasi label menggunakan LabelEncoder
labelencoder = LabelEncoder()

df['waktu_bermain_perhari'] = df['waktu_bermain_perhari'].map({'Ya': 1, 'Tidak': 0})

label_columns = [
    'sering_lalai_kerjakan_tugas',
    'sering_melewatkan_waktu_tidur',
    'kesulitan_mengurangi_waktu',
    'merasa_kekurangan_waktu',
    'pengaruh_tingkat_konsentrasi'
]
for column in label_columns:
    df[column] = df[column].map({'Tidak Pernah': 1, 'Jarang': 2, 'Netral': 3, 'Pernah': 4, 'Selalu': 5})

df['klasifikasi'] = df['klasifikasi'].map({'Ya': 1, 'Tidak': 0})

# Features and target variable
X = df[['waktu_bermain_perhari', 'sering_lalai_kerjakan_tugas',
        'sering_melewatkan_waktu_tidur', 'kesulitan_mengurangi_waktu',
        'merasa_kekurangan_waktu', 'pengaruh_tingkat_konsentrasi']]
y = df['klasifikasi']

# Membagi dataset menjadi data training dan data uji
X_train, _, y_train, _ = train_test_split(X, y, test_size=0.1, random_state=42)

# Fungsi untuk memprediksi
def predict(input_data):
    # Contoh penggunaan model Random Forest (sesuaikan dengan struktur data Anda)
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)

    # Transformasi nilai input menjadi array
    input_data = [input_data[col] if col != 'waktu_bermain_perhari' else input_data[col].map({'Ya': 1, 'Tidak': 0}) for col in X.columns]

    Y_pred = rf_model.predict([input_data])

    return int(Y_pred[0])

def custom_predict(input_data):
    # Kriteria untuk menentukan prediksi khusus
    if input_data['waktu_bermain_perhari'] <= 4:
        return 0  # Tidak ada penurunan nilai akibat bermain game
    elif input_data['waktu_bermain_perhari'] > 4 and all(val in [4, 5] for val in input_data.values()):
        return 1  # Ya, terdapat penurunan nilai akibat terlalu banyak bermain game
    elif any(val in [1, 2, 3] for val in input_data.values()):
        return 0  # Tidak ada penurunan nilai akibat bermain game
    else:
        return predict(input_data)

@app.route('/predict', methods=['POST'])
def prediction_endpoint():
    try:
        # Ambil data dari formulir JSON
        data = request.get_json()

        # Contoh penggunaan model pada data yang baru
        prediction = custom_predict(data)

        # Konversi nilai prediksi ke pesan yang diinginkan
        prediction_message = "Ya, terdapat penurunan nilai akibat terlalu banyak bermain game" if prediction == 1 else "Tidak ada penurunan nilai akibat bermain game."

        # Mengirimkan hasil prediksi sebagai respon JSON
        response_data = {"prediction": int(prediction), "prediction_message": prediction_message}
        return jsonify(response_data)

    except Exception as e:
        error_message = f"Error: {str(e)}"
        return jsonify({"error": error_message}), 500

# Route for serving the index.html file
@app.route('/')
def index():
    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
