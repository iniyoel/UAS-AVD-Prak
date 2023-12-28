function predict() {
    // Mendapatkan nilai input dari formulir
    var waktu_bermain_perhari = parseFloat(document.getElementById('waktu_bermain_perhari').value);
    var sering_lalai_kerjakan_tugas = parseInt(document.getElementById('sering_lalai_kerjakan_tugas').value);
    var sering_melewatkan_waktu_tidur = parseInt(document.getElementById('sering_melewatkan_waktu_tidur').value);
    var kesulitan_mengurangi_waktu = parseInt(document.getElementById('kesulitan_mengurangi_waktu').value);
    var merasa_kekurangan_waktu = parseInt(document.getElementById('merasa_kekurangan_waktu').value);
    var pengaruh_tingkat_konsentrasi = parseInt(document.getElementById('pengaruh_tingkat_konsentrasi').value);

    // Kirim permintaan POST ke endpoint /predict di aplikasi Flask
    fetch('/predict', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({
            waktu_bermain_perhari: waktu_bermain_perhari,
            sering_lalai_kerjakan_tugas: sering_lalai_kerjakan_tugas,
            sering_melewatkan_waktu_tidur: sering_melewatkan_waktu_tidur,
            kesulitan_mengurangi_waktu: kesulitan_mengurangi_waktu,
            merasa_kekurangan_waktu: merasa_kekurangan_waktu,
            pengaruh_tingkat_konsentrasi: pengaruh_tingkat_konsentrasi
        }),
    })
        .then(response => response.json())
        .then(data => {
            Swal.fire({
                title: 'Prediction Result',
                text: data.prediction_message,
                icon: 'info',
                confirmButtonText: 'OK'
            }).then((result) => {
                if (result.isConfirmed) {
                    document.getElementById('predictionResult').innerText = data.prediction_message;
                    document.getElementById('predictionResult').scrollIntoView({
                        behavior: 'smooth',
                        block: 'start'
                    });
                }
            });
        })
        .catch(error => console.error('Error:', error));
}
