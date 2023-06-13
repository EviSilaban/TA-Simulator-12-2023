from flask import Flask,request
app = Flask(__name__)
import pickle
# import pandas as pd
from flask import render_template
import random

with open('modelSVMPSO.pkl', 'rb') as model:
  model = pickle.load(model)

@app.route("/")
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def svmpredict():
   data = { 
        'kepemilikan_rumah' : [request.form["kepemilikan_rumah"]],
        'kepemilikan_lahan' : [request.form["kepemilikan_lahan"]],
        'jenis_dinding' : [request.form["jenis_dinding"]],
        'jenis_atap' : [request.form["jenis_atap"]],
        'fasilitas_buang_air_besar' : [request.form["fasilitas_buang_air_besar"]],
        'ada_tabung_gas' : [request.form["ada_tabung_gas"]],
        'ada_ac' : [request.form["ada_ac"]],
        'ada_pemanas' : [request.form["ada_pemanas"]],
        'ada_telepon' : [request.form["ada_telepon"]],
        'ada_emas' : [request.form["ada_emas"]],
        'ada_sepeda' : [request.form["ada_sepeda"]],
        'ada_mobil' : [request.form["ada_mobil"]],
        'ada_kapal' : [request.form["ada_kapal"]],
        'status_kis' : [request.form["status_kis"]],
        'status_bpjs_mandiri' : [request.form["status_bpjs_mandiri"]],
        'status_asuransi' : [request.form["status_asuransi"]],
        'status_kredit_usaha_rakyat' : [request.form["status_kredit_usaha_rakyat"]],
   }
  # Extract column values
   column_values = list(data.values())

   # Calculate the number of samples
   n_samples = len(column_values[0])

   # Create an empty array to hold the data
   data_array = [[] for _ in range(n_samples)]

   # Convert data to array
   for values in column_values:
      for i, value in enumerate(values):
         data_array[i].append(value)
    # Mengatur seed pada random
   random.seed(42)

   confidence_scores = model.predict_proba(data_array)
   predicted_class = model.predict(data_array)[0]

  # Mendapatkan label kelas dari model
   class_labels = model.classes_

  # Menyimpan confidence score dari predicted class ke dalam variabel
   predicted_class_index = [i for i, label in enumerate(class_labels) if label == predicted_class][0]
   confidence_score_predicted = confidence_scores[0][predicted_class_index] * 100
   formatted_confidence_score_predicted = "{:.2f}".format(confidence_score_predicted)

  # Menampilkan confidence score dari predicted class
   # print(f"Confidence Score (Predicted Class): {confidence_score_predicted:.2f}%")

    
   return render_template('result.html', data = data, result = predicted_class, confidence= formatted_confidence_score_predicted, model = model)