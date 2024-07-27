import os
from flask import Flask, request, send_from_directory, jsonify, redirect, url_for, render_template
from werkzeug.utils import secure_filename
import xmltodict
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import numpy as np

UPLOAD_FOLDER = 'static/files'
TEMPLATE_FOLDER = 'templates'
ALLOWED_EXTENSIONS = {'xml'}

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['TEMPLATE_FOLDER'] = TEMPLATE_FOLDER

def allowed_file(filename):
  return '.' in filename and \
  filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
  
@app.route('/', methods=['GET', 'POST'])
def upload_file():
  if request.method == 'POST':
    if 'file' not in request.files:
      return jsonify('No file part'), 400
    if 'text' not in request.form:
      return jsonify('No text part'), 400
    file = request.files['file']
    text = request.form['text']
    if file.filename == '':
      return jsonify('No selected file'), 400
    if text == '':
      return jsonify('No selected text'), 400
    if not allowed_file(file.filename):
      return jsonify('Only .xml files are allowed'), 400
    if file and allowed_file(file.filename):
      filename = secure_filename(file.filename)
      file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
      send_from_directory(app.config['UPLOAD_FOLDER'], filename)
      return redirect(url_for("output", filename=filename, data_type=text))
  types = open("types.txt").readlines()
  return render_template(os.path.join('upload.html'), types=types)
  
def parse_file(filename, data_type):
  with open(os.path.join(app.config['UPLOAD_FOLDER'], filename)) as file:
    data_dict = xmltodict.parse(file.read())

  health_data = data_dict['HealthData']['Record']
  df = pd.DataFrame(health_data)

  '''df['@creationDate'] = pd.to_datetime(df['@creationDate'])
  df['@startDate'] = pd.to_datetime(df['@startDate'])
  df['@endDate'] = pd.to_datetime(df['@endDate'])'''

  df['@type'] = df['@type'].str.replace('HKQuantityTypeIdentifier', '')

  data_type_records = df[df['@type'] == data_type]

  return data_type_records

def analyze(data_type_records):

  values = data_type_records['@value'].astype(float)

  summary = {
    'count': len(values),
    'mean': np.mean(values),
    'min': np.min(values),
    'max': np.max(values),
    'std': np.std(values)
  }

  return summary

def predict(data_type_records):

  values = data_type_records['@value'].astype(float)[:-1].values.reshape(-1,1)
  previous_values = data_type_records['@value'].astype(float)[1:].values.reshape(-1,1)

  X = previous_values
  y = values

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
  model = LinearRegression()
  model.fit(X_train, y_train)
  y_pred = model.predict(X_test)
  accuracy = model.score(X_test, y_test)

  return {"accuracy": accuracy, "predictions": y_pred.tolist()}

@app.route('/output/<filename>/<data_type>', methods=['GET'])
def output(filename, data_type):
  data_type_records = parse_file(filename, data_type)
  
  if data_type_records.empty:
    return jsonify('No records found for the specified data type'), 400
    
  analytics = analyze(data_type_records)
  prediction = predict(data_type_records)

  output = prediction
  output['analysis'] = analytics

  flatList = []
  
  for element in prediction['predictions']:
    if type(element) is list:
        # Check if type is list than iterate through the sublist
        for item in element:
            flatList.append(item)
    else:
        flatList.append(element)

  return render_template('graph.html', summary=analytics, labels=data_type_records['@creationDate'].tolist(), values1=data_type_records['@value'].astype(float).tolist(), values2=flatList)

if __name__ == '__main__':
  app.run(host='0.0.0.0', port=5000)