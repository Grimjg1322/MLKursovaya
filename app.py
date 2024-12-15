from flask import Flask, render_template, request, redirect, url_for
from models import create_models_and_train
import os
import torch

app = Flask(__name__)

# Путь, куда будут загружаться изображения
UPLOAD_FOLDER = r'C:\Users\Grim_JG\PycharmProjects\PythonProject2\images'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Путь к весам модели
MODEL_WEIGHTS_PATH = r'C:\Users\Grim_JG\PycharmProjects\PythonProject2\yolov5pytorch\best.pt'

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/dense_model')
def dense_model():
    dense_history = create_models_and_train('RailData.csv')[0]
    return render_template('dense_model.html', dense_history=dense_history)

@app.route('/cnn_model')
def cnn_model():
    _, cnn_history = create_models_and_train('RailData.csv')
    return render_template('cnn_model.html', cnn_history=cnn_history)

@app.route('/detect', methods=['GET', 'POST'])
def detect():
    if request.method == 'POST':
        if 'file' not in request.files:
            return redirect(request.url)
        file = request.files['file']
        if file.filename == '':
            return redirect(request.url)
        if file:
            # Сохраняем файл в указанную папку
            filepath = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(filepath)

            # Вызов функции детектирования
            results = perform_detection(filepath)
            return render_template('detect.html', results=results, image_path=file.filename)

    return render_template('detect.html', results=None, image_path=None)

def perform_detection(image_path):
    # Загрузка модели
    model = torch.hub.load('ultralytics/yolov5', 'custom', path=MODEL_WEIGHTS_PATH)  # Указываем путь к весам

    # Выполнение детектирования
    results = model(image_path)  # Используем путь к изображению

    # Возвращаем результаты в формате pandas DataFrame
    return results.pandas().xyxy[0].to_dict(orient='records')

if __name__ == '__main__':
    app.run(debug=True)
