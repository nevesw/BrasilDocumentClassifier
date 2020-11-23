import tensorflow as tf
import numpy as np
import base64
import json
import pytesseract as ocr
from tensorflow import keras
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from flask import Flask, request, jsonify
from enum import Enum
from io import BytesIO
from PIL import Image


app = Flask(__name__)

class ClassifierReturn:
    def __init__(self):
        self.typedoc = None
        self.score = 00.00
        self.isIdentifierDocument = True

def return_prediction(content):
    imgbase64 = content["imagebase64"]
    ocr_text = ocrTesseract(imgbase64)
    #model = load_model('document_detector.h5')
    model = load_model('document_classifier.h5')
    response = ClassifierReturn()
    class_names = np.array(['cnh', 'cpf', 'rgfrente', 'rgverso'])
    img_height = 580
    img_width = 580

   
    img = Image.open(BytesIO(base64.b64decode(imgbase64)))
    img = img.resize((img_height,img_width),Image.ANTIALIAS)


    #img = keras.preprocessing.image.load_img(teste_path, target_size=(img_height, img_width))
    img_array = keras.preprocessing.image.img_to_array(img)
    img_array = tf.expand_dims(img_array, 0) # Create a batch

    predictions = model.predict(img_array)
    score_predictions = tf.nn.softmax(predictions[0])   
    response.typedoc = class_names[np.argmax(score_predictions)]
    response.score = (100 * np.max(score_predictions))
    return json.dumps(response.__dict__)

def ocrTesseract(content):
    #Tesseract OCR via api, ou integrado
    #Tratamento de imagem para melhor extração
    image = Image.open(BytesIO(base64.b64decode(content)))
    ocr.pytesseract.tesseract_cmd = r'C:\Users\USER\AppData\Local\Tesseract-OCR\tesseract.exe'
    phrase = ocr.image_to_string(image, lang='por')
    return phrase

def cropLayout():
    #Algoritimo de crop com layout de coordenadas predefinido
    return True

@app.route("/")
def index():
    return 'Flask Api is running'

@app.route('/imageclassifier/predict/',methods=['POST'])
def document_prediction():
    content = request.json
    results = return_prediction(content)
    return jsonify(results)

if __name__=='__main__':
    app.run()

