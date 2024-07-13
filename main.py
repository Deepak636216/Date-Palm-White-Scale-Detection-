from flask import Flask, request, render_template
from PIL import Image
import numpy as np
from preprocessing import process_image,compute_hsv_features,compute_glcm_features
from joblib import load
app = Flask(__name__)

svm_glcm=load('D:/ML work/web_interface/models/SVM_glcm.joblib')
knn_glcm=load('D:/ML work/web_interface/models/KNN_glcm.joblib')
random_forest_glcm=load('D:/ML work/web_interface/models/random_forest_glcm.joblib')
lightgb_glcm=load('D:/ML work/web_interface/models/lightgbm_glcm.joblib')
svm_glcm_hsv=load('D:/ML work/web_interface/models/svm_glcm_hsv.joblib')
knn_glcm_hsv=load('D:/ML work/web_interface/models/knn_glcm_hsv.joblib')
random_forest_glcm_hsv=load('D:/ML work/web_interface/models/random_forest_glcm_hsv.joblib')
lightgb_glcm_hsv=load('D:/ML work/web_interface/models/lightgbm_glcm_hsv.joblib')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload():
    if 'file' not in request.files:
        return render_template('index.html', error='No file uploaded')

    file = request.files['file']
    image_array = process_image(file)

    if request.form['method'] == 'GLCM':
        texture_values = compute_glcm_features(image_array)
        svm = svm_glcm.predict([texture_values])[0]
        knn = knn_glcm.predict([texture_values])[0]
        random_forest = random_forest_glcm.predict([texture_values])[0]
        lightgb = lightgb_glcm.predict([texture_values])[0]

    elif request.form['method'] == 'GLCM+HSV':
        texture_values = np.concatenate((compute_glcm_features(image_array), compute_hsv_features(image_array)))
        svm = svm_glcm_hsv.predict([texture_values])[0]
        knn = knn_glcm_hsv.predict([texture_values])[0]
        random_forest = random_forest_glcm_hsv.predict([texture_values])[0]
        lightgb = lightgb_glcm_hsv.predict([texture_values])[0]
    else:
        return render_template('index.html', error='Invalid ')
    return render_template('index.html', texture_values=texture_values, svm=svm, knn=knn, random_forest=random_forest, lightgb=lightgb)



if __name__ == '__main__':
    app.run(debug=True)
