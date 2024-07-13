from tqdm import tqdm
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
from io import BytesIO
import cv2

def process_image(file):
    img = Image.open(file)
    img_array = np.array(img)
    return img_array

def compute_glcm_features(image_array):
    gray_img = image_array.mean(axis=2).astype(np.uint8)
    glcm_matrix = graycomatrix(gray_img, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
    contrast = graycoprops(glcm_matrix, 'contrast')
    correlation = graycoprops(glcm_matrix, 'correlation')
    dissimilarity = graycoprops(glcm_matrix, 'dissimilarity')
    energy = graycoprops(glcm_matrix, 'energy')
    homogeneity = graycoprops(glcm_matrix, 'homogeneity')
    return np.array([energy,correlation,dissimilarity,homogeneity,contrast]).flatten()

def compute_hsv_features(image_array):
    hsv_image = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    h, s, v = cv2.split(hsv_image)

    h_mean, h_std = cv2.meanStdDev(h)
    s_mean, s_std = cv2.meanStdDev(s)
    v_mean, v_std = cv2.meanStdDev(v)

    h_skew = float(cv2.mean(cv2.pow(h - h_mean, 3))[0] / (h_std[0] ** 3))
    s_skew = float(cv2.mean(cv2.pow(s - s_mean, 3))[0] / (s_std[0] ** 3))
    v_skew = float(cv2.mean(cv2.pow(v - v_mean, 3))[0] / (v_std[0] ** 3))

    return h_mean[0][0], h_std[0][0], h_skew, s_mean[0][0], s_std[0][0], s_skew, v_mean[0][0], v_std[0][0], v_skew