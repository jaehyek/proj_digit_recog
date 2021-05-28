import cv2
import numpy as np
import matplotlib.pyplot as plt
import pytesseract
from PIL import Image

from makeImageFolder import  get_Image_Value_List_from_json

def main():
    PILim = Image.open(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56265.bmp').convert('L')
    
    gray = np.array(PILim)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')
    
    # Maximize Contrast (Optional)
    
    structuringElement = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    
    imgTopHat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, structuringElement)
    imgBlackHat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, structuringElement)
    
    imgGrayscalePlusTopHat = cv2.add(gray, imgTopHat)
    gray = cv2.subtract(imgGrayscalePlusTopHat, imgBlackHat)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(gray, cmap='gray')
    
    #  Adaptive Thresholding
    img_blurred = cv2.GaussianBlur(gray, ksize=(5, 5), sigmaX=0)
    
    _, img_result = cv2.threshold(img_blurred, thresh=0.0, maxval=255.0, type=cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    
    plt.figure(figsize=(12, 10))
    plt.imshow(img_result, cmap='gray')
    
    chars = pytesseract.image_to_string(gray, lang='kor', config='--psm 7 --oem 1')
    
    result_chars = ''
    has_digit = False
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            if c.isdigit():
                has_digit = True
            result_chars += c
    
    print(result_chars)
    
def read_digit(jsonfile):
    list_image, list_value, dict_json_info = get_Image_Value_List_from_json(jsonfile)
    
    image_first = list_image[0].convert('L')
    image_first = np.array(image_first)
    height, width = image_first.shape

    for pil_image in list_image[1:]:
        pil_image = np.array(pil_image.convert('L'))
        image = cv2.resize(pil_image,(width, height) )
        image_first = cv2.hconcat([image_first, image])

    plt.figure(figsize=(12, 10))
    plt.imshow(image_first, cmap='gray')
    plt.show()
    
    chars = pytesseract.image_to_string(image_first, lang='kor', config='--psm 7 --oem 3')
    result_chars = ''
    for c in chars:
        if ord('가') <= ord(c) <= ord('힣') or c.isdigit():
            result_chars += c

    print(result_chars)
    
if __name__ == '__main__' :
    npred = read_digit(r'D:\proj_gauge\민성기\digit_test\10050-56359.json')