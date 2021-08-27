import codecs, json
import os
import numpy as np
import cv2

from PIL import Image

##########################################################################
def save_dict_to_file(dict_save, filejson,mode='w'):
    with codecs.open(filejson, mode, encoding='utf-8') as f:
        json.dump(dict_save, f, ensure_ascii=False, indent=4)


def read_dict_from_file(filejson):
    with codecs.open(filejson, 'r', encoding='utf-8') as f:
        obj = json.load(f)
        return obj

def read_opencv_file(filename):
    img_array = np.fromfile(filename, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image

def write_opencv_file(image, filename):
    result, encoded_img = cv2.imencode('.jpg', image)
    if result:
        with open(filename, mode='w+b') as f:
            encoded_img.tofile(f)
            return True
    else:
        return False

def show_cvimage(image):
    cv2.imshow('a', image)
    cv2.waitKey()                   # wait을 해야, 이미지가 나온다. PIL은  im.show()으로 바로 나온다.
    cv2.destroyAllWindows()