import sys
import os
from glob import glob
import codecs, json
from PIL import Image
import random
import shutil
import cv2
import numpy as np
import albumentations as A
import random
from imgaug import augmenters as iaa

# https://github.com/albumentations-team/albumentations

def saveimage(sub,dir_digit, basename ):
    filename_jpg = os.path.join(dir_digit, basename + '.jpg')
    sub.save(filename_jpg)


def extractDigit_saveto(file_json, file_bmp, list_dir_digit, digit_type):
    with codecs.open(file_json, 'r', encoding='utf-8') as f:
        dict_bmp_info = json.load(f)
        digitFractNo = int(dict_bmp_info['digitFractNo'])
        digitAllNo = int(dict_bmp_info['digitAllNo'])
        dataValue = int(dict_bmp_info['dataValue'] * 10 ** digitFractNo)
        digitRect = dict_bmp_info['digitRect']
        str_dataValue = f'{dataValue:0{digitAllNo}}'
        
        if digit_type == 'digit_7'  and digitAllNo > 5 :
            return
        elif digit_type == 'digit_normal'  and digitAllNo < 6 :
            return
        
        list_digitRect = digitRect.split('|')[1:]
        list_digitRect = [ aa.split(',') for aa in list_digitRect]
        list_digitRect = [[int(a),int(b),int(c),int(d)]for a,b,c,d in list_digitRect]
        
        img = Image.open(file_bmp)
        if img == None:
            print(f"Can't read a image file :{file_bmp}")
            return
        for index  in range(digitAllNo) :
            x, y, width, height = list_digitRect[index]
            sub = img.crop((x,y,x+width,y+height))
            saveimage(sub,list_dir_digit[int(str_dataValue[index])], os.path.basename(file_json).split('.')[0] + f'_{index}' )



def makeImageFolder(folder_json, folder_digit, digit_type = 'all'):
    try:
        if not os.path.isdir(folder_digit) :
            os.mkdir(folder_digit)
    except:
        pass
    
    # create dir for 0, 1, 2, ..., 9
    list_dir_digit = []
    for num in range(10):
        try:
            dir_digit = os.path.join(folder_digit, f'{num}')
            list_dir_digit.append(dir_digit)
            os.mkdir(dir_digit)
            
        except:
            continue

    list_json = glob(folder_json + r"\*.json")
    
    for file_json in list_json:
        file_bmp = os.path.splitext(file_json)[0] + '.bmp'
        extractDigit_saveto(file_json, file_bmp, list_dir_digit, digit_type)
        

def makeTrainValidFromDigitClass(dir_digit_class, dir_train, dir_valid, train_ratio=0.8):
    try:
        if not os.path.isdir(dir_train):
            os.mkdir(dir_train)
    except:
        pass
    
    for num in range(10):
        try:
            dir_digit = os.path.join(dir_train, f'{num}')
            os.mkdir(dir_digit)
        except:
            continue

    try:
        if not os.path.isdir(dir_valid):
            os.mkdir(dir_valid)
    except:
        pass

    for num in range(10):
        try:
            dir_digit = os.path.join(dir_valid, f'{num}')
            os.mkdir(dir_digit)
        except:
            continue
            
    for index in range(10):
        dir_src = os.path.join(dir_digit_class, str(index))
        dir_train_dest = os.path.join(dir_train, str(index))
        dir_valid_dest = os.path.join(dir_valid, str(index))
        
        list_src = glob(dir_src + r"\*.jpg")
        len_src = len(list_src)
        len_train =  int(train_ratio * len_src)
        len_valid = len_src - len_train
        list_src_index = list(range(len_src))
        list_train_index = random.sample(list_src_index, len_train)
        list_valid_index = list(set(list_src_index) - set(list_train_index))
        
        # copy files to train
        for file_index in list_train_index :
            file_src = list_src[file_index]
            shutil.copy(file_src, dir_train_dest)

        # copy files to valid
        for file_index in list_valid_index:
            file_src = list_src[file_index]
            shutil.copy(file_src, dir_valid_dest)
        

def imagesave(image, file_out):
    result, encoded_img = cv2.imencode('.jpg', image)
    if result:
        with open(file_out, mode='w+b') as f:
            encoded_img.tofile(f)

def imageread(file_in):
    img_array = np.fromfile(file_in, np.uint8)
    image = cv2.imdecode(img_array, cv2.IMREAD_COLOR)
    return image
            
def imageAugmentation(dir_in, dir_out):
    random.seed(42)
    try:
        if not os.path.isdir(dir_out) :
            os.mkdir(dir_out)
    except:
        pass

    list_jpg = glob(dir_in + r"\**\*.jpg", recursive=True)
    
    list_aug = [
        A.CLAHE(),
        A.OpticalDistortion(),
        A.GridDistortion(),
        A.HueSaturationValue(),
        A.GaussNoise(),
        A.MotionBlur(p=.2),
        A.RandomBrightnessContrast(p=0.9),
        # A.InvertImg(),        # Not Accepted
        # A.ISONoise(),        # Not Accepted
        # A.RandomFog(),                  # Not Accepted
        # # A.RandomRain(),      # Not Accepted
        # # A.RandomSnow()      # Not Accepted
    ]
    list_aug_name = [
        'CLAHE',
        'OpticalDist',
        'GridDist',
        'HueSat',
        'GaussNoise',
        'MotionBlur',
        'RandomBright',
        # 'InvertImg',
        # 'IsoNoise',
        # 'RandomFog',
        # # 'RandomRain',
        # # 'RandomSnow'
    ]

    list_aa = [
        # iaa.MedianBlur(k=(3, 11)),      # Not Accepted
        # iaa.Dropout((0.05, 0.06), per_channel=0.5),      # Not Accepted
        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),      # Not Accepted
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        # iaa.ElasticTransformation(alpha=(0.5, 3.5), sigma=0.25),      # Not Accepted
        iaa.PiecewiseAffine(scale=(0.01, 0.02)),
        iaa.EdgeDetect(alpha=0.3),
        # iaa.Sharpen(alpha=(0.0, 1.0)),      # Not Accepted
        # iaa.DirectedEdgeDetect(alpha=0.5, direction=0),      # Not Accepted
    ]
    list_aa_name = [
        # 'MedianBlur',
        # 'Dropout',
        # 'Emboss',
        'AdditiveGaussianNoise',
        # 'ElasticTransformation',
        'PiecewiseAffine',
        'EdgeDetect',
        # 'Sharpen',
        # 'DirectedEdgeDetect',
    ]
    
    
    for jpg in list_jpg :
        jpg_out = jpg.replace(dir_in, dir_out)
        dir_out_jpg = os.path.dirname(jpg_out)
        try:
            if not os.path.exists(dir_out_jpg) :
                os.mkdir(dir_out_jpg)
        except:
            pass
        jpg_out_basename = os.path.splitext(jpg_out)[0]
        image = imageread(jpg)
        imagesave(image, jpg_out_basename + '.jpg')     # save the orignal image.
        for i in range( len(list_aug )) :
            augmented_image = list_aug[i](image=image)['image']
            imagesave(augmented_image, jpg_out_basename + '_' + list_aug_name[i] + '.jpg')
            
        for i in range( len(list_aa )) :
            augmented_image = list_aa[i](image=image)
            imagesave(augmented_image, jpg_out_basename + '_' + list_aa_name[i] + '.jpg')


def extractDigitImage_Value_List(file_json, file_bmp):
    img = Image.open(file_bmp)
    if img == None:
        print(f"Can't read a image file :{file_bmp}")
        return [], []
    
    img = img.convert('RGB')
    with codecs.open(file_json, 'r', encoding='utf-8') as f:
        dict_json_info = json.load(f)
        digitFractNo = int(dict_json_info['digitFractNo'])
        digitAllNo = int(dict_json_info['digitAllNo'])
        dataValue = int(dict_json_info['dataValue'] * 10 ** digitFractNo)
        digitRect = dict_json_info['digitRect']
        str_dataValue = f'{dataValue:0{digitAllNo}}'
        
        list_digitRect = digitRect.split('|')[1:]
        list_digitRect = [aa.split(',') for aa in list_digitRect]
        list_digitRect = [[int(a), int(b), int(c), int(d)] for a, b, c, d in list_digitRect]
        
        
        
        list_image = []
        for index in range(digitAllNo):
            x, y, width, height = list_digitRect[index]
            sub = img.crop((x, y, x + width, y + height))
            list_image.append(sub)
            
    return list_image, [int(aa) for aa in str_dataValue],  dict_json_info
    
def get_Image_Value_List_from_json(file_json):
    list_image, list_value, dict_json_info = extractDigitImage_Value_List( file_json, os.path.splitext(file_json)[0] + '.bmp')
    return list_image, list_value, dict_json_info

    
    
if __name__ == '__main__' :
    # json, jpg 파일이 있는 dir을 지정하고,   출력은 지정한 dir밑에  0 ~9 까지 dir을 만들고 해당 숫자 이미지들이  jpg형태로 저장한다.,
    makeImageFolder(r'D:\proj_gauge\민성기\digitGaugeSamples', r'.\digit_class', 'digit_all')    # digit_7, digit_normal, digit_all
    # makeImageFolder(r'D:\proj_gauge\민성기\digitGaugeSamples_temp', r'.\digit_class')
    
    # Image Augment을 위해  input dir을 지정해 주면, output dir에  이미지 증강 시켜 저장한다.
    imageAugmentation(r'.\digit_class', r'.\digit_class_aug')
    # imageAugmentation(r'D:\proj_gauge\test_class', r'D:\proj_gauge\test_class_aug')
    
    
    # 분류된 input dir을 지정해 주면,    지정한 train dir에,  지정한 valid dir에   비율대로, 이미지를 분산 저장한다.
    makeTrainValidFromDigitClass(r'.\digit_class_aug', r'.\digit_class_train', r'.\digit_class_valid', train_ratio=0.8)
    # makeTrainValidFromDigitClass(r'.\digit_class', r'.\digit_class_train', r'.\digit_class_valid', train_ratio=0.8)


