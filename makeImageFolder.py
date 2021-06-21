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
import imgaug as ia
import torchvision.transforms.functional as TF

from PIL import Image

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

class MyRotationTransform:
    """Rotate by one of the given angles."""

    def __init__(self, angles):
        self.angles = angles

    def __call__(self, x):
        angle = random.choice(self.angles)
        return TF.rotate(x, angle)

def imageAugmentation(dir_in, dir_out):
    random.seed(42)
    try:
        if not os.path.isdir(dir_out) :
            os.mkdir(dir_out)
    except:
        pass

    list_jpg = glob(dir_in + r"\**\*.jpg", recursive=True)

    ## https://github.com/albumentations-team/albumentations
    list_aug = [
        A.CLAHE(),
        A.OpticalDistortion(),
        # A.GridDistortion(),
        # A.HueSaturationValue(),
        A.GaussNoise(),
        A.MotionBlur(p=.2),
        A.RandomBrightnessContrast(p=0.1),
        # A.InvertImg(),        # Not Accepted
        # A.ISONoise(),        # Not Accepted
        # A.RandomFog(),                  # Not Accepted
        # # A.RandomRain(),      # Not Accepted
        # # A.RandomSnow()      # Not Accepted

    ]
    list_aug_name = [
        'CLAHE',
        'OpticalDist',
        # 'GridDist',
        # 'HueSat',
        'GaussNoise',
        'MotionBlur',
        'RandomBright',
        # 'InvertImg',
        # 'IsoNoise',
        # 'RandomFog',
        # # 'RandomRain',
        # # 'RandomSnow',

    ]

    ## https://github.com/aleju/imgaug

    list_aa = [
        # iaa.MedianBlur(k=(3, 11)),      # Not Accepted
        # iaa.Dropout((0.05, 0.06), per_channel=0.5),      # Not Accepted
        # iaa.Emboss(alpha=(0, 1.0), strength=(0, 2.0)),      # Not Accepted
        iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
        iaa.ElasticTransformation(alpha=8.2, sigma=4.0),
        # iaa.ElasticTransformation(alpha=15.5, sigma=4.0),
        # iaa.ElasticTransformation(alpha=22.8, sigma=4.0),
        iaa.PiecewiseAffine(scale=0.015),
        iaa.PiecewiseAffine(scale=0.030),
        iaa.PiecewiseAffine(scale=0.045),
        # iaa.PiecewiseAffine(scale=0.060),
        # iaa.PiecewiseAffine(scale=0.075),
        iaa.EdgeDetect(alpha=0.3),
        # iaa.Sharpen(alpha=(0.0, 1.0)),      # Not Accepted
        # iaa.DirectedEdgeDetect(alpha=0.5, direction=0),      # Not Accepted
        iaa.Affine(scale=0.8, mode='edge', cval=64 ),
        iaa.Affine(scale=1.2, mode='edge'),
        iaa.Affine(rotate=5, cval=64 ),
        iaa.Affine(rotate=10, cval=64 ),
        iaa.Affine(rotate=-5, cval=64 ),
        iaa.Affine(rotate=-10, cval=64 ),
        iaa.Affine(shear=8, cval=64, mode='edge'),
        iaa.Affine(shear=-8, cval=64, mode='edge'),
        iaa.Affine(scale=0.8, rotate=3, mode='edge', cval=64 ),
        iaa.Affine(scale=0.8, rotate=-3, mode='edge', cval=64),
        iaa.Affine(scale=1.2, rotate=3, mode='edge', cval=64),
        iaa.Affine(scale=1.2, rotate=-3, mode='edge', cval=64),
        iaa.GaussianBlur(sigma=1.0),
        # iaa.MaxPooling(kernel_size=2, keep_size=True),
        iaa.Fog(),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(scale=0.8, mode='edge', cval=64 ),]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(scale=1.2, mode='edge'), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(rotate=5, cval=64 ), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(rotate=10, cval=64), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(rotate=-5, cval=64), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(rotate=-10, cval=64 ), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(shear=8, cval=64, mode='edge'), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(shear=-8, cval=64, mode='edge'), ]),

        # iaa.Sequential([iaa.MaxPooling(2, keep_size=True), iaa.Affine(scale=0.8, mode='edge', cval=64), ]),
        # iaa.Sequential([iaa.MaxPooling(2, keep_size=True), iaa.Affine(scale=1.2, mode='edge'), ]),
        # iaa.Sequential([iaa.MaxPooling(2, keep_size=True), iaa.Affine(rotate=5, cval=64), ]),
        # iaa.Sequential([iaa.MaxPooling(2, keep_size=True), iaa.Affine(rotate=10, cval=64), ]),
        # iaa.Sequential([iaa.MaxPooling(2, keep_size=True), iaa.Affine(rotate=-5, cval=64), ]),
        # iaa.Sequential([iaa.MaxPooling(2, keep_size=True), iaa.Affine(rotate=-10, cval=64), ]),
        # iaa.Sequential([iaa.MaxPooling(2, keep_size=True), iaa.Affine(shear=8, cval=64, mode='edge'), ]),
        # iaa.Sequential([iaa.MaxPooling(2, keep_size=True), iaa.Affine(shear=-8, cval=64, mode='edge'), ]),

        # iaa.Sequential([iaa.Fog(), iaa.Affine(scale=0.8, mode='edge', cval=64), ]),
        # iaa.Sequential([iaa.Fog(), iaa.Affine(scale=1.2, mode='edge'), ]),
        # iaa.Sequential([iaa.Fog(), iaa.Affine(rotate=5, cval=64), ]),
        # iaa.Sequential([iaa.Fog(), iaa.Affine(rotate=10, cval=64), ]),
        # iaa.Sequential([iaa.Fog(), iaa.Affine(rotate=-5, cval=64), ]),
        # iaa.Sequential([iaa.Fog(), iaa.Affine(rotate=-10, cval=64), ]),
        # iaa.Sequential([iaa.Fog(), iaa.Affine(shear=8, cval=64, mode='edge'), ]),
        # iaa.Sequential([iaa.Fog(), iaa.Affine(shear=-8, cval=64, mode='edge'), ]),


    ]
    list_aa_name = [
        # 'MedianBlur',
        # 'Dropout',
        # 'Emboss',
        'AdditiveGaussianNoise',
        'ElasticTransformation8',
        # 'ElasticTransformation15',
        # 'ElasticTransformation22',
        'PiecewiseAffine15',
        'PiecewiseAffine30',
        'PiecewiseAffine45',
        # 'PiecewiseAffine60',
        # 'PiecewiseAffine75',
        'EdgeDetect',
        # 'Sharpen',
        # 'DirectedEdgeDetect',
        'scale8',
        'scale12',
        'rotate5',
        'rotate10',
        'rotate_5',
        'rotate_10',
        'shear8',
        'shear_8',
        'scale8rotate3',
        'scale8rotate_3',
        'scale12rotate3',
        'scale12rotate_3',
        'GaussianBlur',
        # 'MaxPooling2',
        'Fog',

        'GaussianBlurscale8',
        'GaussianBlurscale12',
        'GaussianBlurrotate5',
        'GaussianBlurrotate10',
        'GaussianBlurrotate_5',
        'GaussianBlurrotate_10',
        'GaussianBlurshear8',
        'GaussianBlurshear_8',

        # 'MaxPooling2scale8',
        # 'MaxPooling2scale12',
        # 'MaxPooling2rotate5',
        # 'MaxPooling2rotate10',
        # 'MaxPooling2rotate_5',
        # 'MaxPooling2rotate_10',
        # 'MaxPooling2shear8',
        # 'MaxPooling2shear_8',

        # 'Fogscale8',
        # 'Fogscale12',
        # 'Fogrotate5',
        # 'Fogrotate10',
        # 'Fogrotate_5',
        # 'Fogrotate_10',
        # 'Fogshear8',
        # 'Fogshear_8',

    ]

    list_torch_tf = [
        MyRotationTransform(angles=[-10, -5, 0, 5, 10]),
    ]

    list_torch_tf_name = [
        'Rotate',
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

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(image)
        # for i in range(len(list_torch_tf)):
        #     augmented_image = list_aa[i](im_pil)
        #     im_np = np.asarray(augmented_image)
        #     im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        #     imagesave(im_np, jpg_out_basename + '_' + list_torch_tf_name[i] + '.jpg')


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

    # 7-segment digit에 대해 처리.
    makeImageFolder(r'D:\proj_gauge\민성기\digitGaugeSamples', r'.\digit_class_7seg', 'digit_7')  # digit_7, digit_normal, digit_all
    imageAugmentation(r'.\digit_class_7seg', r'.\digit_class_7seg_aug')


    # 1단계 . 아래 2가지 처리를 실행한다.
    # normal digit에 대해 처리.
    # makeImageFolder(r'D:\proj_gauge\민성기\digitGaugeSamples', r'.\digit_class_normal', 'digit_normal')  # digit_7, digit_normal, digit_all
    # imageAugmentation(r'.\digit_class_normal', r'.\digit_class_normal_aug')



    print('job done')

