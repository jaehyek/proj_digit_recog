import sys
import os
from glob import glob
import codecs, json
import shutil
import cv2
import numpy as np
import albumentations as A
import random
from imgaug import augmenters as iaa
import imgaug as ia
import torchvision.transforms.functional as TF


from PIL import Image


def saveimage(sub, dir_digit, basename):
    filename_jpg = os.path.join(dir_digit, basename + '.jpg')
    sub.save(filename_jpg)


def extractDigit_saveto(file_json, file_bmp, list_dir_digit=None):
    with codecs.open(file_json, 'r', encoding='utf-8') as f:
        dict_bmp_info = json.load(f)
        digitFractNo = int(dict_bmp_info['digitFractNo'])
        digitAllNo = int(dict_bmp_info['digitAllNo'])
        dataValue = int(dict_bmp_info['dataValue'] * 10 ** digitFractNo)
        digitRect = dict_bmp_info['digitRect']
        str_dataValue = f'{dataValue:0{digitAllNo}}'
        str_igmsGaugeDataId = dict_bmp_info['igmsGaugeDataId']

        if len(str_dataValue) != digitAllNo:
            if len(str_igmsGaugeDataId) == digitAllNo:
                str_dataValue = str_igmsGaugeDataId
            else:
                print(f'{file_json}')
                raise Exception("improper data format")


        list_digitRect = digitRect.split('|')[1:]
        list_digitRect = [aa.split(',') for aa in list_digitRect]
        list_digitRect = [[int(a), int(b), int(c), int(d)] for a, b, c, d in list_digitRect]

        img = Image.open(file_bmp)
        if img == None:
            print(f"Can't read a image file :{file_bmp}")
            return

        list_image = []
        for index in range(digitAllNo):
            x, y, width, height = list_digitRect[index]
            sub = img.crop((x, y, x + width, y + height))
            if list_dir_digit != None :
                saveimage(sub, list_dir_digit[int(str_dataValue[index])], os.path.basename(file_json).split('.')[0] + f'_{index}{str_dataValue[index]}')
            else:
                list_image.append(sub)

    if list_dir_digit == None :
        return list_image, [int(aa) for aa in str_dataValue], dict_bmp_info


def makeImageFolder(folder_json, folder_digit):

    os.makedirs(folder_digit, exist_ok=True)

    list_dir_digit = [os.path.join(folder_digit, str(num)) for num in range(10) ]
    aa = [os.makedirs(dir_t, exist_ok=True) for dir_t in list_dir_digit ]

    list_json = glob(folder_json + "/*.json")
    print(f'len(list_json) is {len(list_json)}')

    for file_json in list_json:
        file_bmp = os.path.splitext(file_json)[0] + '.bmp'
        extractDigit_saveto(file_json, file_bmp, list_dir_digit)


def makeTrainValidFromDigitClass(dir_digit_class, dir_train, dir_valid, train_ratio=0.8):
    dir_digitname = [str(aa) for aa in range(10)]

    # make dir_train, dir_test
    list_dir = [os.path.join(aa, bb) for aa in [dir_train, dir_valid] for bb in dir_digitname]
    temp = [os.makedirs(kk, exist_ok=True) for kk in list_dir]

    for dir_digit in dir_digitname:
        dir_src = os.path.join(dir_digit_class, dir_digit)
        dir_train_dest = os.path.join(dir_train, dir_digit)
        dir_valid_dest = os.path.join(dir_valid, dir_digit)

        list_src = glob(dir_src + r"/*.jpg")
        len_src = len(list_src)
        len_train = int(train_ratio * len_src)
        len_valid = len_src - len_train
        list_src_index = list(range(len_src))
        list_train_index = random.sample(list_src_index, len_train)
        list_valid_index = list(set(list_src_index) - set(list_train_index))

        # copy files to train
        for file_index in list_train_index:
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


def imageAugmentation(dir_in, dir_out, copy_json = True ):
    random.seed(42)
    os.makedirs(dir_out, exist_ok=True)


    list_jpg = glob(dir_in + r"/**/*.jpg", recursive=True)
    print(f'len(list_jpg) is {len(list_jpg)}')

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
        iaa.Affine(scale=0.8, mode='edge', cval=64),
        iaa.Affine(scale=1.2, mode='edge'),
        iaa.Affine(rotate=5, cval=64),
        iaa.Affine(rotate=10, cval=64),
        iaa.Affine(rotate=-5, cval=64),
        iaa.Affine(rotate=-10, cval=64),
        iaa.Affine(shear=8, cval=64, mode='edge'),
        iaa.Affine(shear=-8, cval=64, mode='edge'),
        iaa.Affine(scale=0.8, rotate=3, mode='edge', cval=64),
        iaa.Affine(scale=0.8, rotate=-3, mode='edge', cval=64),
        iaa.Affine(scale=1.2, rotate=3, mode='edge', cval=64),
        iaa.Affine(scale=1.2, rotate=-3, mode='edge', cval=64),
        iaa.GaussianBlur(sigma=1.0),
        # iaa.MaxPooling(kernel_size=2, keep_size=True),
        # iaa.Fog(seed=1),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(scale=0.8, mode='edge', cval=64), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(scale=1.2, mode='edge'), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(rotate=5, cval=64), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(rotate=10, cval=64), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(rotate=-5, cval=64), ]),
        iaa.Sequential([iaa.GaussianBlur(sigma=1.0), iaa.Affine(rotate=-10, cval=64), ]),
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
        # 'Fog',

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

    for jpg_in in list_jpg:
        # print('.', end='')
        jpg_out = jpg_in.replace(dir_in, dir_out)
        dir_jpg_out = os.path.dirname(jpg_out)

        os.makedirs(dir_jpg_out, exist_ok=True)

        jpg_out_basename = os.path.splitext(jpg_out)[0]
        jpg_in_json = os.path.splitext(jpg_in)[0] + '.json'
        if copy_json != True or not os.path.exists(jpg_in_json) :
            jpg_in_json = None
        image = imageread(jpg_in)
        imagesave(image, jpg_out_basename + '.jpg')  # save the orignal image.
        if jpg_in_json != None:
            shutil.copy(jpg_in_json, jpg_out_basename + '.json')

        for i in range(len(list_aug)):
            augmented_image = list_aug[i](image=image)['image']
            imagesave(augmented_image, jpg_out_basename + '_' + list_aug_name[i] + '.jpg')
            if jpg_in_json != None :
                shutil.copy(jpg_in_json, jpg_out_basename + '_' + list_aug_name[i] + '.json')

        for i in range(len(list_aa)):
            augmented_image = list_aa[i](image=image)
            imagesave(augmented_image, jpg_out_basename + '_' + list_aa_name[i] + '.jpg')
            if jpg_in_json != None :
                shutil.copy(jpg_in_json, jpg_out_basename + '_' + list_aa_name[i] + '.json')

        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        # im_pil = Image.fromarray(image)
        # for i in range(len(list_torch_tf)):
        #     augmented_image = list_aa[i](im_pil)
        #     im_np = np.asarray(augmented_image)
        #     im_np = cv2.cvtColor(im_np, cv2.COLOR_RGB2BGR)
        #     imagesave(im_np, jpg_out_basename + '_' + list_torch_tf_name[i] + '.jpg')


def get_Image_Value_List_from_json(file_json):
    list_image, list_value, dict_json_info = extractDigit_saveto(file_json, os.path.splitext(file_json)[0] + '.bmp')
    return list_image, list_value, dict_json_info


if __name__ == '__main__':
    # json, jpg ????????? ?????? dir??? ????????????,   ????????? ????????? dir??????  0 ~9 ?????? dir??? ????????? ?????? ?????? ???????????????  jpg????????? ????????????.,

    # 7-segment digit??? ?????? ??????.
    # makeImageFolder(r'D:\proj_gauge\?????????\digitGaugeSamples2', r'.\digit_class_seg7', 'digit_7')  # digit_7, digit_normal, digit_all
    # imageAugmentation(r'.\digit_class_seg7', r'.\digit_class_seg7_aug')

    # 1?????? . ?????? 2?????? ????????? ????????????.
    # normal digit??? ?????? ??????.
    # makeImageFolder(r'D:\proj_gauge\?????????\digitGaugeSamples', r'.\digit_class_normal', 'digit_normal')  # digit_7, digit_normal, digit_all
    # imageAugmentation(r'.\digit_class_normal', r'.\digit_class_normal_aug')

    # makeImageFolder(r'D:\proj_gauge\?????????\digitGaugeSamples3', r'.\digit_class_samp3', 'all')  # digit_7, digit_normal, digit_all
    # imageAugmentation(r'.\digit_class_normal', r'.\digit_class_normal_aug')

    # imageAugmentation(r'.\digit_class_samp2', r'.\digit_class_samp2_aug')
    # imageAugmentation(r'.\digit_class_samp3', r'.\digit_class_samp3_aug')

    # makeImageFolder(r'D:\proj_gauge\?????????\digitGaugeSamples', r'./digit_class_sam')
    makeImageFolder(r'D:\proj_gauge\?????????\digitGaugeSamples2', r'./digit_class_sam2')
    # makeImageFolder(r'D:\proj_gauge\?????????\digitGaugeSamples3', r'./digit_class_sam')

    # imageAugmentation(r'D:\proj_gauge\digit_paf_data\digit_class_normal\images', r'D:\proj_gauge\digit_paf_data\digit_class_normal_aug\images', copy_json = True )
    # makeTrainValidFromDigitClass(r'./digit_class_aug', r'./digit_class_aug_train', r'./digit_class_aug_val', 0.8)

    print('job done')

