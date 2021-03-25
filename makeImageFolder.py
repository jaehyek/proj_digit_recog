import sys
import os
from glob import glob
import codecs, json
from PIL import Image
import random
import shutil

def saveimage(sub,dir_digit, basename ):
    filename_jpg = os.path.join(dir_digit, basename + '.jpg')
    sub.save(filename_jpg)


def extractDigit_saveto(file_json, file_bmp, list_dir_digit):
    with codecs.open(file_json, 'r', encoding='utf-8') as f:
        dict_bmp_info = json.load(f)
        digitFractNo = int(dict_bmp_info['digitFractNo'])
        digitAllNo = int(dict_bmp_info['digitAllNo'])
        dataValue = int(dict_bmp_info['dataValue'] * 10 ** digitFractNo)
        digitRect = dict_bmp_info['digitRect']
        str_dataValue = f'{dataValue:0{digitAllNo}}'
        
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
            saveimage(sub,list_dir_digit[int(str_dataValue[index])], os.path.basename(file_json).split('.')[0] )



def makeImageFolder(folder_json, folder_digit):
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
        extractDigit_saveto(file_json, file_bmp, list_dir_digit)
        

def makeTrainValidFromDigitClass(dir_digit_class, dir_train, dir_valid, ratio=(8,2)):
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
        len_train, len_valid = ratio
        len_train =  int(len_train / ( len_train + len_valid) * len_src)
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
        

if __name__ == '__main__' :
    makeImageFolder(r'D:\proj_gauge\민성기\digitGaugeSamples', r'.\digit_class')
    makeTrainValidFromDigitClass(r'.\digit_class', r'.\digit_class_train', r'.\digit_class_valid', ratio=(8, 2))


