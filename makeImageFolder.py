import sys
import os
from glob import glob
import codecs, json
from PIL import Image


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
        if '10061-56397.json' in file_json:
            print('asdf')
        extractDigit_saveto(file_json, file_bmp, list_dir_digit)
        

if __name__ == '__main__' :
    # img = Image.open(r'D:\proj_gauge\민성기\digitGaugeSamples\10050-56265.bmp')
    # cropped_img = img.crop((100,100,200,200))
    # cropped_img.show()
    # cropped_img.save('bb.jpg')
    # num_img = np.array(img)
    # num_img = num_img[:, :, None] * np.ones(3, dtype=int)[None, None, :]

    # pil_image = Image.fromarray(num_img)
    # pil_image.show()


    makeImageFolder(r'D:\proj_gauge\민성기\digitGaugeSamples', r'.\digit_class')


