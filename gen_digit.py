from get_args import get_args
import utils
from glob import  glob
import numpy as np
import os
import cv2
import math
from math import cos, sin
import albumentations as A
from imgaug import augmenters as iaa

'''
0 - 11 - 10
|        |
1        9
|        |
2 - 12 - 8
|        |
3        7
|        |
4 - 5 -  6
'''
NO_keypoint = 13

list_keypoint_digit0 = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11]
list_keypoint_digit1 = [ 6, 7, 8, 9, 10]
list_keypoint_digit2 = [ 0, 11, 10, 9, 8, 12, 2, 3, 4, 5, 6]
list_keypoint_digit3 = [ 0, 11, 10, 9, 8, 12, 2, 7, 6, 5, 4]
list_keypoint_digit4 = [ 0, 1, 2, 12, 10, 9, 8, 7, 6]
list_keypoint_digit5 = [ 10, 11, 0, 1, 2, 12, 8, 7, 6, 5, 4]
list_keypoint_digit6 = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 12]
list_keypoint_digit7 = [ 0, 11, 10, 9, 8, 7, 6]
list_keypoint_digit8 = [ 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
list_keypoint_digit9 = [ 12, 2, 1, 0, 11, 10, 9, 8, 7, 6]

list_limb_digit0 = [(0,1), (1,2),(2,3), (3,4), (4,5), (5,6), (6,7),(7,8), (8,9),(9,10),(10,11), (0,11)]
list_limb_digit1 = [(6,7), (7,8),(8,9), (9,10)]
list_limb_digit2 = [(0,11), (11,10),(10,9), (9,8),(8,12), (12,2),(2,3),(3,4),(4,5),(5,6)]
list_limb_digit3 = [(0,11), (11,10),(10,9), (9,8),(8,7), (7,6),(6,5),(5,4), (2,12),(12,8)]
list_limb_digit4 = [(0,1),(1,2),(2,12),(12,8),(6,7), (7,8),(8,9), (9,10)]
list_limb_digit5 = [(10,11),(11,0),(0,1),(1,2),(2,12),(12,8),(8,7),(7,6),(6,5),(5,4)]
list_limb_digit6 = [(0,1), (1,2),(2,3), (3,4), (4,5), (5,6), (6,7),(7,8), (8,12),(12,2)]
list_limb_digit7 = [(0,11), (11,10),(10,9), (9,8),(8,7),(7,6)]
list_limb_digit8 = [(0,1), (1,2),(2,3), (3,4), (4,5), (5,6), (6,7),(7,8), (8,9),(9,10),(10,11), (0,11), (2,12),(12,8)]
list_limb_digit9 = [(12,8),(2,12),(1,2),(0,1),(0,11),(11,10),(10,9),(9,8),(8,7),(7,6)]


def read_keypoint(args):
    # 여기서는 annotation file에서  annot을 읽어서, list_mp_keypoint에 저장한다.

    list_annofile =  glob(args.dir_digit_anno + '/*.json')
    list_label_np_keypoint = []
    for annofile in list_annofile :
        dict_digit_anno = utils.read_dict_from_file(annofile)
        list_shape_info = dict_digit_anno['shapes']

        np_keypoint = np.ones((NO_keypoint, 2 )) * -1
        label = int(list_shape_info[0]['label'])
        list_coord = np.array([ list_xy for shape_info  in list_shape_info for list_xy in shape_info['points']] )

        if list_coord.shape[0] != len(args.list_list_keypoint_digit[label]) :
            raise Exception("Wrong  keypoint count")
        for index, index_list in enumerate(args.list_list_keypoint_digit[label] ) :
            np_keypoint[index_list] = list_coord[index]

        list_label_np_keypoint.append([label, np_keypoint])

    args.list_label_np_keypoint = list_label_np_keypoint

def normalize_keypoint(args):
    # 여기서는 좌표값들을 0~1 사이의 값으로 변경한다.
    if not hasattr(args, 'list_label_np_keypoint') :
        raise Exception('No list_np_keypoint in args ')
    if not hasattr(args, 'normal') :
        raise Exception('No normal in args ')

    for index, (label, np_keypoint) in enumerate(args.list_label_np_keypoint) :
        np_max = np.max(np_keypoint, axis=0)
        np_temp = np_keypoint.copy()
        np_temp[np_temp == -1 ] = 1000000
        np_min = np.min(np_temp, axis=0)

        if label == 1 :                 # 1은 폭이 좁아서, normalize할 경우 이상하게 나온다. 다른 숫자처럼  width part에 보강이 필요하다.
            np_temp = np_temp[ np_temp[:,0] !=1000000]      # 유효한 숫자만 모은다.
            x_center = np.mean(np_temp, axis=0)[0]       # x center가  x 전체 길이에서  2/3 지점에 존재한다고 가정.
            np_min[0] = x_center - x_center * 2 / 3           # x 폭의 min은  x center에서   diff * 2/3 앞에 있고
            np_max[0] = x_center + x_center / 3               # x 폭의 max은  x center에서  diff / 3 뒤에 있다고 가정.

        np_diff = np_max - np_min
        np_keypoint -= np_min
        np_keypoint *= (args.normal / np_diff )      # 최대 크기를 80%으로 조정.
        np_keypoint += (1. - args.normal) /2                     # 10% <= keypoint < 90% 으로 이동.

        np_keypoint[np_keypoint < 0 ] = -1
        args.list_label_np_keypoint[index][1] = np_keypoint

def rotate_keypoint(args):
    if not hasattr(args, 'list_label_np_keypoint') :
        raise Exception('No list_label_np_keypoint in args ')

    if not hasattr(args, 'list_keypoint_rotate') :
        raise Exception('No list_keypoint_rotate in args ')

    def move_to_origin(l, org):
        x0, y0 = org
        return [(x - x0, y - y0) for x, y in l]

    def rotate_to_x_axis(l, theta):
        return [(x * cos(theta) - y * sin(theta), x * sin(theta) + y * cos(theta)) for x, y in l]

    list_angled_keypoint = []
    for label, np_keypoint in args.list_label_np_keypoint :
        for keypoint_rotate in args.list_keypoint_rotate :
            np_temp = np_keypoint.copy()
            np_temp[np_temp == -1] = 1000
            angle_radian = math.radians(keypoint_rotate)
            np_temp = move_to_origin(np_temp, (0.5, 0.5))
            np_temp = rotate_to_x_axis(np_temp, angle_radian)
            np_temp = np.array(move_to_origin(np_temp, (-0.5, -0.5)))
            np_temp[abs(np_temp) > 2 ] = -1
            list_angled_keypoint.append([label, np_temp])

    args.list_label_np_keypoint +=  list_angled_keypoint

def draw_limbs_on_image(image, np_keypoint, list_limb_digit, font_color, font_thick, keypoint_unit=True):

    image_wh = image.shape[1::-1]
    if keypoint_unit == True:
        np_keypoint *= image_wh
        thick = int(round(image_wh[0] / 20  * font_thick))
    else:
        thick = font_thick

    for (start,end) in list_limb_digit :
        image = cv2.line(image, tuple(np_keypoint[start].round().astype(int)), tuple(np_keypoint[end].round().astype(int)), font_color, thick)

    np_keypoint[np_keypoint < 0 ] = -1
    return image, np_keypoint

# https://github.com/aleju/imgaug

list_aug_geometric = [
    iaa.Identity(),
    iaa.ElasticTransformation(alpha=8.2, sigma=4.0),
    iaa.ElasticTransformation(alpha=15.0, sigma=4.0),
    iaa.Affine(shear=8, cval=64, mode='edge'),
    iaa.Affine(shear=-8, cval=64, mode='edge'),
    iaa.PiecewiseAffine(scale=0.01),
    iaa.PiecewiseAffine(scale=0.05),            # mosaic 변형과 비슷

]

list_aug_non_geo = [
    iaa.Identity(),
    iaa.AdditiveGaussianNoise(loc=0, scale=(0.0, 0.05 * 255), per_channel=0.5),
    iaa.GaussianBlur(sigma=1.0),
    iaa.Dropout(p=0.05),
    iaa.CoarseDropout(0.02, size_percent=0.5),
    iaa.SaltAndPepper(0.1),
    iaa.JpegCompression(compression=10),
    iaa.Cartoon(blur_ksize=1, segmentation_size=0.5,saturation=2.0, edge_prevalence=1.0),
    iaa.MotionBlur(k=5),
    iaa.MultiplyHue(0.),
    iaa.MultiplyHue(1.),
    iaa.ChangeColorTemperature(1000),
    iaa.ChangeColorTemperature(4000),
    iaa.Snowflakes(flake_size=0.1, speed=0.01 ),
    iaa.Rain(speed=0.05, drop_size=0.001),

]


def generate_digits(args):
    if not hasattr(args, 'list_label_np_keypoint') :
        raise Exception('No list_np_keypoint in args ')

    if not hasattr(args, 'dir_backimage') :
        raise Exception('No dir_backimage in args ')

    if not hasattr(args, 'dir_output') :
        raise Exception('No dir_output in args ')

    if not hasattr(args, 'list_list_limb_digit') :
        raise Exception('No list_list_limb_digit in args ')



    for index_keypoint, (label, np_keypoint) in enumerate(args.list_label_np_keypoint) :
        if label != 4  or index_keypoint != 0 :
            continue
        dir_out = os.path.join(args.dir_output, str(label))
        os.makedirs(dir_out, exist_ok=True)
        print(f'index_keypoint is {index_keypoint}/{len(args.list_label_np_keypoint)}')

        for index_c, font_color in enumerate(args.list_font_color) :
            for font_thick in args.list_font_thick :
                for index_back, backimage_cv in enumerate( args.list_backimage_cv ):
                    image_cv = backimage_cv.copy()
                    image_cv, np_keypoint_draw = draw_limbs_on_image(image_cv, np_keypoint.copy(), args.list_list_limb_digit[label],
                                                   font_color, font_thick )
                    # utils.show_cvimage(image_cv)
                    utils.write_opencv_file(image_cv,
                        os.path.join(dir_out, f'img{index_keypoint:02d}b{index_back:02d}c{index_c:02d}t{font_thick:02d}.jpg') )

                    np_keypoint_temp = np_keypoint_draw[ np_keypoint_draw[:,1] >= 0]
                    for index_aug, aug_geometric in enumerate( args.list_aug_geometric ) :

                        image_geo, np_keypoint_geo = aug_geometric(image=image_cv, keypoints=[np_keypoint_temp])

                        if np.any(np_keypoint_temp != np_keypoint_geo[0]) :
                            print(f'keypoints is different : geo {index_aug}')

                        for index_non, aug_non_geo in enumerate( args.list_aug_non_geo ) :
                            image_no_geo, np_keypoint_non_geo = aug_non_geo(image=image_geo, keypoints=np_keypoint_geo)

                            if np.any(np_keypoint_geo[0] != np_keypoint_non_geo[0]):
                                raise Exception(f'keypoints is different : non-geo {index_aug}')

                            # 아래는 변화된 np_keypoint_geo 을 확인한다.
                            # np_keypoint_draw_test = np_keypoint_draw.copy()
                            # np_keypoint_draw_test[np_keypoint_draw_test[:, 1] >= 0] = np_keypoint_non_geo[0]
                            #
                            # image_no_geo, _ = draw_limbs_on_image(image_no_geo, np_keypoint_draw_test, args.list_list_limb_digit[label],
                            #                                                  (0,0,255), 1, keypoint_unit=False)

                            utils.write_opencv_file(image_no_geo,
                                    os.path.join(dir_out,
                                    f'img{index_keypoint:02d}b{index_back:02d}c{index_c:02d}t{font_thick:02d}g{index_aug:02d}n{index_non:02d}.jpg'))


def gen_digit_ready():
    args = get_args()
    list_list_keypoint_digit = [ list_keypoint_digit0, list_keypoint_digit1, list_keypoint_digit2, list_keypoint_digit3, list_keypoint_digit4,
                           list_keypoint_digit5, list_keypoint_digit6, list_keypoint_digit7, list_keypoint_digit8, list_keypoint_digit9]
    list_list_limb_digit = [ list_limb_digit0, list_limb_digit1, list_limb_digit2, list_limb_digit3, list_limb_digit4,
                        list_limb_digit5, list_limb_digit6, list_limb_digit7, list_limb_digit8, list_limb_digit9]

    args.list_list_keypoint_digit = list_list_keypoint_digit
    args.list_list_limb_digit = list_list_limb_digit

    read_keypoint(args)
    normalize_keypoint(args)
    rotate_keypoint(args)

    list_backimage_cv = [utils.read_opencv_file(aa) for aa in glob(args.dir_backimage + r'/*.jpg')]

    list_font_color = [(255, 255, 255), (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255), (0, 255, 255),
                       (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0), (128, 0, 128), (0, 128, 128), (128, 128, 128)]

    list_font_thick = [1, 2, 3, 4, 5]

    args.list_backimage_cv = list_backimage_cv
    args.list_font_color = list_font_color
    args.list_font_thick = list_font_thick
    args.list_aug_geometric = list_aug_geometric
    args.list_aug_non_geo = list_aug_non_geo

    args.len_list_label_np_keypoint = len(args.list_label_np_keypoint)
    args.len_list_font_color = len(args.list_font_color)
    args.len_list_font_thick = len(args.list_font_thick)
    args.len_list_backimage_cv = len(args.list_backimage_cv)
    args.len_list_aug_geometric = len(args.list_aug_geometric)
    args.len_list_aug_non_geo = len(args.list_aug_non_geo)
    args.len_total = args.len_list_label_np_keypoint * args.len_list_font_color * args.len_list_font_thick * \
                   args.len_list_backimage_cv * args.len_list_aug_geometric * args.len_list_aug_non_geo

    args.div_np_keypoint = args.len_list_font_color * args.len_list_font_thick * args.len_list_backimage_cv * \
                           args.len_list_aug_geometric * args.len_list_aug_non_geo
    args.div_font_color = args.len_list_font_thick * args.len_list_backimage_cv * \
                           args.len_list_aug_geometric * args.len_list_aug_non_geo
    args.div_font_thick = args.len_list_backimage_cv * args.len_list_aug_geometric * args.len_list_aug_non_geo
    args.div_backimage = args.len_list_aug_geometric * args.len_list_aug_non_geo
    args.div_aug_geo = args.len_list_aug_non_geo

    return args

def generate_digits_by_index(args, index ) :
    if not hasattr(args, 'list_label_np_keypoint'):
        raise Exception('No list_np_keypoint in args ')

    if not hasattr(args, 'dir_backimage'):
        raise Exception('No dir_backimage in args ')

    if not hasattr(args, 'dir_output'):
        raise Exception('No dir_output in args ')

    if not hasattr(args, 'list_list_limb_digit'):
        raise Exception('No list_list_limb_digit in args ')

    if index >= args.len_total:
        raise Exception('index is not  in args.len_total ')

    ind = index // args.div_font_color
    mod = index % args.div_font_color

    font_color = args.list_font_color[ind]

    ind = mod // args.div_font_thick
    mod = mod % args.div_font_thick

    font_thick = args.list_font_thick[ind]

    ind = mod // args.div_backimage
    mod = mod % args.div_backimage

    backimage = args.list_backimage_cv[ind]

    ind = mod // args.div_aug_geo
    mod = mod % args.div_aug_geo

    aug_geometric = args.list_aug_geometric[ind]

    ind = mod // args.div_aug_non_geo
    mod = mod % args.div_aug_non_geo

    aug_non_geo = args.list_aug_non_geo[ind]
    (label, np_keypoint) = args.list_label_np_keypoint[mod]

    image_cv = backimage.copy()
    image_cv, np_keypoint_draw = draw_limbs_on_image(image_cv, np_keypoint.copy(), args.list_list_limb_digit[label], font_color, font_thick)

    np_keypoint_temp = np_keypoint_draw[np_keypoint_draw[:, 1] >= 0]
    image_geo, np_keypoint_geo = aug_geometric(image=image_cv, keypoints=[np_keypoint_temp])
    image_no_geo, np_keypoint_non_geo = aug_non_geo(image=image_geo, keypoints=np_keypoint_geo)

    np_keypoint_draw_test = np_keypoint_draw.copy()
    np_keypoint_draw_test[np_keypoint_draw_test[:, 1] >= 0] = np_keypoint_non_geo[0]

    return image_no_geo, label, np_keypoint_draw_test

if __name__ == '__main__' :
    args = gen_digit_ready()
    # generate_digits(args)
    image, label, np_keypoint = generate_digits_by_index(args, args.div_np_keypoint * 5 + 100)
    utils.show_cvimage(image)

    print(f'done')