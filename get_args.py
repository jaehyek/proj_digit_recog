import os
import sys
import argparse

def get_args():
    parser = argparse.ArgumentParser( description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter, )

    parser.add_argument('-o', '--output', default='output.json', help='output file')
    parser.add_argument('--dir_output', default=r'D:\proj_gauge\labelme_data\output', help='output file')

    parser.add_argument('--dir_digit_anno', default=r'D:\proj_gauge\labelme_data\annotation', help='annotation file')
    parser.add_argument('--dir_backimage', default=r'D:\proj_gauge\labelme_data\backimage', help='annotation file')
    parser.add_argument('--normal', default=0.7, help='normalize to 1. for heatmap location')

    parser.add_argument('--list_keypoint_rotate', default=[-15, -7, 7, 15], nargs='+', type=int , help='normalize to 1. for heatmap location')

    # parser.add_argument('--stride-apply', default=1, type=int,
    #                     help='apply and reset gradients every n batches')
    # parser.add_argument('--epochs', default=300, type=int,
    #                     help='number of epochs to train')
    # parser.add_argument('--freeze-base', default=0, type=int,
    #                     help='number of epochs to train with frozen base')
    # parser.add_argument('--pre-lr', type=float, default=1e-4,
    #                     help='pre learning rate')
    # parser.add_argument('--update-batchnorm-runningstatistics',
    #                     default=False, action='store_true',
    #                     help='update batch norm running statistics')
    # parser.add_argument('--size_heatvec', default=368, type=int, help='size of heatmap, vectmap')
    # parser.add_argument('--stride_heatvec', default=8, type=int, help='stride of heatmap, vectmap')
    # parser.add_argument('--train_scale', default=(5, 10), type=int, help='stride of heatmap, vectmap')
    # parser.add_argument('--num_pts_between_keypoints', default=10, type=int, help='the number of points between two keypoints ')
    #
    # parser.add_argument('--ema', default=1e-3, type=float,
    #                     help='ema decay constant')
    # parser.add_argument('--debug-without-plots', default=False, action='store_true',
    #                     help='enable debug but dont plot')
    # parser.add_argument('--disable-cuda', action='store_true',
    #                     help='disable CUDA')
    # parser.add_argument('--model_path', default=r'D:\proj_gauge\digit_paf\paf.pt', type=str, metavar='DIR', help='path to where the model saved')
    # parser.add_argument('--vgg19_path', default=r'D:\proj_gauge\digit_paf\vgg19.pt', type=str, metavar='DIR', help='path to where the model saved')
    #
    # parser.add_argument('--category_name', default='digit0', type=str, help='category name inside of annotation to process')
    #
    # parser.add_argument('--model_factor', default=8, type=int, help='image downsampler factor')
    # parser.add_argument('--heatmap_threshold', default=0.15, type=float, help='heatmap  threshold')
    # parser.add_argument('--pafmap_threshold', default=0.01, type=float, help='pafmap threshold')

    args = parser.parse_args()

    return args