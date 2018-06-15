#-*- coding:utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import sys
import time
import os
import glob
def crop_margin(image , resize ):
    """
    주변의 검정색을 지워 버립니다.
    :param path:
    :return:
    """

    """
    file name =1002959_20130627_L.png
    """
    start_time = time.time()
    im = image
    np_img = np.asarray(im)
    mean_pix = np.mean(np_img)
    pix = im.load()
    height, width = im.size  # Get the width and hight of the image for iterating over
    # pix[1000,1000] #Get the RGBA Value of the a pixel of an image
    c_x, c_y = (int(height / 2), int(width / 2))

    for y in range(c_y):
        if sum(pix[c_x, y]) > mean_pix:
            left = (c_x, y)
            break;

    for x in range(c_x):
        if sum(pix[x, c_y]) > mean_pix:
            up = (x, c_y)
            break;

    crop_img = im.crop((up[0], left[1], left[0], up[1]))

    #plt.imshow(crop_img)

    diameter_height = up[1] - left[1]
    diameter_width = left[0] - up[0]

    crop_img = im.crop((up[0], left[1], left[0] + diameter_width, up[1] + diameter_height))
    if not resize is None:
        crop_img.resize(resize , Image.ANTIALIAS)
    end_time = time.time()
    return crop_img




for i in range(2,5):
    paths=glob.glob(os.path.join('/Users/seongjungkim/Downloads/train_{}/train'.format(i) , '*'))
    print '# paths : {}'.format(len(paths))
    # save folder
    root_dir ='/Users/seongjungkim/Downloads/kaggle_540'


    # Margin Crop and Resize
    for i,path in enumerate(paths):
        sys.stdout.write('\r Progress {} / {}'.format(i , len(paths)))
        try:
            sys.stdout.flush()
            img = Image.open(path)
            img = crop_margin(img ,(540,540))

            name = os.path.split(path)[-1]
            pat_code = name.split('_')[0]

            # Make save folder
            savedir = os.path.join(root_dir , pat_code)
            if not os.path.isdir(savedir):
                os.makedirs(savedir)
            savepath = os.path.join(savedir , name)
            # Image Save
            img.save(savepath)

        except Exception as e :
            print e ,'\t' , path


