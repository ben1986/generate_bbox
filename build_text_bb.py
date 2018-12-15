from glob import glob
import cv2
import os
import numpy as np
import re
#import pyson.vision as pv
import numpy as np
#import pyson.utils as pu
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm
import json
REGEX = re.compile('.*(\d{4})-(.+)-(\d{1,4}).png')


        
        
def read_image(path, output_channels=3, resize_factor=1):
    if output_channels == 3:
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    else:
        img = cv2.imread(path, 0)
    
    if resize_factor !=1 :
        img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)
    return img

def get_text_boundingboxes(label_text, ex_border, resize_factor=1):
    ex_border = cv2.morphologyEx(ex_border, cv2.MORPH_CLOSE, np.ones([1, int(50*resize_factor)]))
    ex_border = cv2.morphologyEx(ex_border, cv2.MORPH_CLOSE, np.ones([int(10*resize_factor), 1]))

    dilating = cv2.morphologyEx(label_text, cv2.MORPH_CLOSE, np.ones([int(5*resize_factor), int(300*resize_factor)]))
    
    idxs = np.where(ex_border> 0) 
    dilating[idxs] = 0

    #cnts, hiers = pv.findContours(dilating)
    ret, thresh = cv2.threshold(dilating, 127, 255, 0)
    im2, contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    bb  = []
    mask = np.zeros_like(dilating)
    # Split by excell
    mask = np.zeros_like(dilating)
    #for cnt in cnts:
    for cnt in contours:
        x,y,w,h = cv2.boundingRect(cnt)
        pad = label_text[y:y+h, x:x+w]
        if pad.mean() > 0:
            idxs = np.vstack(np.where(pad==255))
            min_y = idxs[0].min()
            max_y = idxs[0].max()
            min_x = idxs[1].min()
            max_x = idxs[1].max()

            ay = y+min_y
            ax = x + min_x
            by = y+max_y
            bx = x + max_x
            a, b = (ax,ay), (bx, by)
            bb.append((a, b))
            cv2.rectangle(mask, a, b, 255, -1)
    # merge over cell
    mask_merge = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones([1, int(10*resize_factor)]))
    #cnts, _ = pv.findContours(mask_merge)
    ret, thresh1 = cv2.threshold(mask_merge, 127, 255, 0)
    im3, cnts, hierarchy1 = cv2.findContours(thresh1,cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
#     pv.show(mask_merge, dpi=300, size=10)
    return [cv2.boundingRect(cnt) for cnt in cnts]

def get_bb_from_dict_path(dict_path, resize_factor):
#     input = read_image(dict_path['input'], 3, resize_factor)
    label_exborder = read_image(dict_path['label-ex_border'], 1, resize_factor)
    label_text = read_image(dict_path['label-text'], 1, resize_factor)
    
    boxes = get_text_boundingboxes(label_text, label_exborder, resize_factor)
    
    for (x, y, w, h) in boxes:
        a = (x, y)
        b = (x+w, y+h)
    return boxes

def perform(paths, multithread=4):
    input_paths = [path for path in paths if '-input-' in path]
    trainval_set = []
    
    for input_path in input_paths:
        mo = REGEX.search(input_path)
        m1 = mo.group(1)
        m3 = mo.group(3)
        d = {
             'input': input_path,
             #'label-image': input_path.replace('-input-{}'.format(m3), '-image'),
             'label-ex_border': input_path.replace('-input-{}'.format(m3), '-ex_border'),
              'label-text': input_path.replace('-input-', '-text-'),
            }
        
        trainval_set.append(d)
    
        for path in d.values(): assert(os.path.exists(path)), path
    
#     for dict_path in tqdm(trainval_set):
    def fn(dict_path):
        input_path = dict_path['input']
        input_image = read_image(input_path, 1, 1)
        mask = np.zeros_like(input_image)
        boxes = get_bb_from_dict_path(dict_path, 1)
        for (x,y,w,h) in boxes:
            cv2.rectangle(mask, (x, y), (x+w, y+h), 255, 2)


        #### Huynh modify ######
        i = 0
        boxes_mod = {}
        for (x,y,w,h) in boxes:
            #boxes_mod.append((x,y,x+w, y+h))            
            boxes_mod[str(i)] = {'x': x, 'width': w, 'y':y, 'height':h}
            i = i+1

        file_name_ext = input_path.split('/')[-1]
        file_name = file_name_ext.split('.')[0] + '.json'
        file_path = os.path.dirname(input_path)
        json_fullpath = os.path.join(file_path,file_name)
        with open(json_fullpath,'w') as outfile:
            json.dump(boxes_mod,outfile)
        ########################
        #out_path_text_box = input_path.replace('-input-', '-textbox-')
        #cv2.imwrite(out_path_text_box, mask)
        #np.save(out_path_text_box.replace('.png', '.npy'), np.array(boxes))
        #return out_path_text_box
        
    if multithread>1:
        with tqdm(total=len(trainval_set), desc="Executing Pipeline", unit=" Samples") as progress_bar:
            with ThreadPoolExecutor(max_workers=multithread) as executor:
                for result in executor.map(fn, trainval_set):
                    progress_bar.set_description("Processing %s" % result)
                    progress_bar.update(1)
    else:
        with tqdm(total=len(trainval_set), desc="Executing Pipeline", unit=" Samples") as progress_bar:
            for dict_path in trainval_set:
                result =fn(dict_path)
                progress_bar.set_description("Processing %s" % result)
                progress_bar.update(1)