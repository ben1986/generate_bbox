### This code take input images and output bounding box mask
# Parameters: --input_dir /home/vu/Downloads/output1/pdf --output_dir /home/vu/Downloads/output1/mask --mode many

import pdf2image
import cv2
import numpy as np
import numpy as np
import json
from glob import glob
import cv2
import matplotlib.pyplot as plt
import os
import re
from build_text_bb import perform
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
from tqdm import tqdm

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

    cnts, hiers = pv.findContours(dilating)
    bb  = []
    mask = np.zeros_like(dilating)
    # Split by excell
    mask = np.zeros_like(dilating)
    for cnt in cnts:
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
    mask_merge = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones([1, int(50*resize_factor)]))
    cnts, _ = pv.findContours(mask_merge)
    return [cv2.boundingRect(cnt) for cnt in cnts]

def read_pair(dict_path, resize_factor):
    input = read_image(dict_path['input'], 3, resize_factor)
    label_exborder = read_image(dict_path['label-ex_border'], 1, resize_factor)
    label_text = read_image(dict_path['label-text'], 1, resize_factor)
    
    boxes = get_text_boundingboxes(label_text, label_exborder, resize_factor)
    
    for (x, y, w, h) in boxes:
        a = (x, y)
        b = (x+w, y+h)
        cv2.rectangle(input, a, b, (100, 0, 100), 3)
    
#     label = np.stack([label_exborder, label_text, label_text], axis=-1)
    return input#label#np.hstack([input, label])

def is_match(path, str_regex):
    _, name = os.path.split(path)
    r = re.compile(str_regex)
    mo = r.match(name)
    return mo is not None


def load_img(pdf_path):
    img = np.array(pdf2image.convert_from_path(pdf_path, dpi=500)[0])
    return img[...,::-1]

class ExtractInfo:
    def __init__(self, path, nopath, label, name, output_directory, image_mask=None, i=None):
        self.path = path
        self.nopath = nopath
        self.name = name
        self.label = label
        self.output_directory = output_directory
        self.image_mask = image_mask
        self.i = i

class Extract():
    def __init__(self):
        pass

    def extract(self, info):
        """
            args:
                path: path to contain
                nopath: path to background
            return: a mask of size [h,w]
        """
        img = load_img(info.path)
        noimg = load_img(info.nopath)
        x = noimg - img
        x = (x != (0, 0, 0)) * 255
        x = x.astype('uint8')
        return x[..., 0]

    def __call__(self, info):
        if info.image_mask is None:
            output_path = os.path.join(info.output_directory, '{}-{}.png'.format(info.name, info.label))
            if os.path.exists(output_path):
                return output_path
            mask = self.extract(info)
            cv2.imwrite(output_path, mask)
            return output_path
        else:
            output_path = os.path.join(info.output_directory, '{}-{}-{}.png'.format(info.name, info.label, info.i))
            if os.path.exists(output_path) and os.path.exists(output_path.replace('text-', 'input-')):
                return
            text_image = load_img(info.path)
            input_image = text_image + info.image_mask
            mask = self.extract(info)
            cv2.imwrite(output_path, mask)
            cv2.imwrite(output_path.replace('text-', 'input-'), input_image) 
            return output_path

def plot(img):
    img = np.squeeze(img)
    plt.figure(dpi=300, figsize=10)
    plt.imshow(img)
    plt.show()


def create_data_object(pdf_directory, mask_directory, multithread):
    """
        args:
            directory: path to pdfs
        return:
            np.array [w,h,size]
    """
    # create output directory inside the mask directory
    name = os.path.split(pdf_directory)[-1]
    output_directory = mask_directory
    os.makedirs(output_directory, exist_ok=True)
    
    texts_path = []
    paths = glob(os.path.join(pdf_directory, '*.pdf'))
    print(paths)
    # Scan all image and find image type
    for path in paths:
        if is_match(path, r'text_\d*_.*.pdf'):
            texts_path.append(path)
            
        if is_match(path, 'notext.*.pdf'):
            no_text_path = path
        if is_match(path, r'^noborder.*.pdf'):
            noborder_path = path
        if is_match(path, r'^border'):
            border_path = path
        if is_match(path, r'^ex_noborder'):
            ex_noborder_path = path
        if is_match(path, r'^ex_border'):
            ex_border_path = path
        if is_match(path, r'^noimage'):
            no_image_path = path
        if is_match(path, r'^image'):
            image_path = path
            
    assert noborder_path is not None, paths

    infors = []
    #obj_extract_noboder = ExtractInfo(border_path, noborder_path, "noborder", name, output_directory)
    #infors.append(obj_extract_noboder)

    obj_extract_ex_border = ExtractInfo(ex_border_path, ex_noborder_path, "ex_border", name, output_directory)
    infors.append(obj_extract_ex_border)

    #obj_extract_image = ExtractInfo(image_path, no_image_path, "image", name, output_directory)
    #infors.append(obj_extract_image)

    image = load_img(image_path) - load_img(no_image_path)

    for i, text_path in enumerate(texts_path):
        obj_extract_text_i = ExtractInfo(text_path, no_text_path, "text", name, output_directory, image, i)
        infors.append(obj_extract_text_i)

    if multithread>1:
        with tqdm(total=len(infors), desc="Executing Pipeline", unit=" Samples") as progress_bar:
            with ThreadPoolExecutor(max_workers=multithread) as executor:
                for result in executor.map(Extract(), infors):
                    progress_bar.set_description("Processing %s" % result)
                    progress_bar.update(1)
    else:
        with tqdm(total=len(infors), desc="Executing Pipeline", unit=" Samples") as progress_bar:
            for infor in infors:
                result = Extract().__call__(infor)
                progress_bar.set_description("Processing %s" % result)
                progress_bar.update(1)

if __name__ == '__main__':
    import argparse
    parse = argparse.ArgumentParser()
    parse.add_argument("--input_dir")
    parse.add_argument("--output_dir")
    parse.add_argument("--mode", choices=["one", "many"])
    parse.add_argument("--multithread", type=int, default=-1, help="use multi thread")
    args = parse.parse_args()

    for k, v in dict(args.__dict__).items():
        print('{}: {}'.format(k, v))
    if args.mode == "many":
        os.makedirs(args.output_dir, exist_ok=True)
        pdf_dirs = glob(os.path.join(args.input_dir, "*"))
        #print("pdf dirs {}".format(pdf_dirs))
        for pdf_dir in pdf_dirs:
            print("Executing:", pdf_dir,'...')
            name = os.path.split(pdf_dir)[-1]
            output_mask_dir = os.path.join(args.output_dir, name)
            create_data_object(pdf_dir, output_mask_dir, multithread=args.multithread)
    elif args.mode == "one":
        create_data_object(args.input_dir, args.output_dir)
    else:
        raise "--mode can be one/many"

    str_paths = os.path.join(args.output_dir,'*/*.png')
    print("Get paths from: ", str_paths)
    image_paths = glob(str_paths)
    perform(image_paths, multithread=args.multithread)