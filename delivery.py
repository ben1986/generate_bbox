# This file will dump json file to json_ouptut_path
import os
import glob
from glob import glob
import json
import shutil


input_path = "/media/vu/DATA/working/projects/invoice/workspace/pycharm/training/venv/workspace/generate_bbox/output"
ouptut_path = "/media/vu/DATA/working/projects/invoice/workspace/pycharm/training/venv/workspace/generate_bbox/delivery"
json_name = 'label.json'
num_of_type = 2

def _load_db(path_json):
    read_json = lambda path: json.load(open(path, 'r'))

    if type(path_json) is list:
        if len(path_json) == 1:
            return _load_db(path_json[0])
        else:
            paths_0, labels_0 = self._load_db(path_json[0])
            paths_rest, labels_rest = self._load_db(path_json[1:])
            return paths_0 + paths_rest, labels_0 + labels_rest
    else:
        img_dir = os.path.split(path_json)[0]
        d = read_json(path_json)
        paths = [os.path.join(img_dir, path) for path in list(d.keys())]
        labels = list(d.values())
        return paths, labels

if __name__ == '__main__':
    #json_path = os.path.dirname(json_ouptut_path)
    os.makedirs(ouptut_path, exist_ok=True)
    pdf_dirs = glob(os.path.join(input_path,"*"))
    json_data_all = {}
    for pdf_dir in pdf_dirs:
        print("Prcessing: ", pdf_dir, "....")
        ## copy image *input* from pdf_dir to output path
        list_json = glob(os.path.join(pdf_dir,'*input*json'))
        list_image = glob(os.path.join(pdf_dir, '*input*png'))
        list_sub_json = list_json[0:num_of_type]
        list_sub_image = list_image[0:num_of_type]

        for json_file, image_file in zip(list_sub_json,list_sub_image):
            # copy image file
            shutil.copy2(image_file,ouptut_path)
            #print('json file', json_file)
            #print('image file', image_file)
            name = json_file.split('/')[-1].split('.')[0]
            json_data = json.load(open(json_file,'r'))
            json_data_all[name + '.png'] = json_data
    with open(os.path.join(ouptut_path, json_name), 'w') as outfile:
        json.dump(json_data_all, outfile)




