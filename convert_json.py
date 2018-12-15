import json
import os


json_in = 'delivery/label.json' # this json in format as mizuho json.
json_out = 'delivery/label_f1.json'

json_data_out = {}
json_data_in = json.load(open(json_in, 'r'))
for image_name in json_data_in:
    #print(image_name)
    bboxes_out = []
    for key in json_data_in[image_name]:
        bbox = json_data_in[image_name][key]
        xmin = bbox['x']
        xmax = bbox['x'] + bbox['width']
        ymin = bbox['y']
        ymax = bbox['y'] + bbox['height']
        bbox_out = []
        bbox_out.append([xmin, ymin])
        bbox_out.append([xmax, ymin])
        bbox_out.append([xmax, ymax])
        bbox_out.append([xmin, ymax])
        bboxes_out.append(bbox_out)
    #print(bboxes_out)
    dict = {}
    dict['line'] = bboxes_out
    json_data_out[image_name] = dict

## print json dataout
with open(json_out, 'w') as outfile:
    json.dump(json_data_out, outfile)