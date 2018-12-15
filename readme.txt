1. Run create_data_from_pdf_multithread.py
   Params: --input_dir /media/vu/DATA/working/projects/invoice/workspace/pycharm/training/venv/workspace/generate_bbox/input
           --output_dir /media/vu/DATA/working/projects/invoice/workspace/pycharm/training/venv/workspace/generate_bbox/output
           --mode many

   Funcs: generate images and json file, stored in output_dir

2. delivery.py
   input_path = "/media/vu/DATA/working/projects/invoice/workspace/pycharm/training/venv/workspace/generate_bbox/output"
   ouptut_path = "/media/vu/DATA/working/projects/invoice/workspace/pycharm/training/venv/workspace/generate_bbox/delivery"
   json_name = 'label.json'

   Func: combine many json file in 1), creating one json file
3. Convert json
   Convert from json format 1 (similar to mizuho format, with bbox as x,y,w,h)
         to json format 2 (with bbox as (xmin, ymin), (xmax, ymin), (xmax, ymax), (xmin, ymax))
