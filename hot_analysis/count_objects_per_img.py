# get average number of objects per image
import os.path as osp
import json
import plotly.express as px
import plotly.io as pio

version = '1'
dir = '/is/cluster/work/stripathi/pycharm_remote/dca_contact/hot_analysis/'
out_dir_hico = osp.join(dir, f'filtered_data/v_{version}/hico')
out_dir_vcoco = osp.join(dir, f'filtered_data/v_{version}/vcoco')

imgwise_obj_dict_hico = osp.join(out_dir_hico, 'object_per_image_dict.json')
imgwise_obj_dict_vcoco = osp.join(out_dir_vcoco, 'object_per_image_dict.json')

with open(imgwise_obj_dict_hico, 'r') as fp:
    imgwise_obj_dict_hico = json.load(fp)
with open(imgwise_obj_dict_vcoco, 'r') as fp:
    imgwise_obj_dict_vcoco = json.load(fp)

# combine the dicts
imgwise_obj_dict = imgwise_obj_dict_hico.copy()
imgwise_obj_dict.update(imgwise_obj_dict_vcoco)

# get average length of object in the object per image key
avg_obj_per_img = sum([len(v) for v in imgwise_obj_dict.values()]) / len(imgwise_obj_dict)
print(f'Average number of objects per image: {avg_obj_per_img}')

# get average searately for hico and vcoco
avg_obj_per_img_hico = sum([len(v) for v in imgwise_obj_dict_hico.values()]) / len(imgwise_obj_dict_hico)
print(f'Average number of objects per image in hico: {avg_obj_per_img_hico}')

avg_obj_per_img_vcoco = sum([len(v) for v in imgwise_obj_dict_vcoco.values()])  / len(imgwise_obj_dict_vcoco)
print(f'Average number of objects per image in vcoco: {avg_obj_per_img_vcoco}')


