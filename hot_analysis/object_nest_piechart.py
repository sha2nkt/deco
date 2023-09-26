# make a nestest plotly pie chart given a list of labels and a list of values
import os.path as osp
import json
import plotly.express as px
import plotly.io as pio

version = '1'
dir = '/is/cluster/work/stripathi/pycharm_remote/dca_contact/hot_analysis/'
out_dir_hico = osp.join(dir, f'filtered_data/v_{version}/hico')
out_dir_vcoco = osp.join(dir, f'filtered_data/v_{version}/vcoco')

objectwise_img_dict_hico = osp.join(out_dir_hico, 'imgnames_per_object_dict.json')
objectwise_img_dict_vcoco = osp.join(out_dir_vcoco, 'imgnames_per_object_dict.json')

with open(objectwise_img_dict_hico, 'r') as fp:
    objectwise_img_dict_hico = json.load(fp)
    # replace underscore with space
    objectwise_img_dict_hico = {k.replace('_', ' '): v for k, v in objectwise_img_dict_hico.items()}
    # replace values with lenght of values
    objectwise_img_dict_hico = {k: len(v) for k, v in objectwise_img_dict_hico.items()}
with open(objectwise_img_dict_vcoco, 'r') as fp:
    objectwise_img_dict_vcoco = json.load(fp)
    # replace underscore with space
    objectwise_img_dict_vcoco = {k.replace('_', ' '): v for k, v in objectwise_img_dict_vcoco.items()}
    # replace values with lenght of values
    objectwise_img_dict_vcoco = {k: len(v) for k, v in objectwise_img_dict_vcoco.items()}

# combine the dicts
objectwise_img_dict = objectwise_img_dict_hico.copy()
objectwise_img_dict.update(objectwise_img_dict_vcoco)


categorization = {"transportation": ['motorcycle','bicycle','boat','car','truck','bus','train','airplane',],
                  "accessories": ['backpack', 'tie', 'handbag', 'baseball glove'],
                  "furniture": ['bench','chair','couch','bed','toilet','dining table'],
                  'everyday objects': ['book','umbrella','cell phone','laptop','kite','suitcase','bottle','remote',
                                       'toothbrush','teddy bear','scissors','keyboard','hair drier','traffic light',
                                       'fire hydrant','stop sign','tv','vase','parking meter','clock','potted plant','mouse',],
                  'sports equipment': ['frisbee','sports ball','tennis racket','baseball bat','skateboard','snowboard','skis','surfboard',],
                  'food items': ['banana','cake','apple','carrot','pizza','donut','hot dog','sandwich','broccoli','orange'],
                  'kitchen appliances': ['knife', 'spoon', 'cup', 'wine glass', 'oven', 'fork', 'bowl', 'refrigerator', 'toaster', 'sink', 'microwave',]}

# get total lengths of each category
objectwise_img_dict_categorized = {}
for k, v in categorization.items():
    objectwise_img_dict_categorized[k] = sum([objectwise_img_dict[obj] for obj in v])

# reverse categorization
categorization_rev = {}
for k, v in categorization.items():
    for obj in v:
        categorization_rev[obj] = k



data = dict(
    character=list(categorization.keys()) + list(categorization_rev.keys()),
    parent= ["objects"] * len(categorization.keys()) + list(categorization_rev.values()),
    value =[0] * len(categorization.keys()) + list(objectwise_img_dict.values()),
)

# save data as pandas
import pandas as pd
df = pd.DataFrame(data)
df.to_csv(osp.join(dir, 'object_nest_piechart_data.csv'))


fig = px.sunburst(
    data,
    names='character',
    parents='parent',
    values='value',
)

# chage font size of the innermost level
fig.update_traces(textfont_size=30)

# save the figure
out_path = osp.join(dir, "object_nest_piechart.html")
fig.write_html(out_path)
# Save plot as PNG wihtout transparent background
out_path = osp.join(dir, "object_nest_piechart.png")
fig.write_image(out_path,
                format='png',
                width=2000, height=1000, scale=1, engine='kaleido')


# Set layout
fig.update_layout(
    margin=dict(t=0, l=0, r=0, b=0),
    plot_bgcolor='rgba(0, 0, 0, 0)',
    paper_bgcolor='rgba(0, 0, 0, 0)'
)

# Save plot as HTML with transparent background
out_path = osp.join(dir, "object_nest_piechart_transparent.html")
pio.write_html(fig, file=out_path, auto_open=False, include_plotlyjs='cdn', config=dict(displayModeBar=False))

# Save plot as PNG with transparent background
out_path = osp.join(dir, "object_nest_piechart_transparent.png")
pio.write_image(fig, file=out_path,
                format='png',
                width=2000, height=1000, scale=1, engine='kaleido')

import matplotlib.pyplot as plt
import seaborn as sns

# make a bar plot of objectwise_img_dict

# sort the dict
objectwise_img_dict = {k: v for k, v in sorted(objectwise_img_dict.items(), key=lambda item: item[1], reverse=True)}

# save as pandas
import pandas as pd
df = pd.DataFrame.from_dict(objectwise_img_dict, orient='index', columns=['count'])
df.to_csv(osp.join(dir, 'image_per_object_category.csv'))


plt.figure(figsize=(20, 10))
sns.barplot(x=list(objectwise_img_dict.keys()), y=list(objectwise_img_dict.values()))
# make horizontal grid lines
plt.grid(axis='y', alpha=0.5)

# Add x-axis and y-axis labels
plt.xticks(rotation=45, ha='right', fontsize=15)
plt.yticks(fontsize=15)
plt.show()
# avoid clipping of x-axis labels
plt.tight_layout()

# save the figure
out_path = osp.join(dir, "image_per_object_category.png")
plt.savefig(out_path, transparent=False)
out_path = osp.join(dir, "image_per_object_category_transparent.png")
plt.savefig(out_path, transparent=True)

