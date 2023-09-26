import csv
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import gridspec
import numpy as np
import seaborn as sns

# Input/output files
CSV_FOLDER = Path(__file__).parent / "documents-export-2023-03-07"
IMAGE_PER_OBJECT_FILE = CSV_FOLDER / "image_per_object_category.csv"
PIE_CHART_DATA_FILE = CSV_FOLDER / "object_nest_piechart_data.csv"
OUTPUT = 'output.pdf'

# Some configuration
FONTSIZE = 20
FONTSIZE_PIE = 25
COLOR_RANGE = (0.15, 0.85)  # to avoid having exterme values
CUTOFF = 30  # min number of counts for an object to be considered
PIE_INNER_RADIUS = 0.3  # value used for the nested pie chart
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)  # low grid opacity

# Categories and associated objects
categorization_unordered = {
    "transportation": [
        'motorcycle', 'bicycle', 'boat', 'car', 'truck', 'bus', 'train', 'airplane',],
    "accessories": [
        'backpack', 'tie', 'handbag', 'baseball glove'],
    "furniture": [
        'bench', 'chair', 'couch', 'bed', 'toilet', 'dining table'],
    'everyday objects': [
        'book', 'umbrella', 'cell phone', 'laptop', 'kite', 'suitcase', 'bottle', 'remote',
        'toothbrush', 'teddy bear', 'scissors', 'keyboard', 'hair drier', 'traffic light',
        'fire hydrant', 'stop sign', 'tv', 'vase', 'parking meter', 'clock', 'potted plant',
        'mouse'],
    'sports equipment': [
        'frisbee', 'sports ball', 'tennis racket', 'baseball bat',
        'skateboard', 'snowboard', 'skis', 'surfboard'],
    'food items': [
        'banana', 'cake', 'apple', 'carrot', 'pizza', 'donut', 'hot dog',
        'sandwich', 'broccoli', 'orange'],
    'kitchen appliances': [
        'knife', 'spoon', 'cup', 'wine glass', 'oven', 'fork', 'bowl',
        'refrigerator', 'toaster', 'sink', 'microwave']}

# Colors for each category
cat_colors = {
    "transportation": plt.cm.spring,
    "accessories": plt.cm.Greys,
    "furniture": plt.cm.Purples,
    "everyday objects": plt.cm.Reds,
    "sports equipment": plt.cm.Blues,
    "food items": plt.cm.Wistia,
    "kitchen appliances":  plt.cm.summer}

# Read CSV files
objects = {}
with open(IMAGE_PER_OBJECT_FILE, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        objects[row['']] = {}
        objects[row['']]['count'] = int(row['count'])

# In each caterory, order object by count in descending order
categorization = {
    cat: sorted(val, key=lambda _: objects[_]["count"])
    for cat, val in categorization_unordered.items()}

# Some sanity checks :)
assert list(categorization.keys()) == list(categorization_unordered.keys())
assert list(categorization.keys()) == list(cat_colors.keys())
for cat in categorization:
    assert len(categorization[cat]) == len(categorization_unordered[cat])
    assert set(categorization[cat]) == set(categorization_unordered[cat])

# Cutoff objects and categorization
selected_objects = [o for o, v in objects.items() if v["count"] > CUTOFF]
selected_categorization = {}
for k, v in categorization.items():
    selected_categorization[k] = [o for o in v if o in selected_objects]

# Add a color for each object
# All objects in one category use the same category color
for cat, objs in selected_categorization.items():
    colors = [
        cat_colors[cat](x)
        for x in np.linspace(COLOR_RANGE[0], COLOR_RANGE[1], len(selected_categorization[cat]))]
    color_index = 0
    for obj in objs:
        objects[obj]['color'] = colors[color_index]
        color_index += 1

# Plot: histogramm
fig = plt.figure(figsize=(25, 22))
gs = gridspec.GridSpec(1, 1)
gs.update(wspace=0.15, hspace=0.15)
axes = []
for elem in gs:
    axes.append(fig.add_subplot(elem))
axes[0].grid(axis='x', zorder=0)
axes[0].set_xticks(range(0, 500, 50))
sns.barplot(
    x=list(v["count"] for o, v in objects.items() if o in selected_objects),
    y=list(selected_objects),
    ax=axes[0],
    zorder=3,
    palette=list(v["color"] for o, v in objects.items() if o in selected_objects))
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
plt.xticks(rotation=0, fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.xlabel('# images', labelpad=30, fontsize=FONTSIZE)

# Plot: pie chart
right_inset_ax = fig.add_axes([.3, .1, .5, .5], facecolor='k')
outer_colors = [c(0.5) for c in cat_colors.values()]
labels = [word.replace(' ', '\n') for word in list(selected_categorization)]
# Need to be orderd by category
vals = []
groups = {cat: 0 for cat in selected_categorization.keys()}
inner_colors = []
for cat, members in selected_categorization.items():
    for _ in members:
        vals.append(objects[_]["count"])
        inner_colors.append(objects[_]["color"])
        groups[cat] += objects[_]["count"]
right_inset_ax.pie(
    list(groups.values()),
    radius=1,
    colors=outer_colors,
    #    labels=labels,
    autopct='%.1f%%',
    pctdistance=0.82,
    #    labeldistance=1.5,
    textprops={'color': 'white', 'fontsize': FONTSIZE},
    wedgeprops={'width': PIE_INNER_RADIUS, 'edgecolor': 'w'})
right_inset_ax.pie(
    vals,
    radius=1-PIE_INNER_RADIUS,
    colors=inner_colors,
    wedgeprops={'width': PIE_INNER_RADIUS, 'edgecolor': 'w'})

# Plot: add legend and save
plt.legend(
    labels, bbox_to_anchor=(1.2, 1.45),
    loc='upper right', prop={'size': 1.5*FONTSIZE})
plt.savefig(OUTPUT)
# plt.show()
