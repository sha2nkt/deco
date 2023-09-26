import csv
from pathlib import Path

from matplotlib import gridspec
from matplotlib import pyplot as plt
import seaborn as sns

# Input/output files
CSV_FOLDER = Path(__file__).parent / "documents-export-2023-03-07"
CONTACT_FILE = CSV_FOLDER / "partwise_contact_graph.csv"
OUTPUT = 'output.pdf'

# Some configuration
FONTSIZE = 30
plt.rcParams['grid.color'] = (0.5, 0.5, 0.5, 0.1)  # low grid opacity

# Read CSV file
body_parts = []
body_part_counts = []
with open(CONTACT_FILE, newline='', encoding="utf-8") as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        body_parts.append(row[''])
        body_part_counts.append(int(row['count']))

# Plot
fig = plt.figure(figsize=(25, 22))
gs = gridspec.GridSpec(1, 1)
gs.update(wspace=0.15, hspace=0.15)
axes = []
for elem in gs:
    axes.append(fig.add_subplot(elem))
axes[0].grid(axis='y', zorder=0)
sns.barplot(
    x=body_parts, y=body_part_counts, ax=axes[0], zorder=3,
    facecolor=(0.2, 0.4, 0.6, 0.6))
axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=90)
plt.xticks(rotation=45, ha='right', fontsize=FONTSIZE)
plt.yticks(fontsize=FONTSIZE)
plt.ylabel('# images', labelpad=30, fontsize=FONTSIZE)
plt.savefig(OUTPUT)
# plt.show()
