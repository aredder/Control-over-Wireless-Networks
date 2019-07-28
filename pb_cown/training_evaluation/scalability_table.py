import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Qt5Agg')

data = np.array([[39.90, 80.15,  120.17],
                [64.92, 134.25,  314.84],
                [188, 381.93, 1022.60]])

raw_data = data.copy()
data[2, :] = data[2, :] - data[1, :]
data[1, :] = data[1, :] - data[0, :]

columns = ('N = 4, M = 3', 'N = 8, M = 6', 'N = 12, M = 9')
rows = ['Random Agent', 'Iterative Agent', 'Optimal loss']

values = np.arange(0, 1100, 100)
value_increment = 1

# Get some pastel shades for the colors
# colors = plt.cm.Set1(np.linspace(0, 0.5, len(rows)))
colors = ['indianred', 'royalblue', 'sandybrown']
n_rows = len(data)

index = np.arange(len(columns)) + 0.3
bar_width = 0.4

# Initialize the vertical-offset for the stacked bar chart.
y_offset = np.zeros(len(columns))

# Plot bars and create text labels for the table
cell_text = []
for row in range(n_rows):
    plt.bar(index, data[row], bar_width, bottom=y_offset, color=colors[row])
    y_offset = y_offset + data[row]
    cell_text.append(['%.2f' % x for x in y_offset])

for row in range(3):
    plt.plot(index, raw_data[row, :], color='black', linestyle='-', linewidth=3.5, marker='o', markersize=5.5)
    plt.plot(index, raw_data[row, :], color=colors[row], linestyle='-', linewidth=2, marker='o', markersize=1.5)

# Reverse colors and text labels to display the last value at the top.
colors = colors[::-1]
cell_text.reverse()




# Add a table at the bottom of the axes
the_table = plt.table(cellText=cell_text,
                      rowLabels=rows,
                      rowColours=colors,
                      colLabels=columns,
                      loc='bottom',
                      cellLoc='center')
the_table.scale(1, 1.5)

# Adjust layout to make room for the table:
plt.subplots_adjust(left=0.2, bottom=0.2)
plt.ylabel('Average control loss per stage')
plt.yticks(values * value_increment, ['%d' % val for val in values])
plt.xticks([])
plt.show()
plt.savefig("scale_plot.pdf", bbox_inches='tight')
