import os
from plotHelper import smooth
import matplotlib.pyplot as plt
from matplotlib.ticker import MultipleLocator
import numpy as np
from scipy.ndimage.filters import gaussian_filter1d

# Directory containing the data files
data_dir = './data4plots'

# Get the list of files in the directory
files = os.listdir(data_dir)

# Initialize lists to store the data
x_values = []
y_values = []

# Read data from each file
for file in files:
    file_path = os.path.join(data_dir, file)
    with open(file_path, 'r') as f:
        # Read each line as an integer and append to the y_values list, but ignore the last line
        y_values.append([int(line.strip()) for line in f.readlines()])

        # Generate x values based on the number of data points in each file
        x_values.append(list(range(len(y_values[-1]))))

# Plot the data and each line should have a different colors
colors = ['orange', 'purple', 'pink', 'teal', 'gold', 'red', 'blue', 'green'] # Define a list of colors
for i in range(len(files)):
    plt.plot(x_values[i], y_values[i], label=files[i], color=colors[i % len(colors)])  # Use color from the list based on index

# Add legends, labels, and title
plt.legend()
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel('Game Number')
plt.ylabel('Score')
plt.title('Scores of Games')
# save
plt.savefig('./plots/plot.png')
# Show the plot
# plt.show()

# Calculate the mean score of all games
mean_scores = [sum(scores) / len(scores) for scores in y_values]

# create a new plot to show the mean scores

plt.figure()
# Create a bar plot with different colors for each file
# Create a bar plot with different colors for each file and add comments for each bar
plt.bar(range(len(files)), mean_scores, color=colors, tick_label=files)

# Add legends, labels, and title

plt.xlabel('Agents')
plt.ylabel('Mean Score')
plt.title('Mean Scores of Games of {}'.format(len(y_values[0])))

# Save the plot
plt.savefig('./plots/barPlot.png')

# difine the size of the plot
plt.figure(figsize=(20, 6))
# Generate a plot for each file in ./data4plots

for i in range(len(files)):
    plt.figure(figsize=(16, 6))  # Adjust the figure size to make it wider
    # Smooth the data using the smooth function
    # y_values[i] = smooth(y_values[i], 0.1)
    y_values[i] = gaussian_filter1d(y_values[i], sigma=0.1)
    plt.plot(x_values[i], y_values[i], label=files[i], color=colors[i % len(colors)])
    # draw a mean value line on the plot and add a comment
    mean_score = sum(y_values[i]) / len(y_values[i])
    plt.axhline(y=mean_score, color='gold', linestyle='-', label='Mean Score: {:.2f}'.format(mean_score))
    plt.legend()
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(1)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlabel('Game Number')
    plt.ylabel('Score')
    plt.title('Scores of Games - {}'.format(files[i]))
    plt.savefig('./plots/plot_{}.png'.format(i))
    plt.close()

# Show the plot
# plt.show()