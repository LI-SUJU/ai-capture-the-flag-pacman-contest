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
# for i in range(len(files)):
#     plt.plot(x_values[i], y_values[i], label=files[i], color=colors[i % len(colors)])  # Use color from the list based on index

# # Add legends, labels, and title
# plt.legend()
# x_major_locator = MultipleLocator(1)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# y_major_locator = MultipleLocator(1)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlabel('Game Number')
# plt.ylabel('Score')
# plt.title('Scores of Games')
# # save
# plt.savefig('./plots/plot.png')
# Show the plot
# plt.show()

# Calculate the mean score of all games
mean_scores = [sum(scores) / len(scores) for scores in y_values]
# calculate variance of scores
variance_scores = [np.var(scores) for scores in y_values]

# create a new plot to show the mean scores

# plt.figure()
# # Create a bar plot with different colors for each file
# # Create a bar plot with different colors for each file and add comments for each bar
# plt.bar(range(len(files)), mean_scores, color=colors, tick_label=files)

# # Add legends, labels, and title

# plt.xlabel('Agents')
# plt.ylabel('Mean Score')
# plt.title('Mean Scores of Games of {}'.format(len(y_values[0])))

# # Save the plot
# plt.savefig('./plots/barPlot.png')

# Generate a plot for each file in ./data4plots

for i in range(len(files)):
    # difine the size of the plot
    plt.figure(figsize=(20, 8))
# set font size
    plt.rcParams.update({'font.size': 15})  # Adjust the figure size to make it wider
    # Smooth the data using the smooth function
    # y_values[i] = smooth(y_values[i], 0.1)
    y_values[i] = gaussian_filter1d(y_values[i], sigma=0.1)
    plt.plot(x_values[i], y_values[i], label=files[i], color='teal')
    # draw a mean value line on the plot and add a comment
    mean_score = sum(y_values[i]) / len(y_values[i])
    plt.axhline(y=mean_score, color='orange', linestyle='-', label='Mean Score: {:.2f}'.format(mean_score))
    # draw a y=0 dashed line
    plt.axhline(y=0, color='gray', linestyle='--', label='score: 0')
    plt.text(x_values[i][-1] - 2, mean_score, f'Mean: {mean_score:.2f}', color='orange')
    plt.text(x_values[i][-1] + 2.5, mean_score, f'Variance: {variance_scores[i]:.2f}', color='orange')
    plt.legend()
    x_major_locator = MultipleLocator(1)
    ax = plt.gca()
    ax.xaxis.set_major_locator(x_major_locator)
    y_major_locator = MultipleLocator(1)
    ax.yaxis.set_major_locator(y_major_locator)
    plt.xlabel('Game No.')
    plt.ylabel('Score')
    plt.title('Scores of Games - {}'.format(files[i]))
    plt.savefig('./plots/plot_{}.png'.format(i))

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
# for i in range(len(files)):
#     plt.plot(x_values[i], y_values[i], label=files[i], color=colors[i % len(colors)])  # Use color from the list based on index

# # Add legends, labels, and title
# plt.legend()
# x_major_locator = MultipleLocator(1)
# ax = plt.gca()
# ax.xaxis.set_major_locator(x_major_locator)
# y_major_locator = MultipleLocator(1)
# ax.yaxis.set_major_locator(y_major_locator)
# plt.xlabel('Game Number')
# plt.ylabel('Score')
# plt.title('Scores of Games')
# # save
# plt.savefig('./plots/plot.png')
# Show the plot


# Directory containing the data files
data_dir = './data4MCTSCompare'

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

# Calculate mean and variance for each line
mean_values = [np.mean(y) for y in y_values]
variance_values = [np.var(y) for y in y_values]

# difine the size of the plot
plt.figure(figsize=(20, 8))
# set font size
plt.rcParams.update({'font.size': 15})
# Plot each line with different colors and add legend
for i in range(len(files)):
    plt.plot(x_values[i], y_values[i], label=files[i], color=colors[i % len(colors)])

# Add horizontal lines for mean values
for i in range(len(files)):
    plt.axhline(y=mean_values[i], color=colors[i % len(colors)], linestyle='--')
# add mean value text and variance for each line, don't let the text overlap
for i in range(len(files)):
    plt.text(x_values[i][-1] - 2, mean_values[i], f'Mean: {mean_values[i]:.2f}', color=colors[i % len(colors)])
    plt.text(x_values[i][-1] + 2.5, mean_values[i], f'Variance: {variance_values[i]:.2f}', color=colors[i % len(colors)])

# Add legends, labels, and title
plt.legend()
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel('Game No.')
plt.ylabel('Score')
plt.title('Comparison of MCTS Algorithms with Different Parameters')

# Save the plot
plt.savefig('./plots/MCTSCompare.png')

# Directory containing the data files
data_dir = './data4AStarCompare'

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

# Calculate mean and variance for each line
mean_values = [np.mean(y) for y in y_values]
variance_values = [np.var(y) for y in y_values]
# difine the size of the plot
plt.figure(figsize=(20, 8))
# set font size
plt.rcParams.update({'font.size': 15})
# Plot each line with different colors and add legend
for i in range(len(files)):
    plt.plot(x_values[i], y_values[i], label=files[i], color=colors[i % len(colors)])

# Add horizontal lines for mean values, show the mean value of each line

for i in range(len(files)):
    plt.axhline(y=mean_values[i], color=colors[i % len(colors)], linestyle='--')
# add mean value text for each line
for i in range(len(files)):
    plt.text(x_values[i][-1] + 1, mean_values[i], f'Mean: {mean_values[i]:.2f}', color=colors[i % len(colors)])
# Add variance text for each line
for i in range(len(files)):
    plt.text(x_values[i][-1] + 1, mean_values[i], f'Variance: {variance_values[i]:.2f}', color=colors[i % len(colors)])

# Add legends, labels, and title
plt.legend()
x_major_locator = MultipleLocator(1)
ax = plt.gca()
ax.xaxis.set_major_locator(x_major_locator)
y_major_locator = MultipleLocator(1)
ax.yaxis.set_major_locator(y_major_locator)
plt.xlabel('Game No.')
plt.ylabel('Score')
plt.title('Comparison of A* Algorithms with Different Parameters')

# Save the plot
plt.savefig('./plots/AStarCompare.png')


# Directory containing the data files
data_dir = './data4BestOfMCTS&AStar'

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

# compare the variance of the two algorithms, and show the result as a bar plot
# Calculate variance for each line
variance_values = [np.var(y) for y in y_values]
# difine the size of the plot
plt.figure(figsize=(20, 8))
# set font size
plt.rcParams.update({'font.size': 15})
# Plot each line with different colors and add legend
plt.bar(files, variance_values, color=colors)
plt.xlabel('Algorithm')
plt.ylabel('Variance')
plt.title('Comparison of Variance between MCTS and A*')
plt.savefig('./plots/comparisonOfVariance.png')
#


# Show the plot
# plt.show()