import os

# Set the path to the directory containing your Python files
# set the directory to ./myTeam
directory = './myTeam.py'

# Set the path to the output file where the graph will be saved
output_file = './plots'

# Change the current working directory to the directory containing your Python files
os.chdir(directory)

# Run Pyreverse command to generate the class inheritance graph
os.system(f'pyreverse -o png -p myTeam -f ALL')

# Move the generated graph file to the desired location
os.rename('classes_myTeam.png', output_file)