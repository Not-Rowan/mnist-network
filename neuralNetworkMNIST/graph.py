import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
import numpy as np
import warnings

# ignore warnings
#warnings.simplefilter("ignore")

# Read the file initially
data = None
while data is None or len(data) == 0:
    try:
        data = pd.read_csv('accuracy.csv', header=None)
    except pd.errors.EmptyDataError:
        pass

# Initialize global variables
secondsPassed = 0

# Initialize lists to store data
train_accuracies = []
test_accuracies = []

# Extract training and testing accuracies along with timestamps
for row in data[0]:
    if row.startswith('Train:'):
        train_accuracies.append(float(row.split()[1]))
    elif row.startswith('Test:'):
        test_accuracies.append(float(row.split()[1]))

# Initialize the plot with initial data
fig, ax = plt.subplots(figsize=(8, 6))

# Plot training accuracy
train_line, = ax.plot(train_accuracies, label='Training Accuracy', color='blue')

# Display the latest test accuracy as a horizontal red line (if available)
test_line = None  # Initialize test line as None
if len(test_accuracies) > 0:
    test_y_data = [test_accuracies[-1]] * len(train_accuracies)  # Create y-data for test accuracy
    test_line, = ax.plot(test_y_data, linestyle='--', color='red', label='Testing Accuracy')

# Add titles and legend
if len(test_accuracies) > 0 and len(train_accuracies) > 0:
    ax.set_title(f'Current Training Accuracy: {train_accuracies[-1]}%\nCurrent Testing Accuracy: {test_accuracies[-1]}%')
elif len(train_accuracies) > 0:
    ax.set_title(f'Current Training Accuracy: {train_accuracies[-1]}%')
else:
    ax.set_title('No Data Available Currently')

ax.set_xlabel('Seconds Passed')
ax.set_ylabel('Accuracy (%)')
ax.set_ylim(0, 100)
ax.legend()

# Function to update the plot
def update_plot(frame):
    global data, train_accuracies, test_accuracies, secondsPassed

    # Read the file again to check for new values
    updated_data = pd.read_csv('accuracy.csv', header=None)

    # Check if there are new values
    if len(updated_data) > len(data):
        # Extract the new values
        new_rows = updated_data.iloc[len(data):]

        # Update training and testing accuracies along with timestamps
        for row in new_rows[0]:
            if row.startswith('Train:'):
                train_accuracies.append(float(row.split()[1]))
            elif row.startswith('Test:'):
                test_accuracies.append(float(row.split()[1]))

        # Update the plot based on data availability
        if len(train_accuracies) > 0:
            train_line.set_ydata(train_accuracies)
            train_line.set_xdata(range(len(train_accuracies)))
            ax.set_xlim(0, len(train_accuracies))

            if len(test_accuracies) > 0:
                test_y_data = [test_accuracies[-1]] * len(train_accuracies)
                test_line.set_ydata(test_y_data)
                test_line.set_xdata(range(len(train_accuracies)))
                ax.set_title(f'Current Training Accuracy: {train_accuracies[-1]}%\nCurrent Testing Accuracy: {test_accuracies[-1]}%')
            else:
                ax.set_title(f'Current Training Accuracy: {train_accuracies[-1]}%')
        else:
            ax.set_title('No Data Available Currently')

        # Update the data variable
        data = updated_data

    # update the seconds passed and set the x axis value to seconds passed
    secondsPassed += 1
    ax.set_xlabel(f'Seconds Passed: {secondsPassed}')

    return train_line, test_line

# Animate the plot
ani = FuncAnimation(fig, update_plot, interval=1000)

# Show the plot
plt.show()
