import tkinter as tk
import numpy as np
import pandas as pd
import math

# Constants
CANVAS_WIDTH = 280
CANVAS_HEIGHT = 280
PIXEL_SIZE = 10
MIN_LINE_WIDTH = 5

# Global variables
LINE_WIDTH = MIN_LINE_WIDTH

# Create a window
window = tk.Tk()
window.title('Draw')

# Create a canvas
canvas = tk.Canvas(window, width=CANVAS_WIDTH, height=CANVAS_HEIGHT, bg='white')
canvas.pack()

# Initialize variables to store previous coordinates
prev_x = None
prev_y = None

# Initialize variables to store drawn pixels and their intensities
drawn_pixels = {}

# Function to handle mouse movement while button is held down
def draw_line(event):
    global prev_x, prev_y

    x, y = event.x, event.y

    if prev_x is None and prev_y is None:
        prev_x, prev_y = x, y
        return

    # Draw a smooth line from the previous coordinates to the current coordinates of the mouse
    canvas.create_line(prev_x, prev_y, x, y, width=LINE_WIDTH, fill='black', capstyle="round")

    # Update the drawn pixels dictionary with the pixel values and intensities
    draw_pixels_along_line(prev_x, prev_y, x, y)

    # Update previous coordinates
    prev_x, prev_y = x, y

def draw_pixels_along_line(x1, y1, x2, y2):
    # Calculate the distance between the two points
    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)

    # Update the drawn pixels dictionary with the pixels along the line
    for i in range(int(distance)):
        fraction = i / distance
        new_x = int(x1 * (1 - fraction) + x2 * fraction)
        new_y = int(y1 * (1 - fraction) + y2 * fraction)
        # Draw pixels around the line based on line width
        for j in range(-LINE_WIDTH//2, LINE_WIDTH//2 + 1):
            for k in range(-LINE_WIDTH//2, LINE_WIDTH//2 + 1):
                pixel_x = new_x + j
                pixel_y = new_y + k
                # Ensure the pixel is within the canvas bounds
                if 0 <= pixel_x < CANVAS_WIDTH and 0 <= pixel_y < CANVAS_HEIGHT:
                    drawn_pixels[(pixel_x, pixel_y)] = 255

# Bind the canvas to the function
canvas.bind('<B1-Motion>', draw_line)

# Reset prev_x and prev_y when mouse button is released
def reset_prev(event):
    global prev_x, prev_y
    prev_x = None
    prev_y = None

canvas.bind('<ButtonRelease-1>', reset_prev)

# Create a function to save the drawing to a CSS file
def save():
    # Create a blank array to store pixel values
    pixel_values = np.zeros((28, 28), dtype=np.uint8)

    # Iterate through the drawn pixels and update the array
    for (x, y), _ in drawn_pixels.items():
        x_idx = x // 10  # Convert canvas coordinates to array indices
        y_idx = y // 10
        if 0 <= x_idx < 28 and 0 <= y_idx < 28:  # Ensure the pixel is within the canvas bounds
            pixel_values[y_idx, x_idx] = 255  # Set the pixel value

    # Flatten the array and save it to a CSV file
    pixel_values_flat = pixel_values.flatten()
    pd.DataFrame([pixel_values_flat]).to_csv('userInput.csv', index=False, header=False)
    window.title('Draw - Image Saved')

# Create a function to clear the canvas
def clear():
    canvas.delete('all')
    drawn_pixels.clear()  # Clear the drawn pixels dictionary
    window.title('Draw')

# Function to update line width
def update_line_width(val):
    global LINE_WIDTH
    LINE_WIDTH = max(MIN_LINE_WIDTH, int(val))

# Create a Scale widget to adjust line width
line_width_scale = tk.Scale(window, from_=MIN_LINE_WIDTH, to=50, orient=tk.HORIZONTAL, label="Line Width", command=update_line_width)
line_width_scale.pack()

# Create the "Save" button
save_button = tk.Button(window, text='Save', command=save)
save_button.pack(side=tk.LEFT)

# Create the "Clear" button
clear_button = tk.Button(window, text='Clear', command=clear)
clear_button.pack(side=tk.LEFT)

# Run the window
window.mainloop()
