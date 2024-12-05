import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle, FancyArrow

# Initialize figure and axis
fig, ax = plt.subplots(figsize=(12, 6))
ax.set_xlim(0, 12)
ax.set_ylim(0, 6)
ax.axis('off')

# Function to draw labeled rectangles (components)
def add_component(ax, xy, width, height, label, color='lightblue'):
    x, y = xy
    rect = Rectangle((x, y), width, height, edgecolor='black', facecolor=color, lw=1.5)
    ax.add_patch(rect)
    ax.text(x + width / 2, y + height / 2, label, ha='center', va='center', fontsize=10, wrap=True)

# Function to add arrows (connections)
def add_arrow(ax, start, end, text=None):
    ax.annotate(text, xy=end, xytext=start, arrowprops=dict(arrowstyle="->", lw=1.5), fontsize=10, ha='center')

# Add components
add_component(ax, (1, 4), 2, 1, "Manure Collection", "lightgreen")
add_component(ax, (4, 4), 2, 1, "Anaerobic Digestion Tank", "lightblue")
add_component(ax, (7, 4), 2, 1, "Gas Storage Tank", "orange")
add_component(ax, (10, 4), 2, 1, "Biogas Engine", "lightyellow")
add_component(ax, (4, 1), 2, 1, "Digestate Storage", "brown")

# Add arrows to indicate flow
add_arrow(ax, (3, 4.5), (4, 4.5), "Manure")
add_arrow(ax, (6, 4.5), (7, 4.5), "Biogas")
add_arrow(ax, (9, 4.5), (10, 4.5), "Biogas to Engine")
add_arrow(ax, (5, 4), (5, 2), "By-product")

# Add labels for final energy
ax.text(11.5, 4.5, "Energy Output", fontsize=10, ha='left', va='center', color='black')

# Display diagram
plt.title("Simplified Biogas Plant Diagram", fontsize=14)
plt.show()
