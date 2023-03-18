import cairo
import numpy as np
import Shapes

# Set WIDTH and HEIGHT of context
WIDTH = 1000
HEIGHT = 1000

# Create the surface on which all objects are drawn
surface = cairo.ImageSurface(cairo.FORMAT_RGB24, WIDTH, HEIGHT)
# Create the context which holds the objects and surface
context = cairo.Context(surface)

# Flip the context/surface y-axis (pycairo (0,0) is at top-left by default)
# (0,0) is now at bot-left
matrix = cairo.Matrix(yy=-1, y0=surface.get_height())
context.transform(matrix)

rect1 = Shapes.My_Rectangle(
    ref_pt="center", ref_pt_x=50, ref_pt_y=950, width=25, height=25
)

for color1 in np.linspace(0, 1, 11):
    rect1.draw_fill(context, color1, 1, 1)
    rect1.translate(0, -50)

rect1.move_xy("center", 50, 950)
rect1.translate(50, 0)

for color2 in np.linspace(0, 1, 11):
    rect1.draw_fill(context, 1, color2, 1)
    rect1.translate(0, -50)

rect1.move_xy("center", 50, 950)
rect1.translate(100, 0)

for color3 in np.linspace(0, 1, 11):
    rect1.draw_fill(context, 1, 1, color3)
    rect1.translate(0, -50)

surface.write_to_png("C:\\Users\\jon21\\Downloads\\test.png")
