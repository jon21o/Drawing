import cairo
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
    ref_pt="center", ref_pt_x=200, ref_pt_y=200, width=50, height=50
)
rect2 = Shapes.My_Rectangle(
    ref_pt="bot_left",
    ref_pt_x=rect1.top_right.loc[dict(dim="x")],
    ref_pt_y=rect1.top_right.loc[dict(dim="y")],
    width=50,
    height=50,
)

rect1.draw_fill(context, 1, 0, 0)
rect2.draw_fill(context, 0, 1, 0)

surface.write_to_png("C:\\Users\\jon21\\Downloads\\test.png")
