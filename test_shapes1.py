import Shapes_numpy

# create objects but don't draw them yet
# this gives us the matrices so we can refer to points
canvas_rect = Shapes_numpy.My_Rectangle(
    ref_pt="center",
    ref_pt_x=canvas.width / 2,
    ref_pt_y=canvas.height / 2,
    width=canvas.width,
    height=canvas.height,
)
rect = Shapes_numpy.My_Rectangle(
    ref_pt="center", ref_pt_x=100, ref_pt_y=100, width=100, height=100
)
line_top = Shapes_numpy.My_Line_Coord(
    [[rect.bbox_left_x, rect.bbox_top_y], [rect.bbox_right_x, rect.bbox_top_y]]
)
line_bot = Shapes_numpy.My_Line_Coord(
    [[rect.bbox_left_x, rect.bbox_bot_y], [rect.bbox_right_x, rect.bbox_bot_y]]
)
ellipse = Shapes_numpy.My_Ellipse(
    ref_pt="center", ref_pt_x=100, ref_pt_y=100, x_radius=25, y_radius=75
)

# start drawing objects on canvas, but get handles to them for clipping/masking with other objects
# when an object is used in a clip/mask, it is removed from the canvas at that time
canvas_rect_obj = path(canvas_rect.connect_pts(), stroke="none", fill="white")
ellipse1_obj = path(
    ellipse.connect_pts(), stroke="black", stroke_width=5, fill="white"
)
rect1_obj = path(
    rect.connect_pts(),
    mask=mask([canvas_rect_obj, ellipse1_obj]),
    stroke="none",
    fill="blue",
)

ellipse.translate(50, 0)
ellipse2_obj = path(
    ellipse.connect_pts(), stroke="black", stroke_width=5, fill="none"
)
