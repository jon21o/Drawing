import Shapes_numpy

rect = Shapes_numpy.My_Rectangle(
    ref_pt="mid_bot", ref_pt_x=100, ref_pt_y=100, width=100, height=100
)

print(rect.bbox_right_x)
print(rect.bbox_left_x)

# rect1_obj = path(
#     rect.connect_pts(),
#     mask=mask([canvas_rect_obj, ellipse1_obj]),
#     stroke="none",
#     fill="blue",
# )
