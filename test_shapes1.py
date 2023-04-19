import Shapes_numpy

rect1 = Shapes_numpy.My_Rectangle(
    ref_pt="left_bot", ref_pt_x=10, ref_pt_y=10, width=100, height=20
)

path(
    rect1.connect_pts(), stroke="none", fill="green",
)

rect2 = Shapes_numpy.My_Rectangle(
    ref_pt="left_bot",
    ref_pt_x=rect1.bbox_left_x,
    ref_pt_y=rect1.bbox_bot_y + 1,
    width=100,
    height=1,
)

while rect2.bbox_top_y < rect1.bbox_top_y:
    path(
        rect2.connect_pts(), stroke="none", fill="brown",
    )
    rect2.translate(0, 2)

trap1 = Shapes_numpy.My_Trapezoid(
    ref_pt="top_left",
    ref_pt_x=rect1.bbox_left_x + 3,
    ref_pt_y=rect1.bbox_top_y,
    bot_width=3,
    top_width=5,
    height=10,
)

path(
    trap1.connect_pts(), stroke="none", fill="white",
)

line1 = Shapes_numpy.My_Line_Coord(
    [
        [rect1.bbox_left_x, rect1.bbox_top_y],
        [trap1.pts_x["pt1"], trap1.pts_y["pt1"]],
        [trap1.pts_x["pt0"], trap1.pts_y["pt0"]],
        [trap1.pts_x["pt3"], trap1.pts_y["pt3"]],
        [trap1.pts_x["pt2"], trap1.pts_y["pt2"]],
        [rect1.bbox_right_x, rect1.bbox_top_y],
    ]
)

path(
    line1.connect_pts(), stroke="blue", stroke_width=1,
)
