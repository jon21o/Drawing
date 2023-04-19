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

# line_width = 2

# rect3 = Shapes_numpy.My_Rectangle(
#     ref_pt="bot_left",
#     ref_pt_x=rect1.bbox_left_x,
#     ref_pt_y=rect1.bbox_top_y,
#     width=rect1.width,
#     height=line_width,
# )

# path(
#     rect3.connect_pts(), stroke="none", fill="blue",
# )


# rect4 = Shapes_numpy.My_Rectangle(
#     ref_pt="top_left",
#     ref_pt_x=rect1.bbox_left_x + 3,
#     ref_pt_y=rect1.bbox_top_y + line_width,
#     width=5,
#     height=10,
# )

# path(
#     rect4.connect_pts(), stroke="none", fill="blue",
# )

# rect5 = Shapes_numpy.My_Rectangle(
#     ref_pt="mid_bot",
#     ref_pt_x=rect4.centroid_x,
#     ref_pt_y=rect4.bbox_bot_y + line_width,
#     width=rect4.width - line_width * 2,
#     height=rect4.height - line_width,
# )

# path(
#     rect5.connect_pts(), stroke="none", fill="white",
# )


def box_with_fill(
    on_obj,
    x_offset_from_left,
    width_of_box,
    height_of_box,
    line_width=2,
    line_color="blue",
):
    shapes = []
    shapes.append(
        Shapes_numpy.My_Rectangle(
            ref_pt="bot_left",
            ref_pt_x=on_obj.bbox_left_x,
            ref_pt_y=on_obj.bbox_top_y,
            width=on_obj.width,
            height=line_width,
        )
    )

    path(
        shapes[0].connect_pts(), stroke="none", fill=line_color,
    )

    shapes.append(
        Shapes_numpy.My_Rectangle(
            ref_pt="mid_top",
            ref_pt_x=on_obj.bbox_left_x + x_offset_from_left,
            ref_pt_y=on_obj.bbox_top_y + line_width,
            width=width_of_box,
            height=height_of_box,
        )
    )

    path(
        shapes[1].connect_pts(), stroke="none", fill=line_color,
    )

    shapes.append(
        Shapes_numpy.My_Rectangle(
            ref_pt="mid_bot",
            ref_pt_x=shapes[1].centroid_x,
            ref_pt_y=shapes[1].bbox_bot_y + line_width,
            width=shapes[1].width - line_width * 2,
            height=shapes[1].height - line_width,
        )
    )

    path(
        shapes[2].connect_pts(), stroke="none", fill="white",
    )
    return shapes


shapes1 = box_with_fill(
    on_obj=rect1,
    x_offset_from_left=30,
    width_of_box=15,
    height_of_box=10,
    line_width=1,
)
