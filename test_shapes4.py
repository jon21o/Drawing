import Shapes_numpy

rect1 = Shapes_numpy.My_Rectangle(
    ref_pt="mid_top", ref_pt_x=100, ref_pt_y=100, width=150, height=50
)

rect1_drawn = path(rect1.connect_pts(), stroke="none", fill="blue")

rect2 = Shapes_numpy.My_Rectangle(
    ref_pt="mid_top", ref_pt_x=80, ref_pt_y=100, width=60, height=30
)

rect2_drawn = path(rect1.connect_pts(), stroke="none", fill="blue")


trap = Shapes_numpy.My_Trapezoid(
    ref_pt="mid_top",
    ref_pt_x=80,
    ref_pt_y=100,
    top_width=30,
    bot_width=10,
    height=20,
)

trap_group = group()
trap_drawn1 = path(trap.connect_pts(), stroke="none", fill="red")
trap_group.append(trap_drawn1)
trap.translate(20, 0)
trap_drawn2 = path(trap.connect_pts(), stroke="none", fill="red")
trap_group.append(trap_drawn2)

objs = apply_path_operation("difference", [rect1_drawn, trap_drawn1])
objs = apply_path_operation("difference", [objs[0], trap_drawn2])

obj2 = duplicate(objs[0], fill="green")

# path_difference = path_effect(
#     "Boolean_operation", operand_path=trap_drawn, operation="difference"
# )
# rect_drawn.apply_path_effect(path_difference)

path_offset = path_effect("offset", offset=1)
obj2.apply_path_effect(path_offset)
obj2.z_order("bottom")
