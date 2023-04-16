import Shapes_numpy

start_x = 100
start_y = 100
positions = {new_list: [] for new_list in ["rect", "par", "trap", "circ", "ellipse", "right_tri", "isos_tri", "equil_tri"]}
for position in ["center", "bot_left", "mid_left", "top_left", "mid_top", "top_right", "mid_right", "bot_right", "mid_bot"]:
    rect = Shapes_numpy.My_Rectangle(ref_pt=position, ref_pt_x=start_x, ref_pt_y=start_y, width=15, height=20)
    path(rect.connect_pts())
    path(rect.connect_bbox_pts())
    positions["rect"].append([start_x, start_y])
    par = Shapes_numpy.My_Parallelogram(ref_pt=position, ref_pt_x=start_x, ref_pt_y=start_y + 100, width=25, angle=15, height=20)
    path(par.connect_pts())
    path(par.connect_bbox_pts())
    positions["par"].append([start_x, start_y + 100])
    trap = Shapes_numpy.My_Trapezoid(ref_pt=position, ref_pt_x=start_x, ref_pt_y=start_y + 200, bot_width=25, angle=15, height=20)
    path(trap.connect_pts())
    path(trap.connect_bbox_pts())
    positions["trap"].append([start_x, start_y + 200])
    circ = Shapes_numpy.My_Circle(ref_pt=position, ref_pt_x=start_x, ref_pt_y=start_y + 300, radius=20)
    path(circ.connect_pts())
    path(circ.connect_bbox_pts())
    positions["circ"].append([start_x, start_y + 300])
    ellipse = Shapes_numpy.My_Ellipse(ref_pt=position, ref_pt_x=start_x, ref_pt_y=start_y + 400, x_radius=20, y_radius=10)
    path(ellipse.connect_pts())
    path(ellipse.connect_bbox_pts())
    positions["ellipse"].append([start_x, start_y + 400])
    if (position == "mid_top") | (position == "top_right"):
        pass
    else:
        right_tri = Shapes_numpy.My_RightTriangle(ref_pt=position, ref_pt_x=start_x, ref_pt_y=start_y + 500, width=25, height=20)
        path(right_tri.connect_pts())
        path(right_tri.connect_bbox_pts())
        positions["right_tri"].append([start_x, start_y + 500])
    if (position == "top_left") | (position == "top_right"):
        pass
    else:
        isos_tri = Shapes_numpy.My_IsoscelesTriangle(ref_pt=position, ref_pt_x=start_x, ref_pt_y=start_y + 600, width=25, height=40)
        path(isos_tri.connect_pts())
        path(isos_tri.connect_bbox_pts())
        positions["isos_tri"].append([start_x, start_y + 600])
    if (position == "top_left") | (position == "top_right"):
        pass
    else:
        equil_tri = Shapes_numpy.My_EquilateralTriangle(ref_pt=position, ref_pt_x=start_x, ref_pt_y=start_y + 700, width=25)
        path(equil_tri.connect_pts())
        path(equil_tri.connect_bbox_pts())
        positions["equil_tri"].append([start_x, start_y + 700])
    start_x += 100

shapes = ["rect", "par", "trap", "circ", "ellipse", "right_tri", "isos_tri", "equil_tri"]
colors = ["purple", "blue", "green", "cyan", "orange", "yellow", "pink", "blue"]

for shape in shapes:
    ref_pts = Shapes_numpy.My_Line_Coord(positions[shape])
    for pt in range(ref_pts._matrix.shape[1] - 1):
        path(ref_pts.connect_segment(pt, pt + 1), stroke=colors[pt])
