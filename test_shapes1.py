import Shapes_numpy

rect = Shapes_numpy.My_Rectangle(ref_pt="bot_left", ref_pt_x=100, ref_pt_y=100, width=25, height=25)
rect2 = Shapes_numpy.My_Rectangle(ref_pt="center", ref_pt_x=100, ref_pt_y=100, width=25, height=25)
ellipse = Shapes_numpy.My_Ellipse(ref_pt="center", ref_pt_x=100, ref_pt_y=100, x_radius=25, y_radius=50)

path(rect.connect_pts(), stroke="brown", stroke_width=5)
path(rect2.connect_pts())
path(ellipse.connect_pts(), stroke="blue", stroke_width=5)
