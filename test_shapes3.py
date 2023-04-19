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

# because we can't use handles to objects more than once in clip/masks, lets set the handles to
# strings which we can evaluate in the script later at multiple points, so we don't have to
# retype these all the time
canvas_mask_white = (
    'path(canvas_rect.connect_pts(), stroke="none", fill="white")'
)
canvas_mask_black = (
    'path(canvas_rect.connect_pts(), stroke="none", fill="black")'
)
ellipse_full_fill_clip = "path(ellipse.connect_pts())"
ellipse_outline2_mask_black = 'path(ellipse.connect_pts(),stroke="black",stroke_width=5,fill="none",clip_path=clip_path(eval(ellipse_full_fill_clip)),mask=mask([eval(canvas_mask_black),eval(rect_background_mask_white),eval(line2_top_mask_white),]),)'
rect_background_clip = "path(rect.connect_pts())"
rect_background_mask_white = (
    'path(rect.connect_pts(), stroke="none", fill="white")'
)
rect_background_mask_black = (
    'path(rect.connect_pts(), stroke="none", fill="black")'
)
line1_top_mask_white = (
    'path(line_top.connect_pts(), stroke="white", stroke_width=2.5)'
)
line1_top_mask_black = (
    'path(line_top.connect_pts(), stroke="black", stroke_width=2.5)'
)
line2_top_mask_white = (
    'path(line_top.connect_pts(), stroke="white", stroke_width=5)'
)
line2_top_mask_black = (
    'path(line_top.connect_pts(), stroke="black", stroke_width=5)'
)
line1_bot_mask_white = (
    'path(line_bot.connect_pts(), stroke="white", stroke_width=2.5)'
)
line1_bot_mask_black = (
    'path(line_bot.connect_pts(), stroke="black", stroke_width=2.5)'
)
line2_bot_mask_white = (
    'path(line_bot.connect_pts(), stroke="white", stroke_width=5)'
)
line2_bot_mask_black = (
    'path(line_bot.connect_pts(), stroke="black", stroke_width=5)'
)

# actually start drawing objects on canvas
# when an object is used in a clip/mask, it is removed from the canvas at that time
# and cannot be used again

# here we will draw a background filled rectangle with an unfilled ellipse on top of it
# the goal is to have two lines of varying line width follow the path of the filled rectangle
# and unfilled ellipse, with one line laying on top of the other

# shapes are drawn in order, with ones drawn later drawn on top of ones drawn earlier

# draw the background rectangle
path(rect.connect_pts(), stroke="none", fill="red")
# draw a line on top of the rectangle, this will go behind our ellipse
path(
    line_top.connect_pts(),
    stroke="green",
    stroke_width=2.5,
    mask=mask([eval(canvas_mask_white), eval(rect_background_mask_black)]),
)
# draw the first ellipse outline, which simulates our first line following the inner part of the ellipse
path(
    ellipse.connect_pts(),
    stroke="blue",
    stroke_width=5,
    fill="none",
    clip_path=clip_path(eval(ellipse_full_fill_clip)),
    mask=mask(
        [
            eval(canvas_mask_black),
            eval(rect_background_mask_white),
            eval(line2_top_mask_white),
        ]
    ),
)
# draw the first line on the bottom
path(
    line_bot.connect_pts(),
    stroke="blue",
    stroke_width=5,
    clip_path=clip_path(eval(ellipse_full_fill_clip)),
    mask=mask([eval(canvas_mask_black), eval(rect_background_mask_white),]),
)
# draw the second ellipse outline, which simulates our 2nd line following the inner part of the ellipse
path(
    ellipse.connect_pts(),
    stroke="green",
    stroke_width=2.5,
    fill="none",
    clip_path=clip_path(eval(ellipse_full_fill_clip)),
    mask=mask(
        [
            eval(canvas_mask_black),
            eval(rect_background_mask_white),
            eval(line2_top_mask_white),
        ]
    ),
)
# draw the second line on top
path(
    line_top.connect_pts(),
    stroke="blue",
    stroke_width=5,
    mask=mask(
        [
            eval(canvas_mask_white),
            eval(line1_top_mask_black),
            eval(rect_background_mask_black),
        ]
    ),
)
# draw the second line on the bottom
path(
    line_bot.connect_pts(),
    stroke="green",
    stroke_width=2.5,
    clip_path=clip_path(eval(ellipse_full_fill_clip)),
    mask=mask([eval(canvas_mask_black), eval(rect_background_mask_white),]),
)
# we need to mask what's left of the inner part of our ellipse
path(
    ellipse.connect_pts(),
    stroke="none",
    fill="white",
    mask=mask(
        [
            eval(canvas_mask_white),
            eval(rect_background_mask_white),
            eval(ellipse_outline2_mask_black),
            eval(line2_bot_mask_black),
        ]
    ),
)

save_file("C:\\Users\\jon21\\Downloads\\test.png")
