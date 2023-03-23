import math
import numpy as np
import xarray as xr
import cairo


def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def scale_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])


def rotation_matrix(r):
    return np.array([[math.cos(math.radians(r)), -math.sin(math.radians(r)), 0], [math.sin(math.radians(r)), math.cos(math.radians(r)), 0], [0, 0, 1],])


bezier_curve_approx = (4 / 3) * math.tan(math.pi / 8)


class My_Shape:
    def __init__(self, xy_pairs, rotation=0):
        # xy_pairs = [[x1, y1], [x2, y2], [x3, y3], etc.]
        num_pts = len(xy_pairs)
        self._rotation = rotation
        self._matrix = xr.DataArray(
            np.ones((3, num_pts)), dims=["dim", "pt"], coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, num_pts + 1)],},
        )
        for i, xy_pair in enumerate(xy_pairs):
            self._matrix.loc[dict(dim="x", pt="pt{}".format(i + 1))] = xy_pair[0]
            self._matrix.loc[dict(dim="y", pt="pt{}".format(i + 1))] = xy_pair[1]

    # region #### Functions
    def translate(self, tx, ty):
        self._matrix = xr.DataArray(
            translation_matrix(tx, ty) @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)],},
        )

    def scale(self, sx, sy):
        self._matrix = xr.DataArray(
            translation_matrix(self.centroid.loc[dict(dim="x")].values, self.centroid.loc[dict(dim="y")].values,)
            @ scale_matrix(sx, sy)
            @ translation_matrix(-self.centroid.loc[dict(dim="x")].values, -self.centroid.loc[dict(dim="y")].values,)
            @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)],},
        )

    def rotate(self, r):
        self._matrix = xr.DataArray(
            translation_matrix(self.centroid.loc[dict(dim="x")].values, self.centroid.loc[dict(dim="y")].values,)
            @ rotation_matrix(r)
            @ translation_matrix(-self.centroid.loc[dict(dim="x")].values, -self.centroid.loc[dict(dim="y")].values,)
            @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)],},
        )
        self._rotation += r

    def move_bbox_xy_to(self, ref_pt, ref_pt_x, ref_pt_y):
        if ref_pt == "center":
            self.translate(
                ref_pt_x - self.centroid.loc[dict(dim="x")], ref_pt_y - self.centroid.loc[dict(dim="y")],
            )
        if ref_pt == "bot_left":
            self.translate(
                ref_pt_x - self.bbox_bot_left.loc[dict(dim="x")], ref_pt_y - self.bbox_bot_left.loc[dict(dim="y")],
            )
        if ref_pt == "mid_left":
            self.translate(
                ref_pt_x - self.bbox_mid_left.loc[dict(dim="x")], ref_pt_y - self.bbox_mid_left.loc[dict(dim="y")],
            )
        if ref_pt == "top_left":
            self.translate(
                ref_pt_x - self.bbox_top_left.loc[dict(dim="x")], ref_pt_y - self.bbox_top_left.loc[dict(dim="y")],
            )
        if ref_pt == "mid_top":
            self.translate(
                ref_pt_x - self.bbox_mid_top.loc[dict(dim="x")], ref_pt_y - self.bbox_mid_top.loc[dict(dim="y")],
            )
        if ref_pt == "top_right":
            self.translate(
                ref_pt_x - self.bbox_top_right.loc[dict(dim="x")], ref_pt_y - self.bbox_top_right.loc[dict(dim="y")],
            )
        if ref_pt == "mid_right":
            self.translate(
                ref_pt_x - self.bbox_mid_right.loc[dict(dim="x")], ref_pt_y - self.bbox_mid_right.loc[dict(dim="y")],
            )
        if ref_pt == "bot_right":
            self.translate(
                ref_pt_x - self.bbox_bot_right.loc[dict(dim="x")], ref_pt_y - self.bbox_bot_right.loc[dict(dim="y")],
            )
        if ref_pt == "mid_bot":
            self.translate(
                ref_pt_x - self.bbox_mid_bot.loc[dict(dim="x")], ref_pt_y - self.bbox_mid_bot.loc[dict(dim="y")],
            )

    def move_bbox_x_to(self, ref_pt, ref_pt_x):
        if ref_pt == "center":
            self.translate(
                ref_pt_x - self.centroid.loc[dict(dim="x")], 0,
            )
        if ref_pt == "bot_left":
            self.translate(
                ref_pt_x - self.bbox_bot_left.loc[dict(dim="x")], 0,
            )
        if ref_pt == "mid_left":
            self.translate(
                ref_pt_x - self.bbox_mid_left.loc[dict(dim="x")], 0,
            )
        if ref_pt == "top_left":
            self.translate(
                ref_pt_x - self.bbox_top_left.loc[dict(dim="x")], 0,
            )
        if ref_pt == "mid_top":
            self.translate(
                ref_pt_x - self.bbox_mid_top.loc[dict(dim="x")], 0,
            )
        if ref_pt == "top_right":
            self.translate(
                ref_pt_x - self.bbox_top_right.loc[dict(dim="x")], 0,
            )
        if ref_pt == "mid_right":
            self.translate(
                ref_pt_x - self.bbox_mid_right.loc[dict(dim="x")], 0,
            )
        if ref_pt == "bot_right":
            self.translate(
                ref_pt_x - self.bbox_bot_right.loc[dict(dim="x")], 0,
            )
        if ref_pt == "mid_bot":
            self.translate(
                ref_pt_x - self.bbox_mid_bot.loc[dict(dim="x")], 0,
            )

    def move_bbox_y_to(self, ref_pt, ref_pt_y):
        if ref_pt == "center":
            self.translate(
                0, ref_pt_y - self.centroid.loc[dict(dim="y")],
            )
        if ref_pt == "bot_left":
            self.translate(
                0, ref_pt_y - self.bbox_bot_left.loc[dict(dim="y")],
            )
        if ref_pt == "mid_left":
            self.translate(
                0, ref_pt_y - self.bbox_mid_left.loc[dict(dim="y")],
            )
        if ref_pt == "top_left":
            self.translate(
                0, ref_pt_y - self.bbox_top_left.loc[dict(dim="y")],
            )
        if ref_pt == "mid_top":
            self.translate(
                0, ref_pt_y - self.bbox_mid_top.loc[dict(dim="y")],
            )
        if ref_pt == "top_right":
            self.translate(
                0, ref_pt_y - self.bbox_top_right.loc[dict(dim="y")],
            )
        if ref_pt == "mid_right":
            self.translate(
                0, ref_pt_y - self.bbox_mid_right.loc[dict(dim="y")],
            )
        if ref_pt == "bot_right":
            self.translate(
                0, ref_pt_y - self.bbox_bot_right.loc[dict(dim="y")],
            )
        if ref_pt == "mid_bot":
            self.translate(
                0, ref_pt_y - self.bbox_mid_bot.loc[dict(dim="y")],
            )

    def draw_stroke(self, context, color1, color2, color3):
        context.set_source_rgb(color1, color2, color3)
        for pt in self._matrix.coords["pt"]:
            if pt.values == "pt1":
                context.move_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values, self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
            else:
                context.line_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values, self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
        context.stroke()

    def draw_fill(self, context, color1, color2, color3):
        context.set_source_rgb(color1, color2, color3)
        for pt in self._matrix.coords["pt"]:
            if pt.values == "pt1":
                context.move_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values, self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
            else:
                context.line_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values, self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
        context.fill()

    # endregion

    # region #### Properties
    @property
    def centroid(self):
        return self._matrix.mean(dim="pt")

    @property
    def rotation(self):
        return self._rotation

    @property
    def pts(self):
        return {"pt{}".format(pt): self._matrix.loc[dict(pt="pt{}".format(pt))] for pt in range(1, self._matrix.sizes["pt"] + 1)}

    @property
    def mid_pts(self):
        mid_pts_dict = {
            "mid_pt_{}_{}".format(pt, pt + 1): (self._matrix.loc[dict(pt="pt{}".format(pt + 1))] - self._matrix.loc[dict(pt="pt{}".format(pt))]) / 2
            for pt in range(1, self._matrix.sizes["pt"])
        }
        mid_pts_dict["mid_pt_{}_{}".format(1, self._matrix.sizes["pt"])] = (
            self._matrix.loc[dict(pt="pt{}".format(1))] - self._matrix.loc[dict(pt="pt{}".format(self._matrix.sizes["pt"]))]
        ) / 2
        return mid_pts_dict

    @property
    def bbox_bot_left(self):
        return self._matrix.min(dim="pt")

    @property
    def bbox_mid_left(self):
        temp_array = np.array(
            [
                self._matrix.min(dim="pt").loc[dict(dim="x")],
                ((self._matrix.max(dim="pt").loc[dict(dim="y")] - self._matrix.min(dim="pt").loc[dict(dim="y")]) / 2)
                + self._matrix.min(dim="pt").loc[dict(dim="y")],
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_top_left(self):
        temp_array = np.array([self._matrix.min(dim="pt").loc[dict(dim="x")], self._matrix.max(dim="pt").loc[dict(dim="y")], 1,])
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_mid_top(self):
        temp_array = np.array(
            [
                ((self._matrix.max(dim="pt").loc[dict(dim="x")] - self._matrix.min(dim="pt").loc[dict(dim="x")]) / 2)
                + self._matrix.min(dim="pt").loc[dict(dim="x")],
                self._matrix.max(dim="pt").loc[dict(dim="y")],
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_top_right(self):
        return self._matrix.max(dim="pt")

    @property
    def bbox_mid_right(self):
        temp_array = np.array(
            [
                self._matrix.max(dim="pt").loc[dict(dim="x")],
                ((self._matrix.max(dim="pt").loc[dict(dim="y")] - self._matrix.min(dim="pt").loc[dict(dim="y")]) / 2)
                + self._matrix.min(dim="pt").loc[dict(dim="y")],
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_bot_right(self):
        temp_array = np.array([self._matrix.max(dim="pt").loc[dict(dim="x")], self._matrix.min(dim="pt").loc[dict(dim="y")], 1,])
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_mid_bot(self):
        temp_array = np.array(
            [
                ((self._matrix.max(dim="pt").loc[dict(dim="x")] - self._matrix.min(dim="pt").loc[dict(dim="x")]) / 2)
                + self._matrix.min(dim="pt").loc[dict(dim="x")],
                self._matrix.min(dim="pt").loc[dict(dim="y")],
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    # endregion


class My_Rectangle(My_Shape):
    def __init__(self, ref_pt, ref_pt_x, ref_pt_y, width, height, rotation=0):
        # pt1 --> bot left
        # pt2 --> top left
        # pt3 --> top right
        # pt4 --> bot right
        self._rotation = rotation
        self._matrix = xr.DataArray(np.ones((3, 4)), dims=["dim", "pt"], coords={"dim": ["x", "y", "z"], "pt": ["pt1", "pt2", "pt3", "pt4"],},)

        # start with ref_pt = "bot_left" and translate to different ref_pts below
        self._matrix.loc[dict(dim="x", pt="pt1")] = ref_pt_x
        self._matrix.loc[dict(dim="y", pt="pt1")] = ref_pt_y
        self._matrix.loc[dict(dim="x", pt="pt2")] = ref_pt_x
        self._matrix.loc[dict(dim="y", pt="pt2")] = ref_pt_y + height
        self._matrix.loc[dict(dim="x", pt="pt3")] = ref_pt_x + width
        self._matrix.loc[dict(dim="y", pt="pt3")] = ref_pt_y + height
        self._matrix.loc[dict(dim="x", pt="pt4")] = ref_pt_x + width
        self._matrix.loc[dict(dim="y", pt="pt4")] = ref_pt_y

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(-width / 2, -height / 2)
        elif ref_pt == "mid_left":
            self.translate(0, -height / 2)
        elif ref_pt == "top_left":
            self.translate(0, -height)
        elif ref_pt == "mid_top":
            self.translate(-width / 2, -height)
        elif ref_pt == "top_right":
            self.translate(-width, -height)
        elif ref_pt == "mid_right":
            self.translate(-width, -height / 2)
        elif ref_pt == "bot_right":
            self.translate(-width, 0)
        elif ref_pt == "mid_bot":
            self.translate(-width / 2, 0)
        # endregion


class My_Circle(My_Shape):
    def __init__(self, ref_pt, ref_pt_x, ref_pt_y, radius, rotation=0):
        # pt1 --> left
        # pt2 --> top
        # pt3 --> right
        # pt4 --> bot
        self._rotation = rotation
        self._matrix = xr.DataArray(np.ones((3, 4)), dims=["dim", "pt"], coords={"dim": ["x", "y", "z"], "pt": ["pt1", "pt2", "pt3", "pt4"],},)

        # start with ref_pt = "left" and translate to different ref_pts below
        self._matrix.loc[dict(dim="x", pt="pt1")] = ref_pt_x
        self._matrix.loc[dict(dim="y", pt="pt1")] = ref_pt_y
        self._matrix.loc[dict(dim="x", pt="pt2")] = ref_pt_x + radius
        self._matrix.loc[dict(dim="y", pt="pt2")] = ref_pt_y + radius
        self._matrix.loc[dict(dim="x", pt="pt3")] = ref_pt_x + (radius * 2)
        self._matrix.loc[dict(dim="y", pt="pt3")] = ref_pt_y
        self._matrix.loc[dict(dim="x", pt="pt4")] = ref_pt_x + radius
        self._matrix.loc[dict(dim="y", pt="pt4")] = ref_pt_y - radius

        self._control_pts = xr.DataArray(
            np.ones((3, 8)), dims=["dim", "pt"], coords={"dim": ["x", "y", "z"], "pt": ["pt1", "pt2", "pt3", "pt4", "pt5", "pt6", "pt7", "pt8"],},
        )

        self._control_pts.loc[dict(dim="x", pt="pt1")] = self._matrix.loc[dict(dim="x", pt="pt1")]
        self._control_pts.loc[dict(dim="y", pt="pt1")] = self._matrix.loc[dict(dim="y", pt="pt1")] + (bezier_curve_approx * self.y_radius)
        self._control_pts.loc[dict(dim="x", pt="pt2")] = self._matrix.loc[dict(dim="x", pt="pt2")] - (bezier_curve_approx * self.x_radius)
        self._control_pts.loc[dict(dim="y", pt="pt2")] = self._matrix.loc[dict(dim="y", pt="pt2")]
        self._control_pts.loc[dict(dim="x", pt="pt3")] = (bezier_curve_approx * self.x_radius) + self._matrix.loc[dict(dim="x", pt="pt2")]
        self._control_pts.loc[dict(dim="y", pt="pt3")] = self._matrix.loc[dict(dim="y", pt="pt2")]
        self._control_pts.loc[dict(dim="x", pt="pt4")] = self._matrix.loc[dict(dim="x", pt="pt3")]
        self._control_pts.loc[dict(dim="y", pt="pt4")] = self._matrix.loc[dict(dim="y", pt="pt3")] + (bezier_curve_approx * self.y_radius)
        self._control_pts.loc[dict(dim="x", pt="pt5")] = self._matrix.loc[dict(dim="x", pt="pt3")]
        self._control_pts.loc[dict(dim="y", pt="pt5")] = self._matrix.loc[dict(dim="y", pt="pt3")] - (bezier_curve_approx * self.y_radius)
        self._control_pts.loc[dict(dim="x", pt="pt6")] = self._matrix.loc[dict(dim="x", pt="pt4")] + (bezier_curve_approx * self.x_radius)
        self._control_pts.loc[dict(dim="y", pt="pt6")] = self._matrix.loc[dict(dim="y", pt="pt4")]
        self._control_pts.loc[dict(dim="x", pt="pt7")] = self._matrix.loc[dict(dim="x", pt="pt4")] - (bezier_curve_approx * self.x_radius)
        self._control_pts.loc[dict(dim="y", pt="pt7")] = self._matrix.loc[dict(dim="y", pt="pt4")]
        self._control_pts.loc[dict(dim="x", pt="pt8")] = self._matrix.loc[dict(dim="x", pt="pt1")]
        self._control_pts.loc[dict(dim="y", pt="pt8")] = self._matrix.loc[dict(dim="y", pt="pt1")] - (bezier_curve_approx * self.y_radius)

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(-radius, 0)
        elif ref_pt == "top":
            self.translate(-radius, -radius)
        elif ref_pt == "right":
            self.translate(-2 * radius, 0)
        elif ref_pt == "bot":
            self.translate(radius, radius)
        # endregion

    def translate(self, tx, ty):
        self._matrix = xr.DataArray(
            translation_matrix(tx, ty) @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)],},
        )
        self._control_pts = xr.DataArray(
            translation_matrix(tx, ty) @ self._control_pts.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._control_pts.sizes["pt"] + 1)],},
        )

    def scale(self, sx, sy):
        self._matrix = xr.DataArray(
            translation_matrix(self.centroid.loc[dict(dim="x")].values, self.centroid.loc[dict(dim="y")].values,)
            @ scale_matrix(sx, sy)
            @ translation_matrix(-self.centroid.loc[dict(dim="x")].values, -self.centroid.loc[dict(dim="y")].values,)
            @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)],},
        )
        self._control_pts = xr.DataArray(
            translation_matrix(self.centroid.loc[dict(dim="x")].values, self.centroid.loc[dict(dim="y")].values,)
            @ scale_matrix(sx, sy)
            @ translation_matrix(-self.centroid.loc[dict(dim="x")].values, -self.centroid.loc[dict(dim="y")].values,)
            @ self._control_pts.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._control_pts.sizes["pt"] + 1)],},
        )

    def rotate(self, r):
        self._matrix = xr.DataArray(
            translation_matrix(self.centroid.loc[dict(dim="x")].values, self.centroid.loc[dict(dim="y")].values,)
            @ rotation_matrix(r)
            @ translation_matrix(-self.centroid.loc[dict(dim="x")].values, -self.centroid.loc[dict(dim="y")].values,)
            @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)],},
        )
        self._control_pts = xr.DataArray(
            translation_matrix(self.centroid.loc[dict(dim="x")].values, self.centroid.loc[dict(dim="y")].values,)
            @ rotation_matrix(r)
            @ translation_matrix(-self.centroid.loc[dict(dim="x")].values, -self.centroid.loc[dict(dim="y")].values,)
            @ self._control_pts.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt{}".format(x) for x in range(1, self._control_pts.sizes["pt"] + 1)],},
        )
        self._rotation += r

    def draw_fill(self, context, color1, color2, color3):
        context.set_source_rgb(color1, color2, color3)
        context.move_to(
            self._matrix.loc[dict(dim="x", pt="pt1")].values, self._matrix.loc[dict(dim="y", pt="pt1")].values,
        )
        context.curve_to(
            self._control_pts.loc[dict(dim="x", pt="pt1")],
            self._control_pts.loc[dict(dim="y", pt="pt1")],
            self._control_pts.loc[dict(dim="x", pt="pt2")],
            self._control_pts.loc[dict(dim="y", pt="pt2")],
            self._matrix.loc[dict(dim="x", pt="pt2")].values,
            self._matrix.loc[dict(dim="y", pt="pt2")].values,
        )
        context.curve_to(
            self._control_pts.loc[dict(dim="x", pt="pt3")],
            self._control_pts.loc[dict(dim="y", pt="pt3")],
            self._control_pts.loc[dict(dim="x", pt="pt4")],
            self._control_pts.loc[dict(dim="y", pt="pt4")],
            self._matrix.loc[dict(dim="x", pt="pt3")].values,
            self._matrix.loc[dict(dim="y", pt="pt3")].values,
        )
        context.curve_to(
            self._control_pts.loc[dict(dim="x", pt="pt5")],
            self._control_pts.loc[dict(dim="y", pt="pt5")],
            self._control_pts.loc[dict(dim="x", pt="pt6")],
            self._control_pts.loc[dict(dim="y", pt="pt6")],
            self._matrix.loc[dict(dim="x", pt="pt4")].values,
            self._matrix.loc[dict(dim="y", pt="pt4")].values,
        )
        context.curve_to(
            self._control_pts.loc[dict(dim="x", pt="pt7")],
            self._control_pts.loc[dict(dim="y", pt="pt7")],
            self._control_pts.loc[dict(dim="x", pt="pt8")],
            self._control_pts.loc[dict(dim="y", pt="pt8")],
            self._matrix.loc[dict(dim="x", pt="pt1")].values,
            self._matrix.loc[dict(dim="y", pt="pt1")].values,
        )
        context.fill()

    def draw_stroke(self, context, color1, color2, color3):
        context.set_source_rgb(color1, color2, color3)
        context.move_to(
            self._matrix.loc[dict(dim="x", pt="pt1")].values, self._matrix.loc[dict(dim="y", pt="pt1")].values,
        )
        context.curve_to(
            self._control_pts.loc[dict(dim="x", pt="pt1")],
            self._control_pts.loc[dict(dim="y", pt="pt1")],
            self._control_pts.loc[dict(dim="x", pt="pt2")],
            self._control_pts.loc[dict(dim="y", pt="pt2")],
            self._matrix.loc[dict(dim="x", pt="pt2")].values,
            self._matrix.loc[dict(dim="y", pt="pt2")].values,
        )
        context.curve_to(
            self._control_pts.loc[dict(dim="x", pt="pt3")],
            self._control_pts.loc[dict(dim="y", pt="pt3")],
            self._control_pts.loc[dict(dim="x", pt="pt4")],
            self._control_pts.loc[dict(dim="y", pt="pt4")],
            self._matrix.loc[dict(dim="x", pt="pt3")].values,
            self._matrix.loc[dict(dim="y", pt="pt3")].values,
        )
        context.curve_to(
            self._control_pts.loc[dict(dim="x", pt="pt5")],
            self._control_pts.loc[dict(dim="y", pt="pt5")],
            self._control_pts.loc[dict(dim="x", pt="pt6")],
            self._control_pts.loc[dict(dim="y", pt="pt6")],
            self._matrix.loc[dict(dim="x", pt="pt4")].values,
            self._matrix.loc[dict(dim="y", pt="pt4")].values,
        )
        context.curve_to(
            self._control_pts.loc[dict(dim="x", pt="pt7")],
            self._control_pts.loc[dict(dim="y", pt="pt7")],
            self._control_pts.loc[dict(dim="x", pt="pt8")],
            self._control_pts.loc[dict(dim="y", pt="pt8")],
            self._matrix.loc[dict(dim="x", pt="pt1")].values,
            self._matrix.loc[dict(dim="y", pt="pt1")].values,
        )
        context.stroke()

    @property
    def radius(self):
        current_rotation = self._rotation
        self.rotate(-current_rotation)
        radius = abs(self.centroid.loc[dict(dim="x")] - self._matrix.loc[dict(dim="x", pt="pt1")])
        self.rotate(current_rotation)
        return radius

    @property
    def x_radius(self):
        current_rotation = self._rotation
        self.rotate(-current_rotation)
        x_radius = abs(self.centroid.loc[dict(dim="x")] - self._matrix.loc[dict(dim="x", pt="pt1")])
        self.rotate(current_rotation)
        return x_radius

    @property
    def y_radius(self):
        current_rotation = self._rotation
        self.rotate(-current_rotation)
        y_radius = abs(self.centroid.loc[dict(dim="y")] - self._matrix.loc[dict(dim="y", pt="pt2")])
        self.rotate(current_rotation)
        return y_radius

    @property
    def bbox_bot_left(self):
        temp_array = np.array(
            [
                self.centroid.loc[dict(dim="x")]
                - math.sqrt(
                    self.x_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2
                ),
                self.centroid.loc[dict(dim="y")]
                - math.sqrt(
                    self.x_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2
                ),
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_mid_left(self):
        temp_array = np.array(
            [
                self.centroid.loc[dict(dim="x")]
                - math.sqrt(
                    self.x_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2
                ),
                self.centroid.loc[dict(dim="y")],
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_top_left(self):
        temp_array = np.array(
            [
                self.centroid.loc[dict(dim="x")]
                - math.sqrt(
                    self.x_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2
                ),
                self.centroid.loc[dict(dim="y")]
                + math.sqrt(
                    self.x_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2
                ),
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_mid_top(self):
        temp_array = np.array(
            [
                self.centroid.loc[dict(dim="x")],
                self.centroid.loc[dict(dim="y")]
                + math.sqrt(
                    self.x_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2
                ),
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_top_right(self):
        temp_array = np.array(
            [
                self.centroid.loc[dict(dim="x")]
                + math.sqrt(
                    self.x_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2
                ),
                self.centroid.loc[dict(dim="y")]
                + math.sqrt(
                    self.x_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2
                ),
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_mid_right(self):
        temp_array = np.array(
            [
                self.centroid.loc[dict(dim="x")]
                + math.sqrt(
                    self.x_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2
                ),
                self.centroid.loc[dict(dim="y")],
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_bot_right(self):
        temp_array = np.array(
            [
                self.centroid.loc[dict(dim="x")]
                + math.sqrt(
                    self.x_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2
                ),
                self.centroid.loc[dict(dim="y")]
                - math.sqrt(
                    self.x_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2
                ),
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)

    @property
    def bbox_mid_bot(self):
        temp_array = np.array(
            [
                self.centroid.loc[dict(dim="x")],
                self.centroid.loc[dict(dim="y")]
                - math.sqrt(
                    self.x_radius ** 2 * (math.sin(math.radians(self.rotation))) ** 2 + self.y_radius ** 2 * (math.cos(math.radians(self.rotation))) ** 2
                ),
                1,
            ]
        )
        return xr.DataArray(temp_array, dims=["dim"], coords={"dim": ["x", "y", "z"],},)


class My_Ellipse(My_Circle):
    def __init__(self, ref_pt, ref_pt_x, ref_pt_y, x_radius, y_radius):
        super().__init__(ref_pt, ref_pt_x, ref_pt_y, 1)
        self.scale(x_radius, y_radius)

    @property
    def radius(self):
        print("An ellipse has a major and minor axis. Use x_radius and y_radius.")
