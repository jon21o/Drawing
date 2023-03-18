import math
import numpy as np
import xarray as xr
import cairo


def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def scale_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])


def rotation_matrix(r):
    return np.array(
        [
            [math.cos(math.radians(r)), -math.sin(math.radians(r)), 0],
            [math.sin(math.radians(r)), math.cos(math.radians(r)), 0],
            [0, 0, 1],
        ]
    )


class My_Shape:
    def __init__(self, xy_pairs, rotation=0):
        # xy_pairs = [[x1, y1], [x2, y2], [x3, y3], etc.]
        num_pts = len(xy_pairs)
        self._rotation = rotation
        self._matrix = xr.DataArray(
            np.ones((3, num_pts)),
            dims=["dim", "pt"],
            coords={
                "dim": ["x", "y", "z"],
                "pt": ["pt{}".format(x) for x in range(1, num_pts + 1)],
            },
        )
        for i, xy_pair in enumerate(xy_pairs):
            self._matrix.loc[dict(dim="x", pt="pt{}".format(i + 1))] = xy_pair[0]
            self._matrix.loc[dict(dim="y", pt="pt{}".format(i + 1))] = xy_pair[1]

    # region #### Functions
    def translate(self, tx, ty):
        self._matrix = xr.DataArray(
            translation_matrix(tx, ty) @ self._matrix.values,
            dims=["dim", "pt"],
            coords={
                "dim": ["x", "y", "z"],
                "pt": [
                    "pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)
                ],
            },
        )

    def scale(self, sx, sy):
        self._matrix = xr.DataArray(
            scale_matrix(sx, sy) @ self._matrix.values,
            dims=["dim", "pt"],
            coords={
                "dim": ["x", "y", "z"],
                "pt": [
                    "pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)
                ],
            },
        )

    def rotate(self, r):
        self._matrix = xr.DataArray(
            rotation_matrix(r) @ self._matrix.values,
            dims=["dim", "pt"],
            coords={
                "dim": ["x", "y", "z"],
                "pt": [
                    "pt{}".format(x) for x in range(1, self._matrix.sizes["pt"] + 1)
                ],
            },
        )
        self._rotation += r

    def draw_stroke(self, context, color1, color2, color3):
        context.set_source_rgb(color1, color2, color3)
        for pt in self._matrix.coords["pt"]:
            if pt.values == "pt1":
                context.move_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values,
                    self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
            else:
                context.line_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values,
                    self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
        context.stroke()

    def draw_fill(self, context, color1, color2, color3):
        context.set_source_rgb(color1, color2, color3)
        for pt in self._matrix.coords["pt"]:
            if pt.values == "pt1":
                context.move_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values,
                    self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
            else:
                context.line_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values,
                    self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
        context.fill()

    # endregion

    # region #### Properties
    @property
    def centroid(self):
        return self._matrix.mean(dim="pt")

    # endregion


class My_Rectangle(My_Shape):
    def __init__(self, ref_pt, ref_pt_x, ref_pt_y, width, height):
        # pt1 --> bot left
        # pt2 --> top left
        # pt3 --> top right
        # pt4 --> bot right
        self._width = width
        self._height = height
        self._matrix = xr.DataArray(
            np.ones((3, 4)),
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt1", "pt2", "pt3", "pt4"],},
        )

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

    def move_xy(self, ref_pt, ref_pt_x, ref_pt_y):
        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(
                ref_pt_x - self.centroid.loc[dict(dim="x")],
                ref_pt_y - self.centroid.loc[dict(dim="y")],
            )

        # endregion

    # region #### points on the rectangle
    @property
    def bot_left(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt1")]

    @property
    def mid_left(self):
        return (
            self._matrix.loc[dict(dim=["x", "y"], pt="pt2")]
            - self._matrix.loc[dict(dim=["x", "y"], pt="pt1")]
        ) / 2

    @property
    def top_left(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt2")]

    @property
    def mid_top(self):
        return (
            self._matrix.loc[dict(dim=["x", "y"], pt="pt3")]
            - self._matrix.loc[dict(dim=["x", "y"], pt="pt2")]
        ) / 2

    @property
    def top_right(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt3")]

    @property
    def mid_right(self):
        return (
            self._matrix.loc[dict(dim=["x", "y"], pt="pt4")]
            - self._matrix.loc[dict(dim=["x", "y"], pt="pt3")]
        ) / 2

    @property
    def bot_right(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt4")]

    @property
    def mid_bot(self):
        return (
            self._matrix.loc[dict(dim=["x", "y"], pt="pt1")]
            - self._matrix.loc[dict(dim=["x", "y"], pt="pt4")]
        ) / 2

    @property
    def bot_right(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt4")]

    # endregion

