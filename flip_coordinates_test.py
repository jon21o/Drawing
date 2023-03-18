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


class My_Polygon:
    def __init__(self, xy_pairs):
        # xy_pairs = [[x1, y1], [x2, y2], [x3, y3], etc.]
        num_pts = len(xy_pairs)
        if num_pts < 3:
            raise Exception("Please supply at least 3 points to make a polygon!")
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

    @property
    def centroid(self):
        return self._matrix.mean(dim="pt")

    def translate(self, tx, ty):
        self._matrix = xr.DataArray(
            translation_matrix(tx, ty) @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt1", "pt2", "pt3", "pt4"],},
        )

    def flip_y_inplace(self):
        self._matrix = xr.DataArray(
            translation_matrix(
                self.centroid.loc[dict(dim="x")].values,
                self.centroid.loc[dict(dim="y")].values,
            )
            @ scale_matrix(1, -1)
            @ translation_matrix(
                -self.centroid.loc[dict(dim="x")].values,
                -self.centroid.loc[dict(dim="y")].values,
            )
            @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt1", "pt2", "pt3", "pt4"],},
        )

    def correct_coordinates(self, WIDTH, HEIGHT):
        self._matrix = xr.DataArray(
            translation_matrix(
                self.centroid.loc[dict(dim="x")].values,
                HEIGHT
                - self._matrix.loc[dict(dim="y", pt="pt1")].values
                + (
                    self._matrix.max(dim="pt").values[1]
                    - self.centroid.loc[dict(dim="y")].values
                ),
            )
            @ scale_matrix(1, -1)
            @ translation_matrix(
                -self.centroid.loc[dict(dim="x")].values,
                -self.centroid.loc[dict(dim="y")].values,
            )
            @ self._matrix.values,
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt1", "pt2", "pt3", "pt4"],},
        )

    def draw(self, ctx, cx, cy, cz):
        ctx.set_source_rgb(cx, cy, cz)
        for pt in self._matrix.coords["pt"]:
            if pt.values == "pt1":
                ctx.move_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values,
                    self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
            else:
                ctx.line_to(
                    self._matrix.loc[dict(dim="x", pt=pt)].values,
                    self._matrix.loc[dict(dim="y", pt=pt)].values,
                )
        ctx.fill()


class My_Rectangle:
    def __init__(self, ref_pt, ref_pt_x, ref_pt_y, width, height, rotation=0):
        # pt1 --> bot left
        # pt2 --> top left
        # pt3 --> top right
        # pt4 --> bot right
        self._matrix = xr.DataArray(
            np.ones((3, 4)),
            dims=["dim", "pt"],
            coords={"dim": ["x", "y", "z"], "pt": ["pt1", "pt2", "pt3", "pt4"],},
        )
        # region #### reference site definitions
        if ref_pt == "center":
            self._matrix.loc[dict(dim="x", pt="pt1")] = ref_pt_x - (width / 2)
            self._matrix.loc[dict(dim="y", pt="pt1")] = ref_pt_y - (height / 2)
            self._matrix.loc[dict(dim="x", pt="pt2")] = ref_pt_x - (width / 2)
            self._matrix.loc[dict(dim="y", pt="pt2")] = ref_pt_y + (height / 2)
            self._matrix.loc[dict(dim="x", pt="pt3")] = ref_pt_x + (width / 2)
            self._matrix.loc[dict(dim="y", pt="pt3")] = ref_pt_y + (height / 2)
            self._matrix.loc[dict(dim="x", pt="pt4")] = ref_pt_x + (width / 2)
            self._matrix.loc[dict(dim="y", pt="pt4")] = ref_pt_y - (height / 2)
        elif ref_pt == "bot_left":
            self._matrix.loc[dict(dim="x", pt="pt1")] = ref_pt_x
            self._matrix.loc[dict(dim="y", pt="pt1")] = ref_pt_y
            self._matrix.loc[dict(dim="x", pt="pt2")] = ref_pt_x
            self._matrix.loc[dict(dim="y", pt="pt2")] = ref_pt_y + height
            self._matrix.loc[dict(dim="x", pt="pt3")] = ref_pt_x + width
            self._matrix.loc[dict(dim="y", pt="pt3")] = ref_pt_y + height
            self._matrix.loc[dict(dim="x", pt="pt4")] = ref_pt_x + width
            self._matrix.loc[dict(dim="y", pt="pt4")] = ref_pt_y
        elif ref_pt == "top_left":
            self._matrix.loc[dict(dim="x", pt="pt1")] = ref_pt_x
            self._matrix.loc[dict(dim="y", pt="pt1")] = ref_pt_y - height
            self._matrix.loc[dict(dim="x", pt="pt2")] = ref_pt_x
            self._matrix.loc[dict(dim="y", pt="pt2")] = ref_pt_y
            self._matrix.loc[dict(dim="x", pt="pt3")] = ref_pt_x + width
            self._matrix.loc[dict(dim="y", pt="pt3")] = ref_pt_y
            self._matrix.loc[dict(dim="x", pt="pt4")] = ref_pt_x + width
            self._matrix.loc[dict(dim="y", pt="pt4")] = ref_pt_y - height
        elif ref_pt == "top_right":
            self._matrix.loc[dict(dim="x", pt="pt1")] = ref_pt_x - width
            self._matrix.loc[dict(dim="y", pt="pt1")] = ref_pt_y - height
            self._matrix.loc[dict(dim="x", pt="pt2")] = ref_pt_x - width
            self._matrix.loc[dict(dim="y", pt="pt2")] = ref_pt_y
            self._matrix.loc[dict(dim="x", pt="pt3")] = ref_pt_x
            self._matrix.loc[dict(dim="y", pt="pt3")] = ref_pt_y
            self._matrix.loc[dict(dim="x", pt="pt4")] = ref_pt_x
            self._matrix.loc[dict(dim="y", pt="pt4")] = ref_pt_y - height
        elif ref_pt == "bot_right":
            self._matrix.loc[dict(dim="x", pt="pt1")] = ref_pt_x - width
            self._matrix.loc[dict(dim="y", pt="pt1")] = ref_pt_y
            self._matrix.loc[dict(dim="x", pt="pt2")] = ref_pt_x - width
            self._matrix.loc[dict(dim="y", pt="pt2")] = ref_pt_y + height
            self._matrix.loc[dict(dim="x", pt="pt3")] = ref_pt_x
            self._matrix.loc[dict(dim="y", pt="pt3")] = ref_pt_y + height
            self._matrix.loc[dict(dim="x", pt="pt4")] = ref_pt_x
            self._matrix.loc[dict(dim="y", pt="pt4")] = ref_pt_y
        # endregion

    # region #### points on the rectangle
    @property
    def pt1_bot_left(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt1")]

    @property
    def pt12_mid_left(self):
        return (
            self._matrix.loc[dict(dim=["x", "y"], pt="pt2")]
            - self._matrix.loc[dict(dim=["x", "y"], pt="pt1")]
        ) / 2

    @property
    def pt2_top_left(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt2")]

    @property
    def pt23_mid_top(self):
        return (
            self._matrix.loc[dict(dim=["x", "y"], pt="pt3")]
            - self._matrix.loc[dict(dim=["x", "y"], pt="pt2")]
        ) / 2

    @property
    def pt3_top_right(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt3")]

    @property
    def pt34_mid_right(self):
        return (
            self._matrix.loc[dict(dim=["x", "y"], pt="pt4")]
            - self._matrix.loc[dict(dim=["x", "y"], pt="pt3")]
        ) / 2

    @property
    def pt4_bot_right(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt4")]

    @property
    def pt41_mid_bot(self):
        return (
            self._matrix.loc[dict(dim=["x", "y"], pt="pt1")]
            - self._matrix.loc[dict(dim=["x", "y"], pt="pt4")]
        ) / 2

    @property
    def pt4_bot_right(self):
        return self._matrix.loc[dict(dim=["x", "y"], pt="pt4")]

    # endregion


WIDTH = 1000
HEIGHT = 1000

surface = cairo.ImageSurface(cairo.FORMAT_RGB24, WIDTH, HEIGHT)
ctx = cairo.Context(surface)
# ctx.scale(100, 100)

# rect = My_Rectangle(ref_pt="center", ref_pt_x=1, ref_pt_y=1, width=0.1, height=0.1)

poly1 = My_Polygon([[100, 900], [100, 950], [150, 950], [150, 900]])
poly2 = My_Polygon([[100, 110], [150, 160], [200, 160], [150, 110]])

poly1.draw(ctx, 0.8, 0.8, 1)
poly2.correct_coordinates(WIDTH, HEIGHT)
print(poly1._matrix)
print(poly2._matrix)
poly2.draw(ctx, 0.2, 0.2, 1)

surface.write_to_png("C:\\Users\\jon21\\Downloads\\test.png")

# @classmethod
# def from_diameter(cls, diameter):
#     return cls(radius=diameter / 2)

# @radius.setter
# def radius(self, value):
#     self._radius = float(value)


# Return the sine value of 30 degrees
# print(math.sin(math.radians(30)))

# print(rect.pt1_bot_left.values)
# print(rect.pt2_top_left.values)
# print(rect.pt3_top_right.values)
# print(rect.pt4_bot_right.values)
# print(rect.pt12_mid_left.values)
