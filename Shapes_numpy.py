import math
import numpy as np


def translation_matrix(tx, ty):
    return np.array([[1, 0, tx], [0, 1, ty], [0, 0, 1]])


def scale_matrix(sx, sy):
    return np.array([[sx, 0, 0], [0, sy, 0], [0, 0, 1]])


def shear_matrix(shx, shy):
    return np.array([[1, shy, 0], [shx, 1, 0], [0, 0, 1]])


def rotation_matrix(r):
    return np.array(
        [
            [math.cos(math.radians(r)), -math.sin(math.radians(r)), 0],
            [math.sin(math.radians(r)), math.cos(math.radians(r)), 0],
            [0, 0, 1],
        ]
    )


bezier_curve_approx = (4 / 3) * math.tan(math.pi / 8)


class My_Shape:
    def __init__(self, xy_pairs):
        # xy_pairs = [[x1, y1], [x2, y2], [x3, y3], etc.]
        num_pts = len(xy_pairs)
        self._rotation = 0
        self._matrix = np.ones((3, num_pts))
        for i, xy_pair in enumerate(xy_pairs):
            self._matrix[0, i] = xy_pair[0]
            self._matrix[1, i] = xy_pair[1]

    # region #### Functions

    def calc_bbox_pts(self):
        if (self.__class__.__name__ == "My_Circle") | (
            self.__class__.__name__ == "My_Ellipse"
        ):
            self._bbox_pts = np.ones((3, 8))

            center_x = self.centroid_x
            center_y = self.centroid_y

            x1 = math.sqrt(
                self.x_radius ** 2
                * (math.cos(math.radians(self.rotation))) ** 2
                + self.y_radius ** 2
                * (math.sin(math.radians(self.rotation))) ** 2
            )
            y1 = math.sqrt(
                self.x_radius ** 2
                * (math.sin(math.radians(self.rotation))) ** 2
                + self.y_radius ** 2
                * (math.cos(math.radians(self.rotation))) ** 2
            )

            # pt0 --> bot left
            # pt1 --> mid left
            # pt2 --> top left
            # pt3 --> mid top
            # pt4 --> top right
            # pt5 --> mid right
            # pt6 --> bot right
            # pt7 --> mid bot
            self._bbox_pts[0, 0] = center_x - x1
            self._bbox_pts[1, 0] = center_y - y1
            self._bbox_pts[0, 1] = center_x - x1
            self._bbox_pts[1, 1] = center_y
            self._bbox_pts[0, 2] = center_x - x1
            self._bbox_pts[1, 2] = center_y + y1
            self._bbox_pts[0, 3] = center_x
            self._bbox_pts[1, 3] = center_y + y1
            self._bbox_pts[0, 4] = center_x + x1
            self._bbox_pts[1, 4] = center_y + y1
            self._bbox_pts[0, 5] = center_x + x1
            self._bbox_pts[1, 5] = center_y
            self._bbox_pts[0, 6] = center_x + x1
            self._bbox_pts[1, 6] = center_y - y1
            self._bbox_pts[0, 7] = center_x
            self._bbox_pts[1, 7] = center_y - y1
        else:
            self._bbox_pts = np.ones((3, 8))

            center_x = self.centroid_x
            center_y = self.centroid_y
            x_min = self._matrix.min(axis=1)[0]
            y_min = self._matrix.min(axis=1)[1]
            x_max = self._matrix.max(axis=1)[0]
            y_max = self._matrix.max(axis=1)[1]

            self._bbox_pts[0, 0] = x_min
            self._bbox_pts[1, 0] = y_min
            self._bbox_pts[0, 1] = x_min
            self._bbox_pts[1, 1] = center_y
            self._bbox_pts[0, 2] = x_min
            self._bbox_pts[1, 2] = y_max
            self._bbox_pts[0, 3] = center_x
            self._bbox_pts[1, 3] = y_max
            self._bbox_pts[0, 4] = x_max
            self._bbox_pts[1, 4] = y_max
            self._bbox_pts[0, 5] = x_max
            self._bbox_pts[1, 5] = center_y
            self._bbox_pts[0, 6] = x_max
            self._bbox_pts[1, 6] = y_min
            self._bbox_pts[0, 7] = center_x
            self._bbox_pts[1, 7] = y_min

    def translate(self, tx, ty):
        self._matrix = translation_matrix(tx, ty) @ self._matrix
        self._bbox_pts = translation_matrix(tx, ty) @ self._bbox_pts
        if (self.__class__.__name__ == "My_Circle") | (
            self.__class__.__name__ == "My_Ellipse"
        ):
            self._control_pts = translation_matrix(tx, ty) @ self._control_pts

    def scale(self, sx, sy):
        self._matrix = (
            translation_matrix(self.centroid_x, self.centroid_y)
            @ scale_matrix(sx, sy)
            @ translation_matrix(-self.centroid_x, -self.centroid_y)
            @ self._matrix
        )
        self._bbox_pts = (
            translation_matrix(self.centroid_x, self.centroid_y,)
            @ scale_matrix(sx, sy)
            @ translation_matrix(-self.centroid_x, -self.centroid_y,)
            @ self._bbox_pts,
        )
        if (self.__class__.__name__ == "My_Circle") | (
            self.__class__.__name__ == "My_Ellipse"
        ):
            self._control_pts = (
                translation_matrix(self.centroid_x, self.centroid_y,)
                @ scale_matrix(sx, sy)
                @ translation_matrix(-self.centroid_x, -self.centroid_y,)
                @ self._control_pts,
            )

    def shear(self, shx, shy):
        self._matrix = (
            translation_matrix(self.centroid_x, self.centroid_y,)
            @ shear_matrix(shx, shy)
            @ translation_matrix(-self.centroid_x, -self.centroid_y,)
            @ self._matrix,
        )
        self._bbox_pts = (
            translation_matrix(self.centroid_x, self.centroid_y,)
            @ shear_matrix(shx, shy)
            @ translation_matrix(-self.centroid_x, -self.centroid_y,)
            @ self._bbox_pts,
        )

        if (self.__class__.__name__ == "My_Circle") | (
            self.__class__.__name__ == "My_Ellipse"
        ):
            self._control_pts = (
                translation_matrix(self.centroid_x, self.centroid_y,)
                @ shear_matrix(shx, shy)
                @ translation_matrix(-self.centroid_x, -self.centroid_y,)
                @ self._control_pts,
            )

    def rotate(self, r):
        self._rotation += r
        self._matrix = (
            translation_matrix(self.centroid_x, self.centroid_y,)
            @ rotation_matrix(r)
            @ translation_matrix(-self.centroid_x, -self.centroid_y,)
            @ self._matrix,
        )
        self.calc_bbox_pts()
        if (self.__class__.__name__ == "My_Circle") | (
            self.__class__.__name__ == "My_Ellipse"
        ):
            self._control_pts = (
                translation_matrix(self.centroid_x, self.centroid_y,)
                @ rotation_matrix(r)
                @ translation_matrix(-self.centroid_x, -self.centroid_y,)
                @ self._control_pts,
            )

    def move_bbox_xy_to(self, ref_pt, ref_pt_x, ref_pt_y):
        if ref_pt == "center":
            self.translate(
                ref_pt_x - self.centroid_x, ref_pt_y - self.centroid_y,
            )
        if ref_pt == "bot_left":
            self.translate(
                ref_pt_x - self.bbox_left_x, ref_pt_y - self.bbox_bot_y,
            )
        if ref_pt == "mid_left":
            self.translate(
                ref_pt_x - self.bbox_left_x, ref_pt_y - self.centroid_y,
            )
        if ref_pt == "top_left":
            self.translate(
                ref_pt_x - self.bbox_left_x, ref_pt_y - self.bbox_top_y,
            )
        if ref_pt == "mid_top":
            self.translate(
                ref_pt_x - self.centroid_x, ref_pt_y - self.bbox_top_y,
            )
        if ref_pt == "top_right":
            self.translate(
                ref_pt_x - self.bbox_right_x, ref_pt_y - self.bbox_top_y,
            )
        if ref_pt == "mid_right":
            self.translate(
                ref_pt_x - self.bbox_right_x, ref_pt_y - self.centroid_y,
            )
        if ref_pt == "bot_right":
            self.translate(
                ref_pt_x - self.bbox_right_x, ref_pt_y - self.bbox_bot_y,
            )
        if ref_pt == "mid_bot":
            self.translate(
                ref_pt_x - self.centroid_x, ref_pt_y - self.bbox_bot_y,
            )

    def move_bbox_x_to(self, ref_pt, ref_pt_x):
        if ref_pt == "center":
            self.translate(
                ref_pt_x - self.centroid_x, 0,
            )
        if ref_pt == "bot_left":
            self.translate(
                ref_pt_x - self.bbox_left_x, 0,
            )
        if ref_pt == "mid_left":
            self.translate(
                ref_pt_x - self.bbox_left_x, 0,
            )
        if ref_pt == "top_left":
            self.translate(
                ref_pt_x - self.bbox_left_x, 0,
            )
        if ref_pt == "mid_top":
            self.translate(
                ref_pt_x - self.centroid_x, 0,
            )
        if ref_pt == "top_right":
            self.translate(
                ref_pt_x - self.bbox_right_x, 0,
            )
        if ref_pt == "mid_right":
            self.translate(
                ref_pt_x - self.bbox_right_x, 0,
            )
        if ref_pt == "bot_right":
            self.translate(
                ref_pt_x - self.bbox_right_x, 0,
            )
        if ref_pt == "mid_bot":
            self.translate(
                ref_pt_x - self.centroid_x, 0,
            )

    def move_bbox_y_to(self, ref_pt, ref_pt_y):
        if ref_pt == "center":
            self.translate(
                0, ref_pt_y - self.centroid_y,
            )
        if ref_pt == "bot_left":
            self.translate(
                0, ref_pt_y - self.bbox_bot_y,
            )
        if ref_pt == "mid_left":
            self.translate(
                0, ref_pt_y - self.centroid_y,
            )
        if ref_pt == "top_left":
            self.translate(
                0, ref_pt_y - self.bbox_top_y,
            )
        if ref_pt == "mid_top":
            self.translate(
                0, ref_pt_y - self.bbox_top_y,
            )
        if ref_pt == "top_right":
            self.translate(
                0, ref_pt_y - self.bbox_top_y,
            )
        if ref_pt == "mid_right":
            self.translate(
                0, ref_pt_y - self.centroid_y,
            )
        if ref_pt == "bot_right":
            self.translate(
                0, ref_pt_y - self.bbox_bot_y,
            )
        if ref_pt == "mid_bot":
            self.translate(
                0, ref_pt_y - self.bbox_bot_y,
            )

    def move_pt_xy_to(self, ref_pt_x, ref_pt_y, pt_x, pt_y):
        self.translate(
            pt_x - ref_pt_x, pt_y - ref_pt_y,
        )

    def move_pt_x_to(self, ref_pt_x, pt_x):
        self.translate(
            pt_x - ref_pt_x, 0,
        )

    def move_pt_y_to(self, ref_pt_y, pt_y):
        self.translate(
            0, pt_y - ref_pt_y,
        )

    def connect_pts(self):
        pts_path = []
        pts_path.extend(["M", self._matrix[0, 0], self._matrix[1, 0]])
        for pt in range(1, self._matrix.shape[1]):
            pts_path.extend(["L", self._matrix[0, pt], self._matrix[1, pt]])
        pts_path.append("Z")
        return pts_path

    def connect_segment(self, pt1, pt2):
        pts_path = []
        pts_path.extend(["M", self._matrix[0, pt1], self._matrix[1, pt1]])
        pts_path.extend(["L", self._matrix[0, pt2], self._matrix[1, pt2]])
        return pts_path

    def connect_bbox_pts(self):
        pts_path = []
        pts_path.extend(["M", self._bbox_pts[0, 0], self._bbox_pts[1, 0]])
        for pt in range(1, self._bbox_pts.shape[1]):
            pts_path.extend(["L", self._bbox_pts[0, pt], self._bbox_pts[1, pt]])
        pts_path.append("Z")
        return pts_path

    # endregion

    # region #### Properties
    @property
    def centroid_x(self):
        return self._matrix.mean(axis=1)[0]

    @property
    def centroid_y(self):
        return self._matrix.mean(axis=1)[1]

    @property
    def rotation(self):
        return self._rotation

    @property
    def pts_x(self):
        "pts start at 0"
        return {
            "pt{}".format(pt): self._matrix[0, pt]
            for pt in range(self._matrix.shape[1])
        }

    @property
    def pts_y(self):
        "pts start at 0"
        return {
            "pt{}".format(pt): self._matrix[1, pt]
            for pt in range(self._matrix.shape[1])
        }

    @property
    def mid_pts_x(self):
        # create pts 0 through end
        mid_pts_dict = {
            "mid_pt_{}_{}".format(pt + 1, pt): (
                (self._matrix[0, pt + 1] - self._matrix[0, pt]) / 2
            )
            + self._matrix[0, pt]
            for pt in range(self._matrix.shape[1])
        }
        # have to handle last mid point separately because connects first and last point
        mid_pts_dict["mid_pt_{}_{}".format(self._matrix.shape[1], 0)] = (
            (self._matrix[0, 0] - self._matrix[0, -1]) / 2
        ) + self._matrix[0, -1]
        return mid_pts_dict

    @property
    def mid_pts_y(self):
        mid_pts_dict = {
            "mid_pt_{}_{}".format(pt + 1, pt): (
                (self._matrix[1, pt + 1] - self._matrix[1, pt]) / 2
            )
            + self._matrix[1, pt]
            for pt in range(self._matrix.shape[1])
        }
        mid_pts_dict["mid_pt_{}_{}".format(self._matrix.shape[1], 0)] = (
            (self._matrix[1, 0] - self._matrix[1, -1]) / 2
        ) + self._matrix[1, -1]
        return mid_pts_dict

    @property
    def bbox_left_x(self):
        return self._bbox_pts[0, 0]

    @property
    def bbox_right_x(self):
        return self._bbox_pts[0, 4]

    @property
    def bbox_bot_y(self):
        return self._bbox_pts[1, 0]

    @property
    def bbox_top_y(self):
        return self._bbox_pts[1, 4]

    # endregion


class My_Line_Coord(My_Shape):
    def __init__(self, xy_pairs):
        super().__init__(xy_pairs)

    def connect_pts(self):
        pts_path = []
        pts_path.extend(["M", self._matrix[0, 0], self._matrix[1, 0]])
        for pt in range(1, self._matrix.shape[1]):
            pts_path.extend(["L", self._matrix[0, pt], self._matrix[1, pt]])
        return pts_path


class My_Line_Relative(My_Shape):
    def __init__(self, xy_pairs, angular=False):
        super().__init__(xy_pairs)
        self._angular = angular

    # def draw_stroke(self, context, color1, color2, color3, alpha=1):
    #     context.set_source_rgba(color1, color2, color3, alpha)
    #     if self.angular == False:
    #         for pt in self._matrix.coords["pt"]:
    #             if pt == "pt1":
    #                 context.move_to(
    #                     self._matrix.loc[dict(dim="x", pt=pt)],
    #                     self._matrix.loc[dict(dim="y", pt=pt)],
    #                 )
    #             else:
    #                 context.rel_line_to(
    #                     self._matrix.loc[dict(dim="x", pt=pt)],
    #                     self._matrix.loc[dict(dim="y", pt=pt)],
    #                 )
    #         context.stroke()
    #     elif self.angular == True:
    #         for pt in self._matrix.coords["pt"]:
    #             if pt == "pt1":
    #                 context.move_to(
    #                     self._matrix.loc[dict(dim="x", pt=pt)],
    #                     self._matrix.loc[dict(dim="y", pt=pt)],
    #                 )
    #             else:
    #                 context.rel_line_to(
    #                     math.cos(
    #                         math.radians(
    #                             self._matrix.loc[dict(dim="x", pt=pt)]
    #                         )
    #                     )
    #                     * self._matrix.loc[dict(dim="y", pt=pt)],
    #                     math.sin(
    #                         math.radians(
    #                             self._matrix.loc[dict(dim="x", pt=pt)]
    #                         )
    #                     )
    #                     * self._matrix.loc[dict(dim="y", pt=pt)],
    #                 )
    #         context.stroke()


class My_Rectangle(My_Shape):
    def __init__(self, ref_pt, ref_pt_x, ref_pt_y, width, height):
        # pt1 --> bot left
        # pt2 --> top left
        # pt3 --> top right
        # pt4 --> bot right
        self._rotation = 0
        self._matrix = np.ones((3, 4))
        self._width = width
        self._height = height

        # start with ref_pt = "bot_left" and translate to different ref_pts below
        self._matrix[0, 0] = ref_pt_x
        self._matrix[1, 0] = ref_pt_y
        self._matrix[0, 1] = ref_pt_x
        self._matrix[1, 1] = ref_pt_y + height
        self._matrix[0, 2] = ref_pt_x + width
        self._matrix[1, 2] = ref_pt_y + height
        self._matrix[0, 3] = ref_pt_x + width
        self._matrix[1, 3] = ref_pt_y

        self.calc_bbox_pts()

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

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height


class My_Parallelogram(My_Shape):
    def __init__(
        self,
        ref_pt,
        ref_pt_x,
        ref_pt_y,
        width=None,
        height=None,
        angle=None,
        offset=None,
    ):
        # pt1 --> bot left
        # pt2 --> top left
        # pt3 --> top right
        # pt4 --> bot right
        self._rotation = 0
        self._matrix = np.ones((3, 4))
        self._width = width
        self._height = height
        self._angle = angle
        self._offset = offset

        # start with ref_pt = "bot_left" and translate to different ref_pts below
        if (width != None) & (height != None) & (offset != None):
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x + offset
            self._matrix[1, 1] = ref_pt_y + height
            self._matrix[0, 2] = ref_pt_x + width + offset
            self._matrix[1, 2] = ref_pt_y + height
            self._matrix[0, 3] = ref_pt_x + width
            self._matrix[1, 3] = ref_pt_y
        elif (width != None) & (height != None) & (angle != None):
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x + (
                math.tan(math.radians(angle)) * height
            )
            self._matrix[1, 1] = ref_pt_y + height
            self._matrix[0, 2] = (
                ref_pt_x + width + (math.tan(math.radians(angle)) * height)
            )
            self._matrix[1, 2] = ref_pt_y + height
            self._matrix[0, 3] = ref_pt_x + width
            self._matrix[1, 3] = ref_pt_y

        self.calc_bbox_pts()

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(
                self._matrix[0, 0] - self.centroid_x,
                self._matrix[1, 0] - self.centroid_y,
            )
        elif ref_pt == "mid_left":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 1]) / 2,
                (self._matrix[1, 0] - self._matrix[1, 1]) / 2,
            )
        elif ref_pt == "top_left":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 1],
                self._matrix[1, 0] - self._matrix[1, 1],
            )
        elif ref_pt == "mid_top":
            self.translate(
                self._matrix[0, 3]
                - self._matrix[0, 2]
                + (self._matrix[0, 0] - self._matrix[0, 3]) / 2,
                self._matrix[1, 0] - self._matrix[1, 2],
            )
        elif ref_pt == "top_right":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 2],
                self._matrix[1, 0] - self._matrix[1, 2],
            )
        elif ref_pt == "mid_right":
            self.translate(
                self._matrix[0, 0]
                - self._matrix[0, 3]
                + (self._matrix[0, 3] - self._matrix[0, 2]) / 2,
                (self._matrix[1, 3] - self._matrix[1, 2]) / 2,
            )
        elif ref_pt == "bot_right":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 3], 0,
            )
        elif ref_pt == "mid_bot":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 3]) / 2, 0,
            )
        # endregion

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def angle(self):
        return self._angle

    @property
    def offset(self):
        return self._offset


class My_RightTriangle(My_Shape):
    def __init__(
        self, ref_pt, ref_pt_x, ref_pt_y, width=None, height=None, angle=None
    ):
        # pt1 --> bot left
        # pt2 --> top left
        # pt3 --> bot right
        self._rotation = 0
        self._matrix = np.ones((3, 3))
        self._width = width
        self._height = height
        self._angle = angle

        # start with ref_pt = "bot_left" and translate to different ref_pts below
        # 90 deg is at bot_left
        if (width != None) & (height != None):
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x
            self._matrix[1, 1] = ref_pt_y + height
            self._matrix[0, 2] = ref_pt_x + width
            self._matrix[1, 2] = ref_pt_y
        elif (width != None) & (angle != None):
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x
            self._matrix[1, 1] = ref_pt_y + (
                math.tan(math.radians(angle)) * width
            )
            self._matrix[0, 2] = ref_pt_x + width
            self._matrix[1, 2] = ref_pt_y

        self.calc_bbox_pts()

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(
                -(self.centroid_x - self._matrix[0, 0]),
                -(self.centroid_y - self._matrix[1, 0]),
            )
        elif ref_pt == "mid_left":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 1]) / 2,
                (self._matrix[1, 0] - self._matrix[1, 1]) / 2,
            )
        elif ref_pt == "top_left":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 1],
                self._matrix[1, 0] - self._matrix[1, 1],
            )
        elif ref_pt == "mid_right":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 2]) / 2,
                (self._matrix[1, 0] - self._matrix[1, 1]) / 2,
            )
        elif ref_pt == "bot_right":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 2], 0,
            )
        elif ref_pt == "mid_bot":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 2]) / 2, 0,
            )
        # endregion

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def angle(self):
        return self._angle


class My_IsoscelesTriangle(My_Shape):
    def __init__(
        self, ref_pt, ref_pt_x, ref_pt_y, width=None, height=None, angle=None
    ):
        # pt1 --> bot left
        # pt2 --> mid top
        # pt3 --> bot right
        self._rotation = 0
        self._matrix = np.ones((3, 3))
        self._width = width
        self._height = height
        self._angle = angle

        # start with ref_pt = "bot_left" and translate to different ref_pts below
        if (width != None) & (height != None):
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x + (width / 2)
            self._matrix[1, 1] = ref_pt_y + height
            self._matrix[0, 2] = ref_pt_x + width
            self._matrix[1, 2] = ref_pt_y
        elif (width != None) & (angle != None):
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x + (width / 2)
            self._matrix[1, 1] = ref_pt_y + (
                math.tan(math.radians(angle)) * (width / 2)
            )
            self._matrix[0, 2] = ref_pt_x + width
            self._matrix[1, 2] = ref_pt_y

        self.calc_bbox_pts()

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(
                self._matrix[0, 0] - self.centroid_x,
                self._matrix[1, 0] - self.centroid_y,
            )
        elif ref_pt == "mid_left":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 1]) / 2,
                (self._matrix[1, 0] - self._matrix[1, 1]) / 2,
            )
        elif ref_pt == "mid_top":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 1]),
                (self._matrix[1, 0] - self._matrix[1, 1]),
            )
        elif ref_pt == "mid_right":
            self.translate(
                self._matrix[0, 0]
                - self._matrix[0, 1]
                + (self._matrix[0, 1] - self._matrix[0, 2]) / 2,
                (self._matrix[1, 0] - self._matrix[1, 1]) / 2,
            )
        elif ref_pt == "bot_right":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 2]), 0,
            )
        elif ref_pt == "mid_bot":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 2]) / 2, 0,
            )
        # endregion

    @property
    def width(self):
        return self._width

    @property
    def height(self):
        return self._height

    @property
    def angle(self):
        return self._angle


class My_EquilateralTriangle(My_Shape):
    def __init__(self, ref_pt, ref_pt_x, ref_pt_y, width=None):
        # pt1 --> bot left
        # pt2 --> mid top
        # pt3 --> bot right
        self._rotation = 0
        self._matrix = np.ones((3, 3))
        self._width = width

        # start with ref_pt = "bot_left" and translate to different ref_pts below
        self._matrix[0, 0] = ref_pt_x
        self._matrix[1, 0] = ref_pt_y
        self._matrix[0, 1] = ref_pt_x + (width / 2)
        self._matrix[1, 1] = ref_pt_y + (
            math.tan(math.radians(60)) * (width / 2)
        )
        self._matrix[0, 2] = ref_pt_x + width
        self._matrix[1, 2] = ref_pt_y

        self.calc_bbox_pts()

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(
                self._matrix[0, 0] - self.centroid_x,
                self._matrix[1, 0] - self.centroid_y,
            )
        elif ref_pt == "mid_left":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 1]) / 2,
                (self._matrix[1, 0] - self._matrix[1, 1]) / 2,
            )
        elif ref_pt == "mid_top":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 1]),
                (self._matrix[1, 0] - self._matrix[1, 1]),
            )
        elif ref_pt == "mid_right":
            self.translate(
                self._matrix[0, 0]
                - self._matrix[0, 1]
                + (self._matrix[0, 1] - self._matrix[0, 2]) / 2,
                (self._matrix[1, 0] - self._matrix[1, 1]) / 2,
            )
        elif ref_pt == "bot_right":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 2], 0,
            )
        elif ref_pt == "mid_bot":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 2]) / 2, 0,
            )
        # endregion

    @property
    def width(self):
        return self._width


class My_Trapezoid(My_Shape):
    def __init__(
        self,
        ref_pt,
        ref_pt_x,
        ref_pt_y,
        bot_width=None,
        top_width=None,
        height=None,
        angle=None,
    ):
        # pt1 --> bot left
        # pt2 --> top left
        # pt3 --> top right
        # pt4 --> bot right
        self._rotation = 0
        self._matrix = np.ones((3, 4))
        self._bot_width = bot_width
        self._top_width = top_width
        self._height = height
        self._angle = angle

        # start with ref_pt = "bot_left" and translate to different ref_pts below
        if (bot_width != None) & (top_width != None) & (height != None):
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x + (bot_width - top_width) / 2
            self._matrix[1, 1] = ref_pt_y + height
            self._matrix[0, 2] = (
                ref_pt_x + top_width + (bot_width - top_width) / 2
            )
            self._matrix[1, 2] = ref_pt_y + height
            self._matrix[0, 3] = ref_pt_x + bot_width
            self._matrix[1, 3] = ref_pt_y
        elif (bot_width != None) & (angle != None) & (height != None):
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x + (
                math.tan(math.radians(angle)) * height
            )
            self._matrix[1, 1] = ref_pt_y + height
            self._matrix[0, 2] = (
                ref_pt_x + bot_width - (math.tan(math.radians(angle)) * height)
            )
            self._matrix[1, 2] = ref_pt_y + height
            self._matrix[0, 3] = ref_pt_x + bot_width
            self._matrix[1, 3] = ref_pt_y
        elif (top_width != None) & (angle != None) & (height != None):
            bot_width = (
                top_width - 2 * math.tan(math.radians(180 - angle)) * height
            )
            self._matrix[0, 0] = ref_pt_x
            self._matrix[1, 0] = ref_pt_y
            self._matrix[0, 1] = ref_pt_x + (
                math.tan(math.radians(angle)) * height
            )
            self._matrix[1, 1] = ref_pt_y + height
            self._matrix[0, 2] = (
                ref_pt_x + bot_width - (math.tan(math.radians(angle)) * height)
            )
            self._matrix[1, 2] = ref_pt_y + height
            self._matrix[0, 3] = ref_pt_x + bot_width
            self._matrix[1, 3] = ref_pt_y

        self.calc_bbox_pts()

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(
                self._matrix[0, 0] - self.centroid_x,
                self._matrix[1, 0] - self.centroid_y,
            )
        elif ref_pt == "mid_left":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 1]) / 2,
                (self._matrix[1, 0] - self._matrix[1, 1]) / 2,
            )
        elif ref_pt == "top_left":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 1],
                self._matrix[1, 0] - self._matrix[1, 1],
            )
        elif ref_pt == "mid_top":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 3]) / 2,
                self._matrix[1, 0] - self._matrix[1, 2],
            )
        elif ref_pt == "top_right":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 2],
                self._matrix[1, 0] - self._matrix[1, 2],
            )
        elif ref_pt == "mid_right":
            self.translate(
                self._matrix[0, 0]
                - self._matrix[0, 3]
                + (self._matrix[0, 3] - self._matrix[0, 2]) / 2,
                (self._matrix[1, 3] - self._matrix[1, 2]) / 2,
            )
        elif ref_pt == "bot_right":
            self.translate(
                self._matrix[0, 0] - self._matrix[0, 3], 0,
            )
        elif ref_pt == "mid_bot":
            self.translate(
                (self._matrix[0, 0] - self._matrix[0, 3]) / 2, 0,
            )
        # endregion

    @property
    def bot_width(self):
        return self._bot_width

    @property
    def top_width(self):
        return self._top_width

    @property
    def height(self):
        return self._height

    @property
    def angle(self):
        return self._angle


class My_Circle(My_Shape):
    def __init__(
        self, ref_pt, ref_pt_x, ref_pt_y, radius,
    ):
        # pt1 --> left
        # pt2 --> top
        # pt3 --> right
        # pt4 --> bot
        self._rotation = 0
        self._matrix = np.ones((3, 4))
        self._radius = radius
        self._x_radius = radius
        self._y_radius = radius

        # start with ref_pt = "mid_left" and translate to different ref_pts below
        self._matrix[0, 0] = ref_pt_x
        self._matrix[1, 0] = ref_pt_y
        self._matrix[0, 1] = ref_pt_x + radius
        self._matrix[1, 1] = ref_pt_y + radius
        self._matrix[0, 2] = ref_pt_x + (2 * radius)
        self._matrix[1, 2] = ref_pt_y
        self._matrix[0, 3] = ref_pt_x + radius
        self._matrix[1, 3] = ref_pt_y - radius

        self.calc_control_pts()
        self.calc_bbox_pts()

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(-radius, 0)
        elif ref_pt == "bot_left":
            self.translate(0, radius)
        elif ref_pt == "top_left":
            self.translate(0, -radius)
        elif ref_pt == "mid_top":
            self.translate(-radius, -radius)
        elif ref_pt == "top_right":
            self.translate(2 * -radius, -radius)
        elif ref_pt == "mid_right":
            self.translate(2 * -radius, 0)
        elif ref_pt == "bot_right":
            self.translate(2 * -radius, radius)
        elif ref_pt == "mid_bot":
            self.translate(-radius, radius)
        # endregion

    def calc_control_pts(self):
        self._control_pts = np.ones((3, 8))

        self._control_pts[0, 0] = self._matrix[0, 0]
        self._control_pts[1, 0] = self._matrix[1, 0] + (
            bezier_curve_approx * self.y_radius
        )
        self._control_pts[0, 1] = self._matrix[0, 1] - (
            bezier_curve_approx * self.x_radius
        )
        self._control_pts[1, 1] = self._matrix[1, 1]
        self._control_pts[0, 2] = (
            bezier_curve_approx * self.x_radius
        ) + self._matrix[0, 1]
        self._control_pts[1, 2] = self._matrix[1, 1]
        self._control_pts[0, 3] = self._matrix[0, 2]
        self._control_pts[1, 3] = self._matrix[1, 2] + (
            bezier_curve_approx * self.y_radius
        )
        self._control_pts[0, 4] = self._matrix[0, 2]
        self._control_pts[1, 4] = self._matrix[1, 2] - (
            bezier_curve_approx * self.y_radius
        )
        self._control_pts[0, 5] = self._matrix[0, 3] + (
            bezier_curve_approx * self.x_radius
        )
        self._control_pts[1, 5] = self._matrix[1, 3]
        self._control_pts[0, 6] = self._matrix[0, 3] - (
            bezier_curve_approx * self.x_radius
        )
        self._control_pts[1, 6] = self._matrix[1, 3]
        self._control_pts[0, 7] = self._matrix[0, 0]
        self._control_pts[1, 7] = self._matrix[1, 0] - (
            bezier_curve_approx * self.y_radius
        )

    def connect_pts(self):
        pts_path = []
        pts_path.extend(
            ["M", self._matrix[0, 0], self._matrix[1, 0],]
        )
        pts_path.extend(
            [
                "C",
                self._control_pts[0, 0],
                self._control_pts[1, 0],
                self._control_pts[0, 1],
                self._control_pts[1, 1],
                self._matrix[0, 1],
                self._matrix[1, 1],
            ]
        )
        pts_path.extend(
            [
                "C",
                self._control_pts[0, 2],
                self._control_pts[1, 2],
                self._control_pts[0, 3],
                self._control_pts[1, 3],
                self._matrix[0, 2],
                self._matrix[1, 2],
            ]
        )
        pts_path.extend(
            [
                "C",
                self._control_pts[0, 4],
                self._control_pts[1, 4],
                self._control_pts[0, 5],
                self._control_pts[1, 5],
                self._matrix[0, 3],
                self._matrix[1, 3],
            ]
        )
        pts_path.extend(
            [
                "C",
                self._control_pts[0, 6],
                self._control_pts[1, 6],
                self._control_pts[0, 7],
                self._control_pts[1, 7],
                self._matrix[0, 0],
                self._matrix[1, 0],
            ]
        )
        return pts_path

    def connect_segment(self, pt1, pt2):
        pts_path = []
        pts_path.extend(["M", self._matrix[0, pt1], self._matrix[1, pt1]])
        pts_path.extend(
            [
                "C",
                self._control_pts[0, pt1],
                self._control_pts[1, pt1],
                self._control_pts[0, pt2],
                self._control_pts[1, pt2],
                self._matrix[0, pt2],
                self._matrix[1, pt2],
            ]
        )
        return pts_path

    @property
    def radius(self):
        return self._radius

    @property
    def x_radius(self):
        return self._x_radius

    @property
    def y_radius(self):
        return self._y_radius


class My_Ellipse(My_Circle):
    def __init__(
        self, ref_pt, ref_pt_x, ref_pt_y, x_radius, y_radius,
    ):
        # pt1 --> left
        # pt2 --> top
        # pt3 --> right
        # pt4 --> bot
        self._rotation = 0
        self._matrix = np.ones((3, 4))
        self._x_radius = x_radius
        self._y_radius = y_radius

        # start with ref_pt = "mid_left" and translate to different ref_pts below
        self._matrix[0, 0] = ref_pt_x
        self._matrix[1, 0] = ref_pt_y
        self._matrix[0, 1] = ref_pt_x + x_radius
        self._matrix[1, 1] = ref_pt_y + y_radius
        self._matrix[0, 2] = ref_pt_x + (2 * x_radius)
        self._matrix[1, 2] = ref_pt_y
        self._matrix[0, 3] = ref_pt_x + x_radius
        self._matrix[1, 3] = ref_pt_y - y_radius

        self.calc_control_pts()
        self.calc_bbox_pts()

        # region #### reference site definitions
        if ref_pt == "center":
            self.translate(-x_radius, 0)
        elif ref_pt == "bot_left":
            self.translate(0, y_radius)
        elif ref_pt == "top_left":
            self.translate(0, -y_radius)
        elif ref_pt == "mid_top":
            self.translate(-x_radius, -y_radius)
        elif ref_pt == "top_right":
            self.translate(2 * -x_radius, -y_radius)
        elif ref_pt == "mid_right":
            self.translate(2 * -x_radius, 0)
        elif ref_pt == "bot_right":
            self.translate(2 * -x_radius, y_radius)
        elif ref_pt == "mid_bot":
            self.translate(-x_radius, y_radius)
        # endregion

    @property
    def radius(self):
        print(
            "An ellipse has a major and minor axis. Use x_radius and y_radius."
        )

