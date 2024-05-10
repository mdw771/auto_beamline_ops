"""Measurement methods"""

import copy
from typing import Callable, List  # , Sequence

import matplotlib.pyplot as plt
import numpy as np
import scipy.interpolate
import scipy.ndimage as ndi

import autobl.steering.configs as cfg
from autobl import util


class Measurement:
    """
    Measurement class
    """

    def __init__(self, *args, **kwargs):
        """
        Initialize the
        """

    def measure(self, *args, **kwargs):
        """
        Measure the sample
        """


class SimulatedMeasurement(Measurement):
    """
    Simulate a measurement
    """

    def __init__(self, *args, f: Callable = None, data: np.ndarray = None, **kwargs):
        super().__init__(*args, **kwargs)
        assert (f is not None) or (data is not None)
        self.f = f
        self.data = data
        self.interp = None
        if self.data is not None:
            if isinstance(self.data, (list, tuple)):
                x_data, y_data = self.data
                x_data = util.to_numpy(x_data)
                y_data = util.to_numpy(y_data)
            else:
                x_data = np.arange(len(data))
                y_data = util.to_numpy(self.data)
            self.interp = lambda pts: scipy.interpolate.interpn(
                x_data, y_data, pts, bounds_error=False, fill_value=None
            )

    def measure(self, x, *args, **kwargs):
        """
        Take measurement.

        :param x: Tensor. A list of points where the values are measured. Tensor
            shape should be [n_pts, n_dims].
        :return:
        """
        if self.f is not None:
            return util.to_tensor(self.f(x))
        elif self.data is not None:
            x = util.to_numpy(x)
            x = util.to_tensor(self.interp(x))
            return x
        else:
            raise ValueError("f or data cannot both be None.")


class FlyScanSingleValueSimulationMeasurement(Measurement):
    """
    Fly scan simulator for techniques where each exposure measures only a single
    intensity value. The simulator's perform measurement method takes in a
    continuous scan path defined by a list of vertices; the scan path is formed
    by linearly connecting the vertices sequentially. The simulator
    automatically split the scan path into exposures based on the setting of
    exposure time in `sample_params`. Between exposures, there can be dead times
    where no data is acquired based on the setting of dead time. For each
    exposure, several points are sampled along the path, whose interval is
    determined by the setting of `step_size_for_integration_nm` in
    `sample_params`. The intensities sampled at all sampling points are averaged
    as the measurement for that exposure. The positions of all sampling points
    are averaged as the position for that exposure.
    """

    def __init__(self, configs: cfg.FlyScanSimulationConfig):
        """
        The constructor.
        """
        super().__init__()
        self.configs = configs
        self.measured_values = None
        self.measured_positions = None
        self.eps = 1e-6
        self.points_to_sample_all_exposures = []

    def measure(self, vertex_list, *args, vertex_unit="pixel", **kwargs):
        """
        Perform measurement given a fly scan path defined by a list of vertices.

        :param vertex_list: list[list[float, float]]. A list of vertex positions
            that define the scan path, ordered in (y, x). The total number of
            segments is `len(vertex_list) - 1`; for segment i, probe moves from
            `vertex_list[i]` to `vertex_list[i + 1]`.
        :param vertex_unit: str. Can be 'pixel' or 'nm'.
        :return list[float]: measured values at all exposures. The positions of
            the exposures can be retrieved from the `measured_positions`
            attribute.
        """
        vertex_list = np.asarray(vertex_list)
        if vertex_unit == "nm":
            vertex_list = vertex_list / self.configs.setup_params.psize_nm

        self.build_sampling_points(vertex_list)

        meas_value_list = []
        meas_pos_list = []
        for pts_to_sample in self.points_to_sample_all_exposures:
            if len(pts_to_sample) == 0:
                continue
            sampled_vals = self.get_interpolated_values_from_image(pts_to_sample)
            meas_value_list.append(np.mean(sampled_vals))
            meas_pos_list.append(np.mean(pts_to_sample, axis=0))
        self.measured_values = np.array(meas_value_list)
        self.measured_positions = np.stack(meas_pos_list, axis=0)
        return self.measured_values

    def build_sampling_points(self, vertex_list):
        """
        Given vertices, compute the points that are sampled along the scan path
        (between vertices)
        """
        points_to_sample_all_exposures = []
        i_input_segment = 0
        length_covered_in_current_segment = 0
        pt_coords = vertex_list[0]
        while i_input_segment < len(vertex_list) - 1:
            length_exposed = 0
            length_dead = 0
            length_sampling = 0
            # Add live segment.
            points_to_sample_current_exposure = [pt_coords]
            while (
                length_exposed
                < self.configs.setup_params.exposure_length_pixel - self.eps
            ):
                if i_input_segment + 1 >= len(vertex_list):
                    break
                current_direction = (
                    vertex_list[i_input_segment + 1] - vertex_list[i_input_segment]
                )
                current_seg_length = np.linalg.norm(current_direction)
                current_direction = current_direction / current_seg_length
                if (
                    length_covered_in_current_segment
                    + self.configs.step_size_for_integration_pixel
                    - length_sampling
                    <= current_seg_length
                ):
                    pt_coords = pt_coords + current_direction * (
                        self.configs.step_size_for_integration_pixel - length_sampling
                    )
                    points_to_sample_current_exposure.append(pt_coords)
                    length_covered_in_current_segment += (
                        self.configs.step_size_for_integration_pixel - length_sampling
                    )
                    length_exposed += (
                        self.configs.step_size_for_integration_pixel - length_sampling
                    )
                    length_sampling = 0
                else:
                    if i_input_segment + 1 >= len(vertex_list):
                        break
                    pt_coords = pt_coords + current_direction * (
                        current_seg_length - length_covered_in_current_segment
                    )
                    i_input_segment += 1
                    length_exposed += (
                        current_seg_length - length_covered_in_current_segment
                    )
                    length_sampling += (
                        current_seg_length - length_covered_in_current_segment
                    )
                    length_covered_in_current_segment = 0
            # Update variables for dead segment.
            while length_dead < self.configs.setup_params.dead_length_pixel - self.eps:
                if i_input_segment + 1 >= len(vertex_list):
                    break
                current_direction = (
                    vertex_list[i_input_segment + 1] - vertex_list[i_input_segment]
                )
                current_seg_length = np.linalg.norm(current_direction)
                current_direction = current_direction / current_seg_length
                if (
                    length_covered_in_current_segment
                    + self.configs.setup_params.dead_length_pixel
                    - length_dead
                    <= current_seg_length
                ):
                    pt_coords = pt_coords + current_direction * (
                        self.configs.setup_params.dead_length_pixel - length_dead
                    )
                    length_covered_in_current_segment += (
                        self.configs.setup_params.dead_length_pixel - length_dead
                    )
                    break
                else:
                    if i_input_segment + 1 >= len(vertex_list):
                        break
                    pt_coords = vertex_list[i_input_segment + 1]
                    i_input_segment += 1
                    length_dead += (
                        current_seg_length - length_covered_in_current_segment
                    )
                    length_covered_in_current_segment = 0
            points_to_sample_all_exposures.append(points_to_sample_current_exposure)
        self.points_to_sample_all_exposures = points_to_sample_all_exposures

    def plot_sampled_points(self):
        """
        Plot the points that will be sampled along the path
        """
        pts = np.concatenate(self.points_to_sample_all_exposures, axis=0)

        fig, ax = plt.subplots(1, 1)
        ax.scatter(pts[:, 1], pts[:, 0])
        plt.show()
        return fig, ax

    def get_interpolated_values_from_image(self, point_list, normalize_probe=True):
        """
        Obtain interpolated values from the image at given locations.

        :param point_list: list[list[float, float]]. List of point positions.
        :return: list[float].
        """
        if not isinstance(point_list, np.ndarray):
            point_list = np.array(point_list)
        y = point_list[:, 0]
        x = point_list[:, 1]
        if self.configs.setup_params.probe is None:
            # If probe function is not given, assume delta function.
            return ndi.map_coordinates(
                self.configs.sample_params.image, [y, x], order=1, mode="nearest"
            )
        else:
            # Prepare a list of coordinates that include n by m region around
            # each sampled point, where (n, m) is the probe shape.
            sampled_vals = []
            probe = self.configs.setup_params.probe
            if normalize_probe:
                probe = probe / np.sum(probe)
            for this_y, this_x in point_list:
                # Expand around each point to capture probe area
                this_y_all = np.linspace(
                    this_y - probe.shape[0] / 2.0 + 0.5,
                    this_y + probe.shape[0] / 2.0 - 0.5,
                    probe.shape[0],
                )
                this_x_all = np.linspace(
                    this_x - probe.shape[1] / 2.0 + 0.5,
                    this_x + probe.shape[1] / 2.0 - 0.5,
                    probe.shape[1],
                )

                # Combine expansions in x and y to get total expanded area
                xx, yy = np.meshgrid(this_x_all, this_y_all)
                yy = yy.reshape(-1)
                xx = xx.reshape(-1)

                # Get image interpolated values from the image
                vals = ndi.map_coordinates(
                    self.configs.sample_params.image, [yy, xx], order=1, mode="nearest"
                )
                # ... and multiply them with the probe weights
                val = np.sum(vals * probe.reshape(-1))
                sampled_vals.append(val)
            return sampled_vals


class FlyScanPathGenerator:
    """
    Fly scan path generator for raster scans
    """

    def __init__(self, shape, psize_nm=None, return_coordinates_type="pixel"):
        """
        Fly scan path generator.

        :param shape: list[int, int]. Shape of the support in pixel.
        :param psize_nm: float. Pixel size in nm.
        :param return_coordinates_type: str. Can be 'pixel' or 'nm'. Sets the
            unit of the returned coordinates.
        """
        self.shape = shape
        self.psize_nm = psize_nm
        self.return_coordinates_type = return_coordinates_type
        self.generated_path = []
        self.dead_segment_mask = []

    def plot_path(self):
        """
        Plot the flyscan path
        """
        fig = plt.figure()
        plt.plot(self.generated_path[:, 1], self.generated_path[:, 0])
        plt.show()

    def generate_raster_scan_path(
        self,
        pos_top_left: List[float],
        pos_bottom_right: List[float],
        vertical_spacing: float,
    ) -> List[List[float]]:
        """
        Generate a raster (regular) scan path starting at pos_top_left and
        scanning (down) towards pos_bottom_right. Creates long left-right scan
        segments and short vertical (down) segments. Vertical segments are not
        scanned (dead).

        :param pos_top_left: list[float, float]. Top left vertex of the scan
            grid in pixel. Coordinates are defined with regards to the support.
        :param pos_bottom_right: list[float, float]. Bottom right vertex of the
            scan grid in pixel.
        :param vertical_spacing: float. Spacing of adjacent scan lines in pixel.
        :return: list[list[float, float]]. A list of vertices that define the
            scan path.
        """
        # Indicates whether the current vertex is on the left or right of the grid.
        current_side = 0
        current_point = copy.deepcopy(np.array(pos_top_left))
        self.generated_path.append(copy.deepcopy(current_point))
        while True:
            if current_side == 0:
                current_point[1] = pos_bottom_right[1]
            else:
                current_point[1] = pos_top_left[1]
            current_side = 1 - current_side
            self.generated_path.append(copy.deepcopy(current_point))
            if current_point[0] + vertical_spacing > pos_bottom_right[0]:
                break
            current_point[0] += vertical_spacing
            self.generated_path.append(copy.deepcopy(current_point))
        self.generated_path = np.stack(self.generated_path)

        self.dead_segment_mask = np.ones(len(self.generated_path) - 1, dtype="bool")
        self.dead_segment_mask[1::2] = False

        if self.return_coordinates_type == "nm":
            return self.generated_path * self.psize_nm
        return self.generated_path
