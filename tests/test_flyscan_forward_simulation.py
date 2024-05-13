"""
Tests for various flyscan measurements (raster scans and arbitrary paths)
"""
import os
import sys
import argparse

# Get the full path to the parent directory
parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

# Add the parent directory to the system path
sys.path.append(parent_dir)

import tifffile
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

from autobl.steering import configs
from autobl.steering import measurement


def run_simulation(image, scan_path, probe=None):
    """Run the simulation of the flyscan"""
    sample_params = configs.SpatialSampleParams(image=image, psize_nm=1.0)
    setup_params = configs.FlyScanExperimentSetupParams(
        psize_nm=1.0,
        scan_speed_nm_sec=1.0,
        exposure_sec=0.8,
        deadtime_sec=0.2,
        probe=probe,
    )
    measurement_configs = configs.FlyScanSimulationConfig(
        sample_params=sample_params,
        setup_params=setup_params,
        step_size_for_integration_nm=0.1,
    )

    measurement_interface = measurement.FlyScanSingleValueSimulationMeasurement(
        measurement_configs
    )
    _measured_values = measurement_interface.measure(scan_path, "pixel")
    # measurement_interface.plot_sampled_points()
    _measured_positions = measurement_interface.measured_positions
    return _measured_values, _measured_positions


def build_dir():
    """Create the build directory"""
    if not os.path.exists(os.path.join("gold_data", "test_flyscan_forward_simulation")):
        os.makedirs(os.path.join("gold_data", "test_flyscan_forward_simulation"))


def test_flyscan_forward_simulation_provided_probe(
    generate_gold: bool = False,
    return_results: bool = False,
    skip_comparison: bool = False,
    probe: np.ndarray = np.ones((4, 4)),
):
    """
    Test the flyscan using a provided probe and a raster scan path
    """
    build_dir()
    image = tifffile.imread(os.path.join("data", "xrf", "xrf_2idd_Cs_L.tiff"))
    path_gen = measurement.FlyScanPathGenerator(image.shape)
    scan_path = path_gen.generate_raster_scan_path([1, 1], [133, 131], 1)
    _measured_values, _measured_positions = run_simulation(
        image, scan_path, probe=probe
    )
    if generate_gold:
        np.save(
            os.path.join(
                "gold_data",
                "test_flyscan_forward_simulation",
                "measured_values_provided_probe.npy",
            ),
            _measured_values,
        )
        np.save(
            os.path.join(
                "gold_data",
                "test_flyscan_forward_simulation",
                "measured_positions_provided_probe.npy",
            ),
            _measured_positions,
        )
    else:
        if not skip_comparison:
            measured_values_gold = np.load(
                os.path.join(
                    "gold_data",
                    "test_flyscan_forward_simulation",
                    "measured_values_provided_probe.npy",
                )
            )
            measured_positions_gold = np.load(
                os.path.join(
                    "gold_data",
                    "test_flyscan_forward_simulation",
                    "measured_positions_provided_probe.npy",
                )
            )
            assert np.allclose(_measured_values, measured_values_gold)
            assert np.allclose(_measured_positions, measured_positions_gold)
    if return_results:
        return _measured_values, _measured_positions
    return None


def test_flyscan_forward_simulation_delta_probe(
    generate_gold=False, return_results=False, skip_comparison=False
):
    """
    Test the flyscan forward simulation using a delta-function probe and a
    raster scan path
    """
    build_dir()
    image = tifffile.imread(os.path.join("data", "xrf", "xrf_2idd_Cs_L.tiff"))
    path_gen = measurement.FlyScanPathGenerator(image.shape)
    scan_path = path_gen.generate_raster_scan_path([1, 1], [133, 131], 1)
    _measured_values, _measured_positions = run_simulation(image, scan_path, probe=None)
    if generate_gold:
        np.save(
            os.path.join(
                "gold_data",
                "test_flyscan_forward_simulation",
                "measured_values_delta_probe.npy",
            ),
            _measured_values,
        )
        np.save(
            os.path.join(
                "gold_data",
                "test_flyscan_forward_simulation",
                "measured_positions_delta_probe.npy",
            ),
            _measured_positions,
        )
    else:
        if not skip_comparison:
            measured_values_gold = np.load(
                os.path.join(
                    "gold_data",
                    "test_flyscan_forward_simulation",
                    "measured_values_delta_probe.npy",
                )
            )
            measured_positions_gold = np.load(
                os.path.join(
                    "gold_data",
                    "test_flyscan_forward_simulation",
                    "measured_positions_delta_probe.npy",
                )
            )
            assert np.allclose(_measured_values, measured_values_gold)
            assert np.allclose(_measured_positions, measured_positions_gold)
    if return_results:
        return _measured_values, _measured_positions
    return None


def test_flyscan_forward_simulation_delta_probe_arbitrary_path(
    generate_gold=False, return_results=False, skip_comparison=False
):
    """
    Test the flyscan forward simulation using a delta-function probe and an
    arbitrary path
    """
    build_dir()
    image = tifffile.imread(os.path.join("data", "xrf", "xrf_2idd_Cs_L.tiff"))
    scan_path = [[0, 0], [10, 10], [50, 20], [55, 30], [55, 10], [100, 60]]
    _measured_values, _measured_positions = run_simulation(image, scan_path, probe=None)
    if generate_gold:
        np.save(
            os.path.join(
                "gold_data",
                "test_flyscan_forward_simulation",
                "measured_values_delta_probe_arbitrary_path.npy",
            ),
            _measured_values,
        )
        np.save(
            os.path.join(
                "gold_data",
                "test_flyscan_forward_simulation",
                "measured_positions_delta_probe_arbitrary_path.npy",
            ),
            _measured_positions,
        )
    else:
        if not skip_comparison:
            measured_values_gold = np.load(
                os.path.join(
                    "gold_data",
                    "test_flyscan_forward_simulation",
                    "measured_values_delta_probe_arbitrary_path.npy",
                )
            )
            measured_positions_gold = np.load(
                os.path.join(
                    "gold_data",
                    "test_flyscan_forward_simulation",
                    "measured_positions_delta_probe_arbitrary_path.npy",
                )
            )
            assert np.allclose(_measured_values, measured_values_gold)
            assert np.allclose(_measured_positions, measured_positions_gold)
    if return_results:
        return _measured_values, _measured_positions
    return None


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--generate-gold", action="store_true")
    args = parser.parse_args()

    def show_results(_measured_positions, _measured_values):
        """Plot the results"""
        image = tifffile.imread(os.path.join("data", "xrf", "xrf_2idd_Cs_L.tiff"))
        grid_y, grid_x = np.mgrid[: image.shape[0], : image.shape[1]]
        recon = griddata(_measured_positions, _measured_values, (grid_y, grid_x))

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title("Original image")
        axes[1].imshow(recon)
        axes[1].set_title("Sampled")
        plt.show()
        plt.close(fig)

    measured_values, measured_positions = test_flyscan_forward_simulation_delta_probe(
        generate_gold=args.generate_gold, return_results=True, skip_comparison=True
    )
    if not args.generate_gold:
        show_results(measured_positions, measured_values)

    measured_values, measured_positions = (
        test_flyscan_forward_simulation_provided_probe(
            generate_gold=args.generate_gold, return_results=True, skip_comparison=True
        )
    )
    if not args.generate_gold:
        show_results(measured_positions, measured_values)

    measured_values, measured_positions = (
        test_flyscan_forward_simulation_delta_probe_arbitrary_path(
            generate_gold=args.generate_gold, return_results=True, skip_comparison=True
        )
    )
    if not args.generate_gold:
        show_results(measured_positions, measured_values)
