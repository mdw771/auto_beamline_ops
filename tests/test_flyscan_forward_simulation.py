import os
import argparse

import tifffile
from pathlib import Path
from scipy.interpolate import griddata
import matplotlib.pyplot as plt
import numpy as np

from autobl.steering.configs import *
from autobl.steering.measurement import *
from autobl.image_proc import DenseReconstructor


def run_simulation(image, scan_path, probe=None):
    sample_params = SpatialSampleParams(
        image=image,
        psize_nm=1.0
    )
    setup_params = FlyScanExperimentSetupParams(
        psize_nm=1.0,
        scan_speed_nm_sec=1.0,
        exposure_sec=0.8,
        deadtime_sec=0.2,
        probe=probe
    )
    measurement_configs = FlyScanSimulationConfig(
        sample_params=sample_params,
        setup_params=setup_params,
        step_size_for_integration_nm=0.1,
    )

    measurement_interface = FlyScanSingleValueSimulationMeasurement(measurement_configs)
    measured_values = measurement_interface.measure(scan_path, 'pixel')
    # measurement_interface.plot_sampled_points()
    measured_positions = measurement_interface.measured_positions
    return measured_values, measured_positions


def build_dir():
    if not os.path.exists(os.path.join('gold_data', 'test_flyscan_forward_simulation')):
        os.makedirs(os.path.join('gold_data', 'test_flyscan_forward_simulation'))


def test_flyscan_forward_simulation_provided_probe(generate_gold=False, return_results=False, skip_comparison=False):
    build_dir()
    image = tifffile.imread(os.path.join('data', 'xrf', 'xrf_2idd_Cs_L.tiff'))
    path_gen = FlyScanPathGenerator(image.shape)
    scan_path = path_gen.generate_raster_scan_path([1, 1], [133, 131], 1)
    measured_values, measured_positions = run_simulation(image, scan_path, probe=np.ones([4, 4]))
    if generate_gold:
        np.save(os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_values_provided_probe.npy'),
                measured_values)
        np.save(os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_positions_provided_probe.npy'),
                measured_positions)
    else:
        if not skip_comparison:
            measured_values_gold = np.load(
                os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_values_provided_probe.npy'))
            measured_positions_gold = np.load(
                os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_positions_provided_probe.npy'))
            assert np.allclose(measured_values, measured_values_gold)
            assert np.allclose(measured_positions, measured_positions_gold)
    if return_results:
        return measured_values, measured_positions
    return None


def test_flyscan_forward_simulation_delta_probe(generate_gold=False, return_results=False, skip_comparison=False):
    build_dir()
    image = tifffile.imread(os.path.join('data', 'xrf', 'xrf_2idd_Cs_L.tiff'))
    path_gen = FlyScanPathGenerator(image.shape)
    scan_path = path_gen.generate_raster_scan_path([1, 1], [133, 131], 1)
    measured_values, measured_positions = run_simulation(image, scan_path, probe=None)
    if generate_gold:
        np.save(os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_values_delta_probe.npy'),
                measured_values)
        np.save(os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_positions_delta_probe.npy'),
                measured_positions)
    else:
        if not skip_comparison:
            measured_values_gold = np.load(
                os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_values_delta_probe.npy'))
            measured_positions_gold = np.load(
                os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_positions_delta_probe.npy'))
            assert np.allclose(measured_values, measured_values_gold)
            assert np.allclose(measured_positions, measured_positions_gold)
    if return_results:
        return measured_values, measured_positions
    return None


def test_flyscan_forward_simulation_delta_probe_arbitrary_path(generate_gold=False, return_results=False, skip_comparison=False):
    build_dir()
    image = tifffile.imread(os.path.join('data', 'xrf', 'xrf_2idd_Cs_L.tiff'))
    scan_path = [[0, 0], [10, 10], [50, 20], [55, 30], [55, 10], [100, 60]]
    measured_values, measured_positions = run_simulation(image, scan_path, probe=None)
    if generate_gold:
        np.save(os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_values_delta_probe_arbitrary_path.npy'),
                measured_values)
        np.save(os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_positions_delta_probe_arbitrary_path.npy'),
                measured_positions)
    else:
        if not skip_comparison:
            measured_values_gold = np.load(
                os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_values_delta_probe_arbitrary_path.npy'))
            measured_positions_gold = np.load(
                os.path.join('gold_data', 'test_flyscan_forward_simulation', 'measured_positions_delta_probe_arbitrary_path.npy'))
            assert np.allclose(measured_values, measured_values_gold)
            assert np.allclose(measured_positions, measured_positions_gold)
    if return_results:
        return measured_values, measured_positions
    return None


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    def show_results(measured_positions, measured_values):
        image = tifffile.imread(os.path.join('data', 'xrf', 'xrf_2idd_Cs_L.tiff'))
        grid_y, grid_x = np.mgrid[:image.shape[0], :image.shape[1]]
        recon = DenseReconstructor(method='linear').reconstruct(measured_positions, measured_values, (grid_y, grid_x))

        fig, axes = plt.subplots(1, 2)
        axes[0].imshow(image)
        axes[0].set_title('Orignal image')
        axes[1].imshow(recon)
        axes[1].set_title('Sampled')
        plt.show()
        plt.close(fig)

    measured_values, measured_positions = test_flyscan_forward_simulation_delta_probe(
        generate_gold=args.generate_gold,
        return_results=True,
        skip_comparison=True
    )
    if not args.generate_gold:
        show_results(measured_positions, measured_values)

    measured_values, measured_positions = test_flyscan_forward_simulation_provided_probe(
        generate_gold=args.generate_gold,
        return_results=True,
        skip_comparison=True
    )
    if not args.generate_gold:
        show_results(measured_positions, measured_values)

    measured_values, measured_positions = test_flyscan_forward_simulation_delta_probe_arbitrary_path(
        generate_gold=args.generate_gold,
        return_results=True,
        skip_comparison=True
    )
    if not args.generate_gold:
        show_results(measured_positions, measured_values)
