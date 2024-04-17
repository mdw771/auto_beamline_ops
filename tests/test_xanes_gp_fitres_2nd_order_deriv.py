import os
import argparse
import pickle

import gpytorch.kernels
import numpy as np
import botorch
import torch
import pandas as pd
import tqdm
import matplotlib.pyplot as plt

import autobl.steering
from autobl.steering.configs import *
from autobl.steering.measurement import *
from autobl.steering.acquisition import *
from autobl.steering.optimization import *
from autobl.util import *


def rms(actual, true):
    return np.sqrt(np.mean((actual - true) ** 2))


def create_intermediate_figure(n_target_measurements, n_plot_interval=20):
    n_plots = int(np.ceil(n_target_measurements / n_plot_interval))
    n_cols = 3
    n_rows = int(np.ceil(n_plots / n_cols))
    fig, ax = plt.subplots(n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3), squeeze=False)
    return fig, ax


def update_intermediate_figure(guide, energies, data, n_measured, i_plot, axes):
    n_rows = len(axes)
    n_cols = len(axes[0])
    # guide.plot_posterior(energies)
    guide.plot_posterior(energies, ax=axes[i_plot // n_cols][i_plot % n_cols])
    axes[i_plot // n_cols][i_plot % n_cols].plot(to_numpy(energies), data, label='Truth', color='gray', alpha=0.6)
    axes[i_plot // n_cols][i_plot % n_cols].set_title('{} points'.format(n_measured))
    axes[i_plot // n_cols][i_plot % n_cols].legend()


def create_intermediate_data_dict(energies, data):
    intermediate_data_dict = {
        'energies': to_numpy(energies),
        'data': to_numpy(data),
        'n_measured_list': [],
        'mu_list': [],
        'sigma_list': []
    }
    return intermediate_data_dict


def update_intermediate_data_dict(guide, config, energies, n_measured, intermediate_data_dict):
    mu, sigma = guide.get_posterior_mean_and_std(energies[:, None])
    mu = mu.squeeze()
    sigma = sigma.squeeze()
    intermediate_data_dict['n_measured_list'].append(n_measured)
    intermediate_data_dict['mu_list'].append(to_numpy(mu))
    intermediate_data_dict['sigma_list'].append(to_numpy(sigma))
    return intermediate_data_dict


def save_intermediate_data_dict(intermediate_data_dict, config, guide, save_dir):
    fname = get_save_name_prefix(config, guide)
    fname = fname + '_intermediate_data.pkl'
    fname = os.path.join(save_dir, fname)
    pickle.dump(intermediate_data_dict, open(fname, 'wb'))


def create_convergence_figure_and_data():
    fig, ax = plt.subplots(1, 1)
    return fig, ax, [], []


def update_convergence_data(guide, energies, data, n_measured, n_measured_list, metric_list):
    mu, _ = guide.get_posterior_mean_and_std(energies[:, None])
    mu = mu.squeeze()
    metric = rms(mu.detach().cpu().numpy(), data)
    n_measured_list.append(n_measured)
    metric_list.append(metric)
    return n_measured_list, metric_list


def plot_convergence(fig, ax, n_measured_list, metric_list):
    ax.plot(n_measured_list, metric_list)
    ax.set_xlabel('Points measured')
    ax.set_ylabel('RMS')


def get_save_name_prefix(config, guide):
    data_name = os.path.splitext(os.path.basename(data_path))[0]

    acquisition_info = config.acquisition_function_class.__name__
    if config.acquisition_function_class in [GradientAwarePosteriorStandardDeviation,
                                             FittingResiduePosteriorStandardDeviation]:
        acquisition_info += '_phi_{}'.format(guide.acquisition_function.phi)
    if config.acquisition_function_class == ComprehensiveAigmentedAcquisitionFunction:
        acquisition_info += '_gradOrder_{}_phiG_{}_phiR_{}'.format(guide.acquisition_function.gradient_order,
                                                                   guide.acquisition_function.phi_g,
                                                                   guide.acquisition_function.phi_r)
        if guide.acquisition_function.gradient_order == 2:
            acquisition_info += '_phiG2_{}'.format(guide.acquisition_function.phi_g2)

    kernel_info = '{}_lengthscale_{:.3f}'.format(guide.model.covar_module.__class__.__name__,
                                                 guide.unscale_by_normalizer_bounds(
                                                     guide.model.covar_module.lengthscale.item()
                                                 ))
    if isinstance(guide.model.covar_module, gpytorch.kernels.MaternKernel):
        kernel_info += '_nu_{}'.format(guide.model.covar_module.nu)

    optimizer_info = config.optimizer_class.__name__

    save_name_prefix = '_'.join([data_name, acquisition_info, kernel_info, optimizer_info])
    return save_name_prefix


def run_simulated_experiment(config, x_init, y_init, energies, data, instrument,
                             n_target_measurements=None, plot_graphs=False):
    guide = autobl.steering.guide.GPExperimentGuide(config)
    guide.build(x_init, y_init)

    if n_target_measurements is None:
        n_target_measurements = len(data) - len(x_init)
    n_plot_interval = 5
    n_measured = len(x_init)

    if plot_graphs:
        fig_intermediate, axes_intermediate = create_intermediate_figure(n_target_measurements, n_plot_interval)
        fig_conv, axes_conv, n_measured_list, metric_list = create_convergence_figure_and_data()
        intermediate_data_dict = create_intermediate_data_dict(energies, data)
        i_plot = 0
    candidate_list = []
    for i in tqdm.trange(n_target_measurements):
        candidates = guide.suggest().double()
        candidate_list.append(candidates.squeeze().detach().cpu().numpy())
        y_new = instrument.measure(candidates).unsqueeze(-1)
        guide.update(candidates, y_new)
        n_measured += len(candidates)
        if plot_graphs:
            intermediate_data_dict = update_intermediate_data_dict(guide, config, energies, n_measured, intermediate_data_dict)
            if i % n_plot_interval == 0:
                update_intermediate_figure(guide, energies, data, n_measured, i_plot, axes_intermediate)
                i_plot += 1
            update_convergence_data(guide, energies, data, n_measured, n_measured_list, metric_list)

    if plot_graphs:
        plot_convergence(fig_conv, axes_conv, n_measured_list, metric_list)
        plt.show()
    return candidate_list


def test_xanes_gp_fitres_2nd_order_deriv(generate_gold=False, debug=False):

    plot_graphs = debug
    set_random_seed(123)

    data_path = 'data/xanes/Sample1_50C_XANES.csv'
    data_all_spectra = pd.read_csv(data_path, header=None)

    data = data_all_spectra.iloc[len(data_all_spectra) // 2].to_numpy()
    energies = data_all_spectra.iloc[0].to_numpy()
    energies = torch.tensor(energies)

    instrument = SimulatedMeasurement(data=(energies[None, :], data))
    n_init = 10
    x_init = torch.linspace(energies[0], energies[-1], n_init).double().reshape(-1, 1)
    y_init = instrument.measure(x_init).reshape(-1, 1)

    ref_spectra_0 = torch.tensor(data_all_spectra.iloc[1].to_numpy())
    ref_spectra_1 = torch.tensor(data_all_spectra.iloc[-1].to_numpy())
    ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
    ref_spectra_x = torch.linspace(0, 1, ref_spectra_y.shape[-1])

    config = XANESExperimentGuideConfig(
        dim_measurement_space=1,
        num_candidates=1,
        model_class=botorch.models.SingleTaskGP,
        model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
        acquisition_function_class=ComprehensiveAigmentedAcquisitionFunction,
        acquisition_function_params={'gradient_order': 2,
                                     'differentiation_method': 'numerical',
                                     'reference_spectra_x': ref_spectra_x,
                                     'reference_spectra_y': ref_spectra_y,
                                     'phi_r': 100,
                                     'phi_g': 2e-2,
                                     'phi_g2': 3e-4},
        override_kernel_lengthscale=7,
        lower_bounds=torch.tensor([energies[0]]),
        upper_bounds=torch.tensor([energies[-1]]),
        optimizer_class=TorchOptimizer,
        optimizer_params={'torch_optimizer': torch.optim.Adam, 'torch_optimizer_options': {'maxiter': 20}}
    )

    candidate_list = run_simulated_experiment(
        config, x_init, y_init, energies, data, instrument,
        n_target_measurements=70,
        plot_graphs=plot_graphs
    )

    # CI
    candidate_list = np.stack(candidate_list)
    gold_dir = os.path.join('gold_data', 'test_xanes_gp_fitres_2nd_order_deriv')
    if generate_gold:
        if not os.path.exists(gold_dir):
            os.makedirs(gold_dir)
        np.save(os.path.join(gold_dir, 'candidates.npy'), candidate_list)

    if not debug:
        gold_data = np.load(os.path.join(gold_dir, 'candidates.npy'), allow_pickle=True)
        print('=== Current ===')
        print(candidate_list)
        print('=== Reference ===')
        print(gold_data)
        assert np.allclose(candidate_list, gold_data)


def test_xanes_gp_fitres_2nd_order_deriv_with_weight_func_ybco_data(generate_gold=False, debug=False):

    plot_graphs = debug
    set_random_seed(123)

    data_path = 'data/xanes/YBCO3data.csv'
    data_all_spectra = pd.read_csv(data_path, header=0)
    # Only keep 8920 - 9080 eV
    data_all_spectra = data_all_spectra.iloc[14:232]
    data = data_all_spectra['YBCO_epararb.0001'].to_numpy()
    ref_spectra_0 = torch.tensor(data_all_spectra['YBCO_epara.0001'].to_numpy())
    ref_spectra_1 = torch.tensor(data_all_spectra['YBCO_eparc.0001'].to_numpy())
    energies = data_all_spectra['energy'].to_numpy()
    energies = torch.tensor(energies)

    instrument = SimulatedMeasurement(data=(energies[None, :], data))
    n_init = 20
    x_init = torch.linspace(energies[0], energies[-1], n_init).double().reshape(-1, 1)
    y_init = instrument.measure(x_init).reshape(-1, 1)

    ref_spectra_y = torch.stack([ref_spectra_0, ref_spectra_1], dim=0)
    ref_spectra_x = torch.linspace(0, 1, ref_spectra_y.shape[-1])

    config = XANESExperimentGuideConfig(
        dim_measurement_space=1,
        num_candidates=1,
        model_class=botorch.models.SingleTaskGP,
        model_params={'covar_module': gpytorch.kernels.MaternKernel(2.5)},
        noise_variance=1e-6,
        override_kernel_lengthscale=7,
        lower_bounds=torch.tensor([energies[0]]),
        upper_bounds=torch.tensor([energies[-1]]),
        acquisition_function_class=ComprehensiveAigmentedAcquisitionFunction,
        acquisition_function_params={'gradient_order': 2,
                                     'differentiation_method': 'numerical',
                                     'reference_spectra_x': ref_spectra_x,
                                     'reference_spectra_y': ref_spectra_y,
                                     'phi_r': 1e1,
                                     'phi_g': 1e-2,  # 2e-2,
                                     'phi_g2': 1e-4,  # 3e-4
                                     'addon_term_lower_bound': 1e-2,
                                     },
        optimizer_class=TorchOptimizer,
        optimizer_params={'torch_optimizer': torch.optim.Adam, 'torch_optimizer_options': {'maxiter': 20}},
        n_updates_create_acqf_weight_func=5,
        acqf_weight_func_floor_value=0.1,
        acqf_weight_func_post_edge_gain=3.0
    )

    candidate_list = run_simulated_experiment(
        config, x_init, y_init, energies, data, instrument,
        n_target_measurements=70,
        plot_graphs=plot_graphs
    )

    # CI
    candidate_list = np.stack(candidate_list)
    gold_dir = os.path.join('gold_data', 'test_xanes_gp_fitres_2nd_order_deriv_with_weight_func_ybco_data')
    if generate_gold:
        if not os.path.exists(gold_dir):
            os.makedirs(gold_dir)
        np.save(os.path.join(gold_dir, 'candidates.npy'), candidate_list)

    if not debug:
        gold_data = np.load(os.path.join(gold_dir, 'candidates.npy'), allow_pickle=True)
        print('=== Current ===')
        print(candidate_list)
        print('=== Reference ===')
        print(gold_data)
        assert np.allclose(candidate_list, gold_data)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--generate-gold', action='store_true')
    args = parser.parse_args()

    # test_xanes_gp_fitres_2nd_order_deriv(generate_gold=args.generate_gold, debug=True)
    test_xanes_gp_fitres_2nd_order_deriv_with_weight_func_ybco_data(generate_gold=args.generate_gold, debug=True)
