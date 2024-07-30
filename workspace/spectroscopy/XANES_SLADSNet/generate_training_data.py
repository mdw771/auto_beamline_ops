import os
import logging

import numpy as np
import sklearn.preprocessing
import h5py
import scipy
import sklearn
import tqdm
import matplotlib.pyplot as plt

import autobl.steering
import autobl.steering.io_util
from autobl.steering.measurement import SimulatedMeasurement
from autobl.util import reconstruct_spectrum


class CombinedDataset:
    """
    A dataset that combines multiple datasets. Samples are indexed based on the order
    by which the member datasets are given.
    """

    def __init__(self, datasets):
        self.datasets = datasets
        self._length = 0
        self.length_list = []
        for d in self.datasets:
            self._length += len(d)
            self.length_list.append(len(d))
        self.bins = [0] + list(np.cumsum(self.length_list))

    def map_global_index_to_local(self, ind):
        """
        Find the dataset index and the local index in that dataset given a global index.
        """
        dset_ind = np.digitize(ind, self.bins, right=False) - 1
        local_ind = ind - self.bins[dset_ind]
        return dset_ind, local_ind

    def __len__(self):
        return self._length

    def __getitem__(self, ind, return_energies=False):
        dset_ind, local_ind = self.map_global_index_to_local(ind)
        if return_energies:
            return self.datasets[dset_ind][local_ind], self.datasets[dset_ind].energies_ev
        return self.datasets[dset_ind][local_ind]


class TrainingDataGenerator:

    def __init__(self, 
                 dataset: autobl.steering.io_util.SpectroscopyDataset | CombinedDataset, 
                 sampling_ratios: tuple[float, ...] = (0.05, 0.1, 0.15, 0.2, 0.25, 0.3, 0.4, 0.5, 0.6),
                 recon_pixel_size_ev: float = 1.0,
                 min_num_points: int = 10,
                 output_path: str = 'data/data_train.h5'):
        self.dataset = dataset
        self.sampling_ratios = sampling_ratios
        self.instrument = None
        self.recon_psize_ev = recon_pixel_size_ev
        self.min_num_points = min_num_points
        self.output_path = output_path
        self.f = None
        self.i_sample = 0
        
        self.y_true = None
        self.x_true = None
        self.x_true_norm = None
        self.y_measured = None
        self.x_measured = None
        self.x_measured_norm = None
        self.x_interp = None
        self.y_true_interp = None
        self.lb_ev, self.ub_ev = None, None
        self.n_recon_pixels = None
        self.r0 = None
        
        self.standardizer = sklearn.preprocessing.StandardScaler()
        self.metric = lambda y1, y2: np.mean((y1 - y2) ** 2)
        
    def build_file(self):
        if not os.path.exists(os.path.dirname(self.output_path)):
            os.makedirs(os.path.dirname(self.output_path))
        if os.path.exists(self.output_path):
            overwrite = input(f'File {self.output_path} already exists. Overwrite? (y/n) ')
            if overwrite not in ['y', 'Y']:
                raise FileExistsError
        self.f = h5py.File(self.output_path, 'w')
        
        n_samples, n_measured_max = self.estimate_training_set_size()
        self.f.create_dataset('spec_id', (n_samples,), dtype=int)
        self.f.create_dataset('x', (n_samples,), dtype='float32')
        self.f.create_dataset('sampling_ratio', (n_samples,), dtype='float32')
        self.f.create_dataset('n_measured', (n_samples,), dtype=int)
        self.f.create_dataset('x_measured', (n_samples, n_measured_max), dtype='float32')
        self.f.create_dataset('y_measured', (n_samples, n_measured_max), dtype='float32')
        self.f.create_dataset('erd', (n_samples,), dtype='f')
        
    def estimate_training_set_size(self):
        n_samples = 0
        n_measured_max = 0
        for i in range(len(self.dataset)):
            _, energies = self.dataset.__getitem__(i, return_energies=True)
            lb_ev, ub_ev = energies.min(), energies.max()
            n_pix = self.get_num_pixels(lb_ev, ub_ev)
            for sampling_ratio in self.sampling_ratios:
                if int(sampling_ratio * n_pix) < self.min_num_points:
                    continue
                n_samples += n_pix
                n_measured_max = max(n_measured_max, int(np.max(self.sampling_ratios) * n_pix))
        return n_samples, n_measured_max
    
    def get_num_pixels(self, lb, ub):
        return int((ub - lb) / self.recon_psize_ev)
    
    def run(self):
        self.build_file()
        for i in range(len(self.dataset)):
            self.build_for_spectrum(i)
            self.run_for_spectrum(i)
        self.f.close()
    
    def build_for_spectrum(self, ind: int):
        self.y_true, self.x_true = self.dataset.__getitem__(ind, return_energies=True)
        
        self.standardizer.fit(self.y_true[:, None])
        self.lb_ev, self.ub_ev = self.x_true.min(), self.x_true.max()
        
        # Standardize truth data
        self.y_true = self.standardizer.transform(self.y_true[:, None]).squeeze()
        
        self.instrument = SimulatedMeasurement(data=(self.x_true[None, :], self.y_true))
        self.n_recon_pixels = self.get_num_pixels(self.lb_ev, self.ub_ev)
        self.x_interp = np.linspace(self.lb_ev, self.ub_ev, self.n_recon_pixels)
        self.x_interp_norm = np.linspace(0, 1, self.n_recon_pixels)
        self.y_true_interp = self.instrument.measure(self.x_interp[:, None]).numpy()
            
    def run_for_spectrum(self, ind: int):
        self.build_for_spectrum(ind)
        for sampling_ratio in tqdm.tqdm(self.sampling_ratios):
            logging.info(f'Spectrum ({ind + 1}/{len(self.dataset)}); sampling ratio: {sampling_ratio}')
            n_measured = int(self.n_recon_pixels * sampling_ratio)
            if n_measured < self.min_num_points:
                logging.info(f'Skipping this sampling ratio because of too few points: {n_measured} < {self.min_num_points}')
                continue
            self.perform_initial_measurements(n_measured)
            y_recon_0 = reconstruct_spectrum(self.x_measured, self.y_measured, self.x_interp, method="linear")
            r0 = self.metric(y_recon_0, self.y_true_interp)
            
            for x, x_norm in zip(self.x_interp, self.x_interp_norm):
                x_new_measured = np.append(self.x_measured, x)
                y_new_measured = np.append(self.y_measured, 
                                           self.instrument.measure(np.array([[x]])).squeeze().numpy()
                                           )
                y_recon = reconstruct_spectrum(x_new_measured, y_new_measured, self.x_interp, method="linear")
                r1 = self.metric(y_recon, self.y_true_interp)
                erd = np.clip(r0 - r1, 0, None)
                self.record_data(ind, x_norm, self.x_measured_norm, self.y_measured, sampling_ratio, erd)
    
    def record_data(self, spec_id, x, x_measured, y_measured, sampling_ratio, erd):
        self.f['spec_id'][self.i_sample] = spec_id
        sorted_inds = np.argsort(x_measured)
        x_measured = x_measured[sorted_inds]
        y_measured = y_measured[sorted_inds]
        self.f['x'][self.i_sample] = x
        n_measured = len(x_measured)
        self.f['n_measured'][self.i_sample] = n_measured
        self.f['sampling_ratio'][self.i_sample] = sampling_ratio
        self.f['x_measured'][self.i_sample, :n_measured] = x_measured
        self.f['y_measured'][self.i_sample, :n_measured] = y_measured
        self.f['erd'][self.i_sample] = erd
        self.i_sample += 1
            
    def perform_initial_measurements(self, n):
        self.x_measured_norm = np.concatenate([np.array([0, 1]), np.random.rand(n - 2)])
        self.x_measured = self.lb_ev + self.x_measured_norm * (self.ub_ev - self.lb_ev)
        self.y_measured = self.instrument.measure(self.x_measured[:, None]).numpy()
    

class TrainingDatasetVisualizer:
    
    def __init__(self, path):
        self.path = path
        self.f = h5py.File(path, 'r')
        self.spectrum_nmeas_combs = []
        
    def build(self):
        spec_ids = self.f['spec_id'][...]
        n_measured_list = self.f['n_measured'][...]
        self.spectrum_nmeas_combs = np.unique(np.stack((spec_ids, n_measured_list), axis=1), axis=0)
        
    def run(self):
        self.build()
        for comb in tqdm.tqdm(self.spectrum_nmeas_combs):
            spec_id, n_measured = comb
            mask = (self.f['spec_id'][...] == spec_id) & (self.f['n_measured'][...] == n_measured)
            ind = np.where(mask)[0][0]
            x_measured, y_measured = self.f['x_measured'][ind], self.f['y_measured'][ind]
            x_measured = x_measured[:n_measured]
            y_measured = y_measured[:n_measured]
            erds = self.f['erd'][...][mask]
            x_interp = np.linspace(0, 1, len(erds))
            
            fig, ax = plt.subplots(1, 1)
            ax.scatter(x_measured, y_measured, label='measured')
            ax2 = ax.twinx()
            ax2.plot(x_interp, erds)
            plt.show()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='[%(asctime)s] - %(message)s')
    
    dset1 = autobl.steering.io_util.YBCORawDataset('data/raw/YBCO/YBCO_epararb.0001')
    dset1.crop_by_energy(8920, 9080)
    dset2 = autobl.steering.io_util.LTORawDataset('data/raw/LiTiO_XANES/rawdata', filename_pattern="LTOsample3.[0-9]*")
    dset2.crop_by_energy(4936, 5006)
    dset3 = autobl.steering.io_util.LTORawDataset('data/raw/LiTiO_XANES/rawdata', filename_pattern="LTOsample1.[0-9]*")
    dset3.crop_by_energy(4936, 5006)
    dset4 = autobl.steering.io_util.NORSpectroscopyDataset(path="data/raw/Pt-XANES/Pt_xmu", file_pattern="10Feb_PtL3_025_042C.xmu", data_column='xmu')
    dset4.crop_by_energy(11400, 11700)
    dset = CombinedDataset([dset1, dset2, dset3, dset4])
    
    data_gen = TrainingDataGenerator(dset,
                                     output_path='slads_data/data_train.h5')
    data_gen.run()

    visualizer = TrainingDatasetVisualizer('slads_data/data_train.h5')
    visualizer.run()
