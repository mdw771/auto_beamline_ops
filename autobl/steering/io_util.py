import glob
import os

import pandas as pd
import numpy as np
import h5py
import torch
from torch.utils.data import Dataset

from autobl.util import reconstruct_spectrum


class SpectroscopyDataset:

    def __init__(self, path, *args, **kwargs):
        self.path = path
        self.data = None
        self.energies_ev = None

    def __len__(self):
        return len(self.data)

    def __getitem__(self, item):
        return self.data[item]
    
    def crop_by_energy(self, min_energy_ev, max_energy_ev):
        mask = (self.energies_ev >= min_energy_ev) & (self.energies_ev <= max_energy_ev)
        self.data = self.data[:, mask]
        self.energies_ev = self.energies_ev[mask]


class ColumnMajorCSVSpectroscopyDataset(SpectroscopyDataset):
    def __init__(self, *args, column_name="YBCO_epararb.0001", **kwargs):
        super().__init__(*args, **kwargs)
        table = pd.read_csv(self.path, header=0)
        self.data = table[column_name].to_numpy()
        self.data = self.data.reshape(1, -1)
        self.energies_ev = table['energy'].to_numpy()


class YBCORawDataset(SpectroscopyDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table = pd.read_table(self.path, comment='#', header=None, sep='\s+')
        self.energies_ev = table[0].to_numpy()
        self.data = np.log(table[4] / table[5]).to_numpy()
        self.data = self.data.reshape(1, -1)
        

class LTORawDataset(SpectroscopyDataset):
    def __init__(self, *args, filename_pattern='*', **kwargs):
        super().__init__(*args, **kwargs)
        filelist = glob.glob(os.path.join(self.path, filename_pattern))
        filelist.sort()
        self.data = []
        for i, fname in enumerate(filelist):
            table = pd.read_table(fname, comment='#', header=None, sep='\s+')
            if i == 0:
                self.energies_ev = table[0].to_numpy()
            else:
                assert len(table[0].to_numpy()) == len(self.energies_ev), \
                        "Inconsistent number of points at {} ({} vs {}).".format(
                            i, len(table[0].to_numpy()), len(self.energies_ev))
            data = np.log(table[2] / table[3]).to_numpy()
            self.data.append(data)
        self.data = np.stack(self.data)


class RowMajorCSVSpectroscopyDataset(SpectroscopyDataset):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        table = pd.read_csv(self.path, header=None)
        self.data = table.iloc[1:].to_numpy()
        self.energies_ev = table.iloc[0].to_numpy()


class NORSpectroscopyDataset(SpectroscopyDataset):
    def __init__(self, *args, file_pattern='*.nor', data_column='norm', **kwargs):
        super().__init__(*args, **kwargs)
        self.data_column = data_column
        flist = glob.glob(os.path.join(self.path, file_pattern))
        flist.sort()
        self.data = []
        for f in flist:
            d, self.energies_ev = self.load_data_from_nor(f)
            self.data.append(d)
        self.data = np.array(self.data)

    def load_data_from_nor(self, path):
        table = read_nor(path)
        data = table[self.data_column].to_numpy()
        energies = table['e'].to_numpy()
        _, unique_inds = np.unique(energies, return_index=True)
        unique_inds = np.sort(unique_inds)
        data = data[unique_inds]
        energies = energies[unique_inds]
        return data, energies


def read_nor(fname):
    with open(fname, 'r') as f:
        lines = f.readlines()
        columns = []
        for i, l in enumerate(lines):
            if not lines[i + 1].startswith('#'):
                columns = l[1:].strip().split()
                break
    table = pd.read_table(fname, comment='#', header=None, sep='\s+')
    table.columns = columns
    return table


class SLADSDataset(Dataset):
    
    def __init__(self, 
                 path, 
                 returns=("x", "x_measured", "y_measured", "y_interp", "n_measured", "erd"), 
                 n_recon_pixels=1000, *args, **kwargs):
        """SLADS dataset. 

        :param path: str. Path to the h5 file.
        :param returns: tuple[str, ...]. A list of value types to return. Returned values follow
            the same order as in this tuple. Valid values are: 
            - 'x' 
            - 'x_measured'
            - 'y_measured'
            - 'erd'
            - 'y_interp'
        """
        super().__init__(*args, **kwargs)
        self.f = h5py.File(path, 'r')
        self.returns = returns
        self.n_recon_pixels = n_recon_pixels
    
    def __len__(self):
        return len(self.f['erd'])
    
    def __getitem__(self, ind):
        return_list = []
        for return_type in self.returns:
            n_measured = self.f['n_measured'][ind]
            if return_type == 'x':
                v = torch.tensor(self.f['x'][ind:ind + 1]).reshape(1, -1).float()
            elif return_type == 'n_measured':
                v = torch.tensor([[n_measured]]).int()
            elif return_type == 'x_measured':
                v = self.f['x_measured'][ind:ind + 1, :]
                v[:, n_measured:] = np.nan
                v = torch.tensor(v).float()
            elif return_type == 'y_measured':
                v = self.f['y_measured'][ind:ind + 1, :]
                v[:, n_measured:] = np.nan
                v = torch.tensor(v).float()
            elif return_type == 'erd':
                v = torch.tensor(self.f['erd'][ind:ind + 1]).reshape(1, -1).float()
            elif return_type == 'y_interp':
                v = reconstruct_spectrum(
                    self.f['x_measured'][ind, :n_measured], 
                    self.f['y_measured'][ind, :n_measured], 
                    np.linspace(0, 1, self.n_recon_pixels),
                    method="linear")
                v = torch.tensor(v.reshape([1, -1])).float()
            else:
                raise ValueError(f"Invalid return type: {return_type}")
            return_list.append(v)
        return tuple(return_list)
