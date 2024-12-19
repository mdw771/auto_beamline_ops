import re
import glob

from autobl.util import *
import autobl.tools.spectroscopy.xanes as xanestools
from XANES_grid_transfer_LTO import LTOGridTransferTester
from XANES_grid_transfer_Pt import PtGridTransferTester

from plot_dynamic_results import DynamicExperimentResultAnalyzer

import logging
logging.basicConfig(level=logging.CRITICAL)
logging.disable(logging.CRITICAL)


if __name__ == '__main__':
    folders = [
        'outputs/grid_transfer_Pt/grid_redoForEach/Pt',
        # 'outputs/grid_transfer_Pt/grid_initOfSelf/Pt',
        # 'outputs/grid_transfer_Pt/grid_selectedRef/Pt',
    ]
    labels = [
        'Adaptively sampled',
        # 'Initial spectrum',
        # 'Reference spectra'
    ]

    normalizer = xanestools.XANESNormalizer(fit_ranges=((11400, 11500), (11650, 11850)), edge_loc=11566.0)
    
    analyzer = DynamicExperimentResultAnalyzer(tester_class=PtGridTransferTester)
    label_val_list = [int(re.findall('\d+', f)[-1]) for f in glob.glob('data/raw/Pt-XANES/Pt_xmu/*.xmu')]
    label_val_list.sort()
    label_list = [str(x) + '$^\circ\!$C' for x in label_val_list]
    analyzer.plot_data(folders[0], transpose=True, plot_func='plot', line_plot_labels=label_list, normalizer=normalizer, output_filename='Pt_grid_transfer_test_data.pdf')
    analyzer.plot_data(folders[0], transpose=True, plot_func='plot', line_plot_labels=label_list, normalizer=None, output_filename='Pt_grid_transfer_test_data_raw.pdf')
    analyzer.compare_rms_across_time(folders, labels, output_filename='Pt_grid_transfer_comparison_rms_across_time.pdf')
    analyzer.compare_calculated_percentages(folders, labels, x_data=label_val_list[1:-1], x_label='Temperature ($\!^\circ\!$C)', 
                                            read_precalculated_percentage_data=False, normalizer=normalizer, output_filename='Pt_grid_transfer_comparison_calculated_percentages.pdf')
    analyzer.compare_maxima_across_time(folders, labels, x_data=label_val_list[1:-1], x_label='Temperature ($\!^\circ\!$C)', normalizer=normalizer, output_filename='Pt_grid_transfer_comparison_maxima_across_time.pdf')
    analyzer.plot_all_spectra(folders[0], list(range(len(label_list) - 2)), label_list[1:-1], normalizer=normalizer, output_filename='Pt_grid_transfer_all_adapt_spectra.pdf')
    # analyzer.plot_spectrum(folders, labels, 0, output_filename='Pt_grid_transfer_estimated_spectra_0.pdf')
    # analyzer.plot_spectrum(folders, labels, 21, output_filename='Pt_grid_transfer_estimated_spectra_21.pdf')

