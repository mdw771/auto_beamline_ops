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
        'outputs/grid_transfer_LTO/grid_redoForEach/50C',
        # 'outputs/grid_transfer_LTO/grid_initOfSelf/50C',
        # 'outputs/grid_transfer_LTO/grid_selectedRef/50C',
    ]
    labels = [
        'Adaptively sampled',
        # 'Initial spectrum',
        # 'Reference spectra'
    ]
    
    normalizer = xanestools.XANESNormalizer(fit_ranges=((4900, 4950), (5050, 5200)), edge_loc=4983) 
    
    analyzer = DynamicExperimentResultAnalyzer(tester_class=LTOGridTransferTester)
    analyzer.plot_data(folders[0], transpose=True, value_range=(None, None), normalizer=normalizer, plot_func='imshow', line_plot_labels=list(range(200)), output_filename='LTO_50C_test_data.pdf')
    analyzer.plot_data(folders[0], transpose=True, value_range=(None, None), plot_raw_data=True, plot_func='imshow', line_plot_labels=list(range(200)), output_filename='LTO_50C_test_data_raw.pdf')
    analyzer.plot_data('outputs/grid_transfer_LTO/grid_initOfSelf/70C', transpose=True, value_range=(None, None), normalizer=normalizer, plot_func='imshow', line_plot_labels=list(range(200)), xtick_interval=5, output_filename='LTO_70C_test_data.pdf')
    analyzer.plot_data('outputs/grid_transfer_LTO/grid_initOfSelf/70C', transpose=True, value_range=(None, None), plot_raw_data=True, plot_func='imshow', line_plot_labels=list(range(200)), xtick_interval=5, output_filename='LTO_70C_test_data_raw.pdf')
    analyzer.compare_rms_across_time(folders, labels, output_filename='LTO_50C_comparison_rms_across_time.pdf',
                                     read_data=False, normalizer=normalizer, fit_normalizer_with_true_data=True)
    analyzer.compare_calculated_percentages(folders, labels, normalizer=normalizer, 
                                            read_precalculated_percentage_data=False, output_filename='LTO_50C_comparison_calculated_percentages.pdf')
    analyzer.plot_spectrum(folders, labels, 80, output_filename='LTO_50C_estimated_spectra_80.pdf', normalizer=normalizer)
    analyzer.plot_spectrum(folders, labels, 90, output_filename='LTO_50C_estimated_spectra_90.pdf', normalizer=normalizer)
    analyzer.plot_all_spectra(folders[0], list(range(128)), None, normalizer=normalizer, output_filename='LTO_50C_grid_transfer_all_adapt_spectra.pdf', plot_func='imshow', value_range=(0, 1.5))
