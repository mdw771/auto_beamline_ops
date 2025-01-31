import os
import glob
import pickle
import contextlib
import logging
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib
import numpy as np

from autobl.util import rms
import autobl.tools.spectroscopy.xanes as xanestools


logging.basicConfig(level=logging.WARNING)


class ResultAnalyzer:
    style_list = ["-", "-.", ":", (0, (3, 5, 1, 5, 1, 5))]

    def __init__(self, output_dir="factory"):
        self.output_dir = output_dir

        matplotlib.rc("font", family="Times New Roman")
        matplotlib.rcParams["font.size"] = 14
        matplotlib.rcParams["pdf.fonttype"] = 42
        
    def load_rms_data(self, f, normalizer=None, rms_normalization_factor=1.0, x_range=None):
        rms_list = []
        n_pts = []
        data = pickle.load(open(f, "rb"))
        data_true = data["data_y"]
        if normalizer is not None:
            data_true = normalizer.apply(data["data_x"], data_true)
        if x_range is not None:
            data_true = data_true[(data["data_x"] >= x_range[0]) & (data["data_x"] <= x_range[1])]
        for i, data_estimated in enumerate(data["mu_list"]):
            if normalizer is not None:
                data_estimated = normalizer.apply(data["data_x"], data_estimated)
            if x_range is not None:
                data_estimated = data_estimated[(data["data_x"] >= x_range[0]) & (data["data_x"] <= x_range[1])]
            r = rms(data_estimated, data_true)
            r = r / rms_normalization_factor
            rms_list.append(r)
            n_pts.append(data["n_measured_list"][i])
        return n_pts, rms_list

    def compare_convergence(
        self,
        file_list,
        labels,
        ref_line_y=0.005,
        add_legend=True,
        output_filename="comparison_convergence.pdf",
        figsize=None,
        auc_range=None,
        rms_normalization_factor=1.0,
        normalizer=None,
    ):
        rms_all_files = []
        n_pts_all_files = []
        for f in file_list:
            rms_list = []
            n_pts = []
            data = pickle.load(open(f, "rb"))
            data_true = data["data_y"]
            if normalizer is not None:
                data_true = normalizer.apply(data["data_x"], data_true)
            n_pts, rms_list = self.load_rms_data(f, normalizer=normalizer, rms_normalization_factor=rms_normalization_factor)
            rms_all_files.append(rms_list)
            n_pts_all_files.append(n_pts)

        # RMS convergence curve
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        for i in range(len(rms_all_files)):
            ax.plot(
                n_pts_all_files[i],
                rms_all_files[i],
                linestyle=self.style_list[i % len(self.style_list)],
                label=labels[i],
            )
        if ref_line_y is not None:
            orig_xlim, orig_ylim = ax.get_xlim(), ax.get_ylim()
            ax.hlines(
                ref_line_y, orig_xlim[0], orig_xlim[1], linestyles="--", colors="black"
            )
            ax.text(
                orig_xlim[0] + (orig_xlim[1] - orig_xlim[0]) * 0.05,
                ref_line_y + (orig_ylim[1] - orig_ylim[0]) * 0.01,
                str(ref_line_y),
                color="black",
            )
            ax.set_xlim(orig_xlim)
            ax.set_ylim(orig_ylim)
        ax.set_xlabel("Points measured")
        ax.set_ylabel("RMS error")
        plt.tight_layout()
        if add_legend:
            ax.legend(loc="upper right", frameon=True, ncol=1, fontsize=16)
        ax.grid(True)
        plt.savefig(os.path.join(self.output_dir, output_filename), bbox_inches="tight")

        # AUC bar chart
        fig, ax = self.plot_auc(n_pts_all_files, rms_all_files, labels, auc_range=auc_range)
        fig.savefig(
            os.path.join(
                self.output_dir, os.path.splitext(output_filename)[0] + "_auc.pdf"
            ),
            bbox_inches="tight",
        )
        
    def calculate_auc(self, n_pts, rms_list):
        auc = np.trapz(rms_list, n_pts)
        return auc
    
    def plot_auc(self, n_pts_all_files, rms_all_files, labels, auc_range=None, fig=None):
        if fig is None:
            fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        else:
            ax = fig.axes
        auc_list = []
        for i in range(len(rms_all_files)):
            slicer = (
                slice(auc_range[0], auc_range[1])
                if auc_range is not None
                else slice(None)
            )
            auc = self.calculate_auc(rms_all_files[i][slicer], n_pts_all_files[i][slicer])
            auc_list.append(auc)
            rect = ax.bar(i, auc, label=labels[i], width=0.8)
            ax.bar_label(rect, fmt="%.2f")
        ax.set_xticks([])
        ax.set_ylabel("Area under the curve")
        ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1] * 1.05))
        return fig, ax
    
    def plot_auc_multipass(self, base_data_filenames, labels, auc_range=None, normalizer=None, output_filename="auc_multipass.pdf"):
        auc_avg_std_all_cases = []
        fig, ax = plt.subplots(1, 1, figsize=(4, 3))
        for i_case, base_data_filename in enumerate(base_data_filenames):
            base_data_dir = os.path.dirname(base_data_filename)
            n_passes = len(glob.glob(os.path.join(base_data_dir + "_pass*")))
            pass_auc_list = []
            for i_pass in range(n_passes):
                folder_name = os.path.join(base_data_dir + f"_pass{i_pass}")
                data_filenames = glob.glob(os.path.join(folder_name, "*.pkl"))
                assert len(data_filenames) == 1, "Multiple data files found in the directory"
                data_filename = data_filenames[0]
                n_pts, rms_list = self.load_rms_data(data_filename, normalizer=normalizer)
                slicer = (
                    slice(auc_range[0], auc_range[1])
                    if auc_range is not None
                    else slice(None)
                )
                auc = self.calculate_auc(n_pts[slicer], rms_list[slicer])
                pass_auc_list.append(auc)
            pass_average_auc = np.mean(pass_auc_list)
            pass_std_auc = np.std(pass_auc_list)
            print(f'{labels[i_case]}: {pass_average_auc:.3f} ± {pass_std_auc:.3f}')
            auc_avg_std_all_cases.append((pass_average_auc, pass_std_auc))
            rect = ax.bar(i_case, pass_average_auc, yerr=pass_std_auc, ecolor="#404040", capsize=5, label=labels[i_case])
            ax.bar_label(rect, fmt="%.2f")
        ax.set_xticks([])
        ax.set_ylabel("Area under the curve")
        ax.set_ylim((ax.get_ylim()[0], ax.get_ylim()[1] * 1.05))
        plt.tight_layout()
        plt.savefig(os.path.join(self.output_dir, output_filename), bbox_inches="tight")
        return fig, ax
    
    def plot_intermediate(
        self,
        filename,
        n_cols=3,
        sharex=False,
        interval=5,
        iter_list=(),
        plot_uncertainty=True,
        plot_measurements=True,
        plot_truth=True,
        label="Measured",
        linestyle=None,
        add_legend=True,
        figsize=None,
        fig=None,
        save=True,
        make_animation=False,
        output_filename="intermediate.pdf",
    ):
        if make_animation:
            import matplotlib

            matplotlib.use("Agg")
            import matplotlib.animation as manimation

            FFMpegWriter = manimation.writers["ffmpeg"]
            metadata = dict(title="Animation", artist="MD")
            writer = FFMpegWriter(fps=5, metadata=metadata)

        data = pickle.load(open(filename, "rb"))
        n = len(data["mu_list"])
        if len(iter_list) == 0:
            iter_list = range(0, n, interval)
        n_plots = len(iter_list)
        n_rows = int(np.ceil(n_plots / n_cols))
        if make_animation:
            fig, ax = plt.subplots(1, 1)
        else:
            if fig is None:
                fig, ax = plt.subplots(
                    n_rows, n_cols, figsize=(n_cols * 4, n_rows * 3) if figsize is None else figsize, squeeze=False,
                    sharex=sharex, sharey=sharex
                )
            else:
                ax = fig.axes
                ax = [
                    ax[i * n_cols : (i + 1) * n_cols]
                    for i in range(ax[0].get_gridspec().nrows)
                ]

        i_plot = 0
        with contextlib.ExitStack() as stack:
            if make_animation:
                stack.enter_context(
                    writer.saving(
                        fig, os.path.join(self.output_dir, output_filename), 100
                    )
                )
            for i, iter in enumerate(iter_list):
                i_col = i_plot % n_cols
                i_row = i_plot // n_cols
                if make_animation:
                    this_ax = ax
                else:
                    this_ax = ax[i_row][i_col]
                if plot_truth:
                    this_ax.plot(
                        data["data_x"],
                        data["data_y"],
                        color="gray",
                        alpha=0.5,
                        linestyle="--",
                        label="Ground truth",
                    )
                this_ax.plot(
                    data["data_x"],
                    data["mu_list"][iter],
                    linewidth=1,
                    linestyle=linestyle,
                    label=label,
                )
                if plot_uncertainty:
                    this_ax.fill_between(
                        data["data_x"],
                        data["mu_list"][iter] - data["sigma_list"][iter],
                        data["mu_list"][iter] + data["sigma_list"][iter],
                        alpha=0.5,
                    )
                if plot_measurements:
                    this_ax.scatter(
                        data["measured_x_list"][iter],
                        data["measured_y_list"][iter],
                        s=14,
                        label="Measured",
                    )
                this_ax.set_title("{} points".format(data["n_measured_list"][iter]))
                this_ax.set_xlabel("Energy (eV)")
                this_ax.set_ylabel("X-ray absorption")
                this_ax.grid(True)
                if i == 0 and add_legend:
                    this_ax.legend(frameon=False)
                i_plot += 1
                if make_animation:
                    writer.grab_frame()
                    this_ax.clear()
        plt.tight_layout()
        if sharex:
            plt.subplots_adjust(hspace=0)  # Remove vertical space between plots
        if save and not make_animation:
            plt.savefig(
                os.path.join(self.output_dir, output_filename), bbox_inches="tight"
            )

    def compare_intermediate(
        self,
        file_list,
        labels,
        n_cols=3,
        figsize=None,
        interval=5,
        sharex=False,
        add_legend=True,
        output_filename="comparison_intermediate.pdf",
    ):
        n = 0
        for filename in file_list:
            data = pickle.load(open(filename, "rb"))
            n = len(data["mu_list"]) if len(data["mu_list"]) > n else n
        n_plots = n // interval + 1
        n_rows = int(np.ceil(n_plots / n_cols))
        if figsize is None:
            figsize = (n_cols * 4, n_rows * 3)
        fig, ax = plt.subplots(n_rows, n_cols, figsize=figsize, sharex=sharex, sharey=sharex)

        for i, filename in enumerate(file_list):
            self.plot_intermediate(
                filename,
                n_cols=n_cols,
                interval=interval,
                plot_uncertainty=False,
                plot_measurements=False,
                plot_truth=(i == 0),
                label=labels[i],
                linestyle=self.style_list[i],
                add_legend=add_legend,
                fig=fig,
                save=False,
            )
        plt.tight_layout()
        if sharex:
            plt.subplots_adjust(hspace=0, wspace=0)  # Remove vertical space between plots
        plt.savefig(os.path.join(self.output_dir, output_filename))

    def compare_estimates(
        self,
        file_list,
        labels,
        at_n_pts=20,
        zoom_in_range_x=None,
        zoom_in_range_y=None,
        add_legend=True,
        figsize=(5, 5),
        normalizer: Optional[xanestools.XANESNormalizer] = None,
        output_filename="comparison_estimate.pdf",
    ):
        fig, ax = plt.subplots(1, 1, figsize=figsize)
        data = pickle.load(open(file_list[0], "rb"))
        true_x, true_y = data["data_x"], data["data_y"]
        if normalizer is not None:
            true_y = normalizer.apply(true_x, true_y)

        def plot_ax(ax, tick_interval=None, add_legend=True, linewidth=1):
            for i, f in enumerate(file_list):
                n_pts, rms_list = self.load_rms_data(f, normalizer=normalizer, x_range=zoom_in_range_x)
                print(f'{labels[i]}: {np.array(rms_list)[np.array(n_pts) == at_n_pts]}')
                
                data = pickle.load(open(f, "rb"))
                at_iter = data["n_measured_list"].index(at_n_pts)
                x, y = data["data_x"], data["mu_list"][at_iter]
                if normalizer is not None:
                    y = normalizer.apply(x, y)
                ax.plot(
                    x,
                    y,
                    linewidth=linewidth,
                    linestyle=self.style_list[i % len(self.style_list)],
                    label=labels[i],
                )
                if i == 0:
                    meas_x, meas_y = (
                        data["measured_x_list"][at_iter],
                        data["measured_y_list"][at_iter],
                    )
                    if normalizer is not None:
                        sorted_inds = np.argsort(meas_x)
                        meas_x = meas_x[sorted_inds]
                        meas_y = meas_y[sorted_inds]
                        meas_y = normalizer.apply(meas_x, meas_y)
                    ax.scatter(meas_x, meas_y, s=4)
            ax.plot(
                true_x,
                true_y,
                color="gray",
                alpha=0.5,
                linestyle="--",
                label="Ground truth",
            )
            if tick_interval is not None:
                ax.set_xticks(
                    np.arange(
                        np.ceil(true_x.min() / tick_interval) * tick_interval,
                        true_x.max(),
                        tick_interval,
                    )
                )
            if add_legend:
                ax.legend()
            ax.set_xlabel("Energy (eV)")
            ax.grid()

        plot_ax(ax, tick_interval=20, add_legend=add_legend)
        if zoom_in_range_x is not None:
            fig_zoom, ax_zoom = plt.subplots(1, 1, figsize=figsize)
            plot_ax(ax_zoom, add_legend=False, linewidth=2)
            ax_zoom.set_xlim(zoom_in_range_x)
            ax_zoom.set_ylim(zoom_in_range_y)
            ax_zoom.tick_params(axis="x", labelsize=22)
            ax_zoom.tick_params(axis="y", labelsize=22)
            ax_zoom.set_xlabel("")
            ax.add_patch(
                patches.Rectangle(
                    (zoom_in_range_x[0], zoom_in_range_y[0]),
                    zoom_in_range_x[1] - zoom_in_range_x[0],
                    zoom_in_range_y[1] - zoom_in_range_y[0],
                    linestyle="--",
                    edgecolor="orange",
                    facecolor="none",
                )
            )
            fig_zoom.savefig(
                os.path.join(
                    self.output_dir, os.path.splitext(output_filename)[0] + "_zoom.pdf"
                ),
                bbox_inches="tight",
            )
        fig.savefig(os.path.join(self.output_dir, output_filename), bbox_inches="tight")
