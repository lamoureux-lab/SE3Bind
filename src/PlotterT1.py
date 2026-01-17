import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sys
import os
from os.path import exists
from collections import Counter
import numpy as np
from scipy import stats
import matplotlib as mpl
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.legend_handler import HandlerTuple


class PlotterT1:
    def __init__(self, experiment=None,
                 logfile_savepath='Log/losses/'):
        """
        Initialize paths and filename prefixes for saving plots.

        :param experiment: current experiment name
        :param logfile_savepath: path to load and save data and figs
        """
        self.experiment = experiment
        self.logfile_savepath = logfile_savepath

        self.loss_save_folder = 'Figs/loss_plots/'+self.experiment+'/'
        os.makedirs(self.loss_save_folder, exist_ok=True)

        self.rmsd_save_folder = 'Figs/RMSD_distribution_plots/'+self.experiment+'/'
        os.makedirs(self.rmsd_save_folder, exist_ok=True)

        self.correlation_save_folder = 'Figs/correlation_plots/'+self.experiment+'/'
        os.makedirs(self.correlation_save_folder, exist_ok=True)

        self.loss_logs_folder = 'Log/losses/'+self.experiment+'/'

        if not experiment:
            print('no experiment name given')
            sys.exit()

    def plot_loss(self, ylim=None, show=False, save=True, plot_combined=False):
        """
        Plot the current experiments loss curve.
        The plot will plot all epochs present in the log file.

        :param ylim: set the upper limit of the y-axis, initial loss can vary widely
        :param show: show the plot in a window
        :param save: save the plot at specified path
        """
        plt.close()
        # Define the file names for train, valid, and test logs
        train_name = f"{self.logfile_savepath}log_loss_TRAINset_{self.experiment}.txt"
        valid_name = f"{self.logfile_savepath}log_loss_VALIDset_{self.experiment}.txt"
        test_name = f"{self.logfile_savepath}log_loss_TESTset_{self.experiment}.txt"
        # print("test name", test_name)

        # Define the names for the different columns in the logs
        column_names = ["Epoch", "Loss", "F_loss", "CE_loss", "Feature0_reg_loss","deltaF", "RMSD"]

        # Define the number of subplots and figure size
        num_subplots = 3
        fig, ax = plt.subplots(num_subplots, figsize=(20, 10))

        # Read the train log and check if the valid and test logs exist
        if os.path.exists(train_name):
            train = pd.read_csv(train_name, sep='\t', header=1, names=column_names, index_col='Epoch')
            valid = pd.read_csv(valid_name, sep='\t', header=1, names=column_names, index_col='Epoch')
            test = pd.read_csv(test_name, sep='\t', header=1, names=column_names, index_col='Epoch')
            train = train[~train.index.duplicated(keep='last')].reset_index()
            valid = valid[~valid.index.duplicated(keep='last')].reset_index()
            test = test[~test.index.duplicated(keep='last')].reset_index()

            # Plot the average RMSD for each dataset
            line_rmsd, = ax[0].plot(train["Epoch"], train["RMSD"])
            ax[0].plot(valid["Epoch"], valid["RMSD"])
            ax[0].plot(test["Epoch"], test["RMSD"])
            ax[0].legend(("train avgRMSD", "valid avgRMSD", "test avgRMSD"))

            # Set the title, ylabel, and grid for the first subplot
            ax[0].set_title(f"Loss: {self.experiment}")
            ax[0].set_ylabel("avgRMSD")
            ax[0].grid(visible=False)
            ax[0].margins(x=0.01)

            # Plot the CE loss for each dataset
            line_loss, = ax[1].plot(train["Epoch"], train["CE_loss"])
            ax[1].plot(valid["Epoch"], valid["CE_loss"])
            ax[1].plot(test["Epoch"], test["CE_loss"])
            ax[1].legend(("train CE_loss", "valid CE_loss", "test CE_loss"))

            # Set the xlabel, ylabel, and grid for the second subplot
            ax[1].set_ylabel("CE loss")
            ax[1].grid(visible=False)
            ax[1].margins(x=0.01)

            # Plot the CE loss for each dataset
            line_loss, = ax[2].plot(train["Epoch"], train["F_loss"])
            ax[2].plot(valid["Epoch"], valid["F_loss"])
            ax[2].plot(test["Epoch"], test["F_loss"])
            ax[2].legend(("train F_loss", "valid F_loss", "test F_loss"))

            # Set the xlabel, ylabel, and grid for the second subplot
            ax[2].set_xlabel("epochs")
            ax[2].set_ylabel("F loss")
            ax[2].grid(visible=False)
            ax[2].margins(x=0.01)

            # Set the ylim if specified
            if ylim:
                ax[0].set_ylim([0, ylim])
                ax[1].set_ylim([0, ylim])
                ax[2].set_ylim([0, ylim])
            else:
                ax[0].set_ylim([0, 15])
                ax[1].set_ylim([0, 15])
                ax[2].set_ylim([0, 15])

            # # Set the ylim specified
            # ax[0].set_ylim([0, ylim[0]])
            # ax[1].set_ylim([0, ylim[1]])
            # ax[2].set_ylim([0, ylim[2]])

            if self.experiment == 'SE3Bind_exp44_B_JT_back2L1loss_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep':
                ax[0].axvline(x=500, color='black', linestyle='--')
                ax[0].axvline(x=1300, color='black', linestyle='--')
                ax[1].axvline(x=500, color='black', linestyle='--')
                ax[1].axvline(x=1300, color='black', linestyle='--')
                ax[2].axvline(x=500, color='black', linestyle='-.', label='start L2 loss')
                ax[2].axvline(x=1300, color='black', linestyle='--', label='L1 again')
                ax[2].axvspan(500, 1300, color='lightgrey', alpha=0.5)
                ax[2].legend()

            # Save the plot if specified
            if save:
                plt.savefig(f"{self.loss_save_folder}/lossplot_{self.experiment}.png")

            # Show the plot if specified and not plotting the combined plot
            if show and not plot_combined:
                plt.show()

            # Return the line for the CE loss
            return line_loss

    def plot_rmsd_distribution(self,
                               plot_streams,
                               plot_epoch=1, show=False, save=True):
        """
        Plot the predicted RMSDs distributions as histogram(s), depending on how many log files exist.

        :param plot_epoch: epoch of training/evalution to plot
        :param show: show the plot in a window
        :param save: save the plot at specified path
        """
        plt.close()
        # Plot RMSD distribution of all samples across epoch
        train_log = self.logfile_savepath+'log_F_TRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        valid_log = self.logfile_savepath+'log_F_VALIDset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        test_log = self.logfile_savepath+'log_F_TESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        avg_trainRMSD, avg_validRMSD, avg_testRMSD = None, None, None

        names = ["Example", "Loss", "F_loss", "CE_loss", "Feature0_reg_loss","deltaF", "deltaG", "RMSD"
                 ,'clusterID', 'structureIDs'
                 ]

        log_list = [train_log, valid_log, test_log]
        plot_cond_list = list(plot_streams.values())
        df_list = []
        avg_RMSDs = []
        subplot_count = 0
        for i in range(len(log_list)):
            if plot_cond_list[i]:
                subplot_count += 1
                plot_df = pd.read_csv(log_list[i], sep="\t", header=0, names=names, index_col="Example")
                plot_df = plot_df[~plot_df.index.duplicated(keep='last')]
                avg_RMSD = str(plot_df['RMSD'].mean())[:4]
                df_list.append(plot_df)
                avg_RMSDs.append(avg_RMSD)
                print(log_list[i])
                print('average RMSD:')
                print(avg_RMSD)
            else:
                df_list.append(None)
                avg_RMSDs.append(None)

        fig, ax = plt.subplots(subplot_count, figsize=(20, 10))
        plt.suptitle('RMSD distribution: epoch' + str(plot_epoch) + ' ' + self.experiment +'\n'
                      + 'train:'+ str(avg_RMSDs[0]) + ' test:' + str(avg_RMSDs[2]))
        plt.xlabel('RMSD', fontdict={'size':20})
        binwidth=1
        xrange=100
        yrange_ratio=1
        bins = np.arange(0, xrange + binwidth, binwidth)

        y_labels = ['Training set',
                    'Validation set',
                    'Testing set']
        colors = ['b', 'r', 'g']
        # ylims = [[0, len(df_list[0])],[0, 1000],[0, 1000]]
        if subplot_count == 1:
            cond_index = [i for i in range(len(df_list)) if df_list[i] is not None][0]
            # print(cond_index)
            ax.hist(df_list[cond_index]['RMSD'].to_numpy(), bins=bins, color=colors[cond_index])
            ax.set_ylabel(y_labels[cond_index])
            ax.grid(visible=True)
            ax.set_xticks(np.arange(0, xrange, 10))
            ax.set_ylim([0, len(df_list[cond_index]) // yrange_ratio + 1])
        else:
            plot_index = 0
            for i in range(len(plot_cond_list)):
                if df_list[i] is not None and plot_cond_list[i] is True:
                    # print(i, y_labels[i], colors[i])
                    ax[plot_index].hist(df_list[i]['RMSD'].to_numpy(), bins=bins, color=colors[i])
                    ax[plot_index].set_ylabel(y_labels[i], labelpad=10*i, fontdict={'size':20})
                    ax[plot_index].grid(visible=True)
                    ax[plot_index].set_xticks(np.arange(0, xrange, 10))
                    ax[plot_index].set_ylim([0, len(df_list[i]) // yrange_ratio + 1])
                    ax[plot_index].set_xlim([0, 50])
                    ax[plot_index].grid(False)
                    # ax[plot_index].title(str(y_labels[i]))
                else:
                    plot_index-=1
                    pass
                plot_index += 1

        if save:
            plt.savefig(self.rmsd_save_folder+'rmsdplot_epoch' + str(plot_epoch) + '_' + self.experiment + '.png')
        if show:
            plt.show()
          
    def plot_deltaF_vs_deltaG_correlation(self,
                                 plot_training=False,
                                 plot_valid=False,
                                 plot_testing=False,
                                 plot_all_combined=False,
                                 plot_crystal=False,
                                 plot_homology=False,
                                 plot_epoch=1, 
                                 show=False, 
                                 save=True,
                                 valid_crystal=False,
                                 valid_homology=False):

        mpl.rcParams.update({
            "font.size": 12,          # base font
            "axes.labelsize": 14,     # axis labels
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,    # thicker axes
        })

        plt.close()
        
        if valid_homology:
            prefix = 'BM_sab_HM'
            valid_log = f"{self.logfile_savepath}log_F_{prefix}VALIDset_epoch{plot_epoch}{self.experiment}.txt"
            title = "HM_validset_combined_deltaG_corr_for_epoch_" + str(plot_epoch)
            # Plot free energy distribution of all samples across epoch
            train_log = self.logfile_savepath+'log_F_TRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
            test_log = self.logfile_savepath+'log_F_TESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"

        if valid_crystal:
            valid_log = self.logfile_savepath+'log_F_VALIDset_epoch' + str(plot_epoch) + self.experiment + ".txt"
            title = "combined_deltaG_corr_for_epoch_" + str(plot_epoch)                                
            # Plot free energy distribution of all samples across epoch
            train_log = self.logfile_savepath+'log_F_TRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
            test_log = self.logfile_savepath+'log_F_TESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        
        if plot_all_combined:
            train_log = self.logfile_savepath+'log_F_TRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
            crystal_valid_log = self.logfile_savepath+'log_F_VALIDset_epoch' + str(plot_epoch) + self.experiment + ".txt"
            homology_valid_log =  f"{self.logfile_savepath}log_F_BM_sab_HMVALIDset_epoch{plot_epoch}{self.experiment}.txt"
            title = 'all_combined_deltaG_corr_for_epoch_' + str(plot_epoch)
            
            log_list = [train_log, crystal_valid_log, homology_valid_log]
            plot_cond_list = [plot_training, plot_crystal, plot_homology]


            df_list = []
            names = ["Example", "Loss", "F_loss", "CE_loss", "Feature0_reg_loss","deltaF", "deltaG", "RMSD"
                    ,'clusterID', 'structureIDs']
            
            colors = ['g', 'b', 'r']
            edge_color = ['darkolivegreen', 'navy', 'firebrick']
            # labels = ["Training set", "Crystal structure validation set", "Homology model validation set"]
            labels = ["Training set", "Validation set (crystal structures)", "Validation set (homology models)"]
            fill_color=['grey', 'darkgrey', 'lightgray']
        
        else:
            # print("current exp::", self.experiment)
            # print("train_log",train_log)
            # print("exists train_log:", exists(train_log))
            # print("exists valid_log:", exists(valid_log))
            # print("exists test_log:", exists(test_log))

            log_list = [train_log, valid_log, test_log]
            plot_cond_list = [plot_training, plot_valid, plot_testing]
            df_list = []
            names = ["Example", "Loss", "F_loss", "CE_loss", "Feature0_reg_loss","deltaF", "deltaG", "RMSD"
                    ,'clusterID', 'structureIDs'
                    ]

            print('plot_cond_list', plot_cond_list)
            
            colors = ['g', 'b', 'r']
            edge_color = ['lightgreen', 'lightblue', 'lightcoral']
            labels = ['Training set', 'Validation set', 'TESTset']
            fill_color=['grey', 'darkgrey', 'lightgrey']

        print('log_list', len(log_list))
        print('plot_cond_list', plot_cond_list)

        for i in range(len(log_list)):
            if plot_cond_list[i] and exists(log_list[i]):
                # print(i, plot_cond_list[i], exists(log_list[i]))

                plot_df = pd.read_csv(log_list[i], sep="\t", header=0,
                                       names=names, index_col="Example")
                plot_df = plot_df[~plot_df.index.duplicated(keep='last')]
                df_list.append(plot_df)
            else:
                df_list.append(None)

        print('df_list lengths', [len(df) if df is not None else None for df in df_list])

        fig = plt.figure(figsize=(5, 5), constrained_layout=True)
        ax = fig.add_subplot(111, aspect=1)

        r_values, sefs = [], []
        # x_lower, x_upper = -25, 0
        x_lower, x_upper = -100, 0
        diag = np.array([x_lower, x_upper])
        
        for i, df in enumerate(df_list):
            if df is not None:
                groundtruth = df['deltaG'].to_numpy()
                predicted = df['deltaF'].to_numpy()

                # Pearson correlation
                mean_groundtruth = np.mean(groundtruth)
                mean_predicted = np.mean(predicted)
                numerator = np.sum((groundtruth - mean_groundtruth) * (predicted - mean_predicted))
                denominator = np.sqrt(
                    np.sum((groundtruth - mean_groundtruth) ** 2) *
                    np.sum((predicted - mean_predicted) ** 2)
                )
                r_value = round(numerator / denominator, 2)

                residuals = groundtruth - predicted
                n = len(residuals)
                sef = round(np.sqrt(np.sum(residuals ** 2) / (n - 1)), 2)

                ax.fill_between(diag, diag - sef, diag + sef,
                                color=fill_color[i], alpha=0.3, lw=0)

                r_values.append(r_value)
                sefs.append(sef)

        # print("r_values", r_values,"sefs",sefs)

        # scatter points on top
        for i, df in enumerate(df_list):
            if df is not None:
                ax.scatter(df['deltaG'], df['deltaF'],
                           alpha=0.8, s=20,
                           color=colors[i],
                           edgecolor=edge_color[i],
                           linewidth=0.6)

                ax.set_xlim([-25, 0])
                ax.set_ylim([-25, 0])

                # ax.set_xlim([-100, 0])
                # ax.set_ylim([-100, 0])

        # axis labels
        ax.set_xlabel(r'Experimental $\Delta G$ (kcal/mol)', labelpad=4)
        ax.set_ylabel(r'Predicted $\Delta G$ (kcal/mol)', labelpad=4)

        # remove spines on top/right for clean look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        # legend entries
        legend_entries = []
        legend_labels = []
        for i, df in enumerate(df_list):
            if df is not None:
                scatter_handle = Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=colors[i],
                                        markeredgecolor=edge_color[i],
                                        markersize=7, lw=0)
                patch_handle = Patch(facecolor=fill_color[i], alpha=0.3, lw=0)
                legend_entries.append((scatter_handle, patch_handle))
                legend_labels.append(f"{labels[i]} (StdErr: {sefs[i]:.2f}, R: {r_values[i]:.2f})")

        # clean legend — no frame
        plt.subplots_adjust(bottom=0.25)   # increase margin so legend doesn't overlap

        fig.legend(
            legend_entries,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),  # place in white margin
            frameon=False,
            ncol=1,
            fontsize=11
        )
        

        # Save figure
        if save:
            outpath = f"{self.correlation_save_folder}/{title}_{self.experiment}.png"
            plt.savefig(outpath, dpi=600, bbox_inches="tight")
            print(f"Saved publication-ready figure: {outpath}")

        if show:
            plt.show()


    def plot_inference_deltaG_correlation(self,
                                 plot_training=False,
                                 plot_valid=False,
                                 plot_testing=False,
                                 plot_all_combined=False,
                                 plot_crystal=False,
                                 plot_homology=False,
                                 plot_epoch=1, 
                                 show=False, 
                                 save=True,
                                 valid_crystal=False,
                                 valid_homology=False):

        mpl.rcParams.update({
            "font.size": 12,          # base font
            "axes.labelsize": 14,     # axis labels
            "xtick.labelsize": 12,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "axes.linewidth": 1.2,    # thicker axes
        })

        plt.close()
        
        # Initialize test log path only
        test_log = self.logfile_savepath+'log_F_TESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        title = "inference_deltaG_corr_for_epoch_" + str(plot_epoch)
        
        log_list = [test_log]
        plot_cond_list = [plot_testing]
        df_list = []
        names = ["Example", "Loss", "F_loss", "CE_loss", "Feature0_reg_loss","deltaF", "deltaG", "RMSD"
                ,'clusterID', 'structureIDs'
                ]

        print('plot_cond_list', plot_cond_list)
        
        colors = ['r']
        edge_color = ['lightcoral']
        labels = ['Inference set']
        fill_color=['lightgrey']

        print('log_list', len(log_list))
        print('plot_cond_list', plot_cond_list)

        for i in range(len(log_list)):
            if plot_cond_list[i] and exists(log_list[i]):
                # print(i, plot_cond_list[i], exists(log_list[i]))

                plot_df = pd.read_csv(log_list[i], sep="\t", header=0,
                                       names=names, index_col="Example")
                plot_df = plot_df[~plot_df.index.duplicated(keep='last')]
                df_list.append(plot_df)
            else:
                df_list.append(None)

        print('df_list lengths', [len(df) if df is not None else None for df in df_list])

        fig = plt.figure(figsize=(5, 5), constrained_layout=True)
        ax = fig.add_subplot(111, aspect=1)

        r_values, sefs = [], []
        # x_lower, x_upper = -25, 0
        x_lower, x_upper = -100, 0
        diag = np.array([x_lower, x_upper])
        
        for i, df in enumerate(df_list):
            if df is not None:
                groundtruth = df['deltaG'].to_numpy()
                predicted = df['deltaF'].to_numpy()

                # Pearson correlation
                mean_groundtruth = np.mean(groundtruth)
                mean_predicted = np.mean(predicted)
                numerator = np.sum((groundtruth - mean_groundtruth) * (predicted - mean_predicted))
                denominator = np.sqrt(
                    np.sum((groundtruth - mean_groundtruth) ** 2) *
                    np.sum((predicted - mean_predicted) ** 2)
                )
                r_value = round(numerator / denominator, 2)

                residuals = groundtruth - predicted
                n = len(residuals)
                sef = round(np.sqrt(np.sum(residuals ** 2) / (n - 1)), 2)

                ax.fill_between(diag, diag - sef, diag + sef,
                                color=fill_color[i], alpha=0.3, lw=0)

                r_values.append(r_value)
                sefs.append(sef)

        # print("r_values", r_values,"sefs",sefs)

        # scatter points on top
        for i, df in enumerate(df_list):
            if df is not None:
                ax.scatter(df['deltaG'], df['deltaF'],
                           alpha=0.8, s=20,
                           color=colors[i],
                           edgecolor=edge_color[i],
                           linewidth=0.6)

                ax.set_xlim([-25, 0])
                ax.set_ylim([-25, 0])

                # ax.set_xlim([-100, 0])
                # ax.set_ylim([-100, 0])

        # axis labels
        ax.set_xlabel(r'Experimental $\Delta G$ (kcal/mol)', labelpad=4)
        ax.set_ylabel(r'Predicted $\Delta G$ (kcal/mol)', labelpad=4)

        # remove spines on top/right for clean look
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)


        # legend entries
        legend_entries = []
        legend_labels = []
        for i, df in enumerate(df_list):
            if df is not None:
                scatter_handle = Line2D([0], [0], marker='o', color='w',
                                        markerfacecolor=colors[i],
                                        markeredgecolor=edge_color[i],
                                        markersize=7, lw=0)
                patch_handle = Patch(facecolor=fill_color[i], alpha=0.3, lw=0)
                legend_entries.append((scatter_handle, patch_handle))
                legend_labels.append(f"{labels[i]} (StdErr: {sefs[i]:.2f}, R: {r_values[i]:.2f})")

        # clean legend — no frame
        plt.subplots_adjust(bottom=0.25)   # increase margin so legend doesn't overlap

        fig.legend(
            legend_entries,
            legend_labels,
            handler_map={tuple: HandlerTuple(ndivide=None)},
            loc="lower center",
            bbox_to_anchor=(0.5, -0.05),  # place in white margin
            frameon=False,
            ncol=1,
            fontsize=11
        )
        

        # Save figure
        if save:
            outpath = f"{self.correlation_save_folder}/{title}_{self.experiment}.png"
            plt.savefig(outpath, dpi=600, bbox_inches="tight")
            print(f"Saved publication-ready figure: {outpath}")

        if show:
            plt.show() 


    def calculate_pearsons_r(self, groundtruth, predicted):
        """
        Calculate Pearson's correlation coefficient (r) between groundtruth and predicted values.

        :param groundtruth: Array of ground truth values
        :param predicted: Array of predicted values
        :return: Pearson's correlation coefficient (r)
        """
        mean_groundtruth = np.mean(groundtruth)
        mean_predicted = np.mean(predicted)

        numerator = np.sum((groundtruth - mean_groundtruth) * (predicted - mean_predicted))
        denominator = np.sqrt(
            np.sum((groundtruth - mean_groundtruth) ** 2) * np.sum((predicted - mean_predicted) ** 2))
        
        return (numerator / denominator).round(2)