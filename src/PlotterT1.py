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

        self.loss_save_folder = 'Figs/FI_loss_plots/'+self.experiment+'/'
        os.makedirs(self.loss_save_folder, exist_ok=True)

        self.rmsd_save_folder = 'Figs/FI_RMSD_distribution_plots/'+self.experiment+'/'
        os.makedirs(self.rmsd_save_folder, exist_ok=True)

        self.correlation_save_folder = 'Figs/FI_correlation_plots/'+self.experiment+'/'
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

    

    def replace_median_deltaG_withavg(self, plot_epoch, trainset=False, validset=False):
        print("replcing medianwith average ")

        # if self.experiment == 'SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep':
        if trainset:
            train_log = self.logfile_savepath+'log_F_TRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
            average_deltaG_file = 'Log/losses/GETaverage_GT_deltaGs_forlogfile/'+"log_F_TRAINset_epoch0GETaverage_GT_deltaGs_forlogfile.txt"
            # print("train_log", train_log)

            print("exists(train_log[i]):", exists(train_log))
            print("exists(average_deltaG_file[i]):", exists(average_deltaG_file))
            # Read the train log and average deltaG fi
            train_df = pd.read_csv(train_log, sep='\t')

            avg_deltaG_df = pd.read_csv(average_deltaG_file, sep='\t')
            # print("avg_deltaG_df", avg_deltaG_df)

            train_df['deltaG'] = avg_deltaG_df['deltaG']
            # ###Find rows where the deltaG values are different
            # different_rows = train_df[train_df['deltaG'] != avg_deltaG_df['deltaG']]

            ### replace deltaG values with the average deltaG in the log file for plotting.
            train_df.to_csv(train_log, sep='\t', index=False)

            pd.set_option('display.max_colwidth', None)
          
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


    def plot_F_loss_allcombined(self, experiments_list,exp_short_names_list, ylim=None, show=False, save=True,):
        plt.close()

        # Define the names for the different columns in the logs
        column_names = ["Epoch", "Loss", "F_loss", "CE_loss", "Feature0_reg_loss", "deltaF", "RMSD"]

        # Initialize lists to store data from all experiments
        all_train_data = []
        all_valid_data = []
        all_test_data = []


        # Create a discrete list of colors sampled from the 'tab10' colormap.
        # Normalize sampling so colors are spread across the colormap even when
        # len(experiments_list) > base palette size. Handles N==1 as a special case.
        N = len(experiments_list)
        cmap = plt.cm.get_cmap('tab10')
        if N <= 1:
            color_list = [cmap(0.0)]
        else:
            color_list = [cmap(i / (N - 1)) for i in range(N)]

        fig, ax = plt.subplots(figsize=(20, 10))

        for i, exp in enumerate(experiments_list):           
            train_name = f"{self.logfile_savepath}{exp}/log_loss_TRAINset_{exp}.txt"
            if os.path.exists(train_name):
                print('os.path.exists(train_name)',os.path.exists(train_name))
                train = pd.read_csv(train_name, sep='\t', header=1, names=column_names, index_col='Epoch')
                train = train[~train.index.duplicated(keep='last')].reset_index()
                # Use precomputed color for experiment i so each curve is distinct
                ax.plot(train["Epoch"], train["F_loss"], label=f"{exp_short_names_list[i]} F_loss", color=color_list[i])
        
        ax.set_ylabel("F_loss")
        ax.set_xlabel("Epoch")
        ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
        ax.grid(visible=False)
        ax.margins(x=0.01)

        ax.axvline(x=1740, color='black', linestyle='-.', label='lr:0.001')
        ax.axvline(x=500, color='black', linestyle='--', label='L2 loss: SE3Bind_L1_L2_L1again_loss')
        ax.axvline(x=1300, color='black', linestyle='--', label='L1 loss again; SE3Bind_L1_L2_L1again_loss')
        ax.axvspan(1300, 1720, color='red', alpha=0.3)
        ax.axhline(y=1, color='black', linestyle='--',)

        if ylim:
            ax.set_ylim([0, ylim])
        else:
            ax.set_ylim([0, 15])
        ax.set_yticks(list(ax.get_yticks()) + [1])

        # fig_rmsd, ax_rmsd = plt.subplots(num_experiments, figsize=(20, 5 * num_experiments))
        
        # for i, exp in enumerate(experiments_list):
        #     train_name = f"{self.logfile_savepath}{exp}/log_loss_TRAINset_{exp}.txt"
        #     if os.path.exists(train_name):
        #         train = pd.read_csv(train_name, sep='\t', header=1, names=column_names, index_col='Epoch')
        #         train = train[~train.index.duplicated(keep='last')].reset_index()

        #         ax_rmsd[i].plot(train["Epoch"], train["RMSD"], label=f"{exp_short_names_list[i]} RMSD")
        #         ax_rmsd[i].set_ylabel("RMSD")
        #         ax_rmsd[i].set_xlabel("Epoch")
        #         ax_rmsd[i].legend()
        #         ax_rmsd[i].grid(visible=False)
        #         ax_rmsd[i].margins(x=0.01)

        #     if ylim:
        #         ax_rmsd[i].set_ylim([0, ylim])
        #     else:
        #         ax_rmsd[i].set_ylim([0, 15])

        # if save:
        #     plt.savefig(f"{self.loss_save_folder}/combined_F_loss_plot.png")
        #     # plt.savefig(f"{self.loss_save_folder}/combined_RMSD_plot.png")

        if show:
            plt.show()

    

    def get_all_metrics(self,
                plot_epoch,
                experiments_list,
                exp_short_names_list,
                plot_training=False,
                plot_valid=False,
                plot_testing=False,
                show=False,
                save=True,
                valid_crystal=False,
                valid_homology=False):
        """
        for specific epochs: Get all metrics, average F_loss, CE_loss, RMSD, 
        MAE for L2loss models
        """
        
        print("testing MAE for L2loss")

        column_names = ["Example", "Loss", "F_loss", "CE_loss", "Feature0_reg_loss",
                        "deltaF", "deltaG", "RMSD",	"clusterID","structureIDs"]
        df_name = ''

        
        metrics_only_combined_df = pd.DataFrame()
        metrics_df = pd.DataFrame()

        newlogfile_savepath = '/Users/Anushriya/Documents/Lamoureux_lab/AntibodyDocking/src/Log/losses/'

        for i, exp in enumerate(experiments_list):
            if plot_training:
                train_name = f"{newlogfile_savepath}{exp}/Log_F_TRAINset_epoch{plot_epoch}{exp}.txt"

                df_name = train_name
            if plot_valid:
                if valid_crystal:
                    valid_name = f"{newlogfile_savepath}{exp}/Log_F_VALIDset_epoch{plot_epoch}{exp}.txt"
                ## for HM valid dataset::
                if valid_homology:
                    valid_name = f"{newlogfile_savepath}{exp}/Log_F_BM_sab_HMVALIDset_epoch{plot_epoch}{exp}.txt"

                df_name = valid_name

                print("valid_name", valid_name)
                print("os.path.exists(df_name)", os.path.exists(df_name))

            if os.path.exists(df_name):
                # print("train_name", df_name)
                print("os.path.exists(df_name)", os.path.exists(df_name))
                df = pd.read_csv(df_name, sep='\t', index_col='Example', )
                # df = pd.read_csv(df_name, sep='\t', header=1, names=column_names, index_col='Example')
                # print('df\n', df)
                df = df[~df.index.duplicated(keep='last')].reset_index()
                # print('df\n', df)

                y_pred = df['deltaF'].to_numpy()
                y_true = df['deltaG'].to_numpy()

                # print("df[F_loss]", df['F_loss'])

                avg_F_loss = round(np.mean(df['F_loss']), 2)
                avg_CE_loss = np.mean(df['CE_loss']).round(2)
                avg_RMSD = np.mean(df['RMSD']).round(2)

                ## For experiments with L2 loss
                if exp == "SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep" \
                    or exp == 'SE3Bind_exp42_B_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v'\
                    or exp == "SE3Bind_exp43_B_wreg1e3_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v"\
                    or exp == "SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2v":

                    ## MAE for models trained with L2 (MSE) loss
                    mae = np.mean(np.abs(y_pred - y_true)).round(2)
                
                else: ## all models with L1 loss
                    mae = avg_F_loss 

                # metrics_df = pd.DataFrame()

                metrics_df = pd.DataFrame({
                    "Model": [exp_short_names_list[i]],
                    "avg_CE_loss": [avg_CE_loss],
                    "avg_F_loss": [avg_F_loss],
                    "MAE": [mae],
                    "avg_RMSD": [avg_RMSD]
                })
                # print(metrics_df)

                # print(f"avg_F_loss: {df["avg_F_loss"]:.3f}")
                # print(f"avg_CE_loss: {df["avg_CE_loss"]:.3f}")
                # print(f"avg_RMSD: {df["avg_RMSD"]:.3f}")

                # print('df\n', df)
            metrics_only_combined_df = pd.concat([metrics_only_combined_df, metrics_df], ignore_index=True)

            print("metrics_only_combined_df\n",metrics_only_combined_df)

        if save:
            if plot_training:
                prefix = 'Trainset_combined_metrics_epoch'

            if plot_valid:
                if valid_crystal:
                    prefix = 'crystalstructure_Validset_combined_metrics_epoch'
                if valid_homology:
                    prefix = 'Homology_model_Validset_combined_metrics_epoch'

            output_path = os.path.join(newlogfile_savepath, f"{prefix}{plot_epoch}.csv")
            metrics_only_combined_df.to_csv(output_path, index=False)
            print(f"Saved combined metrics to {output_path}")

    
    def get_all_epochs_metrics(self,
                plot_epoch,
                evaluate_epoch,
                plot_training=False,
                plot_valid = False,
                show=False,
                save=True,
                valid_crystal=False,
                valid_homology=False):
        
        
        metrics_only_combined_df = pd.DataFrame()
        metrics_df = pd.DataFrame()
        
        for current_epoch in range(evaluate_epoch, plot_epoch + 1, 10):
            print('current_epoch', current_epoch)
            if plot_training:
                train_name = f"{self.logfile_savepath}log_F_TRAINset_epoch{current_epoch}{self.experiment}.txt"
                df_name = train_name

            if plot_valid:
                if valid_homology:
                    prefix = 'BM_sab_HM'
                    valid_name = f"{self.logfile_savepath}log_F_{prefix}VALIDset_epoch{current_epoch}{self.experiment}.txt"

                if valid_crystal:
                    valid_name = f"{self.logfile_savepath}log_F_VALIDset_epoch{current_epoch}{self.experiment}.txt"
                
                df_name = valid_name
                # print("valid_name", valid_name)
                # print("os.path.exists(df_name)", os.path.exists(df_name))

            if os.path.exists(df_name):
                # print("os.path.exists(df_name)", os.path.exists(df_name))
                df = pd.read_csv(df_name, sep='\t', index_col='Example', )
                df = df[~df.index.duplicated(keep='last')].reset_index() # keep last duplicate index, latest outputs
                num_rows = len(df)
                # print('num_rows', num_rows)

                groundtruth = df['deltaG'].to_numpy()
                predicted = df['deltaF'].to_numpy()

                ##Calculate Pearson's correlation coefficient
                # mean_groundtruth = np.mean(groundtruth)
                # mean_predicted = np.mean(predicted)

                pearsons_r_value = self.calculate_pearsons_r(groundtruth, predicted)

                # print(f"Pearson's correlation coefficient (r) for {pearsons_r_value:.3f}")
                # print("np.mean(df['F_loss'])",np.mean(df['F_loss']))
                avg_F_loss= round(np.mean(df['F_loss']),2)
                avg_CE_loss = round(np.mean(df['CE_loss']),2)
                avg_RMSD = round(np.mean(df['RMSD']),2) #.round(2)

                ## For experiments with L2 loss
                if self.experiment == "SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep" \
                    or self.experiment == 'SE3Bind_exp42_B_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v'\
                    or self.experiment == "SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2v" \
                    or self.experiment == "SE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign"\
                    or self.experiment == "SE3Bind_exp51_B_JT_L2losszeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign":

                    ## MAE for models trained with L2 (MSE) loss
                    mae = np.mean(np.abs(predicted - groundtruth)).round(2)
                
                else: ## all models with L1 loss
                    mae = avg_F_loss 

                # metrics_df = pd.DataFrame()
                mse = np.mean((predicted - groundtruth) ** 2)
                rmse = np.sqrt(mse)

                metrics_df = pd.DataFrame({
                                "Epoch": [current_epoch],
                                "avg_CE_loss": [avg_CE_loss],
                                "avg_F_loss": [avg_F_loss],
                                "MAE": [mae],
                                "avg_RMSD": [avg_RMSD],
                                "R": [pearsons_r_value],
                                "RMSE": [rmse],
                                "num_rows": [num_rows]}
                                )
                if current_epoch == 1000:
                    print("metrics_df\n", metrics_df)

            metrics_only_combined_df = pd.concat([metrics_only_combined_df, metrics_df], ignore_index=True)
                # Print the row with the highest R value so far
        # if not metrics_only_combined_df.empty and "R" in metrics_only_combined_df.columns:
        #     max_r_row = metrics_only_combined_df.loc[metrics_only_combined_df["R"].idxmax()]
        #     print("Row with highest R so far:\n", max_r_row)

        # print("metrics_only_combined_df\n", metrics_only_combined_df)

        # if save:
        plt.savefig(f"{self.loss_save_folder}mae_coordrmsd_epoch{plot_epoch}.png")
        print(f"Saved MAE plot to {self.loss_save_folder}mae_coordrmsd_epoch{plot_epoch}.png")



        if save:
            if plot_training:
                prefix = 'Trainset_all_metrics_all_epoch'

            if plot_valid:
                if valid_crystal:
                    prefix = 'crystal_Validset_all_metrics_all_epoch'
                if valid_homology:
                    prefix = 'Homology_model_Validset_all_metrics_all_epoch'
        
            output_path = os.path.join(self.loss_logs_folder, f"{prefix}.csv")
            metrics_only_combined_df.to_csv(output_path, index=False)
            print('output_path', output_path)


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


    def plot_all_epochs_metrics(self,
                                plot_metrics=False,
                                save=False,
                                ):
        # Define experiments and short names
       # Genereate a combined F_loss and avg RMSD of all experiments
        experiments_list = [
            "SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign",
            "SE3Bind_exp40_B_JT_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep_asign",
            "SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep",
            "SE3Bind_exp39_B_JT_NOE0_nozeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_2s2v_200ep_asign",
            "SE3Bind_exp42_B_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v", #no deltaE0 with L2 loss
            # "SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2v", #no deltaE0 with L1first then and L2 loss
            # "SE3Bind_exp36_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L5s4v_200ep_asign",
            # "SE3Bind_exp35_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L2s1v_200ep_asign",
            "SE3Bind_exp33_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_5L3s2v_200ep_asign",
            "SE3Bind_exp34_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_3L3s2v_200ep_asign",
            "SE3Bind_exp45_B_absf0_L1loss_3L3s2v",
            "SE3Bind_exp46_B_noE0_L1loss_3L3s2v",

            "SE3Bind_exp51_B_JT_L2losszeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign",
            "SE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign"
            
            
        ]

        exp_short_names_list = [
            "SE3Bind (F_0 <> 0)",
            "SE3Bind (F_0 > 0)",
            "SE3Bind_L2loss (F_0 > 0) L1->L2loss",
            "4L2s2v_noDeltaE_0",
            "4L2s2v_noDeltaE_0_L2loss_scratch",
            # "4L2s2v_noDeltaE_0_L2loss_L1first",
            # "4L5s4v",
            # "4L2s1v",
            "5L3s2v",
            "3L3s2v",
            "3L3s2v_(F_0 > 0)",
            "3L3s2v_noDeltaE_0",
            
            "SE3Bind (F_0 <> 0) L2loss",
            "SE3Bind (F_0 > 0) L2 loss scratch"


        ]


        # Create the mapping dictionary
        model_name_map = dict(zip(experiments_list, exp_short_names_list))
        
        train_df = pd.read_csv(f"{self.loss_logs_folder}/Trainset_all_metrics_all_epoch.csv")
        crystal_valid = pd.read_csv(f"{self.loss_logs_folder}/crystal_Validset_all_metrics_all_epoch.csv")
        modeller_valid = pd.read_csv(f"{self.loss_logs_folder}/Homology_model_Validset_all_metrics_all_epoch.csv")

        if self.experiment in model_name_map:
            title_prefix = model_name_map[self.experiment]

        if plot_metrics:
            fig, axs = plt.subplots(5, 1, figsize=(16, 12), sharex=True)
            fig.suptitle(f'{title_prefix}', fontsize=16)

            fig, axs = plt.subplots(2, 1, figsize=(16, 8), sharex=True)
            fig.suptitle(f'{title_prefix}', fontsize=16)

            # First subplot: Average CE loss (L0 loss)
            axs[0].plot(train_df['Epoch'], train_df['avg_CE_loss'], label='Train', color='g')
            axs[0].plot(crystal_valid['Epoch'], crystal_valid['avg_CE_loss'], label='Crystal strucutre', color='b')
            axs[0].plot(modeller_valid['Epoch'], modeller_valid['avg_CE_loss'], label='Homology structure', color='r')
            axs[0].set_ylabel('Average $L_0$ loss',fontsize=14)
            # axs[0].set_title('Average $L_0$ loss per Epoch')
            axs[0].set_xticks(list(axs[1].get_xticks())[:-1] + [1900])
            axs[0].grid(False)

            # Second subplot: Average F loss (L1)
            axs[1].plot(train_df['Epoch'], train_df['avg_F_loss'], label='Train', color='g')
            axs[1].plot(crystal_valid['Epoch'], crystal_valid['avg_F_loss'], label='Crystal strucutre', color='b')
            axs[1].plot(modeller_valid['Epoch'], modeller_valid['avg_F_loss'], label='Homology structure', color='r')
            axs[1].set_xlabel('Epoch', fontsize=14)
            axs[1].set_ylabel('Average $L_1$ loss', fontsize=14)
            # axs[1].set_title('Average $L_1$ per Epoch')
            axs[1].grid(False)
            # Set the x-ticks so that the final tick is 1900
            axs[1].set_xticks(np.arange(0, 2000, 200))
            axs[1].set_xticks(list(axs[1].get_xticks())[:-1] + [1900])
            # Set larger font size for tick labels on both axes
            axs[0].tick_params(axis='both', labelsize=14)
            axs[1].tick_params(axis='both', labelsize=14)

            # Create a single legend for both subplots
            handles, labels_ = axs[0].get_legend_handles_labels()
            fig.legend(handles, labels_, loc='upper right', bbox_to_anchor=(0.99, 0.99))


            plt.tight_layout(rect=[0, 0, 1, 0.97])
        

        if save:
            if plot_metrics:
                # plt.savefig(f"{self.loss_save_folder}combined_metrics.png")
                # print(f"Saved MAE plot to {self.loss_save_folder}combined_metrics.png")

                plt.savefig(f"{self.loss_save_folder}training_curves.png")
                print(f"Saved MAE plot to {self.loss_save_folder}training_curves.png")
        else:
            plt.show()


    # def find_outliers(self, plot_epoch, 
    #                 experiments_list,
    #                 exp_short_names_list, 
    #                 plot_training=False,
    #                 plot_valid = False,
    #                 plot_testing = False,
    #                 show=False, 
    #                 save=True):
        
    #     column_names = ["Example","Loss", "F_loss", "CE_loss", "Feature0_reg_loss", "deltaF","deltaG", "RMSD", "clusterID", "structureIDs"]

    #     def find_outliers_iqr(df, col):
    #             Q1 = df[col].quantile(0.25)  # First quartile (25th percentile)
    #             Q3 = df[col].quantile(0.75)  # Third quartile (75th percentile)
    #             IQR = Q3 - Q1  # Interquartile Range
    #             lower_bound = Q1 - 1.5 * IQR
    #             upper_bound = Q3 + 1.5 * IQR
    #             return df[(df[col] < lower_bound) | (df[col] > upper_bound)].index
        
    #     all_train_data = []
    #     num_experiments = len(experiments_list)

    #     colors = ['g', 'b', 'r']
    #     labels = ['TRAINset', 'VALIDset', 'TESTset']
    #     fill_color=['grey', 'lightgrey', 'lightgrey']

    #     outliers_experi_dict ={}
    #      # IF self.experiment == SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep: replace deltaG with avgdeltaG
        
    #     for i, exp in enumerate(experiments_list):  
    #         if plot_training:         
    #             train_name = f"{self.logfile_savepath}{exp}/Log_F_TRAINset_epoch{plot_epoch}{exp}.txt"
    #             df_name = train_name
    #             labels = labels[0]
    #         if plot_valid:
    #             valid_name = f"{self.logfile_savepath}{exp}/Log_F_VALIDset_epoch{plot_epoch}{exp}.txt"
    #             df_name = valid_name
    #             labels = labels[1]

    #         if os.path.exists(df_name):
    #             print('os.path.exists(df_name)',os.path.exists(df_name))

    #             df = pd.read_csv(df_name, sep='\t', header=1, names=column_names, index_col='Example')
    #             df = df[~df.index.duplicated(keep='last')].reset_index()

    #             # print('df', df)

    #             groundtruth = df['deltaG'].to_numpy()
    #             predicted = df['deltaF'].to_numpy()

    #             spearmanr = stats.spearmanr(groundtruth, predicted)
    #             print(f"Spearman's correlation coefficient (r) for {labels}: {spearmanr.correlation:.3f}")

    #             # kendalltau = stats.kendalltau(groundtruth, predicted)
    #             # print(f"kendalltau's correlation coefficient (r) for {labels}: {kendalltau.correlation:.3f}")
    #             # ## Calculate Pearson's correlation coefficient
    #             # r_value = self.calculate_pearsons_r(groundtruth, predicted)
    #             # denominator = np.sqrt(np.sum((groundtruth - mean_groundtruth)**2) * np.sum((predicted - mean_predicted)**2))
    #             # r_value = numerator / denominator

    #             # print(f"Pearson's correlation coefficient (r) for {labels[i]}: {r_value:.3f}")

    #             # x_lower = min(min(df['deltaF'].to_numpy()) for df in df_list if df is not None)
    #             # x_lower = -20
    #             # x_upper = 0
    #             # diag = np.array([x_lower, x_upper])

    #             ###Calculate the standard error of the fit (SEF)
    #             residuals = groundtruth - predicted
    #             n = len(residuals)
    #             sef = np.sqrt(np.sum(residuals**2) / (n - 1))
    #             outliers = find_outliers_iqr(df, 'deltaF')
    #             df['sef'] = sef
    #             # outlier_examples = df.loc[outliers, ['Example', 'structureIDs', 'sef']]
    #             # print(f"outlier_examples dataframe {outlier_examples}")

    #             # upper_bound = diag + sef
    #             # lower_bound = diag - sef
    #             # ax.fill_between(diag, lower_bound, upper_bound, color=fill_color[i], alpha=0.4, label=f'{labels[i]}± StdError: R={r_value:.3f}')





if __name__ == "__main__":
    print('runs')

    experiment = "SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign"
    p = PlotterT1(experiment=experiment,
                  logfile_savepath='Log/losses/')
    

    
    # Genereate a combined F_loss and avg RMSD of all experiments
    experiments_list = [
        "SE3Bind_exp29_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign",
        "SE3Bind_exp40_B_JT_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep_asign",
        "SE3Bind_exp41_B_JT_L2loss_absDeltaE0_zeroFeatL1_wReg6sum_refModel_4L3s2v_200ep",
        "SE3Bind_exp39_B_JT_NOE0_nozeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_2s2v_200ep_asign",
        "SE3Bind_exp42_B_JT_L2loss_NOE0_nozeroFeatL1_refModel_2s2v", #no deltaE0 with L2 loss
        "SE3Bind_exp33_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_5L3s2v_200ep_asign",
        "SE3Bind_exp34_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_3L3s2v_200ep_asign",
        "SE3Bind_exp45_B_absf0_L1loss_3L3s2v",
        "SE3Bind_exp46_B_noE0_L1loss_3L3s2v",
        "SE3Bind_exp51_B_JT_L2losszeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L3s2v_200ep_asign",
        "SE3Bind_exp50_B_JT_L2loss_absDeltaE0_zeroFeat_wReg6sum_refModel_4L3s2v_200ep_asign"
        

        ## Experiments not included in manuscript
        ## "SE3Bind_exp44_B_L1_L2loss_NOE0_nozeroFeatL1_refModel_2s2v", #no deltaE0 with L1first then and L2 loss
        ## "SE3Bind_exp36_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L5s4v_200ep_asign",
        ## "SE3Bind_exp35_B_JT_zeroFeatL1_wReg3sum_convZerofeat_plr4_ilr2_refModel_4L2s1v_200ep_asign",
        
    ]

    exp_short_names_list = [
        "SE3Bind (F_0 <> 0)",
        "SE3Bind (F_0 > 0)",
        "SE3Bind_L2loss (F_0 > 0) L1->L2loss",
        "4L2s2v_noDeltaE_0",
        "4L2s2v_noDeltaE_0_L2loss_scratch",
        "5L3s2v",
        "3L3s2v",
        "3L3s2v_(F_0 > 0)",
        "3L3s2v_noDeltaE_0",
        "SE3Bind (F_0 <> 0) L2loss",
        "SE3Bind (F_0 > 0) L2 loss scratch"
        ## "4L2s2v_noDeltaE_0_L2loss_L1first",
        ## "4L5s4v",
        ## "4L2s1v",


    ]


    p.plot_F_loss_allcombined(experiments_list, exp_short_names_list, ylim=None, show=True, save=False)
    
    # p.find_outliers(plot_epoch, 
    #                 experiments_list,
    #                 exp_short_names_list,
    #                 plot_training=False,
    #                 plot_valid = True,
    #                 plot_testing = False,
    #                 show=False, 
    #                 save=True)