import numpy as np
import matplotlib.pyplot as plt
import sys
import os
from os.path import exists
import pandas as pd


class PlotterT0:
    def __init__(self, experiment=None,
                 logfile_savepath='Log/losses/'):
        """
        Initialize paths and filename prefixes for saving plots.

        :param experiment: current experiment name
        :param logfile_savepath: path to load and save data and figs
        """
        self.experiment = experiment
        self.logfile_savepath = logfile_savepath

        self.loss_save_folder = 'Figs/IP_loss_plots/'+self.experiment+'/'
        os.makedirs(self.loss_save_folder, exist_ok=True)

        self.rmsd_save_folder = 'Figs/IP_RMSD_distribution_plots/'+self.experiment+'/'
        os.makedirs(self.rmsd_save_folder, exist_ok=True)

        if not experiment:
            print('no experiment name given')
            sys.exit()

    def plot_loss(self, ylim=None, show=False, save=True, plot_combined=False):
        """
        Plot the current interaction pose (IP) experiments loss curve.
        The plot will plot all epochs present in the log file.

        :param ylim: set the upper limit of the y-axis, initial IP loss can vary widely
        :param show: show the plot in a window
        :param save: save the plot at specified path
        """
        plt.close()
        # Define the file names for train, valid, and test logs
        train_name = f"{self.logfile_savepath}log_loss_TRAINset_{self.experiment}.txt"
        valid_name = f"{self.logfile_savepath}log_loss_VALIDset_{self.experiment}.txt"
        test_name = f"{self.logfile_savepath}log_loss_TESTset_{self.experiment}.txt"

        # Define the names for the different columns in the logs
        column_names = ["Epoch", "Loss", "avgRMSD"]

        # Define the number of subplots and figure size
        num_subplots = 2
        fig, ax = plt.subplots(num_subplots, figsize=(20, 10))

        # Read the train log and check if the valid and test logs exist
        if os.path.exists(train_name):
            train = pd.read_csv(train_name, sep="\t", header=1, names=column_names)
            valid = pd.read_csv(valid_name, sep="\t", header=1, names=column_names)
            test = pd.read_csv(test_name, sep="\t", header=1, names=column_names)

            # Plot the average RMSD for each dataset
            line_rmsd, = ax[0].plot(train["Epoch"], train["avgRMSD"])
            ax[0].plot(valid["Epoch"], valid["avgRMSD"])
            ax[0].plot(test["Epoch"], test["avgRMSD"])
            ax[0].legend(("train avgRMSD", "valid avgRMSD", "test avgRMSD"))

            # Set the title, ylabel, and grid for the first subplot
            ax[0].set_title(f"Loss: {self.experiment}")
            ax[0].set_ylabel("avgRMSD")
            ax[0].grid(visible=False)
            ax[0].margins(x=0.01)

            # Plot the CE loss for each dataset
            line_loss, = ax[1].plot(train["Epoch"], train["Loss"])
            ax[1].plot(valid["Epoch"], valid["Loss"])
            ax[1].plot(test["Epoch"], test["Loss"])
            ax[1].legend(("train loss", "valid loss", "test loss"))

            # Set the xlabel, ylabel, and grid for the second subplot
            ax[1].set_xlabel("epochs")
            ax[1].set_ylabel("CE loss")
            ax[1].grid(visible=False)
            ax[1].margins(x=0.01)

            # Set the ylim if specified
            if ylim:
                ax[0].set_ylim([0, ylim])
                ax[1].set_ylim([0, ylim])
            else:
                ax[0].set_ylim([0, 15])
                ax[1].set_ylim([0, 15])

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
        train_log = self.logfile_savepath+'log_RMSDsTRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        valid_log = self.logfile_savepath+'log_RMSDsVALIDset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        test_log = self.logfile_savepath+'log_RMSDsTESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        avg_trainRMSD, avg_validRMSD, avg_testRMSD = None, None, None

        log_list = [train_log, valid_log, test_log]
        plot_cond_list = list(plot_streams.values())
        df_list = []
        avg_RMSDs = []
        subplot_count = 0
        for i in range(len(log_list)):
            if plot_cond_list[i]:
                subplot_count += 1
                plot_df = pd.read_csv(log_list[i], sep='\t', header=0, names=['RMSD'])
                avg_RMSD = str(plot_df['RMSD'].mean())[:4]
                df_list.append(plot_df)
                avg_RMSDs.append(avg_RMSD)
                # print(log_list[i])
                # print('average RMSD:')
                # print(avg_RMSD)
            else:
                df_list.append(None)
                avg_RMSDs.append(None)

        train_avg_rmsd = str(avg_RMSDs[0])[:6]
        valid_avg_rmsd = str(avg_RMSDs[1])[:6]
        if len(avg_RMSDs) > 2:
            test_avg_rmsd = str(avg_RMSDs[2])[:6]
        else:
            test_avg_rmsd = None
        fig, ax = plt.subplots(subplot_count, figsize=(20, 10))
        plt.suptitle('RMSD distribution: epoch' + str(plot_epoch) + ' ' + self.experiment + '\n'
                     + 'train:' + train_avg_rmsd + ' valid:' + valid_avg_rmsd + ' test:' + test_avg_rmsd)

        # fig, ax = plt.subplots(subplot_count, figsize=(20, 10))
        # # plt.suptitle('RMSD distribution: epoch' + str(plot_epoch) + ' ' + self.experiment +'\n'
        # #               + 'train:'+ str(avg_RMSDs[0]) + ' valid:' + str(avg_validRMSD) + ' test:' + str(avg_testRMSD))
        # # plt.legend(['train rmsd', 'valid rmsd', 'test rmsd'])
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
            plt.savefig(self.rmsd_save_folder+'/rmsdplot_epoch' + str(plot_epoch) + '_' + self.experiment + '.png')
        if show:
            plt.show()

    def plot_deltaF_distribution(self,
                                 plot_training=False,
                                 plot_valid = False,
                                 plot_testing = False,
                                 plot_epoch=1, show=False, save=True):
        """
        Plot the labeled free energies of interacting and non-interacting shape pairs as a histogram,
        with a vertical line demarcating the learned `F_0` interaction decision threshold, if applicable.

        :param filename: specify the file to load, default of `None` sets filename to match the project convention
        :param plot_epoch: epoch of training/evalution to plot
        :param xlim: absolute value lower limit of the x-axis.
        :param binwidth: histogram bin width
        :param show: show the plot in a window
        :param save: save the plot at specified path
        :return:
        """
        plt.close()

        # Plot free energy distribution of all samples across epoch
        train_log = self.logfile_savepath+'log_deltaF_TRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        valid_log = self.logfile_savepath+'log_deltaF_VALIDset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        test_log = self.logfile_savepath+'log_deltaF_TESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        subplot_count = 0

        log_list = [train_log, valid_log, test_log]
        plot_cond_list = [plot_training, plot_valid, plot_testing]
        df_list = []
        names = ['deltaF', 'deltaG', 'F_0', 'F_0_prime']

        for i in range(len(log_list)):
            if plot_cond_list[i] and exists(log_list[i]):
                subplot_count += 1
                plot_df = pd.read_csv(log_list[i], sep='\t', header=0, names=names)
                df_list.append(plot_df)
                print(log_list[i])
            else:
                df_list.append(None)

        fig, ax = plt.subplots(subplot_count, figsize=(20, 10))
        plt.suptitle('deltaF vs deltaG distributions: epoch' + str(plot_epoch) + ' ' + self.experiment)
        plt.legend(['train set', 'valid set', 'test set'])
        plt.xlabel('free energy')
        binwidth = 1
        x_upper = 25
        x_lower = -x_upper
        ticks = 5
        bins = np.arange(x_lower, x_upper + binwidth, binwidth)

        y_labels = ['Training set',
                    'Validation set',
                    'Testing set']

        # plt.figure(figsize=(8,6))
        # # if not plot_pub:
        # plt.title('deltaF vs deltaG distribution: epoch' + str(plot_epoch) + ' ' + self.experiment)
        format = 'png'
        # # else:
        # #     format = 'pdf'

        if subplot_count == 1:
            cond_index = [i for i in range(len(df_list)) if df_list[i] is not None][0]
            df = df_list[cond_index]
            # ax.hist(df_list[0]['RMSD'].to_numpy(), bins=bins, color=colors[0])
            y1, x1, _ = ax.hist(df['deltaF'].to_numpy(), bins=bins, rwidth=binwidth, color=['g'], alpha=0.25)
            y2, x2, _ = ax.hist(df['deltaG'].to_numpy(), bins=bins, rwidth=binwidth, color=['r'], alpha=0.25)
            ymax = max(max(y1), max(y2)) + 1
            ax.vlines(df['F_0'].to_numpy()[-1], ymin=0, ymax=ymax, linestyles='dashed', label='F_0', colors='k')
            ax.vlines(df['F_0_prime'].to_numpy()[-1], ymin=0, ymax=ymax, linestyles='solid', label='F_0_prime', colors='k')
            ax.set_ylabel(y_labels[cond_index])
            ax.grid(visible=True)
            ax.set_xticks(np.arange(x_lower, x_upper, ticks))
            ax.set_ylim([0, ymax])
            ax.legend(( 'final F_0', 'final F_0_prime', 'predicted deltaF', 'ground truth deltaG'), prop={'size': 10},
                       loc='upper left')
            ax.grid(visible=False)
            ax.margins(x=None)
        else:
            plot_index = 0
            for i in range(len(plot_cond_list)):
                if df_list[i] is not None and plot_cond_list[i] is True:
                    # print(i, y_labels[i], colors[i])
                    df = df_list[i]
                    y1, x1, _ = ax[plot_index].hist(df['deltaF'].to_numpy(), bins=bins, rwidth=binwidth, color=['g'],
                                        alpha=0.25)
                    y2, x2, _ = ax[plot_index].hist(df['deltaG'].to_numpy(), bins=bins, rwidth=binwidth, color=['r'],
                                        alpha=0.25)
                    ymax = max(max(y1), max(y2)) + 1
                    ax[plot_index].vlines(df['F_0'].to_numpy()[-1], ymin=0, ymax=ymax, linestyles='dashed', label='F_0', colors='k')
                    ax[plot_index].vlines(df['F_0_prime'].to_numpy()[-1], ymin=0, ymax=ymax, linestyles='solid', label='F_0_prime', colors='k')
                    ax[plot_index].set_ylabel(y_labels[i])
                    ax[plot_index].grid(visible=True)
                    ax[plot_index].set_xticks(np.arange(x_lower, x_upper, ticks))
                    ax[plot_index].set_ylim([0, ymax])
                    ax[plot_index].legend(('final F_0', 'final F_0_prime', 'predicted deltaF', 'ground truth deltaG'),
                               prop={'size': 10}, loc='upper left')
                    ax[plot_index].grid(visible=False)
                    ax[plot_index].margins(x=None)
                else:
                    plot_index -= 1
                    pass
                plot_index += 1

        if save:
            plt.savefig('Figs/JT_deltaF_distribution_plots/deltaFvsdeltaGplot_epoch'+ str(plot_epoch) + '_' + self.experiment + '.'+format, format=format)
        if show:
            plt.show()

    def plot_deltaF_vs_deltaG_correlation(self,
                                 plot_training=False,
                                 plot_valid = False,
                                 plot_testing = False,
                                 plot_epoch=1, show=False, save=True):
        """
        Plot correlation of predicted deltaF versus ground truth deltaG.
        A vertical and horizontal line demarcates the learned `F_0` interaction decision threshold, if applicable.

        :param filename: specify the file to load, default of `None` sets filename to match the project convention
        :param plot_epoch: epoch of training/evalution to plot
        :param xlim: absolute value lower limit of the x-axis.
        :param binwidth: histogram bin width
        :param show: show the plot in a window
        :param save: save the plot at specified path
        :return:
        """
        plt.close()

        # Plot free energy distribution of all samples across epoch
        train_log = self.logfile_savepath+'log_deltaF_TRAINset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        valid_log = self.logfile_savepath+'log_deltaF_VALIDset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        test_log = self.logfile_savepath+'log_deltaF_TESTset_epoch' + str(plot_epoch) + self.experiment + ".txt"
        # subplot_count = 0

        # train_log = 'Log/losses/IP_loss/log_deltaF_TRAINset_epoch101B_CLUSTER_lr-4lr-1_BSFI_BSIPpre302_3layer_5c6s6v_k7nrb3lmax2_F0only_lr-2_unshuffled_box50pad100_4438ex.txt'

        log_list = [train_log, valid_log, test_log]
        plot_cond_list = [plot_training, plot_valid, plot_testing]
        df_list = []
        names = ['deltaF', 'deltaG', 'F_0', 'F_0_prime', 'clusterID', 'structureID']

        for i in range(len(log_list)):
            if plot_cond_list[i] and exists(log_list[i]):
                # subplot_count += 1
                plot_df = pd.read_csv(log_list[i], sep='\t', header=0, names=names)
                df_list.append(plot_df)
            else:
                df_list.append(None)

        y_labels = ['Training set',
                    'Validation set',
                    'Testing set']
        format = 'png'

        # if subplot_count == 1:
        fig = plt.figure(constrained_layout=True, figsize=(10,10))
        # Create the main axes
        ax = fig.add_gridspec(top=0.75, right=0.75).subplots()
        ax.set(aspect=1)
        # Create marginal axes
        ax_histx = ax.inset_axes([0, 1.05, 1, 0.25], sharex=ax)
        ax_histy = ax.inset_axes([1.05, 0, 0.25, 1], sharey=ax)

        cond_index = [i for i in range(len(df_list)) if df_list[i] is not None][0]
        df = df_list[cond_index]

        # print(df['clusterID'])

        df['clusterID_numbered'] = [0 if x == "('UniRef50_P15692',)" else 1 if x == "('UniRef50_P06931',)" else 2 for x in df['clusterID']]

        # print(df.iloc[:, -3:])

        # df['clusterID'] = [print(int(''.join(x.split("'")[1])[-5:])) for x in df['clusterID']]
        # print(df['clusterID'].head())
        # print(df['clusterID'])
        # df['clusterID'] = [0 if x == 6931 else 1 if x == 15123 else 2 for x in df['clusterID']]
        # print(df['clusterID'].head())

        # Draw the scatter plot and marginals.
        groundtruth = df['deltaG'].to_numpy()
        # print(groundtruth)
        predicted = df['deltaF'].to_numpy()
        group = df['clusterID_numbered']
        F_0 = df['F_0'].to_numpy()[-1]

        x_lower = -22
        x_upper = 0
        # scatter_hist = scatter_hist(x, y, c, ax, F_0, ax_histx, ax_histy)
        ax_histx.tick_params(axis="x", labelbottom=False)
        ax_histy.tick_params(axis="y", labelleft=False)

        # # the scatter plot:
        # cdict = {0: 'orange', 1: 'purple', 2: 'blue'}
        # # gdict = {0: 'P15692', 1: 'P06931', 2: 'O15123'}
        # gdict = {0: 'VEGF (P15692)', 1: 'PGF (P06931)', 2: 'ANGPT2\n(O15123)'}
        #
        # for g in np.unique(group):
        #     ix = np.where(group == g)
        #     ax.scatter(groundtruth[ix], predicted[ix], c=cdict[g], label=gdict[g], s=100, alpha=0.5)

        ax.scatter(groundtruth, predicted, alpha=0.5)

        # print(F_0)
        ax.vlines(F_0, ymin=F_0, ymax=x_upper, linestyles='dashed', colors='k')
        ax.hlines(F_0, xmin=F_0, xmax=x_upper, linestyles='dashed', label=r'learned $F_{0}$', colors='k')

        binwidth = 0.25
        xymax = max(np.max(np.abs(groundtruth)), np.max(np.abs(predicted)))
        lim = (int(xymax / binwidth) + 1) * binwidth

        bins = np.arange(-lim, lim + binwidth, binwidth)
        y1, x1, _ = ax_histx.hist(groundtruth, bins=bins, color='r', alpha=0.5)
        ax_histx.vlines(F_0, ymin=0, ymax=max(y1), linestyles='dashed', colors='k')
        y2, x2, _ = ax_histy.hist(predicted, bins=bins, orientation='horizontal', color='g', alpha=0.5)
        ax_histy.hlines(F_0, xmin=0, xmax=max(y2), linestyles='dashed', colors='k')

        # plt.suptitle('deltaF vs deltaG correlation: epoch' + str(plot_epoch) + ' ' + self.experiment)
        # plt.legend(['train set', 'valid set', 'test set'])
        plt.ylabel(r'$\Delta F$', fontsize=18, rotation=0, labelpad=20)
        plt.xlabel('ground truth median $\Delta G$', fontsize=18, labelpad=20)

        diag = np.array([x_lower, x_upper])
        ax.plot(diag, diag, 'k-', alpha=0.75, zorder=0, label='1:1 correlation')

        ax.set_title(y_labels[cond_index]+" epoch "+str(plot_epoch), fontdict={'size': 14})
        ax.grid(visible=True)
        ax.set_xlim([x_lower, x_upper])
        ax.set_ylim([x_lower, x_upper])

        # ax.set_xticks(np.arange(x_lower, x_upper, ticks))
        # ax.set_ylim([0, F_0])
        ax.legend(
            # ('cluster1', '1:1 correlation', 'cluster2', 'cluster3'),
            prop={'size': 14},
                   loc='upper left')
        ax.grid(visible=False)
        ax.margins(x=None)

        if save:
            plt.savefig('Figs/JT_deltaF_distribution_plots/correlationdeltaFvsdeltaGplot_epoch'+ str(plot_epoch) + '_' + self.experiment + '.'+format, format=format)
        if show:
            plt.show()


if __name__ == "__main__":
    loadpath = 'Log/losses/IP_loss/'
    # experiment = 'BF_IP_NEWDATA_CHECK_400pool_30ep'
    experiments_list = [
                       'BF_IP_finaldataset_1000pairs_100ep',
                       'BF_IP_finaldataset_100pairs_100ep',
                       'BS_IP_finaldataset_1000pairs_100ep',
                       'BS_IP_finaldataset_100pairs_100ep',
                       ]
    fig_data = []
    for experiment in experiments_list:
        Plotter = PlotterT0(experiment, logfile_savepath=loadpath)
        line_loss = Plotter.plot_loss(show=True, plot_combined=True)
        # Plotter.plot_rmsd_distribution(plot_epoch=100, show=True)
        fig_data.append(line_loss)
        plt.close()

    # plt.close()
    plt.figure(figsize=(10,5))
    color_style = ['b-', 'b--', 'r-', 'r--']

    for i in range(len(fig_data)):
        plt.plot(fig_data[i].get_data()[0], fig_data[i].get_data()[1], color_style[i], )

    plt.margins(x=None)
    plt.legend(['BruteForce IP 1000pairs', 'BruteForce IP 100pairs', 'BruteSimplified IP 1000pairs', 'BruteSimplified IP 100pairs'])
    plt.ylabel('loss')
    plt.xlabel('epochs')
    plt.xlim([0,100])
    plt.savefig('Figs/IP_loss_plots/sup_combined_IP_loss_plot.pdf', format='pdf')
    plt.show()
