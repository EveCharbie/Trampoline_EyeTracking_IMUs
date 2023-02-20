
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.cm as cm
import pickle
from IPython import embed
import pandas as pd
import os
from scipy.io import savemat, loadmat
import mpl_toolkits.mplot3d.axes3d as p3
import sys
sys.path.append("../metrics/")
from animate_JCS import plot_gymnasium_symmetrized


def load_eye_tracking_metrics(path, file):
    index_ = [idx for idx, char in enumerate(file) if char == '_']
    move_number = file[index_[6]+1: index_[7]]

    file_id = open(path + file, "rb")
    eye_tracking_metrics = pickle.load(file_id)

    return move_number, eye_tracking_metrics

def plot_primary_metrics(df, move_list, subelite_names, elite_names, metric, unit, title, save_path):

    cmap = plt.get_cmap('plasma')
    subelite_color = cmap(0.25)
    elite_color = cmap(0.70)

    plt.figure()
    for i in range(len(df)):
        move_index = move_list.index(df['Acrobatics'][i])
        if df['Expertise'][i] == 'SubElite':
            if df['Name'][i] in subelite_names:
                plt.plot((subelite_names.index(df['Name'][i]) + 1) * -0.1 + move_index * 2, df[metric][i],
                     color=subelite_color, marker='o', markersize=2)
        if df['Expertise'][i] == 'Elite':
            if df['Name'][i] in elite_names:
                plt.plot((elite_names.index(df['Name'][i]) + 1) * 0.1 + move_index * 2, df[metric][i],
                     color=elite_color, marker='o', markersize=2)

    for i in range(len(move_list)):
        means_subelite = []
        means_elite = []
        for j in range(len(subelite_names)):
            index_this_time = np.where(
                np.logical_and(df['Name'] == subelite_names[j], df['Acrobatics'] == move_list[i]))
            subelite_list = list(df[metric][index_this_time[0]])
            means_subelite.append(np.nanmedian(subelite_list))
            plt.plot(i * 2 - 0.1 * (j + 1), means_subelite[j], color='k', marker='o', markersize=3)
        for j in range(len(elite_names)):
            index_this_time = np.where(
                np.logical_and(df['Name'] == elite_names[j], df['Acrobatics'] == move_list[i]))
            elite_list = df[metric][index_this_time[0]]
            means_elite.append(np.nanmedian(elite_list))
            plt.plot(i * 2 + 0.1 * (j + 1), means_elite[j], color='k', marker='o', markersize=3)

        plt.errorbar(i * 2 - 0.45, np.nanmean(means_subelite), yerr=np.nanstd(means_subelite), color='black',
                     marker='o', markersize=5)
        plt.errorbar(i * 2 + 0.45, np.nanmean(means_elite), yerr=np.nanstd(means_elite), color='black', marker='o',
                     markersize=5)

    plt.ylabel(metric + ' [' + unit + ']')
    plt.xlabel('Acrobatics')
    plt.xticks(ticks=[0, 2, 4, 6], labels=[i + '/' for i in move_list])

    plt.xlim(-1.2, 7)
    plt.ylim(np.nanmin(df[metric]) - (np.nanmax(df[metric]) - np.nanmin(df[metric])) * 0.1, np.nanmax(df[metric])  + (np.nanmax(df[metric]) - np.nanmin(df[metric])) * 0.1)
    plt.plot(-2, 0, 'o', color=subelite_color, markersize=2, label='SubElite')
    plt.plot(-2, 0, 'o', color=elite_color, markersize=2, label='Elite')
    plt.plot(-2, 0, 'ok', markersize=3, label='Median per participant')
    plt.plot(-2, 0, 'ok', markersize=5, label='Mean of medians')
    plt.legend(ncol=4, loc='upper center', bbox_to_anchor=(0.5, 1.1))
    plt.subplots_adjust(hspace=0.1, top=0.9)
    plt.savefig(save_path + title + '.png', dpi=300)
    # plt.show()

    return

def primary_plots(df, move_list, subelite_names, elite_names, plot_path):

    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Fixations duration absolute', 's', 'fixation_duration_absolute', f'{plot_path}/')
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Fixations duration relative', 's', 'fixation_duration_relative', f'{plot_path}/')
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Number of fixations', '', 'fixation_number', f'{plot_path}/')
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Quiet eye duration absolute', 's', 'quiet_eye_duration_absolute', f'{plot_path}/')
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Quiet eye duration relative', 's', 'quiet_eye_duration_relative', f'{plot_path}/')
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Eye amplitude', '?', 'eye_amplitude', f'{plot_path}/')
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Neck amplitude', '?', 'neck_amplitude', f'{plot_path}/')
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Maximum eye amplitude', 'rad', 'max_eye_amplitude', f'{plot_path}/')
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Maximum neck amplitude', 'rad', 'max_neck_amplitude', f'{plot_path}/')
    # see what we want to do with the zero from Pupil
    return

def trajectory_plots(df, move_list, subelite_names, elite_names):

    def plot_gaze_trajectory(ax,
                            gaze_position_temporal_evolution_projected,
                            color,
                            ):

        N = len(gaze_position_temporal_evolution_projected[:, 0]) - 1
        for j in range(N):
            ax.plot(
                gaze_position_temporal_evolution_projected[j: j + 2, 0],
                gaze_position_temporal_evolution_projected[j: j + 2, 1],
                gaze_position_temporal_evolution_projected[j: j + 2, 2],
                linestyle='-', color=color, linewidth=0.5,
            )

        return

    bound_side = 3 + 121 * 0.0254 / 2

    plt.close('all')

    fig1 = plt.figure(0)
    ax1 = p3.Axes3D(fig1)
    ax1.set_box_aspect([1, 1, 1])
    plot_gymnasium_symmetrized(bound_side, ax1)

    fig2 = plt.figure(1)
    ax2 = p3.Axes3D(fig2)
    ax2.set_box_aspect([1, 1, 1])
    plot_gymnasium_symmetrized(bound_side, ax2)

    fig3 = plt.figure(2)
    ax3 = p3.Axes3D(fig3)
    ax3.set_box_aspect([1, 1, 1])
    plot_gymnasium_symmetrized(bound_side, ax3)

    fig4 = plt.figure(3)
    ax4 = p3.Axes3D(fig4)
    ax4.set_box_aspect([1, 1, 1])
    plot_gymnasium_symmetrized(bound_side, ax4)

    ax_list = [ax1, ax2, ax3, ax4]

    cmap = plt.get_cmap('plasma')

    for i in range(len(df)):
        print(df['Name'][i])
        print(df['Acrobatics'][i])

        gaze_position_temporal_evolution_projected_symmetrized = np.array(df['Projected gaze orientation facing front wall (PGOS)'][i])
        if df['Name'][i] in subelite_names:
            index = subelite_names.index(df['Name'][i])
            color = cmap(index * 0.35 / len(subelite_names))
        elif df['Name'][i] in elite_names:
            index = elite_names.index(df['Name'][i])
            color = cmap(0.65 + index * 0.35 / len(elite_names))

        index_move = move_list.index(df['Acrobatics'][i])
        print(index_move)
        plt.figure(index_move)
        plot_gaze_trajectory(ax_list[index_move], gaze_position_temporal_evolution_projected_symmetrized, color)

    for i in range(4):
        ax_list[i].view_init(elev=15, azim=-120)
    fig1.savefig(f"{plot_path}/gaze_trajectories_3D_4-.png", dpi=300)
    fig2.savefig(f"{plot_path}/gaze_trajectories_3D_41.png", dpi=300)
    fig3.savefig(f"{plot_path}/gaze_trajectories_3D_42.png", dpi=300)
    fig4.savefig(f"{plot_path}/gaze_trajectories_3D_43.png", dpi=300)
    # plt.show()


# def bar_plots_elites_and_subelites_diff_color(df, type_names, subelite_names, elite_names):
#
#     len_types = len(type_names)
#     cmap = plt.get_cmap('plasma')
#     subelite_colors = []
#     elite_colors = []
#     for i in range(len_types):
#         subelite_colors.append(cmap(0.35 * i / len_types))
#         elite_colors.append(cmap(0.65 + 0.35 * i / len_types))
#
#
#     means_subelite = {key: [] for key in type_names}
#     means_elite = {key: [] for key in type_names}
#     for i in range(len(move_list)):
#         subelite_list = []
#         for j in range(len(subelite_names)):
#             index_this_time = np.where(np.logical_and(df['Name'] == subelite_names[j], df['Acrobatics'] == move_list[i]))
#             for key in type_names:
#                 subelite_list.append(list(df[key][index_this_time[0]] * 100))
#                 means_subelite[key].append(np.nanmean(list(df[key][index_this_time[0]] * 100)))
#         elite_list = []
#         for j in range(len(elite_names)):
#             index_this_time = np.where(np.logical_and(df['Name'] == elite_names[j], df['Acrobatics'] == move_list[i]))
#             for key in type_names:
#                 elite_list.append(list(df[key][index_this_time[0]] * 100))
#                 means_elite[key].append(np.nanmean(list(df[key][index_this_time[0]] * 100)))
#
#     fig = plt.figure()
#     ax = fig.add_axes([0.1, 0.1, 0.85, 0.55])
#     for i in range(len(move_list)):
#         total_subelite = 0
#         for j, key in enumerate(type_names):
#             mean = np.nanmean(means_subelite[key])
#             ax.bar(i*2 - 0.4, mean, bottom=total_subelite, color=subelite_colors[j], width=0.4, label=('Subelite ' + key if i == 0 else None))
#             total_subelite += mean
#         total_elite = 0
#         for j, key in enumerate(type_names):
#             mean = np.nanmean(means_elite[key])
#             ax.bar(i * 2 + 0.4, mean, bottom=total_elite, color=elite_colors[j], width=0.4)
#             total_elite += mean
#
#         ax.bar(i * 2 - 0.4, 100-total_subelite, bottom=total_subelite, color='grey', width=0.4, label=('Other' if i == 0 else None))
#         ax.bar(i * 2 + 0.4, 100-total_elite, bottom=total_elite, color='grey', width=0.4, alpha=0.5)
#     return

def stair_bar_plots(df, type_names, subelite_names, elite_names, pourcentage=True, other=False):

    if pourcentage:
        factor = 100
    else:
        factor = 1

    len_types = len(type_names)
    cmap = plt.get_cmap('plasma')
    colors = []
    for i in range(len_types):
        colors.append(cmap(i / len_types))

    means_subelite = {key: [[] for _ in range(4)] for key in type_names}
    means_elite = {key: [[] for _ in range(4)] for key in type_names}
    for i in range(len(move_list)):
        for j in range(len(subelite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == subelite_names[j], df['Acrobatics'] == move_list[i]))
            for key in type_names:
                means_subelite[key][i].append(np.nanmean(list(df[key][index_this_time[0]] * factor)))
        for j in range(len(elite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == elite_names[j], df['Acrobatics'] == move_list[i]))
            for key in type_names:
                means_elite[key][i].append(np.nanmean(list(df[key][index_this_time[0]] * factor)))

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.7])
    for i in range(len(move_list)):
        total_subelite = 0
        for j, key in enumerate(type_names):
            mean = np.nanmean(means_subelite[key][i])
            std = np.nanstd(means_subelite[key][i])
            ax.bar(i*2 - 0.4 + 0.8/len(type_names)*j, mean, bottom=total_subelite, yerr=std, color=colors[j], width=0.8/len(type_names), label=(key if i == 0 else None))
            total_subelite += mean
        total_elite = 0
        for j, key in enumerate(type_names):
            mean = np.nanmean(means_elite[key][i])
            std = np.nanstd(means_elite[key][i])
            ax.bar(i * 2 + 0.4 + 0.8/len(type_names)*j, mean, bottom=total_elite, yerr=std, color=colors[j], width=0.8/len(type_names))
            total_elite += mean

        if other:
            ax.bar(i * 2 - 0.4 + 0.8/len(type_names)*(j+1), 100-total_subelite, bottom=total_subelite, color='k', width=0.8/len(type_names), label=('Other' if i == 0 else None), alpha=0.1)
            ax.bar(i * 2 + 0.4 + 0.8/len(type_names)*(j+1), 100-total_elite, bottom=total_elite, color='k', width=0.8/len(type_names), alpha=0.1)
    return ax


def bar_plots_error_bar_translated(df, type_names, subelite_names, elite_names, pourcentage=True, other=False):

    if pourcentage:
        factor = 100
    else:
        factor = 1

    len_types = len(type_names)
    cmap = plt.get_cmap('plasma')
    colors = []
    for i in range(len_types):
        colors.append(cmap(i / len_types))

    means_subelite = {key: [[] for _ in range(4)] for key in type_names}
    means_elite = {key: [[] for _ in range(4)] for key in type_names}
    for i in range(len(move_list)):
        for j in range(len(subelite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == subelite_names[j], df['Acrobatics'] == move_list[i]))
            for key in type_names:
                means_subelite[key][i].append(np.nanmean(list(df[key][index_this_time[0]] * factor)))
        for j in range(len(elite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == elite_names[j], df['Acrobatics'] == move_list[i]))
            for key in type_names:
                means_elite[key][i].append(np.nanmean(list(df[key][index_this_time[0]] * factor)))

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.7])
    for i in range(len(move_list)):
        total_subelite = 0
        for j, key in enumerate(type_names):
            mean = np.nanmean(means_subelite[key][i])
            std = np.nanstd(means_subelite[key][i])
            ax.bar(i*2 - 0.4, mean, bottom=total_subelite, color=colors[j], width=0.6, label=(key if i == 0 else None), alpha=0.8)
            ax.plot(np.array([0.1 + i*2 - 0.7 + 0.5/len(type_names)*j, 0.1 + i*2 - 0.7 + 0.5/len(type_names)*j]),
                    np.array([total_subelite + mean - std, total_subelite + mean + std]), color=colors[j])
            total_subelite += mean
        total_elite = 0
        for j, key in enumerate(type_names):
            mean = np.nanmean(means_elite[key][i])
            std = np.nanstd(means_elite[key][i])
            ax.bar(i * 2 + 0.4, mean, bottom=total_elite, color=colors[j], width=0.6, alpha=0.8)
            ax.plot(np.array([0.1 + i*2 + 0.1 + 0.5/len(type_names)*j, 0.1 + i*2 + 0.1 + 0.5/len(type_names)*j]),
                    np.array([total_elite + mean - std, total_elite + mean + std]), color=colors[j])
            total_elite += mean

        if other:
            ax.bar(i * 2 - 0.4, 100-total_subelite, bottom=total_subelite, color='k', width=0.6, label=('Other' if i == 0 else None), alpha=0.1)
            ax.bar(i * 2 + 0.4, 100-total_elite, bottom=total_elite, color='k', width=0.6, alpha=0.1)
    return ax

def bar_plots(df, type_names, subelite_names, elite_names, pourcentage=True, other=False):

    if pourcentage:
        factor = 100
    else:
        factor = 1

    len_types = len(type_names)
    cmap = plt.get_cmap('plasma')
    colors = []
    for i in range(len_types):
        colors.append(cmap(i / len_types))

    means_subelite = {key: [[] for _ in range(4)] for key in type_names}
    means_elite = {key: [[] for _ in range(4)] for key in type_names}
    for i in range(len(move_list)):
        for j in range(len(subelite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == subelite_names[j], df['Acrobatics'] == move_list[i]))
            for key in type_names:
                means_subelite[key][i].append(np.nanmean(list(df[key][index_this_time[0]] * factor)))
        for j in range(len(elite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == elite_names[j], df['Acrobatics'] == move_list[i]))
            for key in type_names:
                means_elite[key][i].append(np.nanmean(list(df[key][index_this_time[0]] * factor)))

    fig = plt.figure()
    ax = fig.add_axes([0.1, 0.1, 0.85, 0.7])
    for i in range(len(move_list)):
        total_subelite = 0
        for j, key in enumerate(type_names):
            mean = np.nanmean(means_subelite[key][i])
            std = np.nanstd(means_subelite[key][i])
            ax.bar(i*2 - 0.4, mean, bottom=total_subelite, yerr=std, color=colors[j], width=0.4, label=(key if i == 0 else None))
            total_subelite += mean
        total_elite = 0
        for j, key in enumerate(type_names):
            mean = np.nanmean(means_elite[key][i])
            std = np.nanstd(means_elite[key][i])
            ax.bar(i * 2 + 0.4, mean, bottom=total_elite, yerr=std, color=colors[j], width=0.4)
            total_elite += mean

        if other:
            ax.bar(i * 2 - 0.4, 100-total_subelite, bottom=total_subelite, color='k', width=0.4, label=('Other' if i == 0 else None), alpha=0.1)
            ax.bar(i * 2 + 0.4, 100-total_elite, bottom=total_elite, color='k', width=0.4, alpha=0.1)
    return ax

def movement_pourcentage_plots(df, move_list, subelite_names, elite_names, plot_path):

    type_names = df.columns[3:]

    ax = bar_plots_error_bar_translated(df, type_names, subelite_names, elite_names, True, True)

    ax.set_ylabel('Proportion of the acrobatics [%]')
    ax.set_xticks(ticks=[0, 2, 4, 6])
    ax.set_xticklabels([i + '/' for i in move_list], fontweight='bold')
    ax.tick_params(axis='x', pad=15)
    # plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.55))
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    for i in range(4):
        ax.text(-0.3 + i*2 - 0.5, -7, 'Subelite')
        ax.text(-0.3 + i*2 + 0.5, -7, 'Elite')
    # plt.show()
    plt.savefig(f"{plot_path}/movement_pourcentage.png", dpi=300)

    return

def movement_blocks_number_plot(table, move_list, subelite_names, elite_names, plot_path):
    def find_clusters(array):
        array = array.astype(int)
        diff = array[1:] - array[:-1]
        clusters = np.where(diff == 1)[0]
        num_clusters = len(clusters)
        if array[0] == 1:
            num_clusters += 1
        return num_clusters
    def find_movement_occurence(table):

        table_occurrence = [table[0]]
        for i in range(1, len(table)):
            table_occurrence += [[table[i][0], table[i][1], table[i][2]]]
            for j in range(3, len(table[i])):
                num_clusters = find_clusters(table[i][j])
                table_occurrence[i] += [num_clusters]
        df_occurence = pd.DataFrame(table_occurrence[1:], columns=table_occurrence[0])
        return df_occurence

    type_names = table[0][3:]
    df_occurence = find_movement_occurence(table)

    ax = bar_plots_error_bar_translated(df_occurence, type_names, subelite_names, elite_names, False, False)

    ax.set_ylabel('Number of movement occurrence')
    ax.set_xticks(ticks=[0, 2, 4, 6])
    ax.set_xticklabels([i + '/' for i in move_list], fontweight='bold')
    ax.tick_params(axis='x', pad=15)
    ax.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    for i in range(4):
        ax.text(-0.3 + i*2 - 0.5, -7, 'Subelite')
        ax.text(-0.3 + i*2 + 0.5, -7, 'Elite')
    plt.savefig(f"{plot_path}/movement_blocks.png", dpi=300)
    # plt.show()

    return

def AOI_pourcentage_plots(df, move_list, subelite_names, elite_names, plot_path):

    type_names = df.columns[3:]

    ax = bar_plots_error_bar_translated(df, type_names, subelite_names, elite_names, True, False)

    ax.set_ylabel('Dwell time [%]')
    ax.set_xticks(ticks=[0, 2, 4, 6])
    ax.set_xticklabels([i + '/' for i in move_list], fontweight='bold')
    ax.tick_params(axis='x', pad=15)
    # plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(0.5, 1.55))
    ax.legend(ncol=3, loc='upper center', bbox_to_anchor=(0.5, 1.25))
    for i in range(4):
        ax.text(-0.3 + i*2 - 0.5, -7, 'Subelite')
        ax.text(-0.3 + i*2 + 0.5, -7, 'Elite')

    # plt.show()
    plt.savefig(f"{plot_path}/AOI_pourcentage.png", dpi=300)

    return

def heatmap_spreading_plots(df, move_list, subelite_names, elite_names, plot_path):

    max_spreading = 0
    for i in range(len(df["Distance from the center of each point of the heatmap"])):
        if np.nanmax(df["Distance from the center of each point of the heatmap"][i]) > max_spreading:
            max_spreading = np.nanmax(df["Distance from the center of each point of the heatmap"][i])

    means_subelite_90 = []
    means_elite_90 = []
    histogram_elite_distance = []
    histogram_subelite_distance = []
    percentile_subelite = []
    percentile_computed_from_all_available_trials_ponderated_subelites = []
    percentile_elite = []
    percentile_computed_from_all_available_trials_ponderated_elites = []
    for i in range(len(move_list)):
        subelite_list_distance = np.zeros((256,))
        num_athletes_subelite = len(subelite_names)
        for j in range(len(subelite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == subelite_names[j], df['Acrobatics'] == move_list[i]))
            means_subelite_90.append(np.nanmean(list(df["Heat map 90th percentile"][index_this_time[0]])))
            distance_this_athlete = np.zeros((256,))
            for k in index_this_time[0]:
                # plt.figure()
                # plt.bar(np.linspace(0, max_spreading, 256),
                #         np.histogram(df["Distance from the center of each point of the heatmap"][k],
                #                      bins=np.linspace(0, max_spreading, 257), density=True)[0] / len(
                #             index_this_time[0]), color='blue')
                # plt.show()
                distance_this_athlete += np.histogram(df["Distance from the center of each point of the heatmap"][k], bins=np.linspace(0, max_spreading, 257), density=True)[0] / len(index_this_time[0])
            if len(index_this_time[0]) == 0:
                percentile_computed_from_all_available_trials_ponderated_subelites += [np.nan]
                num_athletes_subelite -= 1
            else:
                subelite_list_distance += distance_this_athlete
                cumulative_sum_of_histogram_pourcentage = np.cumsum(distance_this_athlete) / np.sum(distance_this_athlete) * 100
                percentile_computed_from_all_available_trials_ponderated_subelites += [np.where(cumulative_sum_of_histogram_pourcentage < 90)[0][-1]]
            # plt.figure()
            # plt.bar(np.linspace(0, max_spreading, 256), distance_this_athlete, color='red')
            # plt.show()

        elite_list_distance = np.zeros((256,))
        num_athletes_elite = len(elite_names)
        for j in range(len(elite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == elite_names[j], df['Acrobatics'] == move_list[i]))
            means_elite_90.append(np.nanmean(list(df["Heat map 90th percentile"][index_this_time[0]])))
            nan_index = np.where(np.isnan(df["Heat map 90th percentile"][index_this_time[0]]))
            index_this_time = [np.delete(index_this_time[0], nan_index)]
            distance_this_athlete = np.zeros((256,))
            for k in index_this_time[0]:
                distance_this_athlete += np.histogram(df["Distance from the center of each point of the heatmap"][k], bins=np.linspace(0, max_spreading, 257), density=True)[0] / len(index_this_time[0])
            if len(index_this_time[0]) == 0:
                percentile_computed_from_all_available_trials_ponderated_elites += [np.nan]
                num_athletes_elite -= 1
            else:
                elite_list_distance += distance_this_athlete
                cumulative_sum_of_histogram_pourcentage = np.cumsum(distance_this_athlete) / np.sum(distance_this_athlete) * 100
                percentile_computed_from_all_available_trials_ponderated_elites += [np.where(cumulative_sum_of_histogram_pourcentage < 90)[0][-1]]

        subelite_list_distance /= num_athletes_subelite
        percentile_subelite += [np.nanmean(np.array(percentile_computed_from_all_available_trials_ponderated_subelites))]

        percentile_elite += [np.nanmean(np.array(percentile_computed_from_all_available_trials_ponderated_elites))]
        elite_list_distance /= len(elite_names)

        histogram_subelite_distance += [subelite_list_distance]
        histogram_elite_distance += [elite_list_distance]

    cmap = cm.get_cmap('plasma')
    actual_cmap = cmap.set_bad(color='white')

    fig, axs = plt.subplots(2, 4)
    for i in range(4):
        vect_subelite = np.zeros((256, 20))
        vect_elite = np.zeros((256, 20))
        imgsize = (256, 20)
        for x in range(imgsize[0]):
            vect_subelite[255-x, :] = histogram_subelite_distance[i][x]
            vect_elite[255-x, :] = histogram_elite_distance[i][x]

        index_percentile_subelite = int(255-percentile_subelite[i]/max_spreading*256)
        vect_subelite[index_percentile_subelite, :] = np.nan
        index_percentile_elite = int(255-percentile_elite[i]/max_spreading*256)
        vect_elite[index_percentile_elite, :] = np.nan

        axs[0, i].imshow(vect_subelite, cmap=actual_cmap)
        axs[1, i].imshow(vect_elite, cmap=actual_cmap)
        axs[1, i].set_xlabel(move_list[i])
        if i == 3:
            ticks = ticker.FuncFormatter(lambda x, pos: '{0:3.3g}'.format(max_spreading - ((x * max_spreading) / 256)))
            axs[0, i].yaxis.set_major_formatter(ticks)
            axs[0, i].yaxis.tick_right()
            axs[0, i].yaxis.set_label_position("right")
            axs[0, i].axes.get_xaxis().set_visible(False)
            axs[0, i].set_ylabel('Distance from the\ncenter of the heatmap\n[cm]')
            axs[1, i].yaxis.set_major_formatter(ticks)
            axs[1, i].yaxis.tick_right()
            axs[1, i].yaxis.set_label_position("right")
            axs[1, i].axes.get_xaxis().set_visible(False)
            axs[1, i].set_ylabel('Distance from the\ncenter of the heatmap\n[cm]')
        else:
            axs[0, i].axes.get_xaxis().set_visible(False)
            axs[1, i].axes.get_xaxis().set_visible(False)
            axs[0, i].axes.get_yaxis().set_visible(False)
            axs[1, i].axes.get_yaxis().set_visible(False)

        for axis in ['top', 'bottom', 'left', 'right']:
            axs[0, i].spines[axis].set_color('white')
            axs[1, i].spines[axis].set_color('white')

        plt.text(-287 + 93*i, 280, str(move_list[i]) + '/')
    plt.text(-320, -145, 'Subelite', rotation=90)
    plt.text(-320, 150, 'Elite', rotation=90)
    plt.subplots_adjust(wspace=-0.8)
    plt.savefig(f"{plot_path}/heatmap_spreading.png", dpi=300)
    # plt.show()

    return

def heatmap_percetiel_plot(df, move_list, elite_names, subelite_names, plot_path):
    plot_primary_metrics(df, move_list, subelite_names, elite_names, 'Heat map 90th percentile', 'cm', 'percetile_heatmaps', f'{plot_path}/')
    return

def timing_plots(df, move_list, subelite_names, elite_names, plot_path):

    move_type_list = ["anticipatory_index", "compensatory_index", "spotting_index", "movement_detection_index", "blinks_index", "fixation_index"]
    move_type_labels = ["Anticipatory movement", "Compensatorymovement", "Spotting", "Movement detection", "Blink", "Fixation"]
    colors = [cm.get_cmap('plasma')(k) for k in [0., 0.35, 0.45, 0.65, 0.75, 0.95]]

    fig, axs = plt.subplots(2, 4)
    fig.set_figheight(15)
    for i in range(4):
        for j in range(len(subelite_names)):
            index_this_time = np.where(np.logical_and(df['Name'] == subelite_names[j], df['Acrobatics'] == move_list[i]))
            for m, idx_this_move in enumerate(index_this_time[0]):
                total_time_move = len(df["anticipatory_index"][idx_this_move])
                dt = 100 / total_time_move
                for k, key in enumerate(move_type_list):
                    int_index = np.where(np.array(df[key][idx_this_move], dtype=np.int64))[0]
                    consecutive = np.split(int_index, np.where(np.diff(int_index) != 1)[0]+1)
                    for l, cons_index in enumerate(consecutive):
                        axs[0, i].plot(dt * cons_index, np.ones((len(df[key][idx_this_move][cons_index]), )) *
                                       (10*j + 1*k + 0.05*m), color=colors[k])

            for j in range(len(elite_names)):
                index_this_time = np.where(
                    np.logical_and(df['Name'] == elite_names[j], df['Acrobatics'] == move_list[i]))
                for m, idx_this_move in enumerate(index_this_time[0]):
                    total_time_move = len(df["anticipatory_index"][idx_this_move])
                    dt = 100 / total_time_move
                    for k, key in enumerate(move_type_list):
                        int_index = np.where(np.array(df[key][idx_this_move], dtype=np.int64))[0]
                        consecutive = np.split(int_index, np.where(np.diff(int_index) != 1)[0] + 1)
                        for l, cons_index in enumerate(consecutive):
                            axs[1, i].plot(dt * cons_index, np.ones((len(df[key][idx_this_move][cons_index]),)) *
                                           (10*j + 1*k + 0.05*m), color=colors[k])

        axs[0, i].set_ylim(-5, 10*len(subelite_names) + 1*len(move_type_list) + 0.05*20)
        axs[1, i].set_ylim(-5, 10*len(elite_names) + 1*len(move_type_list) + 0.05*20)
        axs[1, i].set_xlabel(move_list[i] + '/', fontsize=12)
        axs[0, i].axes.get_yaxis().set_visible(False)
        axs[1, i].axes.get_yaxis().set_visible(False)
        for axis in ['top', 'left', 'right']:
            axs[0, i].spines[axis].set_color('white')
            axs[1, i].spines[axis].set_color('white')

    for k, key in enumerate(move_type_list):
        plt.plot(0, 0, color=colors[k], label=move_type_labels[k])
    plt.legend(ncol=2, loc='upper center', bbox_to_anchor=(-1.30, 2.35), fontsize=12)
    axs[1, 3].text(-235, -20, 'Normalized time [%]', fontsize=12)
    axs[1, 3].text(-425, 153, 'Subelite', fontsize=12, rotation=90)
    axs[1, 3].text(-425, 50, 'Elite', fontsize=12, rotation=90)
    plt.subplots_adjust(hspace=0.1, top=0.9, bottom=0.1)
    plt.savefig(f"{plot_path}/timing_movements.png", dpi=300)
    # plt.show()

    return


### ------------------------ Code beginig ------------------------ ###

GENRATE_DATA_FRAME_FLAG = True # False
name_results = None # "20ms_threshold"  #

if os.path.exists("/home/user"):
    home_path = "/home/user"
elif os.path.exists("/home/fbailly"):
    home_path = "/home/fbailly"
elif os.path.exists("/home/charbie"):
    home_path = "/home/charbie"


if name_results:
    results_path = f"{home_path}/disk/Eye-tracking/Results_{name_results}"
    plot_path = home_path + f"/disk/Eye-tracking/plots_{name_results}"
else:
    results_path = f"{home_path}/disk/Eye-tracking/Results"
    plot_path = home_path + f"/disk/Eye-tracking/plots"

csv_name = home_path + "/disk/Eye-tracking/Trials_name_mapping.csv"
trial_table = np.char.split(pd.read_csv(csv_name, sep="\t").values.astype("str"), sep=",")

primary_table = [["Name", "Expertise", "Acrobatics",
                  "Fixations duration absolute", "Fixations duration relative", "Number of fixations",
                  "Quiet eye duration absolute", "Quiet eye duration relative", "Eye amplitude", "Neck amplitude",
                  "Maximum eye amplitude", "Maximum neck amplitude",
                  # "Eye amplitude 90th percentile", "Neck amplitude 90th percentile"
                  ]]

trajectories_table = [["Name", "Expertise", "Acrobatics", "Projected gaze orientation (PGO)",
                       "Projected gaze orientation facing front wall (PGOS)",
                       "Wall index", "Wall index facing front wall", "Twist side", "Fixations index"]]

AOI_proportions_table = [["Name", "Expertise", "Acrobatics", "Trampoline bed", "Trampoline",
                    "Wall front", "Wall back",
                     "Ceiling", "Wall sides",
                    "Athlete himself", "Blink"]]  # All in proportions (was removed from title for the plots legend)

neck_eye_movements_table = [["Name", "Expertise", "Acrobatics", "Anticipatory movements",
                       "Compensatory movements", "Spotting movements",
                       "Movement detection", "Blinks"]] # all in pourcentage (was removed from title for the plots legend)

neck_eye_movements_indices_table = [["Name", "Expertise", "Acrobatics", "Anticipatory movements index",
                       "Compensatory movements index", "Spotting movements index",
                       "Movement detection index", "Blinks index", 'Fixations index']]

heatmaps_spreading_table = [["Name", "Expertise", "Acrobatics", "Distance from the center of each point of the heatmap", "Heat map 90th percentile"]]

qualitative_table = [["Name", "Expertise", "Acrobatics", "Fixation target", "anticipatory_index", "compensatory_index",
                "spotting_index", "movement_detection_index", "blinks_index", "fixation_index"]]

if GENRATE_DATA_FRAME_FLAG:
    for folder_subject in os.listdir(results_path):
        for folder_move in os.listdir(results_path + '/' + folder_subject):
            if folder_move in ['4-', '41', '42', '43']:
                for file in os.listdir(results_path + '/' + folder_subject + '/' + folder_move):
                    if len(file) > 23:
                        if file[-23:] == "eyetracking_metrics.pkl":

                            path = results_path + '/' + folder_subject + '/' + folder_move + '/'
                            move_number, eye_tracking_metrics = load_eye_tracking_metrics(path, file)

                            # Primary analysis
                            expertise = eye_tracking_metrics["subject_expertise"]
                            subject_name = eye_tracking_metrics["subject_name"]

                            for i_trial in range(len(trial_table)):
                                if trial_table[i_trial][0][0] == subject_name:
                                    twist_side = trial_table[i_trial][0][25]

                            acrobatics = folder_move
                            fixation_duration_absolute = np.mean(eye_tracking_metrics["fixation_duration_absolute"])
                            fixation_duration_relative = np.mean(eye_tracking_metrics["fixation_duration_relative"])
                            number_of_fixation = eye_tracking_metrics["number_of_fixation"]
                            quiet_eye_duration_absolute = eye_tracking_metrics["quiet_eye_duration_absolute"]
                            quiet_eye_duration_relative = eye_tracking_metrics["quiet_eye_duration_relative"]
                            eye_amplitude = eye_tracking_metrics["eye_amplitude"]
                            neck_amplitude = eye_tracking_metrics["neck_amplitude"]
                            max_eye_amplitude = eye_tracking_metrics["max_eye_amplitude"]
                            max_neck_amplitude = eye_tracking_metrics["max_neck_amplitude"]
                            # eye_amplitude_percentile = eye_tracking_metrics["eye_amplitude_percentile"]
                            # neck_amplitude_percentile = eye_tracking_metrics["neck_amplitude_percentile"]

                            primary_table += [[subject_name, expertise, acrobatics,
                                               fixation_duration_absolute, fixation_duration_relative,
                                               number_of_fixation, quiet_eye_duration_absolute,
                                               quiet_eye_duration_relative, eye_amplitude, neck_amplitude,
                                               max_eye_amplitude, max_neck_amplitude]]


                            # Secondary analysis - Trajectory
                            gaze_position_temporal_evolution_projected = eye_tracking_metrics["gaze_position_temporal_evolution_projected"]
                            gaze_position_temporal_evolution_projected_facing_front_wall = eye_tracking_metrics["gaze_position_temporal_evolution_projected_facing_front_wall"]
                            wall_index = eye_tracking_metrics["wall_index"]
                            wall_index_facing_front_wall = eye_tracking_metrics["wall_index_facing_front_wall"]
                            fixation_index = eye_tracking_metrics["fixation_index"]
                            trajectories_table += [[subject_name, expertise, acrobatics,
                                                    gaze_position_temporal_evolution_projected,
                                                    gaze_position_temporal_evolution_projected_facing_front_wall,
                                                    wall_index,
                                                    wall_index_facing_front_wall,
                                                    twist_side,
                                                    fixation_index]]

                            # Secondary analysis - Movements
                            pourcentage_anticipatory = eye_tracking_metrics["pourcentage_anticipatory"]
                            pourcentage_compensatory = eye_tracking_metrics["pourcentage_compensatory"]
                            pourcentage_spotting = eye_tracking_metrics["pourcentage_spotting"]
                            pourcentage_movement_detection = eye_tracking_metrics["pourcentage_movement_detection"]
                            pourcentage_blinks = eye_tracking_metrics["pourcentage_blinks"]
                            neck_eye_movements_table += [[subject_name, expertise, acrobatics,
                                              pourcentage_anticipatory, pourcentage_compensatory, pourcentage_spotting,
                                              pourcentage_movement_detection, pourcentage_blinks]]

                            anticipatory_index = eye_tracking_metrics["anticipatory_index"]
                            compensatory_index = eye_tracking_metrics["compensatory_index"]
                            spotting_index = eye_tracking_metrics["spotting_index"]
                            movement_detection_index = eye_tracking_metrics["movement_detection_index"]
                            blinks_index = eye_tracking_metrics["blinks_index"]
                            neck_eye_movements_indices_table += [[subject_name, expertise, acrobatics,
                                              anticipatory_index, compensatory_index, spotting_index,
                                              movement_detection_index, blinks_index, fixation_index]]

                            # Secondary analysis - AOI proportions
                            trampoline_bed_proportions = eye_tracking_metrics["trampoline_bed_proportions"]
                            trampoline_proportions = eye_tracking_metrics["trampoline_proportions"]
                            wall_front_proportions = eye_tracking_metrics["wall_front_proportions"]
                            wall_back_proportions = eye_tracking_metrics["wall_back_proportions"]
                            ceiling_proportions = eye_tracking_metrics["ceiling_proportions"]
                            side_proportions = eye_tracking_metrics["side_proportions"]
                            self_proportions = eye_tracking_metrics["self_proportions"]
                            blink_proportions = eye_tracking_metrics["blink_proportions"]
                            AOI_proportions_table += [[subject_name, expertise, acrobatics, trampoline_bed_proportions,
                                                       trampoline_proportions, wall_front_proportions,
                                                       wall_back_proportions, ceiling_proportions, side_proportions,
                                                       self_proportions, blink_proportions]]

                            # Secondary analysis - percetile heatmaps
                            percetile_heatmaps = eye_tracking_metrics["percetile_heatmaps"]
                            distance_heatmaps = eye_tracking_metrics["distance_heatmaps"]
                            heatmaps_spreading_table += [[subject_name, expertise, acrobatics, distance_heatmaps, percetile_heatmaps]]

                            # Qualitative analysis
                            fixation_positions = eye_tracking_metrics["fixation_positions"]
                            anticipatory_index = eye_tracking_metrics["anticipatory_index"]
                            compensatory_index = eye_tracking_metrics["compensatory_index"]
                            spotting_index = eye_tracking_metrics["spotting_index"]
                            movement_detection_index = eye_tracking_metrics["movement_detection_index"]
                            blinks_index = eye_tracking_metrics["blinks_index"]

                            qualitative_table += [[subject_name, expertise, acrobatics, fixation_positions,
                                             anticipatory_index, compensatory_index, spotting_index,
                                             movement_detection_index, blinks_index, fixation_index]]


    savemat(f'{plot_path}/primary_table.mat', {'primary_table': primary_table})
    savemat(f'{plot_path}/trajectories_table.mat', {'trajectories_table': trajectories_table})
    savemat(f'{plot_path}/AOI_proportions_table.mat', {'AOI_proportions_table': AOI_proportions_table})
    savemat(f'{plot_path}/neck_eye_movements_table.mat', {'neck_eye_movements_table': neck_eye_movements_table})
    savemat(f'{plot_path}/neck_eye_movements_indices_table.mat', {'neck_eye_movements_indices_table': neck_eye_movements_indices_table})
    savemat(f'{plot_path}/heatmaps_spreading_table.mat', {'heatmaps_spreading_table': heatmaps_spreading_table})
    savemat(f'{plot_path}/qualitative_table.mat', {'qualitative_table': qualitative_table})

    # save as pickle files
    with open(f'{plot_path}/primary_table.pkl', 'wb') as f:
        pickle.dump(primary_table, f)
    with open(f'{plot_path}/trajectories_table.pkl', 'wb') as f:
        pickle.dump(trajectories_table, f)
    with open(f'{plot_path}/AOI_proportions_table.pkl', 'wb') as f:
        pickle.dump(AOI_proportions_table, f)
    with open(f'{plot_path}/neck_eye_movements_table.pkl', 'wb') as f:
        pickle.dump(neck_eye_movements_table, f)
        with open(f'{plot_path}/neck_eye_movements_indices_table.pkl', 'wb') as f:
            pickle.dump(neck_eye_movements_indices_table, f)
    with open(f'{plot_path}/heatmaps_spreading_table.pkl', 'wb') as f:
        pickle.dump(heatmaps_spreading_table, f)
    with open(f'{plot_path}/qualitative_table.pkl', 'wb') as f:
        pickle.dump(qualitative_table, f)

else:
    primary_table = loadmat(f'{plot_path}/primary_table.mat')['primary_table']
    trajectories_table = loadmat(f'{plot_path}/trajectories_table.mat')['trajectories_table']
    AOI_proportions_table = loadmat(f'{plot_path}/AOI_proportions_table.mat')['AOI_proportions_table']
    neck_eye_movements_table = loadmat(f'{plot_path}/neck_eye_movements_table.mat')['neck_eye_movements_table']
    neck_eye_movements_indices_table = loadmat(f'{plot_path}/neck_eye_movements_table.mat')['neck_eye_movements_indices_table']
    heatmaps_spreading_table = loadmat(f'{plot_path}/heatmaps_spreading_table.mat')['heatmaps_spreading_table']
    qualitative_table = loadmat(f'{plot_path}/qualitative_table.mat')['qualitative_table']

    # load the pickle files
    with open(f'{plot_path}/primary_table.pkl', 'rb') as f:
        primary_table = pickle.load(f)
    with open(f'{plot_path}/trajectories_table.pkl', 'rb') as f:
        trajectories_table = pickle.load(f)
    with open(f'{plot_path}/AOI_proportions_table.pkl', 'rb') as f:
        AOI_proportions_table = pickle.load(f)
    with open(f'{plot_path}/neck_eye_movements_table.pkl', 'rb') as f:
        neck_eye_movements_table = pickle.load(f)
    with open(f'{plot_path}/heatmaps_spreading_table.pkl', 'rb') as f:
        heatmaps_spreading_table = pickle.load(f)
    with open(f'{plot_path}/qualitative_table.pkl', 'rb') as f:
        qualitative_table = pickle.load(f)

move_list = ['4-', '41', '42', '43']

primary_data_frame = pd.DataFrame(primary_table[1:], columns=primary_table[0])

subelite_names = []
elite_names = []
for i in range(len(primary_data_frame)):
    if primary_data_frame['Expertise'][i] == 'SubElite':
        if primary_data_frame['Name'][i] not in subelite_names:
            subelite_names.append(primary_data_frame['Name'][i])
    if primary_data_frame['Expertise'][i] == 'Elite':
        if primary_data_frame['Name'][i] not in elite_names:
            elite_names.append(primary_data_frame['Name'][i])

# primary_plots(primary_data_frame, move_list, subelite_names, elite_names, plot_path)
#
# trajectories_data_frame = pd.DataFrame(trajectories_table[1:], columns=trajectories_table[0])
# trajectory_plots(trajectories_data_frame, move_list, subelite_names, elite_names)

movement_pourcentage_data_frame = pd.DataFrame(neck_eye_movements_table[1:], columns=neck_eye_movements_table[0])
movement_pourcentage_plots(movement_pourcentage_data_frame, move_list, subelite_names, elite_names, plot_path)
movement_blocks_number_plot(neck_eye_movements_indices_table, move_list, subelite_names, elite_names, plot_path)

AOI_table_tempo = [['Name', 'Expertise', 'Acrobatics', 'Trampoline bed', 'Trampoline', 'Wall back front', 'Ceiling',
                    'Wall sides', 'Athlete himself', 'Blink']]
for i in range(1, len(AOI_proportions_table)):
    AOI_table_tempo += [[AOI_proportions_table[i][0], AOI_proportions_table[i][1], AOI_proportions_table[i][2],
                            AOI_proportions_table[i][3], AOI_proportions_table[i][4], AOI_proportions_table[i][5] + AOI_proportions_table[i][6],
                            AOI_proportions_table[i][7], AOI_proportions_table[i][8], AOI_proportions_table[i][9],
                            AOI_proportions_table[i][10]]]
AOI_pourcentage_data_frame_tempo = pd.DataFrame(AOI_table_tempo[1:], columns=AOI_table_tempo[0])
AOI_pourcentage_plots(AOI_pourcentage_data_frame_tempo, move_list, subelite_names, elite_names, plot_path)
#
# heatmap_spreading_data_frame = pd.DataFrame(heatmaps_spreading_table[1:], columns=heatmaps_spreading_table[0])
# heatmap_spreading_plots(heatmap_spreading_data_frame, move_list, subelite_names, elite_names, plot_path)
#
# qualitative_data_frame = pd.DataFrame(qualitative_table[1:], columns=qualitative_table[0])
# timing_plots(qualitative_data_frame, move_list, subelite_names, elite_names, plot_path)























