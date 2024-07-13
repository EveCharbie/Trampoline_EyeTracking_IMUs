
import numpy as np
import pingouin as pg
import pickle
import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.patches import Rectangle
import spm1d
from scipy.interpolate import interp1d
from scipy.stats import multivariate_normal
import csv
from IPython import embed


##########################################################################################
# run --- python stats_tests.py > stats_output.txt ---  to save the output to a file
##########################################################################################


PRIMARY_ANALYSIS_FLAG = True
TRAJECTORIES_ANALYSIS_FLAG = True
TRAJECTORIES_HEATMAPS_FLAG = True
GENERATE_EACH_ATHLETE_PGOS_GRAPH = True
AOI_ANALYSIS_FLAG = True
NECK_EYE_ANALYSIS_FLAG = True
SPREADING_HEATMAP_FLAG = True
QUALITATIVE_ANALYSIS_FLAG = True

move_list = ['4-', '41', '42', '43']

pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
pd.set_option('display.width', 150)

if os.path.exists("/home/user"):
    home_path = "/home/user"
elif os.path.exists("/home/fbailly"):
    home_path = "/home/fbailly"
elif os.path.exists("/home/charbie"):
    home_path = "/home/charbie"

results_path = f"{home_path}/disk/Eye-tracking/Results"

with open(home_path + '/disk/Eye-tracking/plots/primary_table.pkl', 'rb') as f:
    primary_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/trajectories_table.pkl', 'rb') as f:
    trajectories_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/AOI_proportions_table.pkl', 'rb') as f:
    AOI_proportions_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/neck_eye_movements_table.pkl', 'rb') as f:
    neck_eye_movements_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/heatmaps_spreading_table.pkl', 'rb') as f:
    heatmaps_spreading_table = pickle.load(f)
with open(home_path + '/disk/Eye-tracking/plots/qualitative_table.pkl', 'rb') as f:
    qualitative_table = pickle.load(f)


subelite_names = []
elite_names = []
index_subelites  = []
index_elites = []
for i in range(len(primary_table)):
    if primary_table[i][1] == 'SubElite':
        if primary_table[i][0] not in subelite_names:
            subelite_names.append(primary_table[i][0])
        index_subelites.append(i)
    elif primary_table[i][1] == 'Elite':
        if primary_table[i][0] not in elite_names:
            elite_names.append(primary_table[i][0])
        index_elites.append(i)

trial_per_athlete_per_move_index = {}
for i in range(1, len(trajectories_table)):
    if trajectories_table[i][0] not in trial_per_athlete_per_move_index.keys():
        basic_dict = {move: [] for move in move_list}
        trial_per_athlete_per_move_index[trajectories_table[i][0]] = {move: [] for move in move_list}
        if trajectories_table[i][2] in move_list:
            trial_per_athlete_per_move_index[trajectories_table[i][0]][trajectories_table[i][2]] += [i]
    else:
        if trajectories_table[i][2] in move_list:
            trial_per_athlete_per_move_index[trajectories_table[i][0]][trajectories_table[i][2]] += [i]


with open(home_path + '/disk/Eye-tracking/plots/nb_trial_per_athlete.csv', 'w') as f:
    writer = csv.writer(f)
    head_title = ['Athlete'] + [move + '/' for move in move_list]
    writer.writerow(head_title)
    for i, name in  enumerate(trial_per_athlete_per_move_index):
        if name in subelite_names:
            line_list = [name] + [trial_per_athlete_per_move_index[name][move] for move in move_list]
            writer.writerow(line_list)
    writer.writerow(['-', '-', '-', '-', '-'])
    for i, name in enumerate(trial_per_athlete_per_move_index):
        if name in elite_names:
            line_list = [name] + [trial_per_athlete_per_move_index[name][move] for move in move_list]
            writer.writerow(line_list)
    f.close()

subelites_nb_athlete_per_move = {move: 0 for move in move_list}
elites_nb_athlete_per_move = {move: 0 for move in move_list}
for name in subelite_names:
    for move in subelites_nb_athlete_per_move.keys():
        if len(trial_per_athlete_per_move_index[name][move]) > 0:
            subelites_nb_athlete_per_move[move] += 1
for name in elite_names:
    for move in elites_nb_athlete_per_move.keys():
        if len(trial_per_athlete_per_move_index[name][move]) > 0:
            elites_nb_athlete_per_move[move] += 1


# Give the mean value to the athelte who did not want to do one move in particular for ANOVAs
primary_table_array = np.array(primary_table)
AOI_proportions_table_array = np.array(AOI_proportions_table)
neck_eye_movements_table_array = np.array(neck_eye_movements_table)
heatmaps_spreading_table_array = np.array(heatmaps_spreading_table)
for i, name in enumerate(trial_per_athlete_per_move_index):
    for j, move in enumerate(trial_per_athlete_per_move_index[name]):
        if len(trial_per_athlete_per_move_index[name][move]) == 0:
            index = (index_elites if name in elite_names else index_subelites)
            primary_table += [[name,
                              'Elite' if name in elite_names else 'SubElite',
                              move,
                              np.nanmean(primary_table_array[index, 3].astype(float)),
                              np.nanmean(primary_table_array[index, 4].astype(float)),
                              np.nanmean(primary_table_array[index, 5].astype(float)),
                              np.nanmean(primary_table_array[index, 6].astype(float)),
                              np.nanmean(primary_table_array[index, 7].astype(float)),
                              np.nanmean(primary_table_array[index, 8].astype(float)),
                              np.nanmean(primary_table_array[index, 9].astype(float)),
                              np.nanmean(primary_table_array[index, 10].astype(float)),
                              ]]
            AOI_proportions_table += [[name,
                                  'Elite' if name in elite_names else 'SubElite',
                                  move,
                                  np.nanmean(AOI_proportions_table_array[index, 3].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 4].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 5].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 6].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 7].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 8].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 9].astype(float)),
                                  np.nanmean(AOI_proportions_table_array[index, 10].astype(float)),
                                  ]]
            neck_eye_movements_table += [[name,
                                  'Elite' if name in elite_names else 'SubElite',
                                  move,
                                  np.nanmean(neck_eye_movements_table_array[index, 3].astype(float)),
                                  np.nanmean(neck_eye_movements_table_array[index, 4].astype(float)),
                                  np.nanmean(neck_eye_movements_table_array[index, 5].astype(float)),
                                  np.nanmean(neck_eye_movements_table_array[index, 6].astype(float)),
                                  np.nanmean(neck_eye_movements_table_array[index, 7].astype(float)),
                                  ]]
            heatmaps_spreading_table += [[name,
                                  'Elite' if name in elite_names else 'SubElite',
                                  move,
                                  None,
                                  np.nanmean(heatmaps_spreading_table_array[index, 4].astype(float)),
                                  np.nanmean(heatmaps_spreading_table_array[index, 5].astype(float)),
                                  np.nanmean(heatmaps_spreading_table_array[index, 6].astype(float)),
                                  ]]

def stats_procedure(name, dataframe):

    # ANOVA
    print(f"Mixed ANOVA for {name}")
    out = pg.mixed_anova(data=dataframe, dv=name, within='Acrobatics', between='Expertise', subject='Name', effsize='np2')
    print(f'{out}\n\n')

    # Post-hocs
    print(f"T-tests for {name} (Holm-Bonferroni corrections will be applied afterwards)")
    out_posthocs = pg.pairwise_tests(data=dataframe, dv=name, within='Acrobatics', between='Expertise', subject='Name',
                            parametric=True)
    print(f'{out_posthocs}\n\n')

    if out['p-unc'][0] < 0.05:
        print(f"Significant effect of Expertise for {name} (p-value = {out['p-unc'][0]}) !!!!!!!!!!!!!!!!!!!!!!!!!!!!!")
    if out['p-unc'][1] < 0.05:
        print(f"Significant effect of Acrobatics for {name} (p-value = {out['p-unc'][1]})")
        H0_reject_acrobatics, pvalues_corrected_acrobatics = pg.multicomp(list(out_posthocs["p-unc"][:6]), alpha=0.05, method='holm')
        for acro_comp in range(6):
            if H0_reject_acrobatics[acro_comp]:
                if out_posthocs["hedges"][acro_comp] > 0:
                    print(f"Green: {out_posthocs['A'][acro_comp]} is significantly higher than {out_posthocs['B'][acro_comp]} for {name} (p-value = {pvalues_corrected_acrobatics[acro_comp]})")
                else:
                    print(f"Red: {out_posthocs['B'][acro_comp]} is significantly smaller than {out_posthocs['A'][acro_comp]} for {name} (p-value = {pvalues_corrected_acrobatics[acro_comp]})")
            else:
                print(f"Gray: No significant difference between {out_posthocs['A'][acro_comp]} and {out_posthocs['B'][acro_comp]} for {name} (p-value = {pvalues_corrected_acrobatics[acro_comp]})")
    if out['p-unc'][2] < 0.05:
        print(f"Significant interaction for {name} (p-value = {out['p-unc'][2]})")
        H0_reject_interactions, pvalues_corrected_interactions = pg.multicomp(list(out_posthocs["p-unc"][7:]), alpha=0.05, method='holm')
        for acro_comp in range(4):
            if H0_reject_interactions[acro_comp]:
                if out_posthocs["hedges"][acro_comp + 7] > 0:
                    print(f"Green: {out_posthocs['Acrobatics'][acro_comp + 7]} is significantly higher for {out_posthocs['A'][acro_comp + 7]} than {out_posthocs['B'][acro_comp + 7]} for {name} (p-value = {pvalues_corrected_acrobatics[acro_comp]})")
                else:
                    print(f"Red: {out_posthocs['Acrobatics'][acro_comp + 7]} is significantly smaller for {out_posthocs['A'][acro_comp + 7]} than {out_posthocs['B'][acro_comp + 7]} for {name} (p-value = {pvalues_corrected_acrobatics[acro_comp]})")
            else:
                print(f"Gray: No significant difference on {out_posthocs['Acrobatics'][acro_comp + 7]} for {name} (p-value = {pvalues_corrected_acrobatics[acro_comp]})")
    return

# ------------------------------------ Primary data frame = Mixed Anova ---------------------------------------- #
if PRIMARY_ANALYSIS_FLAG:
    primary_data_frame = pd.DataFrame(primary_table[1:], columns=primary_table[0])
    primary_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/primary_data_frame.csv")

    with open(home_path + "/disk/Eye-tracking/plots/primary_data_frame.pkl", 'wb') as f:
        pickle.dump(primary_table, f)

    primary_data_frame_temporary = pd.DataFrame(columns=primary_table[0])
    for i, name in enumerate(trial_per_athlete_per_move_index):
        for j, move in enumerate(trial_per_athlete_per_move_index[name]):
            index_this_time = np.where(np.logical_and(primary_data_frame['Name'] == name, primary_data_frame['Acrobatics'] == move))[0]
            df = {'Name': [name],
                  'Expertise': [primary_data_frame['Expertise'][index_this_time[0]]],
                  'Acrobatics': [move],
                  'Fixations duration relative': [np.nanmedian(primary_data_frame['Fixations duration relative'][index_this_time])],
                  'Number of fixations': [np.nanmedian(primary_data_frame['Number of fixations'][index_this_time])],
                  'Quiet eye duration relative': [np.nanmedian(primary_data_frame['Quiet eye duration relative'][index_this_time])],
                  'Quiet eye onset relative': [np.nanmedian(primary_data_frame['Quiet eye onset relative'][index_this_time])],
                  'Eye amplitude': [np.nanmedian(primary_data_frame['Eye amplitude'][index_this_time])],
                  'Neck amplitude': [np.nanmedian(primary_data_frame['Neck amplitude'][index_this_time])],
                  'Maximum eye amplitude': [np.nanmedian(primary_data_frame['Maximum eye amplitude'][index_this_time])],
                  'Maximum neck amplitude': [np.nanmedian(primary_data_frame['Maximum neck amplitude'][index_this_time])]}
            primary_data_frame_temporary = pd.concat([primary_data_frame_temporary, pd.DataFrame(df)])

    primary_data_frame = primary_data_frame_temporary

    """
    Here we do a sensitivity analysis to assess the impact of the normality hypothesis on the results of the stats tests.
    First, a mixed ANOVA is used.
    Then, a T-test post hoc is used to identify the significant differences (Bonferoni correction should be applieds to the p-values).
    Then, a non-parametric post hoc is used (Wilcoxon or Mann-Whiteney) to identify the significant differences (Bonferoni correction should be applieds to the p-values).
    If the results are the same, we report the parametric p-values, otherwise we report the non-parametric p-values.
    """

    stats_procedure('Fixations duration relative', primary_data_frame)
    stats_procedure('Number of fixations', primary_data_frame)
    stats_procedure('Quiet eye duration relative', primary_data_frame)
    stats_procedure('Quiet eye onset relative', primary_data_frame)
    stats_procedure('Eye amplitude', primary_data_frame)
    stats_procedure('Neck amplitude', primary_data_frame)

    duration_data_frame = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics', 'Data type', 'Data measure'])
    for i, name in enumerate(trial_per_athlete_per_move_index):
        for j, move in enumerate(trial_per_athlete_per_move_index[name]):
            index = np.where(np.logical_and(primary_data_frame['Name'] == name, primary_data_frame['Acrobatics'] == move))[0]
            df = {'Name': [name],
                  'Expertise': [primary_data_frame['Expertise']],
                  'Acrobatics': [move],
                  'Data type': ["Fixations duration relative"],
                  'Data measure': [np.array(primary_data_frame['Fixations duration relative'])[index][0]]}
            duration_data_frame = pd.concat([duration_data_frame, pd.DataFrame(df)])

            df = {'Name': [name],
                  'Expertise': [primary_data_frame['Expertise']],
                  'Acrobatics': [move],
                  'Data type': ["Quiet eye duration relative"],
                  'Data measure': [np.array(primary_data_frame['Quiet eye duration relative'])[index][0]]}
            duration_data_frame = pd.concat([duration_data_frame, pd.DataFrame(df)])

    print("Two-way paired measures ANOVA for quiet eye duration vs fixation duration")
    out = pg.rm_anova(data=duration_data_frame, dv='Data measure', within=['Data type', 'Acrobatics'], subject='Name', correction=True, effsize='n2')
    print(f'{out}\n\n')
    print("T-tests for quiet eye duration vs fixation duration")
    out = pg.pairwise_tests(data=duration_data_frame, dv='Data measure', within=['Data type', 'Acrobatics'], subject='Name', parametric=True)
    H0_reject, pvalues_corrected = pg.multicomp(list(out["p-unc"]), alpha=0.05, method='holm')
    print(f'{out}\n\n')
    print(f'H0 rejection: {H0_reject}\n'
          f'p-values: {pvalues_corrected}\n\n')
    print("Wilcoxon and Mann-Whiteney tests for quiet eye duration vs fixation duration")
    out = pg.pairwise_tests(data=duration_data_frame, dv='Data measure', within=['Data type', 'Acrobatics'], subject='Name', parametric=False)
    H0_reject, pvalues_corrected = pg.multicomp(list(out["p-unc"]), alpha=0.05, method='holm')
    print(f'{out}\n\n')
    print(f'H0 rejection: {H0_reject}\n'
          f'p-values: {pvalues_corrected}\n\n')

# -------------------------------- Trajectories data frame = SPM1D ------------------------------------ #

def find_significant_timings(xi_interp, subelites_data, elites_data):

    admissible_timings = np.zeros((len(xi_interp),))
    significant_timings = np.zeros((2,))

    for i in range(len(xi_interp)):
        if np.std(subelites_data[:, i]) > 0 and np.std(elites_data[:, i]) > 0:
            admissible_timings[i] = 1
    begining_of_clusters = np.where(admissible_timings[1:] - admissible_timings[:-1] == 1)[0] +1
    end_of_clusters = np.where(admissible_timings[1:] - admissible_timings[:-1] == -1)[0] +1
    if admissible_timings[0] == 1:
        begining_of_clusters = np.concatenate(([0], begining_of_clusters))
    if admissible_timings[-1] == 1:
        end_of_clusters = np.concatenate((end_of_clusters, [len(xi_interp)]))

    for i in range(len(begining_of_clusters)):
        t = spm1d.stats.ttest2(subelites_data[:, begining_of_clusters[i]:end_of_clusters[i]],
                               elites_data[:, begining_of_clusters[i]:end_of_clusters[i]], equal_var=False)
        ti = t.inference(alpha=0.05, two_tailed=True)
        if ti.h0reject == True:
            if ti.clusters != []:
                ti.plot()
                plt.show()
                clusters = ti.clusters
                for k in range(len(clusters)):
                    cluster_x, cluster_z = clusters[k].get_patch_vertices()
                    significant_timings = np.vstack((significant_timings, np.array([cluster_x[0] + begining_of_clusters[i], cluster_x[1] + begining_of_clusters[i]])))
    if significant_timings.shape == (2,):
        significant_timings = None
    else:
        significant_timings = significant_timings[1:]
        print(f"Significant timings : {significant_timings}")
    return admissible_timings, significant_timings

def plot_gymnasium_unwrapped(axs, j, FLAG_3D=False):

    bound_side = 3 + 121 * 0.0254 / 2

    # Plot trampo bed
    if FLAG_3D:
        X, Y = np.meshgrid([-7 * 0.3048, 7 * 0.3048], [-3.5 * 0.3048, 3.5 * 0.3048])
        Z = np.zeros(X.shape)
        axs[j].plot_surface(X, Y, Z, color="k", alpha=0.4)
    else:
        axs[j].add_patch(Rectangle((-7 * 0.3048, -3.5 * 0.3048), 14 * 0.3048, 7 * 0.3048, facecolor='k', alpha=0.2))
    # Plot vertical lines of the symmetrized gymnasium
    axs[j].plot(np.array([-7.2, -7.2]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([7.2, 7.2]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192), -7.2 - (9.4620-1.2192)]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([7.2 + 9.4620-1.2192, 7.2 + 9.4620-1.2192]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192) - 2*7.2, -7.2 - (9.4620-1.2192) - 2*7.2]), np.array([-bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2, -7.2]), np.array([-bound_side - (9.4620-1.2192), bound_side]), '-k')
    axs[j].plot(np.array([7.2, 7.2]), np.array([-bound_side - (9.4620-1.2192), bound_side]), '-k')
    axs[j].plot(np.array([-7.2, -7.2]), np.array([bound_side, bound_side + 9.4620 - 1.2192]), '-k')
    axs[j].plot(np.array([7.2, 7.2]), np.array([bound_side, bound_side + 9.4620 - 1.2192]), '-k')
    # Plot horizontal lines of the symmetrized gymnasium
    axs[j].plot(np.array([-7.2, 7.2]), np.array([-bound_side, -bound_side]), '-k')
    axs[j].plot(np.array([-7.2, 7.2]), np.array([bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2, 7.2]), np.array([-bound_side - (9.4620-1.2192), -bound_side - (9.4620-1.2192)]), '-k')
    axs[j].plot(np.array([-7.2, 7.2]), np.array([bound_side + 9.4620-1.2192, bound_side + 9.4620-1.2192]), '-k')
    axs[j].plot(np.array([7.2, 7.2 + 9.4620-1.2192]), np.array([-bound_side, -bound_side]), '-k')
    axs[j].plot(np.array([7.2, 7.2 + 9.4620-1.2192]), np.array([bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192), 7.2]), np.array([-bound_side, -bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192), 7.2]), np.array([bound_side, bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192) - 2*7.2, -7.2 - (9.4620-1.2192)]), np.array([-bound_side, -bound_side]), '-k')
    axs[j].plot(np.array([-7.2 - (9.4620-1.2192) - 2*7.2, -7.2 - (9.4620-1.2192)]), np.array([bound_side, bound_side]), '-k')

    axs[j].text(-7.2 - (9.4620-1.2192) - 2*7.2 + 7.2/2 + 1, bound_side + 0.1, "Ceiling", fontsize=10)
    axs[j].text(-7.2 - (9.4620-1.2192) + 1, bound_side + 0.1, "Wall back", fontsize=10)
    axs[j].text(7.2 + 1, bound_side + 0.1, "Wall front", fontsize=10)
    axs[j].text(-7.2 + 7.2/2 + 1, bound_side + 9.4620-1.2192 + 0.1, "Wall left", fontsize=10)
    axs[j].text(-7.2 + 7.2/2 + 0.5, -bound_side - (9.4620-1.2192) - 1, "Wall right", fontsize=10)

    return

def plot_trajectories_data_frame(
        trajectory_curves_per_athelte_per_move,
        significant_timings,
        move,
        title_variable,
        output_filename,
):

    fig, axs = plt.subplots(1, 2, figsize=(18, 6))
    i_subelite = 0
    i_elite = 0
    for i, name in enumerate(trajectory_curves_per_athelte_per_move.keys()):
        if name in subelite_names:
            if trajectory_curves_per_athelte_per_move[name][move] != []:
                axs[0].plot(trajectory_curves_per_athelte_per_move[name][move][0, :], trajectory_curves_per_athelte_per_move[name][move][1, :],
                            color=colors_subelites[i_subelite],
                            marker='.',
                            linestyle='None',
                            markersize=1,
                            label=name)
                if significant_timings[move] is not None:
                    for index in significant_timings[move].shape:
                        index_this_time = np.arange(int(round(significant_timings[move][index, 0])),
                                                    int(round(significant_timings[move][index, 1])))
                        axs[0].plot(trajectory_curves_per_athelte_per_move[name][move][0, significant_timings[move][index_this_time]],
                                    trajectory_curves_per_athelte_per_move[name][move][1, significant_timings[move][index_this_time]],
                                    color=colors_subelites[i_subelite],
                                    linewidth=5)
            i_subelite += 1
    for i, name in enumerate(trajectory_curves_per_athelte_per_move.keys()):
        if name in elite_names:
            if not isinstance(trajectory_curves_per_athelte_per_move[name][move], list):
                axs[1].plot(trajectory_curves_per_athelte_per_move[name][move][0, :], trajectory_curves_per_athelte_per_move[name][move][1, :],
                            color=colors_elites[i_elite],
                            marker='.',
                            linestyle='None',
                            markersize=1,
                            label=name)
                if significant_timings[move] is not None:
                    for index in significant_timings[move].shape:
                        index_this_time = np.arange(int(round(significant_timings[move][index, 0])),
                                                    int(round(significant_timings[move][index, 1])))
                        axs[1].plot(trajectory_curves_per_athelte_per_move[name][move][0, significant_timings[move][index_this_time]],
                                    trajectory_curves_per_athelte_per_move[name][move][1, significant_timings[move][index_this_time]],
                                    color=colors_elites[i_elite],
                                    linewidth=3)
            i_elite += 1

    for j in range(2):
        plot_gymnasium_unwrapped(axs, j)
        axs[j].axis('equal')

    plt.subplots_adjust(right=0.8)
    plt.suptitle(title_variable + move)
    axs[0].legend(bbox_to_anchor=(2.3, 1), loc='upper left', borderaxespad=0.)
    axs[1].legend(bbox_to_anchor=(1.1, 0.5), loc='upper left', borderaxespad=0.)
    plt.savefig(output_filename, dpi=300)
    # plt.show()
    plt.close('all')
    return

def plot_mean_PGOS_per_athlete(name, move, interpolated_unwrapped_trajectory, home_path):
    cmap = plt.get_cmap('plasma')
    colors = [cmap(i/np.shape(interpolated_unwrapped_trajectory)[2]) for i in range(np.shape(interpolated_unwrapped_trajectory)[2])]
    min_RMSE = 1e20
    i_min = 0
    for i in range(len(interpolated_unwrapped_trajectory[0, 0, :])):
        RMSE =  0
        for j in range(len(interpolated_unwrapped_trajectory[0, 0, :]) - 1):
            if i != j:
                nb_nan_elements = max(np.sum(np.isnan(interpolated_unwrapped_trajectory[0, :, i])),
                                        np.sum(np.isnan(interpolated_unwrapped_trajectory[0, :, j])))
                nb_non_nan_elements = 500 - nb_nan_elements
                RMSE += np.nansum(np.abs(interpolated_unwrapped_trajectory[0, :, i] - interpolated_unwrapped_trajectory[0, :, j])) / nb_non_nan_elements
        if RMSE < min_RMSE:
            min_RMSE = RMSE
            i_min = i

    if GENERATE_EACH_ATHLETE_PGOS_GRAPH:
        print("Plotting PGOS for " + name + " " + move)
        fig, ax = plt.subplots()
        axs = [ax]
        plot_gymnasium_unwrapped(axs, 0)
        for i in range(interpolated_unwrapped_trajectory.shape[2]):
            plt.plot(interpolated_unwrapped_trajectory[0, :, i], interpolated_unwrapped_trajectory[1, :, i], color=colors[i], marker='.', markersize=0.5, linestyle='None')
        # plt.plot(mean_trajectory[0, :], mean_trajectory[1, :], '-k')
        plt.plot(interpolated_unwrapped_trajectory[0, :, i_min], interpolated_unwrapped_trajectory[1, :, i_min], '-k', alpha=0.8, label='Representative\ntrajectory')
        plt.title(f"{name} {move}")
        plt.legend()
        plt.savefig(home_path + f"/disk/Eye-tracking/plots/PGOS/multiple_trials_{name}_{move}.png", dpi=300)
        # plt.show()

    # fig, axs = plt.subplots(2, 1)
    # for i in range(interpolated_unwrapped_trajectory.shape[2]):
    #     axs[0].plot(np.arange(500), interpolated_unwrapped_trajectory[0, :, i], '.b', label=f'{i}')
    #     axs[1].plot(np.arange(500), interpolated_unwrapped_trajectory[1, :, i], '.b', label=f'{i}')
    # axs[0].plot(np.arange(500), mean_trajectory[0, :], '-r')
    # axs[1].plot(np.arange(500), mean_trajectory[1, :], '-r')
    # axs[0].plot(np.arange(500), interpolated_unwrapped_trajectory[0, :, i_min], '-m', label=f'{i}')
    # axs[1].plot(np.arange(500), interpolated_unwrapped_trajectory[1, :, i_min], '-m', label=f'{i}')
    # plt.suptitle(f"{name} {move}")
    # plt.show()
    return i_min, interpolated_unwrapped_trajectory[:, :, i_min]

def plot_projection_of_PGOS(name, move, original_trajectory, projected_interpolated_trajectory_curve, home_path):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    axs = [ax]
    plot_gymnasium_unwrapped(axs, 0, FLAG_3D=True)
    plt.plot(original_trajectory[:, 0], original_trajectory[:, 1], original_trajectory[:, 2], '.r')
    plt.plot(projected_interpolated_trajectory_curve[0, :], projected_interpolated_trajectory_curve[1, :], np.zeros((500, )), '.b')
    plt.title(f"{name} {move}")
    plt.savefig(home_path + f"/disk/Eye-tracking/plots/PGOS//plots/PGOS/projection_{name}_{move}.png", dpi=300)
    # plt.show()
    return

def unwrap_gaze_positions(gaze_position, wall_index, bound_side):

    gaze_position_x_y = np.zeros((2, np.shape(wall_index)[0]))
    gaze_position_x_y[:, :] = np.nan
    for i in range(1, len(wall_index)):
        if wall_index[i] == 0:  # trampoline
            gaze_position_x_y[:, i] = gaze_position[i][:2]
        if wall_index[i] == 1:  # wall front
            gaze_position_x_y[:, i] = [gaze_position[i][2] + 7.2, gaze_position[i][1]]
        elif wall_index[i] == 2:  # ceiling
            gaze_position_x_y[:, i] = [-7.2 - (9.4620-1.2192) - 7.2 - gaze_position[i][0], gaze_position[i][1]]
        elif wall_index[i] == 3:  # wall back
            gaze_position_x_y[:, i] = [-7.2 - gaze_position[i][2], gaze_position[i][1]]
        elif wall_index[i] == 4:  # bound right
            gaze_position_x_y[:, i] = [gaze_position[i][0], -bound_side - gaze_position[i][2]]
        elif wall_index[i] == 5:  # bound left
            gaze_position_x_y[:, i] = [gaze_position[i][0], bound_side + gaze_position[i][2]]
    return gaze_position_x_y


def nearest_interp(xi, x, y):
    out = np.zeros((len(xi),))
    for i in range(len(xi)):
        index_closest = np.argmin(np.abs(xi[i]-x))
        out[i] = y[index_closest]
    return out

if TRAJECTORIES_ANALYSIS_FLAG:

    bound_side = 3 + 121 * 0.0254 / 2

    nb_interp_points = 500
    xi_interp = np.linspace(0, 1, nb_interp_points)
    trajectory_curves_per_athelte_per_move = {}
    representatives_curves_3D = {move: {name: [] for name in trial_per_athlete_per_move_index.keys()} for move in move_list}
    mean_RMSD_per_athlete_per_move_subelites = {move: [] for move in move_list}
    mean_RMSD_per_athlete_per_move_elites = {move: [] for move in move_list}
    for j, name in enumerate(trial_per_athlete_per_move_index.keys()):
        trajectory_curves_per_athelte_per_move[name] = {}
        for i, move in enumerate(trial_per_athlete_per_move_index[name].keys()):
            trajectory_curves_per_athelte_per_move[name][move] = []
            index_this_time = trial_per_athlete_per_move_index[name][move]
            trajectory_curves = np.zeros((2, nb_interp_points, 1))
            trajectory_curves_3D = np.zeros((3, nb_interp_points, 1))
            for k in index_this_time:
                x_index_this_time = np.linspace(0, 1, len(trajectories_table[k][4]))
                trajectory_this_time_x = np.reshape(interp1d(x_index_this_time, trajectories_table[k][4][:, 0])(xi_interp), (nb_interp_points, 1))
                trajectory_this_time_y = np.reshape(interp1d(x_index_this_time, trajectories_table[k][4][:, 1])(xi_interp), (nb_interp_points, 1))
                trajectory_this_time_z = np.reshape(interp1d(x_index_this_time, trajectories_table[k][4][:, 2])(xi_interp), (nb_interp_points, 1))
                # Mimic that all athletes twisted on the right side
                wall_index_facing_front_wall = trajectories_table[k][6][:, 0]
                if trajectories_table[k][7] == 'G':
                    trajectory_this_time_y = -trajectory_this_time_y
                    index_4 = np.where(wall_index_facing_front_wall == 4)[0]
                    index_5 = np.where(wall_index_facing_front_wall == 5)[0]
                    if len(index_4) > 0:
                        wall_index_facing_front_wall[index_4] = 5
                    if len(index_5) > 0:
                        wall_index_facing_front_wall[index_5] = 4

                wall_index_closest = nearest_interp(xi_interp, x_index_this_time, wall_index_facing_front_wall)
                trajectory_this_time_3d = np.hstack((trajectory_this_time_x, trajectory_this_time_y, trajectory_this_time_z))
                unwrapped_trajectory_this_time = unwrap_gaze_positions(
                    trajectory_this_time_3d,
                    wall_index_closest,
                    bound_side)
                trajectory_curves = np.concatenate((trajectory_curves, unwrapped_trajectory_this_time[:, :, np.newaxis]), axis=2)
                trajectory_curves_3D = np.concatenate((trajectory_curves_3D, trajectory_this_time_3d.T[:, :, np.newaxis]), axis=2)

            trajectory_curves = trajectory_curves[:, :, 1:]
            trajectory_curves_3D = trajectory_curves_3D[:, :, 1:]

            # plot_projection_of_PGOS(name, move, trajectories_table[k][4], trajectory_curves[:, :, -1], home_path)
            if len(index_this_time) > 0:
                representative_index, representative_trajectory = plot_mean_PGOS_per_athlete(name, move, trajectory_curves, home_path)
                trajectory_curves_per_athelte_per_move[name][move] = representative_trajectory
                plt.close('all')

            if len(index_this_time) > 0:
                representatives_curves_3D[move][name] = trajectory_curves_3D[:, :, representative_index]

            # Compute RMSD
            std_each_node = np.linalg.norm(np.nanstd(trajectory_curves_3D, axis=2), axis=0)
            mean_std = np.nanmean(std_each_node)
            if name in subelite_names:
                mean_RMSD_per_athlete_per_move_subelites[move] += [mean_std]
            elif name in elite_names:
                mean_RMSD_per_athlete_per_move_elites[move] += [mean_std]
            # plt.figure()
            # plt.plot(np.nanstd(trajectory_curves_3D, axis=2).T)
            # plt.show()
            # fig = plt.figure()
            # ax = fig.add_subplot(1, 1, 1, projection='3d')
            # for k in range(trajectory_curves_3D.shape[2]):
            #     ax.plot(trajectory_curves_3D[0, :, k], trajectory_curves_3D[1, :, k] , trajectory_curves_3D[2, :, k],  '-')
            # plt.show()

    colors_subelites = [cm.get_cmap('plasma')(k) for k in np.linspace(0, 0.4, len(subelite_names))]
    colors_elites = [cm.get_cmap('plasma')(k) for k in np.linspace(0.6, 1, len(elite_names))]
    subelites_trajectory_x = {move: np.zeros((len(xi_interp), )) for move in move_list}
    subelites_trajectory_y = {move: np.zeros((len(xi_interp), )) for move in move_list}
    elites_trajectory_x = {move: np.zeros((len(xi_interp), )) for move in move_list}
    elites_trajectory_y = {move: np.zeros((len(xi_interp), )) for move in move_list}
    subelite_representative_curve_3D = {move: np.zeros((3, len(xi_interp), 1)) for move in move_list}
    elite_representative_curve_3D = {move: np.zeros((3, len(xi_interp), 1)) for move in move_list}
    for name in trajectory_curves_per_athelte_per_move.keys():
        for move in trajectory_curves_per_athelte_per_move[name].keys():
            if name in subelite_names:
                if len(trajectory_curves_per_athelte_per_move[name][move]) != 0:
                    subelites_trajectory_x[move] = np.vstack((subelites_trajectory_x[move], trajectory_curves_per_athelte_per_move[name][move][0, :]))
                    subelites_trajectory_y[move] = np.vstack((subelites_trajectory_y[move], trajectory_curves_per_athelte_per_move[name][move][1, :]))
                    subelite_representative_curve_3D[move] = np.concatenate((subelite_representative_curve_3D[move], representatives_curves_3D[move][name][:, :, np.newaxis]), axis=2)
            elif name in elite_names:
                if len(trajectory_curves_per_athelte_per_move[name][move]) != 0:
                    elites_trajectory_x[move] = np.vstack((elites_trajectory_x[move], trajectory_curves_per_athelte_per_move[name][move][0, :]))
                    elites_trajectory_y[move] = np.vstack((elites_trajectory_y[move], trajectory_curves_per_athelte_per_move[name][move][1, :]))
                    elite_representative_curve_3D[move] = np.concatenate((elite_representative_curve_3D[move], representatives_curves_3D[move][name][:, :, np.newaxis]), axis=2)
            else:
                print(f"Probleme de nom: {name} not recognised")

    significant_timings = {move: np.zeros((len(xi_interp),)) for move in move_list}
    for i, move in enumerate(move_list):
        subelites_trajectory_x[move] = subelites_trajectory_x[move][1:, :]
        subelites_trajectory_y[move] = subelites_trajectory_y[move][1:, :]
        elites_trajectory_x[move] = elites_trajectory_x[move][1:, :]
        elites_trajectory_y[move] = elites_trajectory_y[move][1:, :]
        subelite_representative_curve_3D[move] = subelite_representative_curve_3D[move][:, :, 1:]
        elite_representative_curve_3D[move] = elite_representative_curve_3D[move][:, :, 1:]

        admissible_timings_x, significant_timings_x = find_significant_timings(xi_interp, subelites_trajectory_x[move], elites_trajectory_x[move])
        admissible_timings_y, significant_timings_y = find_significant_timings(xi_interp, subelites_trajectory_y[move], elites_trajectory_y[move])

        significant_timings[move] = np.logical_or(significant_timings_x, significant_timings_y)
        if significant_timings[move] is not None:
            print("Significant timings found for SPM1D on PGOS: ", significant_timings[move])

        plot_trajectories_data_frame(trajectory_curves_per_athelte_per_move, significant_timings, move, "Projected gaze trajectory symmetrized (PGOS) ",
                      home_path + '/disk/Eye-tracking/plots/' + f'PGOS_{move}.png')
        plt.close('all')

    print()
    print("Mean intra-MSTD per move for subelites: ")
    for move in move_list:
        print(f"{move}: {np.nanmean(mean_RMSD_per_athlete_per_move_subelites[move])} +- {np.nanstd(mean_RMSD_per_athlete_per_move_subelites[move])}")
    print()
    print("Mean intra-MSTD per move for elites: ")
    for move in move_list:
        print(f"{move}: {np.nanmean(mean_RMSD_per_athlete_per_move_elites[move])} +- {np.nanstd(mean_RMSD_per_athlete_per_move_elites[move])}")
    print()

    # Compute RMSD
    print()
    print("Mean inter-MSTD per move for subelites: ")
    for move in move_list:
        std_each_node_subelites = np.linalg.norm(np.nanstd(subelite_representative_curve_3D[move], axis=2), axis=0)
        print(f"{move}: {np.nanmean(std_each_node_subelites)}")
    print()
    print("Mean inter-MSTD per move for elites: ")
    for move in move_list:
        std_each_node_elites = np.linalg.norm(np.nanstd(elite_representative_curve_3D[move], axis=2), axis=0)
        print(f"{move}: {np.nanmean(std_each_node_elites)}")
    print()


    intra_MSTD_table = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics', 'MSTD'])
    for j, move in enumerate(move_list):
        i_elite = 0
        i_subelite = 0
        for i, name in enumerate(trajectory_curves_per_athelte_per_move.keys()):
            if name in subelite_names:
                df = {'Name': [name],
                      'Expertise': ['Subelite'],
                      'Acrobatics': [move],
                      'MSTD': [mean_RMSD_per_athlete_per_move_subelites[move][i_subelite]]}
                i_subelite += 1
            elif name in elite_names:
                if name == 'MaBo' and move == '43':
                    df = {'Name': [name],
                          'Expertise': ['Elite'],
                          'Acrobatics': [move],
                          'MSTD': [np.nanmean(mean_RMSD_per_athlete_per_move_elites['43'])]}
                else:
                    df = {'Name': [name],
                          'Expertise': ['Elite'],
                          'Acrobatics': [move],
                          'MSTD': [mean_RMSD_per_athlete_per_move_elites[move][i_elite]]}
                i_elite += 1
            intra_MSTD_table = pd.concat([intra_MSTD_table, pd.DataFrame(df)])

    # mixed ANOVA for the MSTD
    stats_procedure('MSTD', intra_MSTD_table)

if TRAJECTORIES_HEATMAPS_FLAG:
    def put_lines_on_fig(ax):
        ax.plot(np.array([0, 0]), np.array([82, 173]), '-k', linewidth=1)
        ax.plot(np.array([144, 144]), np.array([82, 173]), '-k', linewidth=1)
        ax.plot(np.array([226, 226]), np.array([0, 255]), '-k', linewidth=1)
        ax.plot(np.array([370, 370]), np.array([0, 255]), '-k', linewidth=1)
        ax.plot(np.array([453, 453]), np.array([82, 173]), '-k', linewidth=1)

        ax.plot(np.array([226, 370]), np.array([0, 0]), '-k', linewidth=1)
        ax.plot(np.array([0, 453]), np.array([82, 82]), '-k', linewidth=1)
        ax.plot(np.array([0, 453]), np.array([173, 173]), '-k', linewidth=1)
        ax.plot(np.array([226, 370]), np.array([255, 255]), '-k', linewidth=1)

        ax.plot(np.array([277, 277]), np.array([117, 138]), '-k', linewidth=1)
        ax.plot(np.array([320, 320]), np.array([117, 138]), '-k', linewidth=1)
        ax.plot(np.array([277, 320]), np.array([117, 117]), '-k', linewidth=1)
        ax.plot(np.array([277, 320]), np.array([138, 138]), '-k', linewidth=1)

        ax.text((-7.2 - (9.4620 - 1.2192) - 2 * 7.2 + 7.2 / 2 + 1) * 10 + 298.428, (4.5367 + 0.1)*10 + 27.295 + 5, "Ceiling", fontsize=10)
        ax.text((-7.2 - (9.4620 - 1.2192) + 1)*10 + 298.428, (4.5367 + 0.1)*10 + 27.295 + 5, "Wall back", fontsize=10)
        ax.text((7.2 + 1)*10 + 298.428, (4.5367 + 0.1)*10 + 27.295 + 5, "Wall front", fontsize=10)
        ax.text((-7.2 + 7.2 / 2 + 1)*10 + 298.428, (4.5367 + 9.4620 - 1.2192 + 1)*10 + 127.295 + 5, "Wall left", fontsize=10)
        ax.text((-7.2 + 7.2 / 2 + 0.5)*10 + 298.428, (-4.5367 - (9.4620 - 1.2192))*10 + 127.295 - 5, "Wall right", fontsize=10)
        return

    def transform_gaze_unwrapped_to_heatmap(gaze_position_unwrapped, width, height):
        centers = gaze_position_unwrapped * 10
        centers[0, :] += 298.428
        centers[1, :] += 127.295

        scale = 5
        gaussians = []
        for i in range(np.shape(centers)[1]):
            if (not np.isnan(centers[0, i]) and not np.isnan(centers[1, i])):
                s = np.eye(2) * scale
                g = multivariate_normal(mean=(centers[0, i], centers[1, i]), cov=s)
                gaussians.append(g)

        # create a grid of (x,y) coordinates at which to evaluate the kernels
        x = np.arange(0, width)
        y = np.arange(0, height)
        xx, yy = np.meshgrid(x, y)
        xxyy = np.stack([xx.ravel(), yy.ravel()]).T

        # evaluate kernels at grid points
        heatmap_this_time = np.zeros((height, width))
        for g in gaussians:
            heatmap_this_time += g.pdf(xxyy).reshape((height, width))

        return heatmap_this_time

    def plot_heatmaps_unwraped(subelites_heatmaps, elites_heatmaps, output_filename, move, vmin, vmax):
        fig, axs = plt.subplots(1, 2, figsize=(18, 6))
        put_lines_on_fig(axs[0])
        put_lines_on_fig(axs[1])
        f = axs[0].imshow(subelites_heatmaps, cmap=plt.get_cmap('hot_r'), vmin=vmin, vmax=vmax)  # 'hot_r'
        axs[1].imshow(elites_heatmaps, cmap=plt.get_cmap('hot_r'), vmin=vmin, vmax=vmax)
        axs[0].axis('off')
        axs[1].axis('off')
        plt.subplots_adjust(right=0.8)
        plt.suptitle(move)
        cbar_ax = fig.add_axes([0.85, 0.2, 0.02, 0.6])
        fig.colorbar(f, cax=cbar_ax)
        plt.savefig(output_filename, dpi=300)
        # plt.show()
        return

    bound_side = 3 + 121 * 0.0254 / 2
    width_heatmap = 453
    height_heatmap = 255

    heatmap_unwraped_per_athelte_per_move = {}
    heatmap_unwraped_fixations_per_athelte_per_move = {}
    for j, name in enumerate(trial_per_athlete_per_move_index.keys()):
        heatmap_unwraped_per_athelte_per_move[name] = {}
        heatmap_unwraped_fixations_per_athelte_per_move[name] = {}
        for i, move in enumerate(trial_per_athlete_per_move_index[name].keys()):
            heatmap_unwraped_per_athelte_per_move[name][move] = []
            heatmap_unwraped_fixations_per_athelte_per_move[name][move] = []
            index_this_time = trial_per_athlete_per_move_index[name][move]
            heatmap_unwraped = np.zeros((height_heatmap, width_heatmap))
            heatmap_unwraped_fixations = np.zeros((height_heatmap, width_heatmap))
            for k in index_this_time:
                # Mimic that all athletes twisted on the right side
                wall_index_facing_front_wall = trajectories_table[k][6][:, 0]
                SPGO = trajectories_table[k][4]
                if trajectories_table[k][7] == 'G':
                    SPGO[:, 1] = -SPGO[:, 1]
                    index_4 = np.where(wall_index_facing_front_wall == 4)[0]
                    index_5 = np.where(wall_index_facing_front_wall == 5)[0]
                    if len(index_4) > 0:
                        wall_index_facing_front_wall[index_4] = 5
                    if len(index_5) > 0:
                        wall_index_facing_front_wall[index_5] = 4
                gaze_position_unwrapped = unwrap_gaze_positions(SPGO, wall_index_facing_front_wall, bound_side)
                heatmap_unwraped += transform_gaze_unwrapped_to_heatmap(gaze_position_unwrapped, width_heatmap, height_heatmap)
                fixation_index = trajectories_table[k][8].astype(bool)
                heatmap_unwraped_fixations += transform_gaze_unwrapped_to_heatmap(gaze_position_unwrapped[:, fixation_index], width_heatmap, height_heatmap)

            heatmap_unwraped[0:82+1, 0:226+1] = np.nan
            heatmap_unwraped[0:82+1, 370:453+1] = np.nan
            heatmap_unwraped[173:255, 0:226+1] = np.nan
            heatmap_unwraped[173:255, 370:453+1] = np.nan
            heatmap_unwraped /= len(index_this_time)

            heatmap_unwraped_fixations[0:82+1, 0:226+1] = np.nan
            heatmap_unwraped_fixations[0:82+1, 370:453+1] = np.nan
            heatmap_unwraped_fixations[173:255, 0:226+1] = np.nan
            heatmap_unwraped_fixations[173:255, 370:453+1] = np.nan
            heatmap_unwraped_fixations /= len(index_this_time)

            if len(index_this_time) > 0:
                heatmap_unwraped_per_athelte_per_move[name][move] = heatmap_unwraped
                heatmap_unwraped_fixations_per_athelte_per_move[name][move] = heatmap_unwraped_fixations
                # plt.figure()
                # put_lines_on_fig()
                # plt.imshow(heatmap_unwraped, cmap=plt.get_cmap('hot_r'))
                # plt.savefig(home_path + '/disk/Eye-tracking/plots/' + f'PGOS_{move}.png')
                # plt.show()
                # plt.close('all')
            else:
                heatmap_unwraped_per_athelte_per_move[name][move] = None
                heatmap_unwraped_fixations_per_athelte_per_move[name][move] = None

    subelites_heatmaps_trajectory = {move: np.zeros((height_heatmap, width_heatmap)) for move in move_list}
    elites_heatmaps_trajectory = {move: np.zeros((height_heatmap, width_heatmap)) for move in move_list}
    subelites_heatmaps_fixations = {move: np.zeros((height_heatmap, width_heatmap)) for move in move_list}
    elites_heatmaps_fixations = {move: np.zeros((height_heatmap, width_heatmap)) for move in move_list}
    for name in heatmap_unwraped_per_athelte_per_move.keys():
        for move in heatmap_unwraped_per_athelte_per_move[name].keys():
            if name in subelite_names:
                if heatmap_unwraped_per_athelte_per_move[name][move] is not None:
                    subelites_heatmaps_trajectory[move] += heatmap_unwraped_per_athelte_per_move[name][move]
                if heatmap_unwraped_fixations_per_athelte_per_move[name][move] is not None:
                    subelites_heatmaps_fixations[move] += heatmap_unwraped_fixations_per_athelte_per_move[name][move]
            elif name in elite_names:
                if heatmap_unwraped_per_athelte_per_move[name][move] is not None:
                    elites_heatmaps_trajectory[move] += heatmap_unwraped_per_athelte_per_move[name][move]
                if heatmap_unwraped_fixations_per_athelte_per_move[name][move] is not None:
                    elites_heatmaps_fixations[move] += heatmap_unwraped_fixations_per_athelte_per_move[name][move]
            else:
                print(f"Probleme de nom: {name} not recognised")

    for i, move in enumerate(move_list):
        subelites_heatmaps_trajectory[move] /= subelites_nb_athlete_per_move[move]
        elites_heatmaps_trajectory[move] /= elites_nb_athlete_per_move[move]
        subelites_heatmaps_fixations[move] /= subelites_nb_athlete_per_move[move]
        elites_heatmaps_fixations[move] /= elites_nb_athlete_per_move[move]

    vmax_fixations = max(np.nanmax(np.array([np.nanmax(subelites_heatmaps_fixations[move]) for move in move_list])),
                        np.nanmax(np.array([np.nanmax(elites_heatmaps_fixations[move]) for move in move_list])))
    vmax_trajectory = max(np.nanmax(np.array([np.nanmax(subelites_heatmaps_trajectory[move]) for move in move_list])),
                         np.nanmax(np.array([np.nanmax(elites_heatmaps_trajectory[move]) for move in move_list])))

    for i, move in enumerate(move_list):
        plot_heatmaps_unwraped(subelites_heatmaps_trajectory[move], elites_heatmaps_trajectory[move],
                               home_path + '/disk/Eye-tracking/plots/' + f'heatmaps_trajectory_{move}.svg',
                               move, vmin=0, vmax=vmax_trajectory)

        plot_heatmaps_unwraped(subelites_heatmaps_fixations[move], elites_heatmaps_fixations[move],
                               home_path + '/disk/Eye-tracking/plots/' + f'heatmaps_fixations_{move}.svg',
                               move, vmin=0, vmax=vmax_fixations)
        plt.close('all')

# ----------------------------------------- AOI data frame = Mixed ANOVA --------------------------------------------- #

if AOI_ANALYSIS_FLAG:
    AOI_proportions_data_frame = pd.DataFrame(AOI_proportions_table[1:], columns=AOI_proportions_table[0])
    AOI_proportions_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/AOI_proportions_data_frame.csv")

    AOI_proportions_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics', 'Trampoline bed',
                                                            'Trampoline', 'Wall back front', 'Ceiling', 'Wall sides',
                                                             'Blink']) # 'Athlete himself',
    for i, name in enumerate(trial_per_athlete_per_move_index):
        for j, move in enumerate(trial_per_athlete_per_move_index[name]):
            index_this_time = np.where(np.logical_and(AOI_proportions_data_frame['Name'] == name, AOI_proportions_data_frame['Acrobatics'] == move))[0]
            df = {'Name': [name],
                  'Expertise': [AOI_proportions_data_frame['Expertise'][index_this_time[0]]],
                  'Acrobatics': [move],
                  'Trampoline bed': [np.nanmedian(AOI_proportions_data_frame['Trampoline bed'][index_this_time])],
                  'Trampoline': [np.nanmedian(AOI_proportions_data_frame['Trampoline'][index_this_time])],
                  'Wall back front': [
                      np.nanmedian(AOI_proportions_data_frame['Wall front'][index_this_time] + AOI_proportions_data_frame['Wall back'][index_this_time])],
                  'Ceiling': [np.nanmedian(AOI_proportions_data_frame['Ceiling'][index_this_time])],
                  'Wall sides': [np.nanmedian(AOI_proportions_data_frame['Wall sides'][index_this_time])],
                  # 'Athlete himself': [np.nanmedian(AOI_proportions_data_frame['Athlete himself'][index_this_time])],
                  'Blink': [np.nanmedian(AOI_proportions_data_frame['Blink'][index_this_time])]}
            AOI_proportions_table_temporary = pd.concat([AOI_proportions_table_temporary, pd.DataFrame(df)])

    AOI_proportions_data_frame = AOI_proportions_table_temporary

    stats_procedure('Trampoline bed', AOI_proportions_data_frame)
    stats_procedure('Trampoline', AOI_proportions_data_frame)
    stats_procedure('Wall back front', AOI_proportions_data_frame)
    stats_procedure('Ceiling', AOI_proportions_data_frame)
    stats_procedure('Wall sides', AOI_proportions_data_frame)
    # stats_procedure('Athlete himself', AOI_proportions_data_frame)
    stats_procedure('Blink', AOI_proportions_data_frame)


# ------------------------- Neck + eye proportion of movements data frame = Mixed ANOVA ----------------------------- #
if NECK_EYE_ANALYSIS_FLAG:
    neck_eye_movements_data_frame = pd.DataFrame(neck_eye_movements_table[1:], columns=neck_eye_movements_table[0])
    neck_eye_movements_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/neck_eye_movements_data_frame.csv")

    neck_eye_movements_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics',
                                                               'Anticipatory movements', 'Compensatory movements',
                                                               'Spotting movements', 'Movement detection', 'Blinks'])
    for i, name in enumerate(trial_per_athlete_per_move_index):
        for j, move in enumerate(trial_per_athlete_per_move_index[name]):
            index_this_time = np.where(np.logical_and(neck_eye_movements_data_frame['Name'] == name, neck_eye_movements_data_frame['Acrobatics'] == move))[0]
            df = {'Name': [name],
            'Expertise': [neck_eye_movements_data_frame['Expertise'][index_this_time[0]]],
            'Acrobatics': [move],
            'Anticipatory movements': [np.nanmedian(neck_eye_movements_data_frame['Anticipatory movements'][index_this_time])],
            'Compensatory movements': [np.nanmedian(neck_eye_movements_data_frame['Compensatory movements'][index_this_time])],
            'Spotting movements': [np.nanmedian(neck_eye_movements_data_frame['Spotting movements'][index_this_time])],
            'Movement detection': [np.nanmedian(neck_eye_movements_data_frame['Movement detection'][index_this_time])],
            'Blinks': [np.nanmedian(neck_eye_movements_data_frame['Blinks'][index_this_time])]}
            neck_eye_movements_table_temporary = pd.concat([neck_eye_movements_table_temporary, pd.DataFrame(df)])

    neck_eye_movements_data_frame = neck_eye_movements_table_temporary

    stats_procedure('Anticipatory movements', neck_eye_movements_data_frame)
    stats_procedure('Compensatory movements', neck_eye_movements_data_frame)
    stats_procedure('Spotting movements', neck_eye_movements_data_frame)
    stats_procedure('Movement detection', neck_eye_movements_data_frame)


# ----------------------------------- Heatmap spreading data frame = Mixed ANOVA -------------------------------------- #

if SPREADING_HEATMAP_FLAG:
    heatmaps_spreading_data_frame = pd.DataFrame(heatmaps_spreading_table[1:], columns=heatmaps_spreading_table[0])
    heatmaps_spreading_data_frame.to_csv(home_path + "/disk/Eye-tracking/plots/heatmaps_spreading_data_frame.csv")

    heatmaps_spreading_table_temporary = pd.DataFrame(columns=['Name', 'Expertise', 'Acrobatics',
                                                               'Heat map 90th percentile', 
                                                               'Width of the ellipse comprising 90 percentiles', 
                                                               'Height of the ellipse comprising 90 percentiles'])
    for i, name in enumerate(trial_per_athlete_per_move_index):
        for j, move in enumerate(trial_per_athlete_per_move_index[name]):
            index_this_time = np.where(np.logical_and(heatmaps_spreading_data_frame['Name'] == name, heatmaps_spreading_data_frame['Acrobatics'] == move))[0]
            df = {'Name': [name],
                  'Expertise': [heatmaps_spreading_data_frame['Expertise'][index_this_time[0]]],
                  'Acrobatics': [move],
                  'Heat map 90th percentile': [np.nanmedian(heatmaps_spreading_data_frame['Heat map 90th percentile'][index_this_time])],
                  'Width of the ellipse comprising 90 percentiles': [np.nanmedian(heatmaps_spreading_data_frame['Heatmap width'][index_this_time])],
                  'Height of the ellipse comprising 90 percentiles': [np.nanmedian(heatmaps_spreading_data_frame['Heatmap height'][index_this_time])]}
            heatmaps_spreading_table_temporary = pd.concat([heatmaps_spreading_table_temporary, pd.DataFrame(df)])

    heatmaps_spreading_data_frame = heatmaps_spreading_table_temporary

    stats_procedure('Width of the ellipse comprising 90 percentiles', heatmaps_spreading_data_frame)
    stats_procedure('Height of the ellipse comprising 90 percentiles', heatmaps_spreading_data_frame)


# ---------------------------------------- Qualitative data frame = SPM1D -------------------------------------------- #

def plot_presence(presence_curves_per_athelte, move, xi_interp, index_variable, title_variable, output_file_name, significant_timings):

    fig, axs = plt.subplots(2, 1)
    i_subelite = 0
    i_elite = 0
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in subelite_names:
            if presence_curves_per_athelte[name][move] != []:
                axs[0].plot(xi_interp * 100, presence_curves_per_athelte[name][move][index_variable], color=colors_subelites[i_subelite],
                            label=name)
            i_subelite += 1
    for i, name in enumerate(presence_curves_per_athelte.keys()):
        if name in elite_names:
            if presence_curves_per_athelte[name][move] != []:
                axs[1].plot(xi_interp * 100, presence_curves_per_athelte[name][move][index_variable], color=colors_elites[i_elite],
                            label=name)
            i_elite += 1

    if len(np.shape(significant_timings)) > 1:
        for i in range(1, np.shape(significant_timings[move])[0]):
            print(f"Significant difference for {move}, {title_variable}\n")
            axs[0].axvspan(significant_timings[move][i, 0] * 100/len(xi_interp), significant_timings[move][i, 1] * 100, alpha=0.2, facecolor='k')
            axs[1].axvspan(significant_timings[move][i, 0] * 100/len(xi_interp), significant_timings[move][i, 1] * 100, alpha=0.2, facecolor='k')

    axs[0].spines['top'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    axs[0].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    axs[1].set_xlabel("Normalized time [%]")
    plt.subplots_adjust(right=0.8)
    plt.suptitle(title_variable + move)
    plt.savefig(output_file_name, dpi=300)
    # plt.show()
    return

def plot_presence_all_at_the_same_time(presence_curves_per_athelte, move, xi_interp, output_file_name):

    colors = [cm.get_cmap('plasma')(k) for k in np.linspace(0, 1, 6)]

    variable_names = ["Anticipatory movements", "Compensatory movements", "Spotting", "Self-motion detection", "Blink", "Fixation"]

    presence_curves_subelites = {variable_names[index_variable]: np.zeros((len(xi_interp, ))) for index_variable in range(len(variable_names))}
    presence_curves_elites = {variable_names[index_variable]: np.zeros((len(xi_interp, ))) for index_variable in range(len(variable_names))}
    for index_variable, variable_name in enumerate(variable_names):
        for i, name in enumerate(presence_curves_per_athelte.keys()):
            if name in subelite_names:
                if presence_curves_per_athelte[name][move] != []:
                    presence_curves_subelites[variable_names[index_variable]] = np.vstack((
                        presence_curves_subelites[variable_names[index_variable]],
                        presence_curves_per_athelte[name][move][index_variable]))
        for i, name in enumerate(presence_curves_per_athelte.keys()):
            if name in elite_names:
                if presence_curves_per_athelte[name][move] != []:
                    presence_curves_elites[variable_names[index_variable]] = np.vstack((
                        presence_curves_elites[variable_names[index_variable]],
                        presence_curves_per_athelte[name][move][index_variable]))

    presence_curves_subelites = {variable_names[index_variable]: presence_curves_subelites[variable_names[index_variable]][1:, :] for index_variable in range(len(variable_names))}
    presence_curves_elites = {variable_names[index_variable]: presence_curves_elites[variable_names[index_variable]][1:, :] for index_variable in range(len(variable_names))}

    mean_presence_curves_subelites = {variable_names[index_variable]:
                                          np.mean(presence_curves_subelites[variable_names[index_variable]], axis=0)
                                      for index_variable in range(len(variable_names))}
    std_presence_curves_subelites = {variable_names[index_variable]:
                                          np.std(presence_curves_subelites[variable_names[index_variable]], axis=0)
                                      for index_variable in range(len(variable_names))}
    mean_presence_curves_elites = {variable_names[index_variable]:
                                          np.mean(presence_curves_elites[variable_names[index_variable]], axis=0)
                                      for index_variable in range(len(variable_names))}
    std_presence_curves_elites = {variable_names[index_variable]:
                                          np.std(presence_curves_elites[variable_names[index_variable]], axis=0)
                                      for index_variable in range(len(variable_names))}

    fig, axs = plt.subplots(2, 1, figsize=(9, 6))
    for index_variable in range(len(variable_names)):
        axs[0].fill_between(xi_interp * 100,
                            mean_presence_curves_subelites[variable_names[index_variable]] - std_presence_curves_subelites[variable_names[index_variable]],
                            mean_presence_curves_subelites[variable_names[index_variable]] + std_presence_curves_subelites[variable_names[index_variable]],
                            alpha=0.2, facecolor=colors[index_variable])
        axs[0].plot(xi_interp * 100, mean_presence_curves_subelites[variable_names[index_variable]], color=colors[index_variable], label=variable_names[index_variable])
        axs[1].fill_between(xi_interp * 100,
                            mean_presence_curves_elites[variable_names[index_variable]] - std_presence_curves_elites[variable_names[index_variable]],
                            mean_presence_curves_elites[variable_names[index_variable]] + std_presence_curves_elites[variable_names[index_variable]],
                            alpha=0.2, facecolor=colors[index_variable])
        axs[1].plot(xi_interp * 100, mean_presence_curves_elites[variable_names[index_variable]], color=colors[index_variable], label=variable_names[index_variable])
    axs[0].spines['top'].set_visible(False)
    axs[1].spines['top'].set_visible(False)
    axs[0].set_xlim(0, 100)
    axs[1].set_xlim(0, 100)
    axs[0].set_ylim(0, 1)
    axs[1].set_ylim(0, 1)
    # axs[0].legend(bbox_to_anchor=(0.5, 1.3), loc='upper center', ncol=3)
    axs[1].set_xlabel("Normalized time [%]")
    axs[0].set_ylabel("Subelites", fontsize=16)
    axs[1].set_ylabel("Elites", fontsize=16)
    plt.subplots_adjust(top=0.8)
    plt.suptitle(move)
    plt.savefig(output_file_name, dpi=300)
    # plt.show()

    plt.figure(figsize=(15, 3))
    for index_variable in range(len(variable_names)):
        plt.plot(0, 0, color=colors[index_variable], label=variable_names[index_variable])
    plt.legend(ncol=len(variable_names))
    plt.savefig(output_file_name[:-4] + 'legend.png', dpi=300)
    # plt.show()
    plt.close('all')
    return

if QUALITATIVE_ANALYSIS_FLAG:

    nb_interp_points = 500
    xi_interp = np.linspace(0, 1, nb_interp_points)
    presence_curves_per_athelte = {}
    for j, name in enumerate(trial_per_athlete_per_move_index.keys()):
        presence_curves_per_athelte[name] = {}
        for i, move in enumerate(trial_per_athlete_per_move_index[name].keys()):
            presence_curves_per_athelte[name][move] = []
            index_this_time = trial_per_athlete_per_move_index[name][move]
            anticipatory_curve = np.zeros((nb_interp_points, ))
            compensatory_curve = np.zeros((nb_interp_points, ))
            spotting_curve = np.zeros((nb_interp_points, ))
            movement_detection_curve = np.zeros((nb_interp_points, ))
            blink_curve = np.zeros((nb_interp_points, ))
            fixation_curve = np.zeros((nb_interp_points, ))
            for k in index_this_time:
                x_index_this_time = np.linspace(0, 1, len(qualitative_table[k][4]))
                anticipatory_presence_this_time = qualitative_table[k][4].astype(int)
                compensatory_presence_this_time = qualitative_table[k][5].astype(int)
                spotting_presence_this_time = qualitative_table[k][6].astype(int)
                movement_detection_presence_this_time = qualitative_table[k][7].astype(int)
                blink_presence_this_time = qualitative_table[k][8].astype(int)
                fixation_presence_this_time = qualitative_table[k][9].astype(int)

                anticipatory_curve += nearest_interp(xi_interp, x_index_this_time, anticipatory_presence_this_time)
                compensatory_curve += nearest_interp(xi_interp, x_index_this_time, compensatory_presence_this_time)
                spotting_curve += nearest_interp(xi_interp, x_index_this_time, spotting_presence_this_time)
                movement_detection_curve += nearest_interp(xi_interp, x_index_this_time, movement_detection_presence_this_time)
                blink_curve += nearest_interp(xi_interp, x_index_this_time, blink_presence_this_time)
                fixation_curve += nearest_interp(xi_interp, x_index_this_time, fixation_presence_this_time)

            if len(index_this_time) > 0:
                presence_curves_per_athelte[name][move] = [anticipatory_curve/len(index_this_time),
                                                 compensatory_curve/len(index_this_time),
                                                 spotting_curve/len(index_this_time),
                                                 movement_detection_curve/len(index_this_time),
                                                 blink_curve/len(index_this_time),
                                                 fixation_curve/len(index_this_time)]

    colors_subelites = [cm.get_cmap('plasma')(k) for k in np.linspace(0, 0.4, len(subelite_names))]
    colors_elites = [cm.get_cmap('plasma')(k) for k in np.linspace(0.6, 1, len(elite_names))]
    subelites_anticipatory = {move: np.zeros((len(xi_interp))) for move in move_list}
    elites_anticipatory = {move: np.zeros((len(xi_interp))) for move in move_list}
    subelites_compensatory = {move: np.zeros((len(xi_interp))) for move in move_list}
    elites_compensatory = {move: np.zeros((len(xi_interp))) for move in move_list}
    subelites_spotting = {move: np.zeros((len(xi_interp))) for move in move_list}
    elites_spotting = {move: np.zeros((len(xi_interp))) for move in move_list}
    subelites_movement_detection = {move: np.zeros((len(xi_interp))) for move in move_list}
    elites_movement_detection = {move: np.zeros((len(xi_interp))) for move in move_list}
    subelites_blink = {move: np.zeros((len(xi_interp))) for move in move_list}
    subelites_fixation = {move: np.zeros((len(xi_interp))) for move in move_list}
    elites_blink = {move: np.zeros((len(xi_interp))) for move in move_list}
    elites_fixation = {move: np.zeros((len(xi_interp))) for move in move_list}
    for name in presence_curves_per_athelte.keys():
        for move in presence_curves_per_athelte[name].keys():
            if name in subelite_names:
                if presence_curves_per_athelte[name][move] != []:
                    subelites_anticipatory[move] = np.vstack((subelites_anticipatory[move], presence_curves_per_athelte[name][move][0]))
                    subelites_compensatory[move] = np.vstack((subelites_compensatory[move], presence_curves_per_athelte[name][move][1]))
                    subelites_spotting[move] = np.vstack((subelites_spotting[move], presence_curves_per_athelte[name][move][2]))
                    subelites_movement_detection[move] = np.vstack((subelites_movement_detection[move], presence_curves_per_athelte[name][move][3]))
                    subelites_blink[move] = np.vstack((subelites_blink[move], presence_curves_per_athelte[name][move][4]))
                    subelites_fixation[move] = np.vstack((subelites_fixation[move], presence_curves_per_athelte[name][move][5]))
            elif name in elite_names:
                if presence_curves_per_athelte[name][move] != []:
                    elites_anticipatory[move] = np.vstack((elites_anticipatory[move], presence_curves_per_athelte[name][move][0]))
                    elites_compensatory[move] = np.vstack((elites_compensatory[move], presence_curves_per_athelte[name][move][1]))
                    elites_spotting[move] = np.vstack((elites_spotting[move], presence_curves_per_athelte[name][move][2]))
                    elites_movement_detection[move] = np.vstack((elites_movement_detection[move] , presence_curves_per_athelte[name][move][3]))
                    elites_blink[move] = np.vstack((elites_blink[move], presence_curves_per_athelte[name][move][4]))
                    elites_fixation[move] = np.vstack((elites_fixation[move], presence_curves_per_athelte[name][move][5]))
            else:
                print(f"Probleme de nom: {name} not recognised")

    admissible_timings_anticipatory = {move: [] for move in move_list}
    significant_timings_anticipatory = {move: [] for move in move_list}
    admissible_timings_compensatory = {move: [] for move in move_list}
    significant_timings_compensatory = {move: [] for move in move_list}
    admissible_timings_spotting = {move: [] for move in move_list}
    significant_timings_spotting = {move: [] for move in move_list}
    admissible_timings_movement_detection = {move: [] for move in move_list}
    significant_timings_movement_detection = {move: [] for move in move_list}
    admissible_timings_blink = {move: [] for move in move_list}
    significant_timings_blink = {move: [] for move in move_list}
    admissible_timings_fixation = {move: [] for move in move_list}
    significant_timings_fixation = {move: [] for move in move_list}
    for i, move in enumerate(move_list):
        subelites_anticipatory[move] = subelites_anticipatory[move][1:]
        subelites_compensatory[move] = subelites_compensatory[move][1:]
        subelites_spotting[move] = subelites_spotting[move][1:]
        subelites_movement_detection[move] = subelites_movement_detection[move][1:]
        subelites_blink[move] = subelites_blink[move][1:]
        subelites_fixation[move] = subelites_fixation[move][1:]
        elites_anticipatory[move] = elites_anticipatory[move][1:]
        elites_compensatory[move] = elites_compensatory[move][1:]
        elites_spotting[move] = elites_spotting[move][1:]
        elites_movement_detection[move] = elites_movement_detection[move][1:]
        elites_blink[move] = elites_blink[move][1:]
        elites_fixation[move] = elites_fixation[move][1:]

        admissible_timings_anticipatory[move], significant_timings_anticipatory[move] = find_significant_timings(xi_interp, subelites_anticipatory[move], elites_anticipatory[move])
        admissible_timings_compensatory[move], significant_timings_compensatory[move] = find_significant_timings(xi_interp, subelites_compensatory[move], elites_compensatory[move])
        admissible_timings_spotting[move], significant_timings_spotting[move] = find_significant_timings(xi_interp, subelites_spotting[move], elites_spotting[move])
        admissible_timings_movement_detection[move], significant_timings_movement_detection[move] = find_significant_timings(xi_interp, subelites_movement_detection[move], elites_movement_detection[move])
        admissible_timings_blink[move], significant_timings_blink[move] = find_significant_timings(xi_interp, subelites_blink[move], elites_blink[move])
        admissible_timings_fixation[move], significant_timings_fixation[move] = find_significant_timings(xi_interp, subelites_fixation[move], elites_fixation[move])

    for i, move in enumerate(trial_per_athlete_per_move_index[name].keys()):
        plot_presence(presence_curves_per_athelte, move, xi_interp, 0, "Anticipatory movements ",
                      home_path + '/disk/Eye-tracking/plots/' + f'anticiaptory_presence_{move}.png',
                      significant_timings_anticipatory[move])
        plot_presence(presence_curves_per_athelte, move, xi_interp, 1, "Compensatory movements ",
                      home_path + '/disk/Eye-tracking/plots/' + f'compensatory_presence_{move}.png',
                      significant_timings_compensatory[move])
        plot_presence(presence_curves_per_athelte, move, xi_interp, 2, "Spotting ",
                      home_path + '/disk/Eye-tracking/plots/' + f'spotting_presence_{move}.png',
                      significant_timings_spotting[move])
        plot_presence(presence_curves_per_athelte, move, xi_interp, 3, "Self-motion detection ",
                      home_path + '/disk/Eye-tracking/plots/' + f'movement_detection_presence_{move}.png',
                      significant_timings_movement_detection[move])
        plot_presence(presence_curves_per_athelte, move, xi_interp, 4, "Blink ",
                      home_path + '/disk/Eye-tracking/plots/' + f'blink_presence_{move}.png',
                      significant_timings_blink[move])
        plot_presence(presence_curves_per_athelte, move, xi_interp, 5, "Fixation ",
                      home_path + '/disk/Eye-tracking/plots/' + f'fixation_presence_{move}.png',
                      significant_timings_fixation[move])

        plot_presence_all_at_the_same_time(presence_curves_per_athelte, move, xi_interp,
                          home_path + '/disk/Eye-tracking/plots/' + f'presence_all_{move}.png')







