import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import pickle

import src.plotter.utils as utils
from src.plotter.main import *


BATCH_RESULT_PATH = "C:/Users/SK3LA/Desktop/WORKSPACE/1_TSP/BATCH_FINALI/@RISULTATI@/res/SAMESERVICEGAUSIGMA2/"
OUTPUT_DIRECTORY_PATH = f"C:/Users/SK3LA/Desktop/WORKSPACE/1_TSP/BATCH_FINALI/@RISULTATI@/plot/SAMESERVICEGAUSIGMA2/aggregated-{BATCH_RESULT_PATH.split('/')[-2]}/"

# slice deadlines: after that time-slot the job is expired
DEADLINES = [1, 5]


def get_results_path(start_path=BATCH_RESULT_PATH):
    to_ret = []

    for file in os.listdir(start_path):
        if os.path.isdir(os.path.join(start_path, file)):
            to_ret.append(os.path.join(start_path, file, 'results.data'))

    return to_ret


def main():
    results_path = get_results_path()

    policies_aggregated_stats = None

    try:
        policies_aggregated_stats = pickle.load(open(f"{OUTPUT_DIRECTORY_PATH}policies_aggregated_stats.pickle", "rb"))
    except FileNotFoundError:
        for r in results_path:
            imported_data = utils.import_serialized_data(r)
            if policies_aggregated_stats is None:
                policies_aggregated_stats = [{'name': imported_data[i]['name'], 'stats_per_slice': []} for i in range(len(imported_data))]

            for policy_index in range(len(imported_data)):
                policies_aggregated_stats[policy_index]['stats_per_slice'].append(
                    {
                     # the way we calculate the server_cap assumes the directory name like "something_NUMBEROFSERVERS"
                     'server_cap': r.replace("\\", "/").split('/')[-2].split('_')[1],
                     'slices': []
                     }
                )

                slices = filter_stats_per_slice(split_stats_per_slice(imported_data[policy_index]['environment_data']))

                policies_aggregated_stats[policy_index]['stats_per_slice'][-1]['slices'].append(
                    {
                        'processed_jobs': sum(slices[0]['processed_jobs']),
                        'incoming_jobs': sum(slices[0]['incoming_jobs']),
                        'lost_jobs': sum(slices[0]['lost_jobs']),
                        'toolate_jobs': sum(slices[0]['processed_jobs']) * sum(slices[0]['wait_time_in_the_system'][DEADLINES[0]:]),
                        'cost': sum(slices[0]['cost']),
                        'cost_component': np.array(slices[0]['cost_component']).sum(0).tolist()
                     }
                )

                policies_aggregated_stats[policy_index]['stats_per_slice'][-1]['slices'].append(
                    {
                        'processed_jobs': sum(slices[1]['processed_jobs']),
                        'incoming_jobs': sum(slices[1]['incoming_jobs']),
                        'lost_jobs': sum(slices[1]['lost_jobs']),
                        'toolate_jobs': sum(slices[1]['processed_jobs']) * sum(slices[1]['wait_time_in_the_system'][DEADLINES[1]:]),
                        'cost': sum(slices[1]['cost']),
                        'cost_component': np.array(slices[1]['cost_component']).sum(0).tolist()
                    }
                )

        os.makedirs(OUTPUT_DIRECTORY_PATH)
        pickle.dump(policies_aggregated_stats, open(f"{OUTPUT_DIRECTORY_PATH}policies_aggregated_stats.pickle", "wb"))

    dataframes = []

    for policy in policies_aggregated_stats:
        servers_list = []
        processed_on_time_percent = [[], []]
        processed_too_late_percent = [[], []]
        lost_percent = [[], []]

        cost_job = [[], []]
        cost_server = [[], []]
        cost_lost = [[], []]
        cost_alloc = [[], []]
        cost_dealloc = [[], []]

        costs = [[], []]

        for s in policy['stats_per_slice']:
            servers_list.append(s['server_cap'])

            lost_percent[0].append((s['slices'][0]['lost_jobs'] * 100) / s['slices'][0]['incoming_jobs'])
            lost_percent[1].append((s['slices'][1]['lost_jobs'] * 100) / s['slices'][1]['incoming_jobs'])

            processed_too_late_percent[0].append((s['slices'][0]['toolate_jobs'] * 100) / s['slices'][0]['incoming_jobs'])
            processed_too_late_percent[1].append((s['slices'][1]['toolate_jobs'] * 100) / s['slices'][1]['incoming_jobs'])

            s['slices'][0]['processed_jobs'] -= s['slices'][0]['toolate_jobs']
            s['slices'][1]['processed_jobs'] -= s['slices'][1]['toolate_jobs']

            processed_on_time_percent[0].append((s['slices'][0]['processed_jobs'] * 100) / s['slices'][0]['incoming_jobs'])
            processed_on_time_percent[1].append((s['slices'][1]['processed_jobs'] * 100) / s['slices'][1]['incoming_jobs'])

            cost_job[0].append(s['slices'][0]['cost_component'][0])
            cost_job[1].append(s['slices'][1]['cost_component'][0])

            cost_server[0].append(s['slices'][0]['cost_component'][1])
            cost_server[1].append(s['slices'][1]['cost_component'][1])

            cost_lost[0].append(s['slices'][0]['cost_component'][2])
            cost_lost[1].append(s['slices'][1]['cost_component'][2])

            cost_alloc[0].append(s['slices'][0]['cost_component'][3])
            cost_alloc[1].append(s['slices'][1]['cost_component'][3])

            cost_dealloc[0].append(s['slices'][0]['cost_component'][4])
            cost_dealloc[1].append(s['slices'][1]['cost_component'][4])

            costs[0].append(s['slices'][0]['cost'])
            costs[1].append(s['slices'][1]['cost'])

        dataframes.append(
            {
                'name': policy['name'],
                'slices': [
                    pd.DataFrame({  # slice high priority
                        'server': servers_list,
                        'processed_on_time': processed_on_time_percent[0],
                        'processed_too_late': processed_too_late_percent[0],
                        'lost': lost_percent[0],
                        'cost_job': cost_job[0],
                        'cost_server': cost_server[0],
                        'cost_lost': cost_lost[0],
                        'cost_alloc': cost_alloc[0],
                        'cost_dealloc': cost_dealloc[0],
                        'cost_sum': costs[0]
                    }),
                    pd.DataFrame({  # slice low priority
                        'server': servers_list,
                        'processed_on_time': processed_on_time_percent[1],
                        'processed_too_late': processed_too_late_percent[1],
                        'lost': lost_percent[1],
                        'cost_job': cost_job[1],
                        'cost_server': cost_server[1],
                        'cost_lost': cost_lost[1],
                        'cost_alloc': cost_alloc[1],
                        'cost_dealloc': cost_dealloc[1],
                        'cost_sum': costs[1]
                    })  # .sort_values(by=["server"]).set_index("server")
                ]
            }
        )

        dataframes[-1]['slices'][0]['server'] = dataframes[-1]['slices'][0].astype(int)
        dataframes[-1]['slices'][1]['server'] = dataframes[-1]['slices'][1].astype(int)
        dataframes[-1]['slices'][0] = dataframes[-1]['slices'][0].sort_values(by=["server"]).set_index("server")
        dataframes[-1]['slices'][1] = dataframes[-1]['slices'][1].sort_values(by=["server"]).set_index("server")

    for dataframe in dataframes:
        for slice_i in range(len(dataframe['slices'])):
            title = f'Policy {dataframe["name"]} - {"High Priority Slice" if slice_i == 0 else "Low Priority Slice"}\n\n'

            # STACKED PLOT SINGLE SLICE
            # ax = dataframe['slices'][slice_i][['processed_on_time', 'processed_too_late', 'lost']].plot(
            #     kind='barh',
            #     stacked=True,
            #     title=title,
            #     figsize=(15, 10),
            #     width=0.8
            # )
            # ax.xaxis.grid(color='gray', linestyle='dashed')
            # xticks = np.arange(0, 101, 10)
            # xlabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
            # plt.xticks(xticks, xlabels)
            # plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower center', ncol=3, borderaxespad=0., frameon=False)
            # plt.savefig(f"{OUTPUT_DIRECTORY_PATH}{dataframe['name']}-slice{slice_i}_stacked")
            # -------------------------

            # BARH PLOT NON STACKED SINGLE SLICE
            ax = dataframe['slices'][slice_i][['processed_on_time', 'processed_too_late', 'lost']].plot(
                kind='bar',
                stacked=False,
                title=title,
                # figsize=(25, 10),
                width=0.8
            )
            ax.yaxis.grid(color='gray', linestyle='dashed')
            yticks = np.arange(0, 101, 10)
            ylabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
            plt.yticks(yticks, ylabels)
            plt.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower center', ncol=3, borderaxespad=0., frameon=False)

            # create a list to collect the plt.patches data
            totals = []

            # find the values and append to list
            for i in ax.patches:
                totals.append(i.get_height())

            # set individual bar lables using above list
            total = sum(totals)

            # set individual bar lables using above list
            for i in ax.patches:
                # get_x pulls left or right; get_height pushes up or down
                ax.text(i.get_x() - .03, i.get_height() + .5,
                        str(round((i.get_height()), 4)) + '%', fontsize=9, color='black', rotation=45)

            plt.savefig(f"{OUTPUT_DIRECTORY_PATH}{dataframe['name']}-slice{slice_i}")
            plt.close()

            # -------------------------

        # TWO SLICES IN ONE FIGURE

        df0 = dataframe['slices'][0][['processed_on_time', 'processed_too_late', 'lost']]
        df1 = dataframe['slices'][1][['processed_on_time', 'processed_too_late', 'lost']]

        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].set_title("High Priority", loc="left")
        axs[1].set_title("Low Priority", loc="left")
        axs[0].set_ylabel("% of Jobs")
        axs[1].set_ylabel("% of Jobs")
        fig.suptitle(f"Policy {dataframe['name']}")
        df0.plot(
            kind='bar',
            stacked=False,
            width=0.8,
            ax=axs[0],
            legend=False
        )
        axs[0].legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower right', ncol=3, borderaxespad=0., frameon=False)
        axs[0].grid(axis='y', which='both', color='grey', linestyle='dashed')
        df1.plot(
            kind='bar',
            stacked=False,
            width=0.8,
            ax=axs[1],
            legend=False
        )
        axs[1].legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower right', ncol=3, borderaxespad=0., frameon=False)
        axs[1].grid(axis='y', which='both', color='grey', linestyle='dashed')
        plt.savefig(f"{OUTPUT_DIRECTORY_PATH}{dataframe['name']}")
        plt.close()

        df0 = dataframe['slices'][0][['cost_job', 'cost_server', 'cost_lost', 'cost_alloc', 'cost_dealloc']]
        df1 = dataframe['slices'][1][['cost_job', 'cost_server', 'cost_lost', 'cost_alloc', 'cost_dealloc']]
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].set_title("High Priority", loc="left")
        axs[1].set_title("Low Priority", loc="left")
        axs[0].set_ylabel("Costs (USD)")
        axs[1].set_ylabel("Costs (USD)")
        fig.suptitle(f"Policy {dataframe['name']}")
        df0.plot(
            kind='bar',
            stacked=False,
            width=0.8,
            ax=axs[0],
            legend=False
        )
        axs[0].legend(bbox_to_anchor=(1.05, 1), ncol=3, loc='lower right', borderaxespad=0., frameon=False)
        axs[0].grid(axis='y', which='both', color='grey', linestyle='dashed')
        df1.plot(
            kind='bar',
            stacked=False,
            width=0.8,
            ax=axs[1],
            legend=False
        )
        axs[1].legend(bbox_to_anchor=(1.05, 1), ncol=3, loc='lower right', borderaxespad=0., frameon=False)
        axs[1].grid(axis='y', which='both', color='grey', linestyle='dashed')
        plt.savefig(f"{OUTPUT_DIRECTORY_PATH}{dataframe['name']}_cost_component")
        plt.close()

        df0 = dataframe['slices'][0][['cost_sum']]
        df1 = dataframe['slices'][1][['cost_sum']]
        fig, axs = plt.subplots(2, 1, constrained_layout=True)
        axs[0].set_title("High Priority", loc="left")
        axs[1].set_title("Low Priority", loc="left")
        axs[0].set_ylabel("Costs (USD)")
        axs[1].set_ylabel("Costs (USD)")
        fig.suptitle(f"Policy {dataframe['name']}")
        df0.plot(
            kind='bar',
            stacked=False,
            ax=axs[0],
            legend=False
        )
        axs[0].legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower right', borderaxespad=0., frameon=False)
        axs[0].grid(axis='y', which='both', color='grey', linestyle='dashed')
        df1.plot(
            kind='bar',
            stacked=False,
            ax=axs[1],
            legend=False
        )
        axs[1].legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower right', borderaxespad=0., frameon=False)
        axs[1].grid(axis='y', which='both', color='grey', linestyle='dashed')
        plt.savefig(f"{OUTPUT_DIRECTORY_PATH}{dataframe['name']}_costs")
        plt.close()

    # ------ uniform view ------- I KNOW, THIS IS NOT THE ELEGANT WAY TO DO IT
    slice_0 = [d['slices'][0] for d in dataframes]
    slice_1 = [d['slices'][1] for d in dataframes]

    df_processed_on_time_slice_0 = \
        pd.DataFrame({
            'server': slice_0[0].index.to_list(),
            '0_SequentialPriorityMultiSliceMdpPolicy': slice_0[0]['processed_on_time'].to_list(),
            '1_SimplifiedPriorityMultiSliceMdpPolicy': slice_0[1]['processed_on_time'].to_list(),
            '2_MultiSliceStaticPolicy': slice_0[2]['processed_on_time'].to_list(),
            '3_MultiSliceStaticPolicy': slice_0[3]['processed_on_time'].to_list(),
                     })

    df_processed_on_time_slice_0['server'] = df_processed_on_time_slice_0['server'].astype(int)
    df_processed_on_time_slice_0 = df_processed_on_time_slice_0.sort_values(by=["server"]).set_index("server")

    ax = df_processed_on_time_slice_0.plot(
        kind='bar',
        stacked=False,
        # figsize=(25, 10),
        width=0.8
    )
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.set_title(f"[High Priority] Processed on time", pad=40)
    yticks = np.arange(0, 101, 10)
    ylabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.yticks(yticks, ylabels)
    ax.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower center', ncol=4, borderaxespad=0., frameon=False)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        if i.get_x() > 20:
            ax.text(i.get_x(), i.get_height() - 5,
                    str(round((i.get_height()), 3)) + '%', fontsize=9, color='white', rotation=180)
        else:
            # get_x pulls left or right; get_height pushes up or down
            ax.text(i.get_x(), i.get_height() + .5,
                    str(round((i.get_height()), 3)) + '%', fontsize=9, color='black', rotation=180)

    plt.savefig(f"{OUTPUT_DIRECTORY_PATH}processed_on_time_slice0_allpolicy")
    plt.close()

    # ---

    df_processed_on_time_slice_1 = \
        pd.DataFrame({
            'server': slice_1[0].index.to_list(),
            '0_SequentialPriorityMultiSliceMdpPolicy': slice_1[0]['processed_on_time'].to_list(),
            '1_SimplifiedPriorityMultiSliceMdpPolicy': slice_1[1]['processed_on_time'].to_list(),
            '2_MultiSliceStaticPolicy': slice_1[2]['processed_on_time'].to_list(),
            '3_MultiSliceStaticPolicy': slice_1[3]['processed_on_time'].to_list(),
        })

    df_processed_on_time_slice_1['server'] = df_processed_on_time_slice_1['server'].astype(int)
    df_processed_on_time_slice_1 = df_processed_on_time_slice_1.sort_values(by=["server"]).set_index("server")

    ax = df_processed_on_time_slice_1.plot(
        kind='bar',
        stacked=False,
        # figsize=(25, 10),
        width=0.8
    )
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.set_title(f"[Low Priority] Processed on time", pad=40)
    yticks = np.arange(0, 101, 10)
    ylabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.yticks(yticks, ylabels)
    ax.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower center', ncol=4, borderaxespad=0., frameon=False)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x(), i.get_height() + .5,
                str(round((i.get_height()), 3)) + '%', fontsize=9, color='black', rotation=45)

    plt.savefig(f"{OUTPUT_DIRECTORY_PATH}processed_on_time_slice1_allpolicy")
    plt.close()

    # ---

    df_processed_too_late_slice_0 = \
        pd.DataFrame({
            'server': slice_0[0].index.to_list(),
            '0_SequentialPriorityMultiSliceMdpPolicy': slice_0[0]['processed_too_late'].to_list(),
            '1_SimplifiedPriorityMultiSliceMdpPolicy': slice_0[1]['processed_too_late'].to_list(),
            '2_MultiSliceStaticPolicy': slice_0[2]['processed_too_late'].to_list(),
            '3_MultiSliceStaticPolicy': slice_0[3]['processed_too_late'].to_list(),
        })

    df_processed_too_late_slice_0['server'] = df_processed_too_late_slice_0['server'].astype(int)
    df_processed_too_late_slice_0 = df_processed_too_late_slice_0.sort_values(by=["server"]).set_index("server")

    ax = df_processed_too_late_slice_0.plot(
        kind='bar',
        stacked=False,
        # figsize=(25, 10),
        width=0.8
    )
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.set_title(f"[High Priority] Processed too late", pad=40)
    yticks = np.arange(0, 101, 10)
    ylabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.yticks(yticks, ylabels)
    ax.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower center', ncol=4, borderaxespad=0., frameon=False)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x(), i.get_height() + .5,
                str(round((i.get_height()), 3)) + '%', fontsize=9, color='black', rotation=45)

    plt.savefig(f"{OUTPUT_DIRECTORY_PATH}processed_too_late_slice0_allpolicy")
    plt.close()

    # ---

    df_processed_too_late_slice_1 = \
        pd.DataFrame({
            'server': slice_1[0].index.to_list(),
            '0_SequentialPriorityMultiSliceMdpPolicy': slice_1[0]['processed_too_late'].to_list(),
            '1_SimplifiedPriorityMultiSliceMdpPolicy': slice_1[1]['processed_too_late'].to_list(),
            '2_MultiSliceStaticPolicy': slice_1[2]['processed_too_late'].to_list(),
            '3_MultiSliceStaticPolicy': slice_1[3]['processed_too_late'].to_list(),
        })

    df_processed_too_late_slice_1['server'] = df_processed_too_late_slice_1['server'].astype(int)
    df_processed_too_late_slice_1 = df_processed_too_late_slice_1.sort_values(by=["server"]).set_index("server")

    ax = df_processed_too_late_slice_1.plot(
        kind='bar',
        stacked=False,
        # figsize=(25, 10),
        width=0.8
    )
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.set_title(f"[Low Priority] Processed too late", pad=40)
    yticks = np.arange(0, 101, 10)
    ylabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.yticks(yticks, ylabels)
    ax.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower center', ncol=4, borderaxespad=0., frameon=False)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x(), i.get_height() + .5,
                str(round((i.get_height()), 3)) + '%', fontsize=9, color='black', rotation=45)

    plt.savefig(f"{OUTPUT_DIRECTORY_PATH}processed_too_late_slice1_allpolicy")
    plt.close()

    # ---

    df_lost_slice_0 = \
        pd.DataFrame({
            'server': slice_0[0].index.to_list(),
            '0_SequentialPriorityMultiSliceMdpPolicy': slice_0[0]['lost'].to_list(),
            '1_SimplifiedPriorityMultiSliceMdpPolicy': slice_0[1]['lost'].to_list(),
            '2_MultiSliceStaticPolicy': slice_0[2]['lost'].to_list(),
            '3_MultiSliceStaticPolicy': slice_0[3]['lost'].to_list(),
        })

    df_lost_slice_0['server'] = df_lost_slice_0['server'].astype(int)
    df_lost_slice_0 = df_lost_slice_0.sort_values(by=["server"]).set_index("server")

    ax = df_lost_slice_0.plot(
        kind='bar',
        stacked=False,
        # figsize=(25, 10),
        width=0.8
    )
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.set_title(f"[High Priority] Lost jobs", pad=40)
    yticks = np.arange(0, 101, 10)
    ylabels = ['{}%'.format(i) for i in np.arange(0, 101, 10)]
    plt.yticks(yticks, ylabels)
    ax.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower center', ncol=4, borderaxespad=0., frameon=False)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x(), i.get_height() + .5,
                str(round((i.get_height()), 3)) + '%', fontsize=9, color='black', rotation=45)

    plt.savefig(f"{OUTPUT_DIRECTORY_PATH}lost_slice0_allpolicy")
    plt.close()


    # ---

    df_lost_slice_1 = \
        pd.DataFrame({
            'server': slice_1[0].index.to_list(),
            '0_SequentialPriorityMultiSliceMdpPolicy': slice_1[0]['lost'].to_list(),
            '1_SimplifiedPriorityMultiSliceMdpPolicy': slice_1[1]['lost'].to_list(),
            '2_MultiSliceStaticPolicy': slice_1[2]['lost'].to_list(),
            '3_MultiSliceStaticPolicy': slice_1[3]['lost'].to_list(),
        })

    df_lost_slice_1['server'] = df_lost_slice_1['server'].astype(int)
    df_lost_slice_1 = df_lost_slice_1.sort_values(by=["server"]).set_index("server")

    ax = df_lost_slice_1.plot(
        kind='bar',
        stacked=False,
        # figsize=(25, 10),
        width=0.8
    )
    ax.yaxis.grid(color='gray', linestyle='dashed')

    ax.set_title(f"[Low Priority] Lost jobs", pad=40)
    yticks = np.arange(0, 101, 10)
    ylabels = ['{}%'.format(i) for i in np.arange(0, 109, 10)]
    plt.yticks(yticks, ylabels)
    ax.legend(bbox_to_anchor=(0., 1.01, 1., .102), loc='lower center', ncol=4, borderaxespad=0., frameon=False)

    # create a list to collect the plt.patches data
    totals = []

    # find the values and append to list
    for i in ax.patches:
        totals.append(i.get_height())

    # set individual bar lables using above list
    total = sum(totals)

    # set individual bar lables using above list
    for i in ax.patches:
        # get_x pulls left or right; get_height pushes up or down
        ax.text(i.get_x(), i.get_height() + .5,
                str(round((i.get_height()), 3)) + '%', fontsize=9, color='black', rotation=45)

    plt.savefig(f"{OUTPUT_DIRECTORY_PATH}lost_slice1_allpolicy")
    plt.close()


if __name__ == '__main__':
    main()
