import os
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import json
import numpy as np

plt.rc('font', family='serif')
plt.rc('font', serif='Times New Roman')


def default():
    WORKER_NUM = [5, 10, 30, 50, 70, 100]
    matplotlib.rcParams['font.family'] = 'Times New Roman'
    plt.rcParams["mathtext.fontset"] = "stix"

    DATASET = 'mnist_all_data_0_niid_unbalanced'
    result_dict = {}
    with open(os.path.join('metrics_mnist_all_data_0_equal_niid.json'), 'r') as load_f:
        load_dict = eval(json.load(load_f))
    result_dict['10'] = load_dict['loss_on_train_data'][:80]

    prop_cycle = plt.rcParams['axes.prop_cycle']
    colors = prop_cycle.by_key()['color']
    COLORS = {"10": colors[0],
              "30": colors[1],
              "50": colors[2],
              "70": colors[3],
              "100": colors[5]}
    LABELS = {"10": r"$K=10$",
              "30": r"$K=30$",
              "50": r"$K=50$",
              "70": r"$K=70$",
              "100": r"$K=100$"}

    plt.figure(figsize=(4, 3))
    for wn, stat in result_dict.items():
        plt.plot(np.arange(len(stat)), np.array(stat), linewidth=1.0, color=COLORS[wn], label=LABELS[wn])

    plt.grid(True)
    # 0: ‘best', 1: ‘upper right', 2: ‘upper left', 3: ‘lower left'
    plt.legend(loc=0, borderaxespad=0., prop={'size': 10})
    plt.xlabel('Round $(T/E)$', fontdict={'size': 10})
    plt.ylabel('Loss', fontdict={'size': 10})
    plt.xticks(fontsize=10)
    plt.yticks(fontsize=10)
    plt.tight_layout()

    before = 15
    a = plt.axes([0.31, 0.6, .3, .3])
    for wn, stat in result_dict.items():
        plt.plot(np.arange(len(stat))[-before:-5], np.array(stat)[-before:-5], linewidth=1.0, color=COLORS[wn],
                 label=LABELS[wn])

    plt.xticks(fontsize=7)
    plt.yticks(fontsize=7)
    fig = plt.gcf()
    fig.savefig(f'{DATASET}_K.pdf')


def acc_plot(metrics, filepath):
    train_acc = metrics['acc_on_train_data']
    eval_acc = metrics['acc_on_eval_data']

    plt.plot(range(len(train_acc)), train_acc, label='train', color='r')
    plt.plot(range(len(eval_acc)), eval_acc, label='eval', color='b')
    plt.xlabel('Rounds')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.title('Accuracy vs. Round')
    plt.savefig(os.path.join(filepath,'train_eval_acc'))
    plt.show()
    plt.clf()


def loss_plot(metrics, filepath):
    train_loss = metrics['loss_on_train_data']
    eval_loss = metrics['loss_on_eval_data']

    plt.plot(range(len(train_loss)), train_loss, label='train', color='r')
    plt.plot(range(len(eval_loss)), eval_loss, label='eval', color='b')
    plt.xlabel('Rounds')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Loss vs. Round')
    plt.savefig(os.path.join(filepath,'train_eval_loss'))
    plt.show()
    plt.clf()


def bin_allocation(bins, round, accuracies):
    for acc in accuracies:
        if acc < 0.33:
            bins[0][round] += 1
        elif acc < 0.66:
            bins[1][round] += 1
        elif acc < 0.9:
            bins[2][round] += 1
        else:
            bins[3][round] += 1

def plot_client_bar(data, title, n_rounds, interval,filepath):
    barWidth = 3
    fig = plt.subplots(figsize=(12, 8))

    br1 = list(range(0, n_rounds, interval))
    br2 = [x + barWidth for x in br1]
    br3 = [x + barWidth for x in br2]
    br4 = [x + barWidth for x in br3]

    plt.bar(br1, data[0], width=barWidth,
            edgecolor='grey', label='<0.33')
    plt.bar(br2, data[1], width=barWidth,
            edgecolor='grey', label='>=0.33, <0.66')
    plt.bar(br3, data[2], width=barWidth,
            edgecolor='grey', label='>=0.66, <0.9')
    plt.bar(br4, data[3], width=barWidth,
            edgecolor='grey', label='>=0.9')

    plt.xlabel('Rounds')
    plt.ylabel('No. of clients')
    plt.xticks(br2)
    plt.title(title)

    plt.legend()
    plt.savefig(os.path.join(filepath, title))
    plt.show()


def client_acc_plot(metrics, filepath):
    client_train_acc = metrics['client_train_acc']
    client_eval_acc = metrics['client_test_acc']

    interval = 20
    n_rounds = len(client_train_acc)
    n_bins = 4

    train_bins = [[0] * (int(n_rounds / interval) + 1) for _ in range(n_bins)]
    eval_bins = [[0] * (int(n_rounds / interval) + 1) for _ in range(n_bins)]

    for i in range(0, n_rounds, interval):
        train_acc = client_train_acc[i]
        eval_acc = client_eval_acc[i]

        round = int(i / interval)

        bin_allocation(train_bins, round, train_acc)
        bin_allocation(eval_bins, round, eval_acc)

    # colors = ['r', 'b', 'g', ]
    plot_client_bar(train_bins, 'Client Train Accuracies',
                    n_rounds, interval,filepath)
    plot_client_bar(eval_bins, 'Client Eval Accuracies',
                    n_rounds, interval, filepath)

    return


def main():

    folders = os.listdir('Results')
    for dir in folders:
        if '.DS' in dir:
            continue
        dirname = os.path.join('Results', dir)
        filename = os.path.join(dirname, 'metrics.json')
        with open(filename, 'r') as f:
            metrics = eval(json.load(f))
        acc_plot(metrics, dirname)
        loss_plot(metrics, dirname)
        client_acc_plot(metrics, dirname)


    # with open('metrics_mnist_all_data_0_equal_niid.json', 'r') as f:
    #     metrics = eval(json.load(f))
    #
    # acc_plot(metrics)
    # loss_plot(metrics)
    # client_acc_plot(metrics)


main()
