"""
Author: Sajad Rezaie (sre@es.aau.dk)
"""
import tensorflow as tf
import numpy as np
import h5py, math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from pylab import *
from nn_model import deep_nn
from inverse_fp import inv_fp_3d
from sklearn.preprocessing import StandardScaler


if __name__ == '__main__':
    top_m = 100
    range_pt = [[0, 50], [-50, 50]]  # [[xmin, xmax], [ymin, ymax]]
    bin_size_ang = [(2, 5), (5, 5), (5, 10)]  # meter, degree
    bin_size_ang_2 = [(5, 10), (10, 5), (15, 10)]  # meter, degree
    ns_s = 100  # number of scanned beam pairs with high RSS for saving
    nb_t = 64  # number of TX antennas
    nb_r = 64  # number of RX antennas
    ns_t = nb_t * nb_r  # Number of total beam pairs
    snr_out_dB = 10  # Threshold snr of outage (dB)
    snr_out = 10 ** (snr_out_dB / 10)
    multilabel = [1, 2 , 5, 7, 11]
    multilabel_2 = [1, 2 , 5, 7, 11]
    top_n = np.concatenate((np.arange(1, 10, 1), np.arange(10, 20, 2), np.arange(20, 51, 5)), axis=0)
    marker_list = ['o', 's', 'D', 'v', '^', '<', '>', '*', 'P', 'X', 'd', '8', 'p', 'h', 'H']
    color_list = [[0,    0.4470,    0.7410],
                    [0.8500,    0.3250,    0.0980],
                    [0.9290,    0.6940,    0.1250],
                    [0.4940,    0.1840,    0.5560],
                    [0.4660,    0.6740,    0.1880],
                    [0.3010,    0.7450,    0.9330],
                    [0.6350,    0.0780,    0.1840],
                    [0.2500,    0.2500,    0.2500]]

    print('Loading data ...')
    with h5py.File('./Datasets/snr_data_1e5_nt64_nr64_nb_100_newbeams_so6_mo10_PPP_alpharx_random_2pi_alphatx_90.hdf5', 'r') as fr:
        data_r = fr['default']
        s = np.shape(data_r)
        data = np.zeros([s[0], s[1]], dtype=float)
        data[:, ] = data_r[:, ]
        print('size:', np.shape(data_r))
    print('Input data loaded successfully')

    x = np.zeros([np.shape(data)[0], 4])
    x[:, 0] = np.arange(np.shape(data)[0])
    x[:, 1:3] = data[:, :2]
    x[:, 3] = data[:, 2 + 3 * ns_s + 2]
    y = data[:, 2]
    y = y.astype(int)

    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)

    ind_train = x_train[:, 0]
    ind_train = ind_train.astype(int)
    ind_val_test = x_val_test[:, 0]
    ind_val_test = ind_val_test.astype(int)
    ind_val = x_val[:, 0]
    ind_val = ind_val.astype(int)
    ind_test = x_test[:, 0]
    ind_test = ind_test.astype(int)

    x_train = x_train[:, 1:]
    x_val = x_val[:, 1:]
    x_val_test = x_val_test[:, 1:]
    x_test = x_test[:, 1:]

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_n = scaler.transform(x_train)
    x_val_n = scaler.transform(x_val)
    x_val_test_n = scaler.transform(x_val_test)

    nb = np.shape(top_n)[0]
    n_rows = np.shape(multilabel)[0] + len(bin_size_ang) + np.shape(multilabel_2)[0] + len(bin_size_ang_2)
    n_rows_1 = np.shape(multilabel)[0] + len(bin_size_ang)
    n_rows_2 = np.shape(multilabel_2)[0] + len(bin_size_ang_2)
    align_0db_t = np.zeros([n_rows, nb])
    align_3db_t = np.zeros([n_rows, nb])
    prob_out_perfect = np.zeros([n_rows, nb])
    prob_out = np.zeros([n_rows, nb])
    ave_perfect_align_rate_t = np.zeros([n_rows, nb])
    ave_achievable_rate_t = np.zeros([n_rows, nb])

    print(scaler.transform(np.transpose([np.arange(0, 50, 5), np.arange(0, 50, 5), np.arange(0, 50, 5)])))

    leg = []
    color_l = []
    marker_l = []
    c = 0

    for k in range(len(bin_size_ang)):
        align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect[c], prob_out[c] = \
            inv_fp_3d(top_m, range_pt, bin_size_ang[k][0], bin_size_ang[k][1], ns_s, ns_t, nb_t, nb_r, top_n, data, ind_train,
                      x_train, y_train, ind_val_test, x_val_test, y_val_test, snr_out)
        leg.append('GIFP (SBS = {} m, ABS = {}°)'.format(bin_size_ang[k][0], bin_size_ang[k][1]))
        color_l.append(color_list[c])
        marker_l.append(marker_list[c])
        c += 1

    for z in range(np.shape(multilabel)[0]):
        h, align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect[c], prob_out[c] = \
            deep_nn(ns_s, ns_t, top_n, data, ind_train, x_train_n, y_train, data, ind_val, x_val_n, y_val, ind_val_test, x_val_test_n,
                    y_val_test, multilabel[z], 0, snr_out,
                    load_name="./Models/model_XYT_1e5_nt64_nr64_newbeams_so6_mo10_PPP_alpharx_random_2pi_alphatx_90_1hot_{}label.h5".format(multilabel[z]))
        leg.append('Deep NN ({}-labeled)'.format(multilabel[z]))
        color_l.append(color_list[c])
        marker_l.append(marker_list[c])
        c += 1

    """ Select first 10,000 samples """
    data = data[:10000, :]

    x = np.zeros([np.shape(data)[0], 4])
    x[:, 0] = np.arange(np.shape(data)[0])
    x[:, 1:3] = data[:, :2]
    x[:, 3] = data[:, 2 + 3 * ns_s + 2]
    y = data[:, 2]
    y = y.astype(int)

    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2, random_state=42)
    x_val, x_test, y_val, y_test = train_test_split(x_val_test, y_val_test, test_size=0.5, random_state=42)

    ind_train = x_train[:, 0]
    ind_train = ind_train.astype(int)
    ind_val_test = x_val_test[:, 0]
    ind_val_test = ind_val_test.astype(int)
    ind_val = x_val[:, 0]
    ind_val = ind_val.astype(int)
    ind_test = x_test[:, 0]
    ind_test = ind_test.astype(int)

    x_train = x_train[:, 1:]
    x_val = x_val[:, 1:]
    x_val_test = x_val_test[:, 1:]
    x_test = x_test[:, 1:]

    scaler = StandardScaler()
    scaler.fit(x_train)
    x_train_n = scaler.transform(x_train)
    x_val_n = scaler.transform(x_val)
    x_val_test_n = scaler.transform(x_val_test)

    for k in range(len(bin_size_ang_2)):
        align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect[c], prob_out[c] = \
            inv_fp_3d(top_m, range_pt, bin_size_ang_2[k][0], bin_size_ang_2[k][1], ns_s, ns_t, nb_t, nb_r, top_n, data, ind_train,
                      x_train, y_train, ind_val_test, x_val_test, y_val_test, snr_out)
        leg.append('GIFP (SBS = {} m, ABS = {}°)'.format(bin_size_ang_2[k][0], bin_size_ang_2[k][1]))
        color_l.append(color_list[c - n_rows_1])
        marker_l.append(marker_list[c - n_rows_1])
        c += 1

    for z in range(np.shape(multilabel_2)[0]):
        h, align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect[c], prob_out[c] = \
            deep_nn(ns_s, ns_t, top_n, data, ind_train, x_train_n, y_train, data, ind_val, x_val_n, y_val, ind_val_test, x_val_test_n,
                    y_val_test, multilabel_2[z], 0, snr_out,
                    load_name="./Models/model_XYT_1e4_nt64_nr64_newbeams_so6_mo10_PPP_alpharx_random_2pi_alphatx_90_1hot_{}label.h5".format(multilabel_2[z]))

        leg.append('Deep NN ({}-labeled)'.format(multilabel_2[z]))
        print(c - n_rows_1)
        color_l.append(color_list[c - n_rows_1])
        marker_l.append(marker_list[c - n_rows_1])
        c += 1

    fontsize = 18
    fontweight = 'bold'
    fontproperties = {'size': fontsize}

    """ MisAlignment Probability 1e4 """
    fig0 = plt.figure(figsize=plt.figaspect(0.5))
    lines = []
    for k in range(n_rows_1, n_rows):
        l, = plt.plot(top_n, 1 - align_0db_t[k], color=color_l[k], marker=marker_l[k])
        lines.append(l)
    plt.yscale('log')
    min_val = 100
    for k in range(n_rows_1, n_rows):
        min_val = np.amin([min_val, np.amin(1 - align_0db_t[k])])
    min_val = np.amax([min_val, 1e-4])
    l = 10 ** (math.floor(math.log10(min_val)))
    plt.ylim(0.25, 1)
    plt.xlim(0, 50.5)
    plt.ylabel('Misalignment Probability', fontproperties)
    plt.xlabel('Number of Beam Pairs Scanned', fontproperties)
    ax = gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_minor_ticks():
        tick.label1.set_fontsize(fontsize)
    legend1 = plt.legend(lines[3:], leg[3:], loc=3, prop=fontproperties)
    plt.legend(lines[:3], leg[:3], bbox_to_anchor=(.42, 0.16, 0.5, 0.08), loc='upper left', prop=fontproperties)
    plt.gca().add_artist(legend1)
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', axis='y', linestyle=':')
    plt.show(block=True)

    """ MisAlignment Probability 1e5 """
    fig0 = plt.figure(figsize=plt.figaspect(0.5))
    lines = []
    for k in range(n_rows_1):
        l, = plt.plot(top_n, 1 - align_0db_t[k], color=color_l[k], marker=marker_l[k])
        lines.append(l)
    plt.yscale('log')
    min_val = 100
    for k in range(n_rows_1):
        min_val = np.amin([min_val, np.amin(1 - align_0db_t[k])])
    min_val = np.amax([min_val, 1e-4])
    l = 10 ** (math.floor(math.log10(min_val)))
    # plt.ylim(math.floor(min_val / l) * l, 1)
    plt.ylim(0.095, 1)
    plt.xlim(0, 50.5)
    plt.ylabel('Misalignment Probability', fontproperties)
    plt.xlabel('Number of Beam Pairs Scanned', fontproperties)
    ax = gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_minor_ticks():
        tick.label1.set_fontsize(fontsize)
    legend1 = plt.legend(lines[3:], leg[3:], loc=3, prop=fontproperties)
    plt.legend(lines[:3], leg[:3], bbox_to_anchor=(.48, 0.45, 0.35, 0.25), loc='upper left', prop=fontproperties)
    plt.gca().add_artist(legend1)
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', axis='y', linestyle=':')
    plt.show(block=True)