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
    bin_size_ang = [(5, 10)]  # (meter, degree)
    ns_s = 100  # number of scanned beam pairs with high RSS for saving
    nb_t = 64  # number of TX antennas
    nb_r = 64  # number of RX antennas
    ns_t = nb_t * nb_r  # Number of total beam pairs
    snr_out_dB = 20  # Threshold snr of outage (dB)
    snr_out = 10 ** (snr_out_dB / 10)
    multilabel = [1, 5]
    top_n = np.concatenate((np.arange(1, 10, 1), np.arange(10, 20, 2), np.arange(20, 51, 5)), axis=0)
    marker_list = ['o', 's', 'v', '^', '<', '>', '*', 'D', 'P', 'X', 'd', '8', 'p', 'h', 'H']
    color_list = [[0, 0.4470, 0.7410],
                  [0.8500, 0.3250, 0.0980],
                  [0.9290, 0.6940, 0.1250],
                  [0.4940, 0.1840, 0.5560],
                  [0.4660, 0.6740, 0.1880],
                  [0.3010, 0.7450, 0.9330],
                  [0.6350, 0.0780, 0.1840],
                  [0.2500, 0.2500, 0.2500]]

    print('Loading 1st dataset ...')
    with h5py.File('./Datasets/snr_data_1e5_nt64_nr64_nb_100_newbeams_so6_mo0_PPP_alpharx_random_2pi_alphatx_90.hdf5', 'r') as fr:
        data_r = fr['default']
        s = np.shape(data_r)
        data1 = np.zeros([s[0], s[1]], dtype=float)
        data1[:, ] = data_r[:, ]
        print('size:', np.shape(data_r))
    print('Input data loaded successfully')

    x1 = np.zeros([np.shape(data1)[0], 4])
    x1[:, 0] = np.arange(np.shape(data1)[0])
    x1[:, 1:3] = data1[:, :2]
    x1[:, 3] = data1[:, 2 + 3 * ns_s + 2]
    y1 = data1[:, 2]
    y1 = y1.astype(int)

    x_train1, x_val_test1, y_train1, y_val_test1 = train_test_split(x1, y1, test_size=0.2, random_state=42)
    x_val1, x_test1, y_val1, y_test1 = train_test_split(x_val_test1, y_val_test1, test_size=0.5, random_state=42)

    ind_train1 = x_train1[:, 0]
    ind_train1 = ind_train1.astype(int)
    ind_val_test1 = x_val_test1[:, 0]
    ind_val_test1 = ind_val_test1.astype(int)
    ind_val1 = x_val1[:, 0]
    ind_val1 = ind_val1.astype(int)
    ind_test1 = x_test1[:, 0]
    ind_test1 = ind_test1.astype(int)

    x_train1 = x_train1[:, 1:]
    x_val1 = x_val1[:, 1:]
    x_val_test1 = x_val_test1[:, 1:]
    x_test1 = x_test1[:, 1:]

    scaler1 = StandardScaler()
    scaler1.fit(x_train1)
    x_train1_n1 = scaler1.transform(x_train1)
    x_val1_n1 = scaler1.transform(x_val1)
    x_val_test1_n1 = scaler1.transform(x_val_test1)

    print('Loading 2nd dataset ...')
    with h5py.File('./Datasets/snr_data_1e5_nt64_nr64_nb_100_newbeams_so6_mo30_PPP_alpharx_random_2pi_alphatx_90.hdf5', 'r') as fr:
        data_r = fr['default']
        s = np.shape(data_r)
        data2 = np.zeros([s[0], s[1]], dtype=float)
        data2[:, ] = data_r[:, ]
        print('size:', np.shape(data_r))
    print('Input data loaded successfully')

    x2 = np.zeros([np.shape(data2)[0], 4])
    x2[:, 0] = np.arange(np.shape(data2)[0])
    x2[:, 1:3] = data2[:, :2]
    x2[:, 3] = data2[:, 2 + 3 * ns_s + 2]
    y2 = data2[:, 2]
    y2 = y2.astype(int)

    x_train2, x_val_test2, y_train2, y_val_test2 = train_test_split(x2, y2, test_size=0.2, random_state=42)
    x_val2, x_test2, y_val2, y_test2 = train_test_split(x_val_test2, y_val_test2, test_size=0.5, random_state=42)

    ind_train2 = x_train2[:, 0]
    ind_train2 = ind_train2.astype(int)
    ind_val_test2 = x_val_test2[:, 0]
    ind_val_test2 = ind_val_test2.astype(int)
    ind_val2 = x_val2[:, 0]
    ind_val2 = ind_val2.astype(int)
    ind_test2 = x_test2[:, 0]
    ind_test2 = ind_test2.astype(int)

    x_train2 = x_train2[:, 1:]
    x_val2 = x_val2[:, 1:]
    x_val_test2 = x_val_test2[:, 1:]
    x_test2 = x_test2[:, 1:]

    x_train2_n1 = scaler1.transform(x_train2)
    x_val2_n1 = scaler1.transform(x_val2)
    x_val_test2_n1 = scaler1.transform(x_val_test2)

    scaler2 = StandardScaler()
    scaler2.fit(x_train2)
    x_train2_n2 = scaler2.transform(x_train2)
    x_val2_n2 = scaler2.transform(x_val2)
    x_val_test2_n2 = scaler2.transform(x_val_test2)

    x_train1_n2 = scaler2.transform(x_train1)
    x_val1_n2 = scaler2.transform(x_val1)
    x_val_test1_n2 = scaler2.transform(x_val_test1)

    nb = np.shape(top_n)[0]
    n_rows = 4*(np.shape(multilabel)[0] + len(bin_size_ang))
    align_0db_t = np.zeros([n_rows, nb])
    align_3db_t = np.zeros([n_rows, nb])
    prob_out_perfect = np.zeros([2, nb])
    prob_out = np.zeros([n_rows, nb])
    ave_perfect_align_rate_t = np.zeros([n_rows, nb])
    ave_achievable_rate_t = np.zeros([n_rows, nb])

    leg = []
    color_l = []
    marker_l = []
    c = 0

    """ Train: 2    -    Test: 2 """
    for k in range(len(bin_size_ang)):
        align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect_, \
        prob_out_ = \
            inv_fp_3d(top_m, range_pt, bin_size_ang[k][0], bin_size_ang[k][1], ns_s, ns_t, nb_t, nb_r, top_n, data2,
                      ind_train2, x_train2, y_train2, ind_val_test2, x_val_test2, y_val_test2, snr_out)
        prob_out[c] = prob_out_
        prob_out_perfect[0] = prob_out_perfect_[0]
        leg.append('GIFP (SBS = {} m, ABS = {}°)'.format(bin_size_ang[k][0], bin_size_ang[k][1]))
        color_l.append(color_list[0])
        marker_l.append(marker_list[0])
        c += 1

    for z in range(np.shape(multilabel)[0]):
        h, align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect_, \
        prob_out_ = \
            deep_nn(ns_s, ns_t, top_n, data2, ind_train2, x_train2_n2, y_train2, data2, ind_val2, x_val2_n2, y_val2, ind_val_test2,
                    x_val_test2_n2, y_val_test2, multilabel[z], 0, snr_out,
                    load_name="./Models/model_XYT_1e5_nt64_nr64_newbeams_so6_mo30_PPP_alpharx_random_2pi_alphatx_90_1hot_{}label.h5".format(multilabel[z]))
        prob_out[c] = prob_out_
        c += 1
        leg.append('Deep NN ({}-labeled)'.format(multilabel[z]))
        color_l.append(color_list[0])
        marker_l.append(marker_list[z+1])

    """ Train: 2    -    Test: 1 """
    for k in range(len(bin_size_ang)):
        align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect_, \
        prob_out_ = \
            inv_fp_3d(top_m, range_pt, bin_size_ang[k][0], bin_size_ang[k][1], ns_s, ns_t, nb_t, nb_r, top_n, data1,
                      ind_train2, x_train2, y_train2, ind_val_test1, x_val_test1, y_val_test1, snr_out)
        prob_out[c] = prob_out_
        prob_out_perfect[1] = prob_out_perfect_[0]
        color_l.append(color_list[1])
        marker_l.append(marker_list[0])
        c += 1

    for z in range(np.shape(multilabel)[0]):
        h, align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect_, \
        prob_out_ = \
            deep_nn(ns_s, ns_t, top_n, data2, ind_train2, x_train2_n2, y_train2, data1, ind_val1, x_val1_n2, y_val1, ind_val_test1,
                    x_val_test1_n2,y_val_test1, multilabel[z], 0, snr_out,
                    load_name="./Models/model_XYT_1e5_nt64_nr64_newbeams_so6_mo30_PPP_alpharx_random_2pi_alphatx_90_1hot_{}label.h5".format(multilabel[z]))
        prob_out[c] = prob_out_
        c += 1
        color_l.append(color_list[1])
        marker_l.append(marker_list[z+1])

    """ Train: 1    -    Test: 2 """
    for k in range(len(bin_size_ang)):
        align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect_, \
        prob_out_ = \
            inv_fp_3d(top_m, range_pt, bin_size_ang[k][0], bin_size_ang[k][1], ns_s, ns_t, nb_t, nb_r, top_n, data2,
                      ind_train1, x_train1, y_train1, ind_val_test2, x_val_test2, y_val_test2, snr_out)
        prob_out[c] = prob_out_
        prob_out_perfect[1] = prob_out_perfect_[0]
        color_l.append(color_list[2])
        marker_l.append(marker_list[0])
        c += 1

    for z in range(np.shape(multilabel)[0]):
        h, align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect_, \
        prob_out_ = \
            deep_nn(ns_s, ns_t, top_n, data1, ind_train1, x_train1_n1, y_train1, data2, ind_val2, x_val2_n1, y_val2, ind_val_test2,
                    x_val_test2_n1,y_val_test2, multilabel[z], 0, snr_out,
                    load_name="./Models/model_XYT_1e5_nt64_nr64_newbeams_so6_mo0_PPP_alpharx_random_2pi_alphatx_90_1hot_{}label.h5".format(multilabel[z]))
        prob_out[c] = prob_out_
        c += 1
        color_l.append(color_list[2])
        marker_l.append(marker_list[z+1])

    """ Train: 1    -    Test: 1 """
    for k in range(len(bin_size_ang)):
        align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect_, \
        prob_out_ = \
            inv_fp_3d(top_m, range_pt, bin_size_ang[k][0], bin_size_ang[k][1], ns_s, ns_t, nb_t, nb_r, top_n, data1,
                      ind_train1, x_train1, y_train1, ind_val_test1, x_val_test1, y_val_test1, snr_out)
        prob_out[c] = prob_out_
        prob_out_perfect[1] = prob_out_perfect_[0]
        color_l.append(color_list[3])
        marker_l.append(marker_list[0])
        c += 1

    for z in range(np.shape(multilabel)[0]):
        h, align_0db_t[c], align_3db_t[c], ave_perfect_align_rate_t[c], ave_achievable_rate_t[c], prob_out_perfect_, \
        prob_out_ = \
            deep_nn(ns_s, ns_t, top_n, data1, ind_train1, x_train1_n1, y_train1, data1, ind_val1, x_val1_n1, y_val1, ind_val_test1,
                    x_val_test1_n1,y_val_test1, multilabel[z], 0, snr_out,
                    load_name="./Models/model_XYT_1e5_nt64_nr64_newbeams_so6_mo0_PPP_alpharx_random_2pi_alphatx_90_1hot_{}label.h5".format(multilabel[z]))
        prob_out[c] = prob_out_
        c += 1
        color_l.append(color_list[3])
        marker_l.append(marker_list[z+1])

    """ Plots """
    fontsize = 18
    fontweight = 'bold'
    fontproperties = {'size': fontsize}

    """ Average 0dB Power Loss Probability """
    lines = []
    lines2 = []
    fig0 = plt.figure(figsize=plt.figaspect(0.5))
    for k in range(n_rows):
        plt.plot(top_n, 1 - align_0db_t[k], color=color_l[k], marker=marker_l[k])
    for k in range(n_rows):
        l, = plt.plot(top_n, 0.001 * (1 - align_0db_t[k]), color=color_l[k])
        lines.append(l)
    for k in range(3):
        l2, = plt.plot(top_n, 0.001 * (1 - align_0db_t[k]), color='black', marker=marker_l[k])
        lines2.append(l2)
    plt.yscale('log')
    min_val = 100
    for k in range(n_rows):
        min_val = np.amin([min_val, np.amin(1 - align_0db_t[k])])
    min_val = np.amax([min_val, 1e-4])
    l = 10 ** (math.floor(math.log10(min_val)))
    # plt.ylim(math.floor(min_val/l)*l, 1)
    plt.ylim(0.03, 1)
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
    legend1 = plt.legend([lines[0], lines[2*(len(multilabel) + 1)], lines[len(multilabel) + 1], lines[3*(len(multilabel) + 1)]],
            ["(λ_train, λ_test) = (30, 30)", "(λ_train, λ_test) = (0, 30)", "(λ_train, λ_test) = (30, 0)", "(λ_train, λ_test) = (0, 0)"], loc=3,
        prop=fontproperties)
    plt.legend(lines2, leg, bbox_to_anchor=(0.5, 0.04, 0.4, 0.21), loc='upper left', prop=fontproperties)
    plt.gca().add_artist(legend1)
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', axis='y', linestyle=':')
    plt.show(block=True)

    """ Average 3dB Power Loss Probability """
    lines = []
    fig0 = plt.figure(figsize=plt.figaspect(0.5))
    for k in range(n_rows):
        l, = plt.plot(top_n, 1 - align_3db_t[k], color=color_l[k], marker=marker_l[k])
        lines.append(l)
    plt.yscale('log')
    min_val = 100
    for k in range(n_rows):
        min_val = np.amin([min_val, np.amin(1 - align_0db_t[k])])
    min_val = np.amax([min_val, 1e-4])
    l = 10 ** (math.floor(math.log10(min_val)))
    plt.ylim(0.01, 1)
    plt.xlim(0, 50.5)
    plt.yticks([0.05, 0.1, 0.2, 0.5, 1])
    plt.ylabel('Average 3dB Power Loss Probability', fontproperties)
    plt.xlabel('Number of Beam Pairs Scanned', fontproperties)
    ax = gca()
    for tick in ax.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax.yaxis.get_minor_ticks():
        tick.label1.set_fontsize(fontsize)
    legend1 = plt.legend([lines[0], lines[len(multilabel) + 1], lines[2 * (len(multilabel) + 1)], lines[3 * (len(multilabel) + 1)]],
        ["(λ_train, λ_test) = (30, 30)", "(λ_train, λ_test) = (30, 0)", "(λ_train, λ_test) = (0, 30)", "(λ_train, λ_test) = (0, 0)"], loc=3,
        prop=fontproperties)
    plt.legend(leg, bbox_to_anchor=(.5, 0.1, 0.4, 0.3), loc='upper left', prop=fontproperties)
    plt.gca().add_artist(legend1)
    plt.grid(True, which='major', linestyle='-')
    plt.grid(True, which='minor', axis='y', linestyle=':')
    plt.show(block=True)