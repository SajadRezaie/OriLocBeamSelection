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
    range_pt = [[0,50], [-50,50]]   # [[xmin, xmax], [ymin, ymax]]
    bin_size_ang = [(5, 10)] # (meter, degree)
    ns_s = 100  # number of scanned beam pairs with high RSS for saving
    nb_t = 64  # number of TX antennas
    nb_r = 64  # number of RX antennas
    ns_t = nb_t * nb_r  # Number of total beam pairs
    snr_out_dB = 20  # Threshold snr of outage (dB)
    snr_out = 10**(snr_out_dB / 10)
    multilabel = [5, 2, 1]
    mob_objs = [0, 5, 10, 20, 30]
    top_n = [40]
    marker_list = ['o', 's', 'D', 'v', '^', '<', '>', '*', 'P', 'X', 'd', '8', 'p', 'h', 'H']
    color_list = [[0,    0.4470,    0.7410],
                    [0.8500,    0.3250,    0.0980],
                    [0.9290,    0.6940,    0.1250],
                    [0.4940,    0.1840,    0.5560],
                    [0.4660,    0.6740,    0.1880],
                    [0.3010,    0.7450,    0.9330],
                    [0.6350,    0.0780,    0.1840],
                    [0.2500,    0.2500,    0.2500]]
    nb = np.shape(mob_objs)[0]
    n_rows = np.shape(multilabel)[0] + 1
    align_0db_t = np.zeros([n_rows, nb])
    align_3db_t = np.zeros([n_rows, nb])
    prob_out_perfect = np.zeros([nb])
    prob_out = np.zeros([n_rows, nb])
    ave_perfect_align_rate_t = np.zeros([nb])
    ave_achievable_rate_t = np.zeros([n_rows, nb])

    c = 0
    for kk in range(len(mob_objs)):
        print('Loading data ...', mob_objs[kk])
        with h5py.File('./Datasets/snr_data_1e5_nt64_nr64_nb_100_newbeams_so6_mo{}_PPP_alpharx_random_2pi_alphatx_90.hdf5'.format(mob_objs[kk]),'r') as fr:
                data_r = fr['default']
                s = np.shape(data_r)
                data = np.zeros([s[0], s[1]], dtype=float)
                data[:, ] = data_r[:, ]
                print('size:', np.shape(data_r))

        print('Input data loaded successfully', mob_objs[kk])

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

        for z in range(np.shape(multilabel)[0]):
            h, align_0db_t_, align_3db_t_, ave_perfect_align_rate_t_, ave_achievable_rate_t_, prob_out_perfect_, prob_out_ = \
                deep_nn(ns_s, ns_t, top_n, data, ind_train, x_train_n, y_train, data, ind_val, x_val_n, y_val, ind_val_test,
                        x_val_test_n,
                        y_val_test, multilabel[z], 0, snr_out,
                        load_name="./Models/model_XYT_1e5_nt64_nr64_newbeams_so6_mo{}_PPP_alpharx_random_2pi_alphatx_90_1hot_{}label.h5".format(mob_objs[kk], multilabel[z]))
            align_0db_t[z, c] = align_0db_t_
            ave_achievable_rate_t[z, c] = ave_achievable_rate_t_
            ave_perfect_align_rate_t[c] = ave_perfect_align_rate_t_[0]

            if prob_out_ <= 25 / 20000:  # This probability is not valid
                prob_out[z, c] = 0
            else:
                prob_out[z, c] = prob_out_

            if prob_out_perfect_[0] <= 25/20000:  # This probability is not valid
                prob_out_perfect[c] = 0
            else:
                prob_out_perfect[c] = prob_out_perfect_[0]
        k = 0

        align_0db_t_, align_3db_t_, ave_perfect_align_rate_t_, ave_achievable_rate_t_, prob_out_perfect_, prob_out_ = \
            inv_fp_3d(top_m, range_pt, bin_size_ang[k][0], bin_size_ang[k][1], ns_s, ns_t, nb_t, nb_r, top_n, data, ind_train,
                      x_train, y_train, ind_val_test, x_val_test, y_val_test, snr_out)
        align_0db_t[n_rows-1, c] = align_0db_t_
        ave_achievable_rate_t[n_rows-1, c] = ave_achievable_rate_t_
        if prob_out_ <= 25 / 20000:
            prob_out[n_rows-1, c] = 0
        else:
            prob_out[n_rows-1, c] = prob_out_
        print('mob_objs: ', mob_objs[kk], prob_out_perfect[c], prob_out[:, c])
        print('****', ave_perfect_align_rate_t[c], ave_achievable_rate_t[:, c])
        c += 1

    leg = []
    color_l = []
    marker_l = []

    for mm in range(np.shape(multilabel)[0]):
        leg.append('Deep NN ({}-labeled)'.format(multilabel[mm]))
    leg.append('GIFP (SBS = {} m, ABS = {}°)'.format(bin_size_ang[k][0], bin_size_ang[k][1]))

    fontsize = 18
    fontweight = 'bold'
    fontproperties = {'size': fontsize}
    fontproperties_leg = {'size': fontsize}

    """ Average Outage Probability and average rate """
    lines = []
    fig2, ax1 = plt.subplots()
    l, = ax1.plot(mob_objs, ave_perfect_align_rate_t, color='black', lineStyle='--')
    lines.append(l)
    for k in range(n_rows):
        l, = ax1.plot(mob_objs, ave_achievable_rate_t[k], color='black', marker=marker_list[k])
        lines.append(l)
    plt.yticks(np.arange(10, 16))
    plt.ylim(10, 15)
    ax2 = ax1.twinx()
    for k in range(n_rows):
        d = prob_out[k]
        c = 0
        for z in range(len(prob_out[k])):
            if d[z] == 0:
                c = z + 1

        ax2.plot(mob_objs[c:], d[c:], color=color_list[3], marker=marker_list[k])
    plt.yscale('log')

    min_val = np.amin(prob_out_perfect)
    min_val = np.amax([min_val, 1e-3])
    plt.ylim(10 ** (math.floor(math.log10(min_val))), 1e-1)
    ax1.set_xlabel('λ', fontproperties)
    ax1.set_ylabel('Spectral Efficiency [bps/Hz]', fontproperties)
    ax2.set_ylabel('Outage Probability', fontproperties, color=color_list[3])
    ax2.tick_params(axis='y', labelcolor=color_list[3])
    for tick in ax1.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax1.yaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax2.xaxis.get_major_ticks():
        tick.label1.set_fontsize(fontsize)
    for tick in ax2.yaxis.get_major_ticks():
        tick.label2.set_fontsize(fontsize)
    for tick in ax2.yaxis.get_minor_ticks():
        tick.label2.set_fontsize(fontsize)
    leg_2 = ['Perfect Alignment']
    leg_2.extend(leg)
    plt.legend(lines, leg_2, bbox_to_anchor=(.2, 0.1, 0.5, 0.3), loc='upper left', prop=fontproperties_leg)
    plt.grid(True)
    plt.show(block=True)