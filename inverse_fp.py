"""
Author: Sajad Rezaie (sre@es.aau.dk)
"""
import numpy as np
import h5py, math
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from channel_model import lut
import random
import math

def inv_fp_3d(top_m, range_pt, bin_size, ang_bin, ns_s, ns_t, nb_t, nb_r, top_n, data, ind_train, x_train, y_train, ind_test, x_test, y_test, snr_out):
    num_cell_x_l = np.floor(range_pt[0][0] / bin_size)
    num_cell_x_r = np.floor(range_pt[0][1] / bin_size)
    num_cell_y_d = np.floor(range_pt[1][0] / bin_size)
    num_cell_y_u = np.floor(range_pt[1][1] / bin_size)
    num_cell_x = int(num_cell_x_r - num_cell_x_l + 1)
    num_cell_y = int(num_cell_y_u - num_cell_y_d + 1)
    num_ang_bin = int(np.floor(360/ang_bin) + 1)

    num_data_cell = np.zeros([num_cell_x, num_cell_y, num_ang_bin], dtype=int)
    freq_data_cell = np.zeros([num_cell_x, num_cell_y, num_ang_bin, ns_t], dtype=int)
    top_beams = np.zeros([num_cell_x, num_cell_y, num_ang_bin, top_m], dtype=int)

    print('Start calculation of frequency for each cell')
    for k in range(np.shape(x_train)[0]):
        p = x_train[k, :]
        x_ind = int(np.floor(p[0] / bin_size) - num_cell_x_l)
        y_ind = int(np.floor(p[1] / bin_size) - num_cell_y_d)
        ang = p[2] % 360
        if ang > 180:
            ang = ang - 180
        ang = math.fabs(ang)
        ang_ind = int(ang / ang_bin)
        beam_ind = y_train[k]
        num_data_cell[x_ind, y_ind, ang_ind] += 1
        freq_data_cell[x_ind, y_ind, ang_ind, beam_ind] += 1

    print('Start finding beam candidates for each cell')
    for k in range(num_cell_x):
        for s in range(num_cell_y):
            for t in range(num_ang_bin):
                if num_data_cell[k, s, t] == 0:
                    # ind_sort = random.sample(np.arange(0, nb_t*nb_r-1).tolist(), top_m)
                    ind_sort = np.random.randint(nb_t * nb_r, size=top_m)
                    top_beams[k, s, t, :] = ind_sort
                else:
                    freq = freq_data_cell[k, s, t, :]
                    ind_sort = np.argsort(freq)[::-1][:top_m]
                    zz = 0
                    for z in range(top_m):
                        if freq[ind_sort[z]] == 0:
                            zz = z
                    if zz != 0:
                        ind_sort[zz:] = np.random.randint(nb_t * nb_r, size=top_m-zz)
                    top_beams[k, s, t, :] = ind_sort
    print('Start processing test data ...')
    ind_sort = np.zeros([np.shape(x_test)[0], top_m], dtype=int)
    for m in range(np.shape(x_test)[0]):
        p = x_test[m, :]
        x_ind = int(np.floor(p[0] / bin_size) - num_cell_x_l)
        y_ind = int(np.floor(p[1] / bin_size) - num_cell_y_d)
        ang = p[2] % 360
        if ang > 180:
            ang = ang - 180
        ang = math.fabs(ang)
        ang_ind = int(ang / ang_bin)
        ind_sort[m] = top_beams[x_ind, y_ind, ang_ind]

    print('Start calculating accuracy ...')
    ind_sort_2 = data[ind_test, 2:2 + ns_s]
    rss_2 = data[ind_test, 2 + ns_s:2 + 2 * ns_s]
    snr_2 = data[ind_test, 2 + 2 * ns_s:2 + 3 * ns_s]
    align_0db = np.zeros([np.shape(top_n)[0]], dtype=int)
    align_3db = np.zeros([np.shape(top_n)[0]], dtype=int)
    outage_perfect_num = np.zeros([1], dtype=int)
    outage_num = np.zeros([np.shape(top_n)[0]], dtype=int)
    ave_achievable_rate = np.zeros([np.shape(top_n)[0]], dtype=float)
    ave_perfect_align_rate = np.zeros([1], dtype=float)
    ave_perfect_snr = np.zeros([1], dtype=float)

    for m in range(np.shape(x_test)[0]):
        ind_s = (ind_sort_2[m, :]).astype(int)
        if ns_s != ns_t:
            d = data[ind_test[m], 2 + 3 * ns_s] * np.ones([ns_t], dtype=float)
            d[ind_s] = rss_2[m, :]
            snr = data[ind_test[m], 2 + 3 * ns_s + 1] * np.ones([ns_t], dtype=float)
            snr[ind_s] = snr_2[m, :]
        else:
            d = np.zeros([ns_t], dtype=float)
            d[ind_s] = rss_2[m, :]
            snr = np.zeros([ns_t], dtype=float)
            snr[ind_s] = snr_2[m, :]
        ave_perfect_align_rate[0] += math.log2(1 + snr[y_test[m]])
        ave_perfect_snr[0] += snr[y_test[m]]
        if snr[y_test[m]] < snr_out:
            outage_perfect_num[0] += 1
        # 0dB and 3dB power loss probability
        for k in range(np.shape(top_n)[0]):
            i_max = int(np.argmax(d[ind_sort[m, 0:top_n[k]]], axis=0))
            rss_top_n = d[ind_sort[m, i_max]]
            ave_achievable_rate[k] += math.log2(1 + snr[ind_sort[m, i_max]])
            if rss_top_n == d[y_test[m]]:
                align_0db[k] += 1
            if rss_top_n > 0.5 * d[y_test[m]]:
                align_3db[k] += 1
            if snr[ind_sort[m, i_max]] < snr_out:
                outage_num[k] += 1

    prob_align_0db = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_align_3db = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_out_perfect = np.zeros([1], dtype=float)
    prob_out = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_ave_achievable_rate = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_ave_perfect_align_rate = np.zeros([1], dtype=float)
    num_iter = float(np.shape(x_test)[0])
    for k in range(np.shape(top_n)[0]):
        prob_align_0db[k] = align_0db[k] / num_iter
        prob_align_3db[k] = float(align_3db[k]) / num_iter
        prob_ave_achievable_rate[k] = float(ave_achievable_rate[k]) / num_iter
        prob_out[k] = float(outage_num[k]) / (num_iter + 1e-10)
    prob_out_perfect[0] = float(outage_perfect_num[0]) / (num_iter + 1e-10)
    prob_out_perfect_r = np.repeat(prob_out_perfect, np.shape(top_n)[0])
    prob_ave_perfect_align_rate[0] = float(ave_perfect_align_rate[0]) / num_iter
    prob_ave_perfect_align_rate_r = np.repeat(prob_ave_perfect_align_rate, np.shape(top_n)[0])
    # print('inv_fp - 0dB power loss: ', 1 - prob_align_0db)
    # print('inv_fp - 3dB power loss: ', 1 - prob_align_3db)
    return prob_align_0db, prob_align_3db, prob_ave_perfect_align_rate_r, prob_ave_achievable_rate, \
           prob_out_perfect, prob_out


def inv_fp_3d_xyz(top_m, range_pt, bin_size, ang_bin, ns_s, ns_t, nb_t, nb_r, top_n, data, ind_train, x_train, y_train, ind_test, x_test, y_test, snr_out):
    num_cell_x_l = np.floor(range_pt[0][0] / bin_size)
    num_cell_x_r = np.floor(range_pt[0][1] / bin_size)
    num_cell_y_d = np.floor(range_pt[1][0] / bin_size)
    num_cell_y_u = np.floor(range_pt[1][1] / bin_size)
    num_cell_x = int(num_cell_x_r - num_cell_x_l + 1)
    num_cell_y = int(num_cell_y_u - num_cell_y_d + 1)
    num_ang_bin = int(np.floor(360/ang_bin) + 1)

    num_data_cell = np.zeros([num_cell_x, num_cell_y, num_ang_bin], dtype=int)
    freq_data_cell = np.zeros([num_cell_x, num_cell_y, num_ang_bin, ns_t], dtype=int)
    top_beams = np.zeros([num_cell_x, num_cell_y, num_ang_bin, top_m], dtype=int)

    print('Start calculation of frequency for each cell')
    for k in range(np.shape(x_train)[0]):
        p = x_train[k, :]
        x_ind = int(np.floor(p[0] / bin_size) - num_cell_x_l)
        y_ind = int(np.floor(p[1] / bin_size) - num_cell_y_d)
        ang = p[2] % 360
        if ang > 180:
            ang = ang - 180
        ang = math.fabs(ang)
        ang_ind = int(ang / ang_bin)
        beam_ind = y_train[k]
        num_data_cell[x_ind, y_ind, ang_ind] += 1
        freq_data_cell[x_ind, y_ind, ang_ind, beam_ind] += 1

    print('Start finding beam candidates for each cell')
    for k in range(num_cell_x):
        for s in range(num_cell_y):
            for t in range(num_ang_bin):
                if num_data_cell[k, s, t] == 0:
                    # ind_sort = random.sample(np.arange(0, nb_t*nb_r-1).tolist(), top_m)
                    ind_sort = np.random.randint(nb_t * nb_r, size=top_m)
                    top_beams[k, s, t, :] = ind_sort
                else:
                    freq = freq_data_cell[k, s, t, :]
                    ind_sort = np.argsort(freq)[::-1][:top_m]
                    zz = 0
                    for z in range(top_m):
                        if freq[ind_sort[z]] == 0:
                            zz = z
                    if zz != 0:
                        ind_sort[zz:] = np.random.randint(nb_t * nb_r, size=top_m-zz)
                    top_beams[k, s, t, :] = ind_sort
    print('Start processing test data ...')
    ind_sort = np.zeros([np.shape(x_test)[0], top_m], dtype=int)
    for m in range(np.shape(x_test)[0]):
        p = x_test[m, :]
        x_ind = int(np.floor(p[0] / bin_size) - num_cell_x_l)
        y_ind = int(np.floor(p[1] / bin_size) - num_cell_y_d)
        ang = p[2] % 360
        if ang > 180:
            ang = ang - 180
        ang = math.fabs(ang)
        ang_ind = int(ang / ang_bin)
        ind_sort[m] = top_beams[x_ind, y_ind, ang_ind]

    print('Start calculating accuracy ...')
    ind_sort_2 = data[ind_test, 3:3 + ns_s]
    rss_2 = data[ind_test, 3 + ns_s:3 + 2 * ns_s]
    snr_2 = data[ind_test, 3 + 2 * ns_s:3 + 3 * ns_s]
    align_0db = np.zeros([np.shape(top_n)[0]], dtype=int)
    align_3db = np.zeros([np.shape(top_n)[0]], dtype=int)
    outage_perfect_num = np.zeros([1], dtype=int)
    outage_num = np.zeros([np.shape(top_n)[0]], dtype=int)
    ave_achievable_rate = np.zeros([np.shape(top_n)[0]], dtype=float)
    ave_perfect_align_rate = np.zeros([1], dtype=float)
    ave_perfect_snr = np.zeros([1], dtype=float)

    for m in range(np.shape(x_test)[0]):
        ind_s = (ind_sort_2[m, :]).astype(int)
        if ns_s != ns_t:
            d = data[ind_test[m], 3 + 3 * ns_s] * np.ones([ns_t], dtype=float)
            d[ind_s] = rss_2[m, :]
            snr = data[ind_test[m], 3 + 3 * ns_s + 1] * np.ones([ns_t], dtype=float)
            snr[ind_s] = snr_2[m, :]
        else:
            d = np.zeros([ns_t], dtype=float)
            d[ind_s] = rss_2[m, :]
            snr = np.zeros([ns_t], dtype=float)
            snr[ind_s] = snr_2[m, :]
        ave_perfect_align_rate[0] += math.log2(1 + snr[y_test[m]])
        ave_perfect_snr[0] += snr[y_test[m]]
        if snr[y_test[m]] < snr_out:
            outage_perfect_num[0] += 1
        # 0dB and 3dB power loss probability
        for k in range(np.shape(top_n)[0]):
            i_max = int(np.argmax(d[ind_sort[m, 0:top_n[k]]], axis=0))
            rss_top_n = d[ind_sort[m, i_max]]
            ave_achievable_rate[k] += math.log2(1 + snr[ind_sort[m, i_max]])
            if rss_top_n == d[y_test[m]]:
                align_0db[k] += 1
            if rss_top_n > 0.5 * d[y_test[m]]:
                align_3db[k] += 1
            if snr[ind_sort[m, i_max]] < snr_out:
                outage_num[k] += 1

    prob_align_0db = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_align_3db = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_out_perfect = np.zeros([1], dtype=float)
    prob_out = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_ave_achievable_rate = np.zeros([np.shape(top_n)[0]], dtype=float)
    prob_ave_perfect_align_rate = np.zeros([1], dtype=float)
    num_iter = float(np.shape(x_test)[0])
    for k in range(np.shape(top_n)[0]):
        prob_align_0db[k] = align_0db[k] / num_iter
        prob_align_3db[k] = float(align_3db[k]) / num_iter
        prob_ave_achievable_rate[k] = float(ave_achievable_rate[k]) / num_iter
        prob_out[k] = float(outage_num[k]) / (num_iter + 1e-10)
    prob_out_perfect[0] = float(outage_perfect_num[0]) / (num_iter + 1e-10)
    prob_out_perfect_r = np.repeat(prob_out_perfect, np.shape(top_n)[0])
    prob_ave_perfect_align_rate[0] = float(ave_perfect_align_rate[0]) / num_iter
    prob_ave_perfect_align_rate_r = np.repeat(prob_ave_perfect_align_rate, np.shape(top_n)[0])
    # print('inv_fp - 0dB power loss: ', 1 - prob_align_0db)
    # print('inv_fp - 3dB power loss: ', 1 - prob_align_3db)
    return prob_align_0db, prob_align_3db, prob_ave_perfect_align_rate_r, prob_ave_achievable_rate, \
           prob_out_perfect, prob_out


if __name__ == '__main__':
    top_m = 100
    range_pt = [[0,50], [-50,50]]   # [[xmin, xmax], [ymin, ymax]]
    bin_size = 1  # meter
    ns_s = 100
    ns_t = 64 * 64
    top_n = np.arange(1, 52, 5)

    print('Loading data ...')
    with h5py.File('snr_data_1e6_nt64_nr64_nb_100_so6_mo10.hdf5', 'r') as fr:
        data_r = fr['default']
        s = np.shape(data_r)
        data = np.zeros([s[0], s[1]], dtype=float)
        data[:, ] = data_r[:, ]
        print('size:', np.shape(data_r))
    print('Input data loaded successfully')

    x = np.zeros([np.shape(data)[0], 3])
    x[:, 0] = np.arange(np.shape(data)[0])
    x[:, 1:] = data[:, :2]
    y = data[:, 2]
    y = y.astype(int)

    x_train, x_val_test, y_train, y_val_test = train_test_split(x, y, test_size=0.2, random_state=42)

    ind_train = x_train[:, 0]
    ind_train = ind_train.astype(int)

    ind_val_test = x_val_test[:, 0]
    ind_val_test = ind_val_test.astype(int)

    x_train = x_train[:, 1:]
    x_val_test = x_val_test[:, 1:]

    align_0db, align_3db, ave_optimal_rate_r, ave_perfect_align_rate_r, ave_achievable_rate = inv_fp(top_m, range_pt,
                bin_size, ns_s, ns_t, top_n, data, ind_train, x_train, y_train, ind_val_test, x_val_test, y_val_test)

    # average 0dB and 3dB Power Loss Probability
    fig1 = plt.figure(figsize=plt.figaspect(0.5))
    plt.plot(top_n, 1 - align_0db, marker='o')
    plt.plot(top_n, 1 - align_3db, marker='s')
    plt.yscale('log')
    plt.ylim(0.001, 1)
    plt.ylabel('Average Power Loss Probability')
    plt.xlabel('Number of Beam Pairs Scanned')
    plt.legend(['0dB Power Loss', '3dB Power Loss'], loc='upper right')
    plt.grid(True)
    plt.show(block=False)
    # average achievable rate and optimal rate
    fig2 = plt.figure(figsize=plt.figaspect(0.5))
    plt.plot(top_n, ave_achievable_rate, marker='o')
    plt.plot(top_n, ave_perfect_align_rate_r, lineStyle='--')
    plt.plot(top_n, ave_optimal_rate_r)
    plt.ylabel('Average Rate (b/s/Hz)')
    plt.xlabel('Number of Beam Pairs Scanned')
    plt.legend(['Achievable Rate', 'Perfect Alignment', 'Optimal'], loc='lower right')
    plt.grid(True)
    plt.show(block=True)




