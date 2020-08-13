"""
Author: Sajad Rezaie (sre@es.aau.dk)
"""
import matplotlib.pyplot as plt
import random, math, time, datetime
import numpy as np
from system_model import *
from channel_model import *
from mpl_toolkits.mplot3d import axes3d, Axes3D
import multiprocessing
import h5py


global range_pt, range_s, tx_coord, stc_coord, num_mob_obj, nb_t, nb_r, noise_p, show_plots, z_order
# Define parameters
c = 3e8  # speed of light meter / s
f = 6e10  # carrier frequency
bw = 1e9  # Bandwidth
n0 = -174  # Noise spectral density (dBm)
nf = 0  # Noise figure (dB)
pl_exp = 2  # Path loss exponent
tp_dbm = 24  # power of transmitter (dbm)
nt = 64  # number of TX antennas
nr = 64  # number of RX antennas
nb_t = nt  # number of beams in dictionary (codebook) at TX
nb_r = nr  # number of beams in dictionary (codebook) at RX
ns_s = 100  # number of scanned beam pairs with high RSS for saving
min_pass_coef = Obj.pass_coef**1
alpha_tx = 90  # BS orientationx
tx_coord = ['cir',np.array([0, 0]), [0.2], 0]
stc_coord = [['rec',np.array([-2, -31]), [1,62], 0], ['rec',np.array([-0.95, 30]), [39.9,1], 0],
             ['rec',np.array([39, -31]), [1,62], 0], ['rec',np.array([-0.95, -31]), [39.9,1], 0],
             ['rec',np.array([15, 15]), [5,1], 0], ['rec',np.array([20, -25]), [1,5], 0]]

lambda_mob_obj = 10  # average number of mobile objects
mob_obj_shape = 'rec'  # shape of mobile objects, 'rec' or 'cir'
range_pt = [[0,40], [-30,30]]   # range of center of mobile objects, [[xmin, xmax], [ymin, ymax]]
range_s = [0.35, 0.6]      # size of each mobile object, [length in x axis, length in y axis]

num_iter = 1e5  # number of iteration
num_sub_section = 1000  # number of divided iterations
num_sub_save = 100  # save generated dataset each "num_sub_save" iterations
show_plots = 1  # flag to show plots


# Calculate transmitter power and noise power
tp = 10 ** (tp_dbm / 10)  # power of transmitter (mW)
noise_p_dbm = n0 + 10*math.log10(bw) + nf  # power of noise at receiver (dBm)
noise_p = 10 ** (noise_p_dbm/10)  # power of noise at receiver (mW)


def rss_realization(kk):
    global range_pt, range_s, tx_coord, stc_coord, num_mob_obj, nb_t, nb_r, noise_p, show_plots, z_order

    """ Define the transmitter (BS) object """
    tic1 = time.perf_counter()
    tx_obj = Obj(tx_coord[0], tx_coord[1], tx_coord[2], tx_coord[3])
    tx_obj.ref_coef = 1

    """ Define the static objects according to their coordinates """
    stc_obj_list = []
    for k in range(len(stc_coord)):
        stc_c = stc_coord[k]
        stc_obj_list.append(StcObj(stc_c[0], stc_c[1], stc_c[2], stc_c[3]))
        if stc_obj_list[len(stc_obj_list) - 1].intersect_obj(tx_obj):
            print(len(stc_obj_list) - 1)
            raise Exception('Error in intersection between transmitter and static object {}'.format(k))
        for m in range(len(stc_obj_list) - 1):
            if stc_obj_list[m].intersect_obj(stc_obj_list[len(stc_obj_list) - 1]):
                raise Exception('Error in intersection between static objects {} and {}'.format(m, k))

    """ Define the receiver (UE) object """
    while 1:
        cnt_pt_x = random.uniform(range_pt[0][0], range_pt[0][1])  # random center point of the UE in x direction
        cnt_pt_y = random.uniform(range_pt[1][0], range_pt[1][1])  # random center point of the UE in y direction
        rx_coord = ['cir', np.array([cnt_pt_x, cnt_pt_y]), [1], 0]
        alpha_rx = random.uniform(0, 2 * math.pi)  # random orientation of the UE
        rx_obj = Obj(rx_coord[0], rx_coord[1], rx_coord[2], rx_coord[3])
        obj_list_dummy = [tx_obj]
        obj_list_dummy.extend(stc_obj_list)
        if check_intersect_obj_list(rx_obj, obj_list_dummy) is False:
            break

    """ Define mobile objects """
    mob_obj_list = []
    obj_list_dummy = [tx_obj, rx_obj]
    obj_list_dummy.extend(stc_obj_list)
    num_mob_obj = np.random.poisson(lambda_mob_obj, 1)  # number of the mobile objects
    while len(mob_obj_list) < num_mob_obj:
        cnt_pt_x = random.uniform(range_pt[0][0],range_pt[0][1])  # random center point of the mobile object in x direction
        cnt_pt_y = random.uniform(range_pt[1][0], range_pt[1][1])  # random center point of the mobile object in x direction
        if mob_obj_shape == 'rec':
            length = range_s[0]
            width = range_s[1]
            ang = random.uniform(0, 180)
            mob_obj1 = MobObj('rec', np.array([cnt_pt_x, cnt_pt_y]), [length,width], ang)
        else:
            radius = range_s[0]
            mob_obj1 = MobObj('cir', np.array([cnt_pt_x, cnt_pt_y]), [radius], 0)
        if check_intersect_obj_list(mob_obj1, obj_list_dummy) is True:
            continue
        mob_obj_list.append(mob_obj1)
        obj_list_dummy.append(mob_obj1)

    """ Find reflection point of each object and check blockage """
    obj_list = stc_obj_list + mob_obj_list  # The list includes all the objects
    ref_params = []
    ref_coef_list = []
    pass_coef_list = []

    """ Blockage of LOS """
    p_list = [tx_obj.pt, rx_obj.pt]
    k_list = [-1]
    pass_coef = check_blockage_obj_list(p_list, obj_list, k_list, min_pass_coef)
    if pass_coef is not None:
        ref_p = {"ref_pt": [tx_obj.pt], "corners_pts": []}
        ref_params.append(ref_p)
        ref_coef_list.append(tx_obj.ref_coef)
        pass_coef_list.append(pass_coef)

    """ 1st order reflection from the static and mobile objects and check blockage """
    for k1 in range(len(obj_list)):
        obj_list_dummy = [obj_list[k1]]
        ref_p = find_ref_pt1(obj_list_dummy, tx_obj, rx_obj)
        if ref_p is not None:
            p_list = [tx_obj.pt]
            p_list.extend(ref_p['ref_pt'])
            p_list.append(rx_obj.pt)
            k_list = [k1]
            pass_coef = check_blockage_obj_list(p_list, obj_list, k_list, min_pass_coef)
            if pass_coef is not None:
                ref_params.append(ref_p)
                ref_coef_list.append(obj_list[k1].ref_coef)
                pass_coef_list.append(pass_coef)

    """ 2nd order reflection from static and mobile objects and check blockage """
    for k1 in range(len(obj_list)):
        for k2 in range(k1, len(obj_list)):
            if k2 == k1 or k2 >= len(stc_obj_list):
                continue
            obj_list_dummy = [obj_list[k1], obj_list[k2]]
            ref_p = find_ref_pt2(obj_list_dummy, tx_obj, rx_obj)
            if ref_p is not None:
                p_list = [tx_obj.pt]
                p_list.extend(ref_p['ref_pt'])
                p_list.append(rx_obj.pt)
                k_list = [k1, k2]
                pass_coef = check_blockage_obj_list(p_list, obj_list, k_list, min_pass_coef)
                if pass_coef is not None:
                    ref_params.append(ref_p)
                    ref_coef_list.append(obj_list[k1].ref_coef * obj_list[k2].ref_coef)
                    pass_coef_list.append(pass_coef)

    """ 3rd order reflection from static and mobile objects and check blockage"""
    for k1 in range(len(obj_list)):
        for k2 in range(k1, len(obj_list)):
            for k3 in range(k2, len(obj_list)):
                if k2 >= len(stc_obj_list) or k3 >= len(stc_obj_list):
                    continue
                obj_list_dummy = [obj_list[k1], obj_list[k2], obj_list[k3]]
                ref_p = find_ref_pt3(obj_list_dummy, tx_obj, rx_obj)
                if ref_p is not None:
                    p_list = [tx_obj.pt]
                    p_list.extend(ref_p['ref_pt'])
                    p_list.append(rx_obj.pt)
                    k_list = [k1, k2, k3]
                    pass_coef = check_blockage_obj_list(p_list, obj_list, k_list, min_pass_coef)
                    if pass_coef is not None:
                        ref_params.append(ref_p)
                        ref_coef_list.append(obj_list[k1].ref_coef * obj_list[k2].ref_coef * obj_list[k3].ref_coef)
                        pass_coef_list.append(pass_coef)

    """ Channel response """
    h_ch, toa_list, aod_list, aoa_list, coef_list = channel_resp(nt, nr, tx_obj.pt, rx_obj.pt, alpha_tx, alpha_rx,
                                                                 ref_params, ref_coef_list, pass_coef_list, c, f, pl_exp)

    """ Codebook Beamforming """
    ut, ur = create_dictionary(nt, nr, nb_t, nb_r)
    ur_h = np.conj(np.transpose(ur))

    """ Add noise """
    noise = math.sqrt(noise_p) * (np.random.normal(0, 1 / math.sqrt(2), [nb_t * nb_r, nr]) +
                                  1j * np.random.normal(0, 1 / math.sqrt(2), [nb_t * nb_r, nr]))
    noise_r = np.reshape(np.sum(np.multiply(np.repeat(ur_h, nb_t, axis=0), noise), axis=1), [nb_t, nb_r])

    """ Received signal strength at each codebook """
    rss = np.square(np.absolute(np.transpose(math.sqrt(tp) * np.matmul(np.matmul(ur_h, h_ch), ut)) + noise_r))
    s_p = np.square(np.absolute(np.transpose(math.sqrt(tp) * np.matmul(np.matmul(ur_h, h_ch), ut))))
    snr = s_p/(noise_p+1e-40)
    rss_2 = np.ravel(rss)
    snr_2 = np.ravel(snr)
    alpha_rx_deg = math.degrees(alpha_rx)

    """ return results """
    if ns_s == nb_t * nb_r:
        ind_sort = np.arange(0, ns_s)
        ave_rss = 0
        ave_snr = 0
        if show_plots == 0:
            return k, rx_obj.pt, ind_sort, rss_2[ind_sort], snr_2[ind_sort], ave_rss, ave_snr, alpha_rx_deg
        else:
            return k, rx_obj.pt, ind_sort, rss_2[ind_sort], snr_2[ind_sort], ave_rss, ave_snr, alpha_rx_deg, tx_obj, \
                   rx_obj, stc_obj_list, mob_obj_list, ref_params
    else:
        ind_sort = np.argsort(-rss_2)[:ns_s]
        ave_rss = (np.sum(rss_2) - np.sum(rss_2[ind_sort]))/(np.shape(rss_2)[0] - np.shape(ind_sort)[0] + 1e-30)
        ave_snr = (np.sum(snr_2) - np.sum(snr_2[ind_sort])) / (np.shape(snr_2)[0] - np.shape(ind_sort)[0] + 1e-30)
        if show_plots == 0:
            return k, rx_obj.pt, ind_sort, rss_2[ind_sort], snr_2[ind_sort], ave_rss, ave_snr, alpha_rx_deg
        else:
            return k, rx_obj.pt, ind_sort, rss_2[ind_sort], snr_2[ind_sort], ave_rss, ave_snr, alpha_rx_deg, tx_obj, \
                   rx_obj, stc_obj_list, mob_obj_list, ref_params


def plot_rss(fig, rss, block = None):
    nb_t = rss.shape[0]
    nb_r = rss.shape[1]
    x = np.zeros(nb_t)
    if nb_t > 1:
        aa = np.arange(0, nb_t)
        aa = 1 - 2 * aa / (nb_t)  # dictionary of spatial frequencies
        for k in range(nb_t):
            x[k] = math.degrees(math.acos(aa[k]))
    y = np.zeros(nb_r)
    if nb_r > 1:
        aa = np.arange(0, nb_r)
        aa = 1 - 2 * aa / (nb_r)  # dictionary of spatial frequencies
        for k in range(nb_r):
            y[k] = math.degrees(math.acos(aa[k]))
    x_g, y_g = np.meshgrid(x, y)  # Set up grid
    x_g = np.transpose(x_g)
    y_g = np.transpose(y_g)
    ax = fig.add_subplot(1, 2, 1)
    pos1 = ax.get_position()  # get the original position
    pos2 = [pos1.x0 - 0.1, pos1.y0, 1.25 * pos1.width, pos1.height]
    ax.set_position(pos2)  # set a new position
    plt.imshow(rss, cmap='cool', interpolation='nearest')
    # ax.plot_surface(x_g, y_g, rss, cmap='jet', linewidth=0.5)
    ax.set_xlabel('Index of RX precoder', fontsize=10, rotation=0); ax.set_ylabel('Index of TX combiner')
    # ax.set_zlabel('RSS (mW)', fontsize=15, rotation=90); ax.yaxis._axinfo['label']['space_factor'] = 3.0
    # set up the axes for the second plot
    ax = fig.add_subplot(1, 2, 2, projection='3d')
    pos1 = ax.get_position()  # get the original position
    pos2 = [pos1.x0 - 0.1, pos1.y0, 1.25 * pos1.width, pos1.height]
    ax.set_position(pos2)  # set a new position
    x = np.ravel(x_g)
    y = np.ravel(y_g)
    z = np.ravel(rss)
    surf = ax.scatter(x, y, z, c=z, cmap='cool', linewidth=0.5)
    ax.set_xlabel('Angle of TX precoder', fontsize=10, rotation=0); ax.set_ylabel('Angle of RX combiner')
    ax.set_zlabel('RSS (mW)', fontsize=15, rotation=90); ax.yaxis._axinfo['label']['space_factor'] = 6.0
    cbaxes = hf.add_axes([0.92, 0.2, 0.03, 0.6]); hf.colorbar(surf, shrink=0.2, aspect=20, cax=cbaxes)
    if block is not False:
        plt.show()
    else:
        plt.show(block=False)


if __name__ == '__main__':
    # global z_order
    pool = multiprocessing.Pool(processes=4)
    # Iteration
    print('\n Start processing ... ')
    tic = time.perf_counter()
    num_i_loop = int(np.ceil(num_iter/num_sub_section))
    save_params = np.zeros([num_i_loop * num_sub_section, 2 + 3 * ns_s + 3], dtype=float)
    for s_loop in range(num_sub_section):
        results = list(pool.map(rss_realization, np.arange(num_i_loop)))
        toc = time.perf_counter()
        print('\nloop index:', s_loop, 'processing time: ', str(datetime.timedelta(seconds=round(toc - tic))),
              'remaining time: ', str(datetime.timedelta(seconds=round((num_sub_section - 1 - s_loop)*(toc - tic)/(s_loop+1)))))
        for k in range(len(results)):
            save_params[s_loop * num_i_loop + k][:2] = results[k][1]  # rx_obj.pt
            save_params[s_loop * num_i_loop + k][2:2 + ns_s] = results[k][2]  # Ind_sort
            save_params[s_loop * num_i_loop + k][2 + ns_s:2 + 2 * ns_s] = results[k][3]  # rss
            save_params[s_loop * num_i_loop + k][2 + 2 * ns_s:2 + 3 * ns_s] = results[k][4]  # snr
            save_params[s_loop * num_i_loop + k][2 + 3 * ns_s] = results[k][5]  # ave_rss
            save_params[s_loop * num_i_loop + k][2 + 3 * ns_s + 1] = results[k][6]  # ave_snr
            save_params[s_loop * num_i_loop + k][2 + 3 * ns_s + 2] = results[k][7]  # ave_snr

        """ Save Dataset """
        if s_loop % num_sub_save == 0 or s_loop == (num_sub_section - 1):
            with h5py.File('snr_data.hdf5', 'w') as f:
                dset = f.create_dataset("default", data=save_params[0:(s_loop + 1) * num_i_loop])
            print('save dataset, s_loop: ', s_loop)

    if show_plots == 1:
        ind_res = len(results)-1
        # Plot objects
        fig1 = plt.figure(); ax1 = fig1.add_subplot(111, aspect='equal')
        ax1.grid(b=True, which='major', axis='both')
        plot_obj(ax1, results[ind_res][8]); plot_obj(ax1, results[ind_res][9])
        for k in range(len(results[ind_res][10])):
            plot_obj(ax1, results[ind_res][10][k])
        for m in range(len(results[ind_res][11])):
            plot_obj(ax1, results[ind_res][11][m])

        """ Plot rays """
        nums = np.zeros((3, 1))
        for k in range(len(results[ind_res][12])):
            if results[ind_res][12][k] is not None:
                r = results[ind_res][12][k]['ref_pt']
                nums[len(r) - 1] += 1
                if len(r) == 1:
                    plot_ray(ax1, results[ind_res][8].pt, r[0], results[ind_res][9].pt, 'm')
                elif len(r) == 2:
                    plot_ray(ax1, results[ind_res][8].pt, r[0], r[1], 'g')
                    plot_ray(ax1, r[0], r[1], results[ind_res][9].pt, 'g')
                else:
                    plot_ray(ax1, results[ind_res][8].pt, r[0], r[1], 'b')
                    plot_ray(ax1, r[1], r[2], results[ind_res][9].pt, 'b')
        print('nums', nums)
        plt.xlim((-5, 45)); plt.ylim((-35, 35))
        plt.show(block=True)

        """ plot RSS for all codebooks """
        if ns_s == nb_t * nb_r:
            hf = plt.figure(figsize=plt.figaspect(0.5))
            plot_rss(hf, np.reshape(results[ind_res][3], [nb_t, nb_r]), True)

