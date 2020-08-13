"""
Author: Sajad Rezaie (sre@es.aau.dk)
"""
import random
import math, cmath
import numpy as np
from sympy.geometry import Point
import matplotlib.pyplot as plt


def path_gains(pos_tx, pos_rx, pos_ref_list, c, f, pl_exp):
    """
    Calculating path gain according to the reflection points
    Inputs:
              pos_tx: position of transmitter
              pos_rx: position of receiver
              pos_ref_list: position of all reflection points for all paths
              c: the speed of light
              f: carrier frequency
              pl_exp: path loss exponent
    Output:
              sigma: path loss exponent
    """
    sigma = []
    for k in range(len(pos_ref_list)):
        ref_list = pos_ref_list[k]['ref_pt']
        d = np.linalg.norm(ref_list[0] - pos_tx)  # Distance between TX and the first reflection point
        for z in range(len(ref_list)-1):
            d += np.linalg.norm(ref_list[z+1] - ref_list[z])  # Distance between zth and (z+1)th reflection points
        d += np.linalg.norm(pos_rx - ref_list[len(ref_list)-1])  # Distance between the last reflection point and RX
        path_g = (c/(4 * math.pi * f * d))**(pl_exp/2)  # Path gain of kth path
        sigma.append(path_g)
    return sigma


def channel_params(pos_tx, pos_rx, alpha_tx, alpha_rx, pos_ref, c):
    """
    Calculating time of arrival, angle of departure and angle of arrival for a path
    Inputs:
              pos_tx: position of transmitter
              pos_rx: position of receiver
              alpha_tx: angle of antenna elements of transmitter
              alpha_rx: angle of antenna elements of receiver
              pos_ref: position of reflection points
              c: the speed of light
    Output:
              toa: time of arrival
              aod: angle of departure
              aoa: angle of arrival
    """
    if pos_tx[0] == pos_ref[0][0] and pos_tx[1] == pos_ref[0][1]:
        toa = np.linalg.norm(pos_tx - pos_rx) / c  # time of arrival
        aod = math.atan2(pos_rx[1] - pos_tx[1], pos_rx[0] - pos_tx[0]) - alpha_tx # angle of departure
        aoa = math.atan2(pos_tx[1] - pos_rx[1], pos_tx[0] - pos_rx[0]) - alpha_rx  # angle of arrival
    else:
        d = np.linalg.norm(pos_ref[0] - pos_tx)
        for z in range(len(pos_ref) - 1):
            d += np.linalg.norm(pos_ref[z + 1] - pos_ref[z])
        d += np.linalg.norm(pos_rx - pos_ref[len(pos_ref)-1])
        toa = d / c  # time of arrival, note: max distance should be below (N*Ts*c)
        aod = math.atan2(pos_ref[0][1] - pos_tx[1], pos_ref[0][0] - pos_tx[0]) - alpha_tx # angle of departure
        aoa = math.atan2(pos_ref[len(pos_ref)-1][1] - pos_rx[1], pos_ref[len(pos_ref)-1][0] - pos_rx[0]) - alpha_rx # angle of arrival
    return toa, aod, aoa


def get_response(N, phi):
    """
    Calculating antenna response
    Inputs:
              N: number of antenna elements
              phi: angle of departure (arrivals)
    Output:
              a: antenna response
    """
    a = np.ones([N, 1], dtype=complex)
    sq = math.sqrt(N)
    for k in range(N):
        a[k] = cmath.exp(-1j * math.pi * phi * k)/sq
    return a


def channel_resp(nt, nr, pos_tx, pos_rx, alpha_tx, alpha_rx, pos_ref_list, ref_coef_list, pass_coef_list, c, f, pl_exp):
    """
    Calculating channel response between TX and RX
    Inputs:
              nt: number of antenna elements at TX
              nr: number of antenna elements at RX
              pos_tx: position of transmitter
              pos_rx: position of receiver
              alpha_tx: angle of antenna elements of transmitter
              alpha_rx: angle of antenna elements of receiver
              pos_ref_list: position of all reflection points for all paths
              ref_coef_list: reflection coefficient of all reflections for all paths
              ref_coef_list: transmission coefficient for all paths
              c: the speed of light
              f: carrier frequency
              pl_exp: path loss exponent
    Outputs:
              h_ch: channel response
              toa_list: list of time of arrival
              aod_list: list of angle of departure
              aoa_list: list of angle of arrivals
              coef_list: list of path coefficient
    """
    sigma_list = path_gains(pos_tx, pos_rx, pos_ref_list, c, f, pl_exp) # path gain of all paths
    h_ch = np.zeros([nr, nt], dtype=complex)
    toa_list = []; aod_list = []; aoa_list = []; coef_list = []
    for k in range(len(pos_ref_list)):
        toa, aod, aoa = channel_params(pos_tx, pos_rx, alpha_tx, alpha_rx, pos_ref_list[k]['ref_pt'], c)
        toa_list.append(toa); aod_list.append(aod); aoa_list.append(aoa)
        a_rx = get_response(nr,math.cos(aoa))
        a_tx = np.transpose(get_response(nt,math.cos(aod)))
        cmp_rand = sigma_list[k] * (np.random.normal(0, 1/math.sqrt(2), 1) + 1j * np.random.normal(0, 1/math.sqrt(2), 1))  # Complex Gaussian random
        coef_list.append(pass_coef_list[k] * ref_coef_list[k] * np.absolute(cmp_rand)[0])
        h_ch += pass_coef_list[k] * ref_coef_list[k] * cmp_rand * np.matmul(a_rx, np.conjugate(a_tx.reshape((1, nt))))  # contribution of kth path to channel response
    h_ch *= math.sqrt(nt*nr)  # Normalization
    return h_ch, toa_list, aod_list, aoa_list, coef_list


def create_dictionary(nt, nr, nb_t, nb_r):
    """
    Calculating beam codebook at TX and RX
    Inputs:
              nt: number of antenna elements at TX
              nr: number of antenna elements at RX
              nb_t: number of antenna beams at TX
              nb_r: number of antenna beams at RX
    Outputs:
              ut: beam codebook at TX
              ur: beam codebook at RX
    """
    ut = np.zeros([nt, nb_t], dtype=complex)
    ur = np.zeros([nr, nb_r], dtype=complex)
    if nb_t > 1:
        aa = np.arange(1, nb_t + 1)
        aa = (2 * aa - 1 - nb_t)/(nb_t)  # dictionary of spatial frequencies
    else:
        aa = np.array([1])
    for m in range(nb_t):
        ut[:, [m]] = get_response(nt, aa[m])

    if nb_r > 1:
        aa = np.arange(1, nb_r + 1)
        aa = (2 * aa - 1 - nb_r) / (nb_r)  # dictionary of spatial frequencies
    else:
        aa = np.array([1])
    for m in range(nb_r):
        ur[:, [m]] = get_response(nr, aa[m])
    return ut, ur


def lut(nb, ang):
    """
    This function Calculates which beam has the highest gain
    Inputs:
              nb: number of antenna beams
              ang: angle of departure (arrivals)
    Outputs:
              ind_max: index of the best beam
              aa: beam codebook at RX
              ang_max:
              list_angut:
    """
    num_ang = np.shape(ang)[0]
    ind_max = np.zeros([num_ang], dtype=int)
    ang_max = np.zeros([num_ang], dtype=float)
    if nb > 1:
        aa = np.arange(1, nb + 1)
        aa = (2 * aa - 1 - nb)/(nb)  # dictionary of spatial frequencies
    else:
        aa = np.array([1])
    num_beams = np.shape(aa)[0]
    for k in range(num_ang):
        a1 = get_response(nb, math.cos(ang[k]))
        a1_h = np.conj(np.transpose(a1[:, 0]))
        mag = np.zeros([num_beams], dtype=float)
        for m in range(num_beams):
            a2 = get_response(nb, aa[m])
            mag[m] = np.absolute(np.matmul(a1_h, a2[:, 0]))
        ind_max[k] = np.argmax(mag)
        ang_max[k] = math.acos(aa[ind_max[k]])
    list_ang = []
    for k in range(nb):
        ii = np.where(ind_max == k)
        list_ang.append(ii)
    return ind_max, aa, ang_max, list_ang


if __name__ == '__main__':
    c = 3e8  # speed of light meter / s
    f = 6e10  # carrier frequency
    pl_exp = 2  # path loss exponent
    nt = 64  # number of TX antennas
    nr = nt  # number of RX antennas
    nb = nt  # number of beams in dictionary (codebook)
    pos_tx = np.array([0, 0])  # TX (BS) position
    pos_rx = np.array([100, 30])  # RX (user) position
    pos_ref = np.array([50,50])  # Reflection point position
    alpha_tx = 0.2  # BS orientation
    alpha_rx = 0.2  # user orientation
    pos_ref_list = [{'ref_pt': pos_tx}, {'ref_pt': pos_ref}]
    ref_coef_list = [1, 0.1]  # list of reflection coefficients
    pass_coef_list = [1, 1]  # list of pass coefficients
    h_ch, toa_list, aod_list, aoa_list, coef_list = \
        channel_resp(nt, nr, pos_tx, pos_rx, alpha_tx, alpha_rx, pos_ref_list, ref_coef_list, pass_coef_list, c, f, pl_exp)
    ang = np.arange(0, 2*math.pi, 0.01)
    ind_max, aa, ang_max, list_ang = lut(nb, ang)
    fig0 = plt.figure(figsize=plt.figaspect(0.5))
    plt.plot(180 / math.pi * ang, ind_max)
    plt.ylabel('Index of Best Beam in the DFT Code book')
    plt.xlabel('Angle (degree)')
    plt.grid(True)
    plt.show(block=True)

