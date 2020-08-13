"""
Author: Sajad Rezaie (sre@es.aau.dk)
"""

import matplotlib.pyplot as plt
import matplotlib.patches as patches
import matplotlib.lines as mlines
import math
from shapely.geometry import LineString
from itertools import count
import random, time
import numpy as np


class Obj:
    ref_coef = 0.4  # reflection coefficient caused by the object
    pass_coef = 0.05  # pass coefficient caused by the object
    ids = count(0)

    def __init__(self, type, point, size, angle=None):
        if type is not 'rec' and type is not 'cir':
            raise Exception('Error in type definition')
        self.type = type   # 'rec', 'cir'
        self.pt = point

        if self.type is 'rec' and not len(size) == 2:
            raise Exception('Error in size dimension')
        if self.type is 'cir' and not len(size) == 1:
            raise Exception('Error in size dimension')
        self.s = size      # [length, width] for 'rec', [length] for 'squ', [radius] for 'cir'

        if angle is None:
            self.angle = 0
        else:
            self.angle = angle

        if self.type is 'rec':
            self.corners = self.find_corners()
        else:
            self.corners = self.pt

        self.num_inst = next(self.ids)

    """ Check intersection between two objects """
    def intersect_obj(self, obj):
        if self.type == 'cir' and obj.type == 'cir':
            d = math.sqrt((self.pt[0] - obj.pt[0])**2 + (self.pt[1] - obj.pt[1])**2)
            if d <= (self.s[0] + obj.s[0]):
                return 1
            else:
                return 0
        elif self.type == 'cir':
            c_ext = obj.find_corners_extended(self.s[0])
            pd = []
            for k in range(4):
                pd.append(perpendicular_dist(c_ext[k], c_ext[(k + 1) % 4], self.pt))
            sx = (obj.s[0] + 2 * self.s[0])
            sy = (obj.s[1] + 2 * self.s[0])
            if pd[0] <= sy and pd[1] <= sx and pd[2] <= sy and pd[3] <= sx:
                return 1
            c = obj.corners
            for k in range(4):
                d = math.sqrt((self.pt[0] - c[k][0]) ** 2 + (self.pt[1] - c[k][1]) ** 2)
                if d <= (self.s[0]):
                    return 1
            return 0

        elif obj.type == 'cir':
            c_ext = self.find_corners_extended(obj.s[0])
            pd = []
            for k in range(4):
                pd.append(perpendicular_dist(c_ext[k], c_ext[(k + 1) % 4], obj.pt))
            sx = (self.s[0] + 2 * obj.s[0])
            sy = (self.s[1] + 2 * obj.s[0])
            if pd[0] <= sy and pd[1] <= sx and pd[2] <= sy and pd[3] <= sx:
                return 1
            c = self.corners
            for k in range(4):
                d = math.sqrt((obj.pt[0] - c[k][0]) ** 2 + (obj.pt[1] - c[k][1]) ** 2)
                if d <= (obj.s[0]):
                    return 1
            return 0
        else:
            c = self.corners
            c2 = obj.corners
            for m in range(4):
                for k in range(4):
                    int_pt = int_4pt(c[m], c[(m + 1) % 4], c2[k], c2[(k + 1) % 4])
                    if int_pt is not None:
                        if check_pt_inner(c[m], c[(m + 1) % 4], int_pt[0], int_pt[1]):
                            if check_pt_inner(c2[k], c2[(k + 1) % 4], int_pt[0], int_pt[1]):
                                return 1
            """ check the obj is inside of the self """
            for k in range(1):
                pd = []
                for m in range(4):
                    pd.append(perpendicular_sign(c[m], c[(m + 1) % 4], c2[k]))
                if pd[0] == pd[2] and pd[1] == pd[3]:
                    return 1
            """ check the self is inside of the obj """
            for m in range(1):
                pd = []
                for k in range(4):
                    pd.append(perpendicular_sign(c2[k],c2[(k+1)%4], c[m]))
                if pd[0] == pd[2] and pd[1] == pd[3]:
                    return 1
            return 0

    """ find coordinates of 4 corners """
    def find_corners(self):
        if self.type != 'rec':
            return self.pt
        ang_rad = math.radians(self.angle)
        corners = [self.pt]
        p1 = [self.pt[0] + round(self.s[0] * math.cos(ang_rad),5),
              self.pt[1] + round(self.s[0] * math.sin(ang_rad),5)]
        corners.append(p1)
        p2 = [self.pt[0] + round(self.s[0] * math.cos(ang_rad) - self.s[1] * math.sin(ang_rad),5),
              self.pt[1] + round(self.s[0] * math.sin(ang_rad) + self.s[1] * math.cos(ang_rad),5)]
        corners.append(p2)
        p3 = [self.pt[0] - round(self.s[1] * math.sin(ang_rad),5),
              self.pt[1] + round(self.s[1] * math.cos(ang_rad),5)]
        corners.append(p3)
        return corners

    """ find coordinates of 4 corners of extended area"""
    def find_corners_extended(self, r):
        if self.type != 'rec':
            return self.pt
        ang_rad = math.radians(self.angle)
        shift_x = np.sign(self.s[0]) * r
        shift_y = np.sign(self.s[1]) * r
        p0 = [self.pt[0] + round(-shift_x * math.cos(ang_rad) - (- shift_y) * math.sin(ang_rad),5),
              self.pt[1] + round(-shift_x * math.sin(ang_rad) + (- shift_y) * math.cos(ang_rad),5)]
        corners_extnd = [p0]
        p1 = [self.pt[0] + round((self.s[0] + shift_x) * math.cos(ang_rad) - (- shift_y) * math.sin(ang_rad), 5),
              self.pt[1] + round((self.s[0] + shift_x) * math.sin(ang_rad) + (- shift_y) * math.cos(ang_rad), 5)]
        corners_extnd.append(p1)
        p2 = [self.pt[0] + round((self.s[0] + shift_x) * math.cos(ang_rad) - (self.s[1] + shift_y) * math.sin(ang_rad),5),
              self.pt[1] + round((self.s[0] + shift_x) * math.sin(ang_rad) + (self.s[1] + shift_y) * math.cos(ang_rad),5)]
        corners_extnd.append(p2)
        p3 = [self.pt[0] + round(-shift_x * math.cos(ang_rad) - (self.s[1] + shift_y) * math.sin(ang_rad),5),
              self.pt[1] + round(-shift_x * math.sin(ang_rad) + (self.s[1] + shift_y) * math.cos(ang_rad),5)]
        corners_extnd.append(p3)
        return corners_extnd

    def find_ref_pt(self, tx_obj, rx_obj):
        if self.type != 'rec':
            return None

        c = self.corners
        cnt = [0, 0]
        for k in range(4):
            cnt[0] += c[k][0]
            cnt[1] += c[k][1]
        cnt[0] /= 4
        cnt[1] /= 4
        for k in range(4):
            pv0 = perpendicular_sign(c[k], c[(k + 1) % 4], tx_obj.pt)
            pv1 = perpendicular_sign(c[k], c[(k + 1) % 4], rx_obj.pt)
            pv2 = perpendicular_sign(c[k], c[(k + 1) % 4], cnt)
            if pv0 != pv1 or pv0 == 0 or pv1 == 0:
                continue
            elif pv0 == pv2:
                continue
            else:
                rx_ref_pt = ref_pt_over_line(c[k], c[(k + 1) % 4],rx_obj.pt)
                int_pt = int_4pt(c[k], c[(k + 1) % 4], tx_obj.pt, rx_ref_pt)
                if int_pt is None:
                    continue
                if check_pt_inner(c[k], c[(k + 1) % 4], int_pt[0], int_pt[1]):
                    if check_pt_inner(tx_obj.pt, rx_ref_pt, int_pt[0], int_pt[1]):
                        ref_pt = []
                        ref_pt.append(int_pt[0])
                        ref_pt.append(int_pt[1])
                        ref_parms = {"ref_pt": ref_pt, "corners_pts": [c[k], c[(k + 1) % 4]]}
                        return ref_parms
        return None


class StcObj(Obj):
    """ Static Object """
    ref_coef = 0.4
    pass


class MobObj(Obj):
    """ Mobile Object """
    ref_coef = 0.4
    pass


def perpendicular_dist(p1,p2,p0):
    """ Calculate distance of a point from a line """
    d = abs((p2[1]-p1[1])*p0[0]-(p2[0]-p1[0])*p0[1]+p2[0]*p1[1]-p2[1]*p1[0])/math.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)
    return d


def perpendicular_val(p1,p2,p0):
    """ Calculate signed distance of a point from a line """
    d = ((p2[1]-p1[1])*p0[0]-(p2[0]-p1[0])*p0[1]+p2[0]*p1[1]-p2[1]*p1[0])/math.sqrt((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)
    return d


def perpendicular_sign(p1,p2,p0):
    """ Calculate the sign of distance of a point from a line """
    d = ((p2[1] - p1[1]) * p0[0] - (p2[0] - p1[0]) * p0[1] + p2[0] * p1[1] - p2[1] * p1[0]) / math.sqrt(
        (p2[1] - p1[1]) ** 2 + (p2[0] - p1[0]) ** 2)
    return np.sign(d)


def mid_pt_over_line(p1,p2,p0):
    """ Calculate the projected point of the point p0 over the line which passes over the points p1 and p2 """
    t = ((p2[0]-p1[0])*(p0[0]-p1[0]) + (p2[1]-p1[1])*(p0[1]-p1[1]))/((p2[1]-p1[1])**2 + (p2[0]-p1[0])**2)
    mid_pt = []
    mid_pt.append(p1[0] + (p2[0]-p1[0])* t)
    mid_pt.append(p1[1] + (p2[1]-p1[1])* t)
    return mid_pt


def ref_pt_over_line(p1,p2,p0):
    """ Calculate the reflected point of the point p0 against the line which passes over the points p1 and p2 """
    mid_pt = mid_pt_over_line(p1, p2, p0)
    ref_pt = [2 * mid_pt[0] - p0[0], 2 * mid_pt[1] - p0[1]]
    return ref_pt


def int_4pt(p1, p2, p3, p4):
    """ Check if the line pass over the p3 and p4 has intersection with the line pass over the p1 and p2 """
    if p1[0] == p2[0]:
        if p3[0] == p4[0]:
            return None
        m2 = (p3[1] - p4[1])/(p3[0] - p4[0])
        c2 = p3[1] - m2 * p3[0]
        int_pt = [p1[0], m2 * p1[0] + c2]
        return int_pt

    if p3[0] == p4[0]:
        m1 = (p1[1] - p2[1])/(p1[0] - p2[0])
        c1 = p1[1] - m1 * p1[0]
        int_pt = [p3[0], m1 * p3[0] + c1]
        return int_pt
    m1 = (p1[1] - p2[1])/(p1[0] - p2[0])
    m2 = (p3[1] - p4[1])/(p3[0] - p4[0])
    if m1 == m2:
        return None
    c1 = p1[1] - m1 * p1[0]
    c2 = p3[1] - m2 * p3[0]
    x_int = (c1 - c2)/(m2 - m1)
    y_int = m1 * x_int + c1
    int_pt = [x_int, y_int]
    return int_pt


def find_ref_pt1(obj_list, tx_obj, rx_obj):
    """ Find the 1st order reflection path between the tx_obj and rx_obj and reflection from the obj_list"""
    num_objs = len(obj_list)
    if num_objs != 1:
        return None
    for o in range(num_objs):
        if obj_list[o].type != 'rec':
            return None

    cnt_objs = np.zeros((num_objs, 2))
    for o in range(len(obj_list)):
        c = obj_list[o].corners
        cnt = [0, 0]
        for k in range(4):
            cnt[0] += c[k][0]
            cnt[1] += c[k][1]
        cnt_objs[o][0] = cnt[0] / 4
        cnt_objs[o][1] = cnt[1] / 4
    for o1 in range(len(obj_list)):
        p_s1 = tx_obj.pt
        c1 = obj_list[o1].corners
        cnt1 = cnt_objs[o1]
        for k1 in range(4):
            pv0_1 = perpendicular_sign(c1[k1], c1[(k1 + 1) % 4], p_s1)
            pv2_1 = perpendicular_sign(c1[k1], c1[(k1 + 1) % 4], cnt1)
            if pv0_1 == 0 or pv0_1 == pv2_1:
                continue
            else:
                p_s2 = ref_pt_over_line(c1[k1], c1[(k1 + 1) % 4], p_s1)
                p_r1 = p_s2
                int_pt1 = int_4pt(c1[k1], c1[(k1 + 1) % 4], rx_obj.pt, p_s2)
                if int_pt1 is None:
                    continue
                if check_pt_inner(c1[k1], c1[(k1 + 1) % 4], int_pt1[0],
                                  int_pt1[1]):
                    if check_pt_inner(rx_obj.pt, p_s2, int_pt1[0], int_pt1[1]):
                        ref_pt = []
                        ref_pt.append(np.array(int_pt1))
                        ref_parms2 = {"ref_pt": ref_pt,
                                     "corners_pts": [c1[k1], c1[(k1 + 1) % 4]]}
                        return ref_parms2
    return None


def find_ref_pt2(obj_list, tx_obj, rx_obj):
    """ Find the 2nd order reflection path between the tx_obj and rx_obj and reflections from the obj_list"""
    num_objs = len(obj_list)
    if num_objs != 2:
        return None
    for o in range(num_objs):
        if obj_list[o].type != 'rec':
            return None

    cnt_objs = np.zeros((num_objs, 2))
    for o in range(len(obj_list)):
        c = obj_list[o].corners
        cnt = [0, 0]
        for k in range(4):
            cnt[0] += c[k][0]
            cnt[1] += c[k][1]
        cnt_objs[o][0] = cnt[0] / 4
        cnt_objs[o][1] = cnt[1] / 4

    for o1 in range(len(obj_list)):
        p_s1 = tx_obj.pt
        c1 = obj_list[o1].corners
        cnt1 = cnt_objs[o1]
        for k1 in range(4):
            pv0_1 = perpendicular_sign(c1[k1], c1[(k1 + 1) % 4], p_s1)
            pv2_1 = perpendicular_sign(c1[k1], c1[(k1 + 1) % 4], cnt1)
            if pv0_1 == 0 or pv0_1 == pv2_1:
                continue
            else:

                for o2 in range(len(obj_list)):
                    if o2 == o1:
                        continue
                    p_s2 = ref_pt_over_line(c1[k1], c1[(k1 + 1) % 4], p_s1)
                    p_r1 = p_s2
                    c2 = obj_list[o2].corners
                    cnt2 = cnt_objs[o2]
                    for k2 in range(4):
                        pv0_2 = perpendicular_sign(c2[k2], c2[(k2 + 1) % 4], p_s2)
                        pv2_2 = perpendicular_sign(c2[k2], c2[(k2 + 1) % 4], cnt2)
                        if pv0_2 == 0 or pv0_2 == pv2_2:
                            continue
                        else:
                            p_s3 = ref_pt_over_line(c2[k2], c2[(k2 + 1) % 4], p_s2)
                            p_r2 = p_s3
                            int_pt2 = int_4pt(c2[k2], c2[(k2 + 1) % 4], rx_obj.pt, p_s3)
                            if int_pt2 is None:
                                continue
                            if check_pt_inner(c2[k2], c2[(k2 + 1) % 4], int_pt2[0], int_pt2[1]):
                                if check_pt_inner(rx_obj.pt, p_s3, int_pt2[0], int_pt2[1]):
                                    int_pt1 = int_4pt(c1[k1], c1[(k1 + 1) % 4], int_pt2, p_s2)
                                    if int_pt1 is None:
                                        continue
                                    if check_pt_inner(c1[k1], c1[(k1 + 1) % 4], int_pt1[0],
                                                      int_pt1[1]):
                                        if check_pt_inner(int_pt2, p_s2, int_pt1[0], int_pt1[1]):
                                            ref_pt = []
                                            ref_pt.append(np.array(int_pt1))
                                            ref_pt.append(np.array(int_pt2))
                                            ref_parms2 = {"ref_pt": ref_pt,
                                                         "corners_pts": [c1[k1], c1[(k1 + 1) % 4]]}
                                            return ref_parms2
    return None


def find_ref_pt3(obj_list, tx_obj, rx_obj):
    """ Find the 3rd order reflection path between the tx_obj and rx_obj and reflections from the obj_list"""
    num_objs = len(obj_list)
    if num_objs != 3:
        return None
    for o in range(num_objs):
        if obj_list[o].type != 'rec':
            return None

    cnt_objs = np.zeros((num_objs, 2))
    for o in range(len(obj_list)):
        c = obj_list[o].corners
        cnt = [0, 0]
        for k in range(4):
            cnt[0] += c[k][0]
            cnt[1] += c[k][1]
        cnt_objs[o][0] = cnt[0] / 4
        cnt_objs[o][1] = cnt[1] / 4

    for o1 in range(len(obj_list)):
        p_s1 = tx_obj.pt
        c1 = obj_list[o1].corners
        cnt1 = cnt_objs[o1]
        for k1 in range(4):
            pv0_1 = perpendicular_sign(c1[k1], c1[(k1 + 1) % 4], p_s1)
            pv2_1 = perpendicular_sign(c1[k1], c1[(k1 + 1) % 4], cnt1)
            if pv0_1 == 0 or pv0_1 == pv2_1:
                continue
            else:

                for o2 in range(len(obj_list)):
                    if o2 == o1:
                        continue
                    p_s2 = ref_pt_over_line(c1[k1], c1[(k1 + 1) % 4], p_s1)
                    p_r1 = p_s2
                    c2 = obj_list[o2].corners
                    cnt2 = cnt_objs[o2]
                    for k2 in range(4):
                        pv0_2 = perpendicular_sign(c2[k2], c2[(k2 + 1) % 4], p_s2)
                        pv2_2 = perpendicular_sign(c2[k2], c2[(k2 + 1) % 4], cnt2)
                        if pv0_2 == 0 or pv0_2 == pv2_2:
                            continue
                        else:

                            for o3 in range(len(obj_list)):
                                if o3 == o1 or o3 == o2:
                                    continue
                                p_s3 = ref_pt_over_line(c2[k2], c2[(k2 + 1) % 4], p_s2)
                                p_r2 = p_s3
                                c3 = obj_list[o3].corners
                                cnt3 = cnt_objs[o3]
                                for k3 in range(4):
                                    pv0_3 = perpendicular_sign(c3[k3], c3[(k3 + 1) % 4], p_s3)
                                    pv1_3 = perpendicular_sign(c3[k3], c3[(k3 + 1) % 4], rx_obj.pt)
                                    pv2_3 = perpendicular_sign(c3[k3], c3[(k3 + 1) % 4], cnt3)
                                    if pv0_3 != pv1_3 or pv0_3 == 0 or pv1_3 == 0 or pv0_3 == pv2_3:
                                        continue
                                    else:
                                        p_s4 = ref_pt_over_line(c3[k3], c3[(k3 + 1) % 4], p_s3)
                                        p_r3 = p_s4
                                        int_pt3 = int_4pt(c3[k3], c3[(k3 + 1) % 4], rx_obj.pt, p_s4)
                                        if int_pt3 is None:
                                            continue
                                        if check_pt_inner(c3[k3], c3[(k3 + 1) % 4], int_pt3[0], int_pt3[1]):
                                            if check_pt_inner(rx_obj.pt, p_s4, int_pt3[0], int_pt3[1]):
                                                int_pt2 = int_4pt(c2[k2], c2[(k2 + 1) % 4], int_pt3, p_s3)
                                                if int_pt2 is None:
                                                    continue
                                                if check_pt_inner(c2[k2], c2[(k2 + 1) % 4], int_pt2[0], int_pt2[1]):
                                                    if check_pt_inner(int_pt3, p_s3, int_pt2[0], int_pt2[1]):
                                                        int_pt1 = int_4pt(c1[k1], c1[(k1 + 1) % 4], int_pt2, p_s2)
                                                        if int_pt1 is None:
                                                            continue
                                                        if check_pt_inner(c1[k1], c1[(k1 + 1) % 4], int_pt1[0],
                                                                          int_pt1[1]):
                                                            if check_pt_inner(int_pt2, p_s2, int_pt1[0], int_pt1[1]):
                                                                ref_pt = []
                                                                ref_pt.append(np.array(int_pt1))
                                                                ref_pt.append(np.array(int_pt2))
                                                                ref_pt.append(np.array(int_pt3))
                                                                ref_parms2 = {"ref_pt": ref_pt,
                                                                             "corners_pts": [c1[k1], c1[(k1 + 1) % 4]]}
                                                                return ref_parms2
    return None


def check_pt_inner(p1, p2, x, y):
    """ check the point (x, y) is between the points p1 and p2 """
    c_x_min = min(p1[0], p2[0])
    c_x_max = max(p1[0], p2[0])
    c_y_min = min(p1[1], p2[1])
    c_y_max = max(p1[1], p2[1])
    if x >= c_x_min and x <= c_x_max and y >= c_y_min and y <= c_y_max:
        return 1
    return 0


def check_intersect_obj_list(obj, obj_list):
    """ Check if the "obj" has any intersection with any of the objects in the "obj_list" """
    for k in range(len(obj_list)):
        if obj_list[k].intersect_obj(obj):
            return True
    return False


def check_blockage(p1, p2, obj):
    """ Check if the line segment between p1 and p2 has intersection with the "obj" """
    if obj.type == 'rec':
        c = obj.corners
        for k in range(4):
            int_pt = int_4pt(p1, p2, c[k], c[(k + 1) % 4])
            if int_pt is not None:
                if check_pt_inner(p1, p2, int_pt[0], int_pt[1]):
                    if check_pt_inner(c[k], c[(k + 1) % 4], int_pt[0], int_pt[1]):
                        return 1
        return 0
    elif obj.type == 'cir':
        pd = perpendicular_dist(p1,p2,obj.pt)
        if pd > obj.s[0]:
            return 0
        mid_pt = mid_pt_over_line(p1, p2, obj.pt)
        if check_pt_inner(p1, p2, mid_pt[0], mid_pt[1]):
            return 1
        else:
            return 0
    else:
        return 0


def check_blockage_obj_list(p_list, obj_list, ind, min_pass_coef):
    """ Check if the path between points in the "p_list" can be blocked by the objects in the "obj_list" """
    pass_coef = 1
    blockage = 0
    num_pts = len(p_list)
    for k in range(num_pts-1):
        p0 = p_list[k]
        p1 = p_list[k + 1]
        for m in range(len(obj_list)):
            match_ind = 0
            for z in range(len(ind)):
                if m == ind[z]:
                    match_ind = 1
                    break
            if match_ind == 0:
                if check_blockage(p0, p1, obj_list[m]):
                    pass_coef *= obj_list[m].pass_coef
                    if pass_coef < min_pass_coef:
                        blockage = 1
                        break
    if blockage == 0:
        return pass_coef
    else:
        return None


def plot_obj(ax, obj):
    """ add object to figure """
    if obj.__class__ == MobObj:
        edge_color = 'b'
    elif obj.__class__ == StcObj:
        edge_color = 'k'
    else:
        edge_color = 'r'

    if obj.type == 'rec':
        ax.add_patch(
            patches.Rectangle(obj.pt, obj.s[0], obj.s[1], angle=obj.angle, linewidth=1, edgecolor=edge_color, facecolor='none'))
    elif obj.type == 'cir' and obj.__class__ == Obj:
        ax.add_patch(patches.Circle(obj.pt, .1, linewidth=1, edgecolor=edge_color, facecolor=edge_color))
    elif obj.type == 'cir':
        ax.add_patch(patches.Circle(obj.pt, obj.s[0], linewidth=1, edgecolor=edge_color, facecolor='none'))


def plot_ray(ax, p0, p1, p2, color='m'):
    """ add ray to figure """
    x, y = np.array([[p0[0],p1[0],p2[0]], [p0[1], p1[1],p2[1]]])
    line = mlines.Line2D(x, y, lw=2, color=color, linestyle= '--' , alpha=0.3)
    ax.add_line(line)


if __name__ == '__main__':
    stc_coord = [['rec',[6, 6], [8,5], 0], ['rec',[-5, -5], [7,3], 45], ['rec',[-5, 2], [10,6], 30]]
    num_mob_obj = 5
    stc_obj_list = []
    range_pt = [[-8,8], [-8,8]]   # [[xmin, xmax], [ymin, ymax]]
    range_s = [1, 4]      # [min length, max length]
    for k in range(len(stc_coord)):
        stc_c = stc_coord[k]
        stc_obj_list.append(StcObj(stc_c[0],stc_c[1], stc_c[2], stc_c[3]))
        for m in range(len(stc_obj_list)-1):
            if stc_obj_list[m].intersect_obj(stc_obj_list[len(stc_obj_list)-1]):
                raise Exception('Error in intersection between static objects {} and {}'.format(m, k))
    mob_obj_list = []
    while len(mob_obj_list) < num_mob_obj:
        cnt_pt_x = random.uniform(range_pt[0][0],range_pt[0][1])
        cnt_pt_y = random.uniform(range_pt[1][0], range_pt[1][1])
        length = random.uniform(range_s[0], range_s[1])
        width = random.uniform(range_s[0], range_s[1])
        ang = random.uniform(0, 180)
        mob_obj1 = MobObj('rec', [cnt_pt_x, cnt_pt_y], [length,width], ang)
        is_intersect = 0
        for k in range(len(stc_obj_list)):
            if stc_obj_list[k].intersect_obj(mob_obj1):
                is_intersect = 1
                break
        if is_intersect == 1:
            continue
        for m in range(len(mob_obj_list)):
            if mob_obj_list[m].intersect_obj(mob_obj1):
                is_intersect = 1
                break
        if is_intersect == 1:
            continue
        mob_obj_list.append(mob_obj1)

    fig1 = plt.figure()
    ax1 = fig1.add_subplot(111, aspect='equal')
    ax1.grid(b=True, which='major', axis='both')
    for k in range(len(stc_obj_list)):
        plot_obj(ax1, stc_obj_list[k])
    for m in range(len(mob_obj_list)):
        plot_obj(ax1, mob_obj_list[m])
    plt.ylim((-15,15))
    plt.xlim((-15,15))
    plt.show()