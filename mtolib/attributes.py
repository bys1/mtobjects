from mtolib.maxtree import MaxTree

import numpy as np

from math import sqrt, exp, log, pi
from matplotlib import pyplot as plt

from scipy.special import gammaincinv, gammainc
from scipy.optimize import curve_fit

MT_HAVE_SIGNIFICANT_DESCENDANT = 4
MT_HAVE_DESCENDANT = 16

PI_DIV = 1 / pi

def get_attributes(mto_struct, tree : MaxTree, object_id):
    """
    tree        : The Max-Tree to search in
    center_id   : The id of the Max-Tree node representing the object.
    """
    xs = []
    ys = []
    zs = []
    ysi = []
    zsi = []

    bg = tree.nodes[object_id].parent
    V_prev = tree.node_attributes[bg].volume
    I = 0

    V_tot = tree.node_attributes[object_id].volume
    print('V_tot', V_tot)
    V_e = 0
    I_e = 0
    R_e = 0

    area = 1
    node_id = object_id
    while node_id != -1:
        node = tree.nodes[node_id]
        atts = tree.node_attributes[node_id]
        R = sqrt(node.area * PI_DIV)

        I += (V_prev - atts.volume) / area
        V_prev = atts.volume
        area = node.area
        V = atts.volume + I * node.area
        
        if V >= V_tot / 2:
            R_e = R
            V_e = V
            I_e = I

        xs.append(R)
        ys.append(V)
        ysi.append(I)

        node_id = one_up(mto_struct, node_id)
    
    I += V_prev / area
    I_e = I / 2
    for R, I_cur in zip(xs, ysi):
        if I_cur > I_e:
            break
        R_e = R

    n = 4
    print('I', I, 'V_e', V_e, 'I_e', I_e)
    while object_id != -1:
        R = sqrt(tree.nodes[object_id].area * PI_DIV)
        L = sersic_luminosity(tree, object_id, n, R_e, I_e) * 4000
        I = sersic_intensity (n, R, R_e, I_e)
        zs.append(L)
        zsi.append(I)
        object_id = one_up(mto_struct, object_id)
    
    xs0 = [R / R_e for R in xs]

    plt.plot(xs0, ysi)
    plt.plot(xs0, zsi)
    plt.ylim(ymin=0, ymax=12000)
    plt.show()

    formula = lambda xs, n: [sersic_intensity(n, x, R_e, I_e) for x in xs]

    """
    cnt = sum([x > 1 for x in xs0])
    plt.plot(xs0[:cnt], ysi[:cnt])
    plt.plot(xs0[:cnt], zsi[:cnt])
    plt.show()
    """

    plt.plot(xs0, [log(y) for y in ysi])
    plt.plot(xs0, [log(z) for z in zsi])
    plt.show()

    popt, _ = curve_fit(formula, xs, ysi, bounds=(0,12))
    n_fit = popt[0]
    print('n_fit =', n_fit)
    sersic_fit = formula(xs, n_fit)

    plt.plot(xs0, ysi)
    plt.plot(xs0, sersic_fit)
    plt.ylim(ymin=0, ymax=12000)
    plt.show()

def sersic_luminosity(tree : MaxTree, node_id, n, R_e, I_e):
    b_n = gammaincinv(2*n, 0.5)
    R = sqrt(tree.nodes[node_id].area * PI_DIV)
    x = b_n * (R / R_e) ** (1/n)
    L = I_e * R_e * R_e * pi * 2 * n * exp(b_n) / (b_n ** (2*n)) * gammainc(2*n, x)
    return L

def sersic_intensity(n, R, R_e, I_e):
    b_n = gammaincinv(2*n, 0.5)
    ex = -b_n * ((R/R_e) ** (1/n) - 1)
    return I_e * exp(ex)

def biggest_object(tree : MaxTree, object_ids):
    objects = set(object_ids.flatten())
    biggest = -1
    area = -1
    for id in objects:
        nodeArea = tree.nodes[id].area
        if nodeArea > area:
            biggest = id
            area = nodeArea
    return biggest

def one_up(mto_struct, node_id):
    if check_flag(mto_struct, node_id, MT_HAVE_SIGNIFICANT_DESCENDANT):
        return mto_struct.main_branches[node_id]
    if check_flag(mto_struct, node_id, MT_HAVE_DESCENDANT):
        return mto_struct.main_power_branches[node_id]
    return -1

def check_flag(mto_struct, idx, flag):
    return mto_struct.flags[idx] & flag != 0