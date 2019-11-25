#!/usr/bin/python
# -*- coding:utf-8 -*-
"""

American Option Pricing by Binomial Tree
Reference: Options, Futures and Derivatives - John C.Hull

@author: Haoyang Wu
"""

import math
import numpy as np


def AmericanOption(S0, K, T, v, r, q, N, CallorPut):
    """

    :param S0:  Spot price
    :param K:   Strike price
    :param T:   Time to maturity   eg: 200/365
    :param v:   Volatility of asset
    :param r:   Risk free interest
    :param q:   Dividend Rate
    :param N:   Iterations
    :param CallorPut:  Option type, include 'Call' & 'Put'
    :return: Option price
    """

    # ------- Bi-Tree Parameters ------------

    dt = T/N
    u = math.exp(v * math.sqrt(dt))      # up move
    d = 1 / u                           # down move
    a = math.exp((r - q) * dt)
    p = (a - d) / (u - d)

    # ------- Bi-Tree Construction ----------

    stock_tree = np.zeros([N + 1, N + 1])
    for j in range(0, N + 1):
        for i in range(0, j + 1):
            stock_tree[i, j] = S0 * (u ** (j - i)) * (d ** i)

    option_tree = np.zeros([N + 1, N + 1])
    if CallorPut == 'call':

        option_tree[:, N] = np.maximum(0, stock_tree[:, N] - K)
        for j in range(N - 1, -1, -1):
            for i in range(0, j + 1):
                option_tree[i, j] = (1 / a) * (p * option_tree[i, j + 1] + (1 - p) * option_tree[i + 1, j + 1])
            option_tree[:, j] = np.maximum(option_tree[:, j], stock_tree[:, j] - K)

    elif CallorPut == 'put':
        option_tree[:, N] = np.maximum(0, K - stock_tree[:, N])
        for j in range(N - 1, -1, -1):
            for i in range(0, j + 1):
                option_tree[i, j] = (1 / a) * (p * option_tree[i, j + 1] + (1 - p) * option_tree[i + 1, j + 1])
            option_tree[:, j] = np.maximum(option_tree[:, j], K - stock_tree[:, j])

    else:
        print('Please input the right option type: ``call`` or ``put`` ')
        return -1

    return option_tree[0, 0]