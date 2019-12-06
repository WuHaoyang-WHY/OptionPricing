#!/usr/bin/python
# -*- coding:utf-8 -*-


"""

BSmodel Pricing(Analytical):

    vanilla: euro call option - vanilla_european_call()
             euro put  option - vanilla_european_put()

    spread: bull call option - bull_call()
            bull put option - bull_put()
            bear call option - bear_call()
            bear put option - bear_put()

    binary:  cash call option - bi_cashcall()
             cash put  option - bi_cashput()
             asset call option - bi_assetcall()
             asset put  option - bi_assetput()

    barrier: down-and-out-call option - down_call(knock='out')
             down-and-in-call  option - down_call(knock='in')
             up-and-out-call   option - up_call(knock='out')
              up-and-in-call   option - up_call(knock='in')
              up-and-out-put   option - up_put(knock='out')
              up-and-in-put    option - up_put(knock='in')
             down-and-out-put  option - down_put(knock='out')
             down-and-in-put   option - down_put(knock='in')


Reference: Options, Futures and Derivatives - John C.Hull

@author: Haoyang Wu
"""

import numpy as np
import scipy.stats as ss


class BSmodel:

    def __init__(self, S0=None, K=None, T=None, v=None, r=None, q=None):
        """

        :param S0:  Spot price
        :param K:   Strike price
        :param T:   Time to maturity  eg: 200/365
        :param v:   Volatility of asset
        :param r:   Risk free interest
        :param q:   Dividend Rate
        """
        self.S0 = S0
        self.K = K

        self.T = T
        self.v = v
        self.r = r
        self.q = q
        self.d1 = self._d1()
        self.d2 = self._d2()

    def _d1(self):
        d1 = (np.log(self.S0 / self.K) + (self.r - self.q + 0.5 * (self.v ** 2)) * self.T) / (self.v * np.sqrt(self.T))

        return d1

    def _d2(self):
        d2 = (np.log(self.S0 / self.K) + (self.r - self.q - 0.5 * (self.v ** 2)) * self.T) / (self.v * np.sqrt(self.T))

        return d2


class VanillaEuro(BSmodel):

    def vanilla_european_call(self):
        c = self.S0 * np.exp(-self.q * self.T) * ss.norm.cdf(self.d1, 0, 1)\
            - self.K * np.exp(-1 * self.r* self.T) * ss.norm.cdf(self.d2, 0, 1)

        return c

    def vanilla_european_put(self):
        p = self.K * np.exp(-1 * self.r * self.T) * ss.norm.cdf(-1 * self.d2, 0, 1)\
            - self.S0 * np.exp(-self.q * self.T) * ss.norm.cdf(-1 * self.d1, 0, 1)

        return p


class SpreadOption(BSmodel):

    def bull_call(self, K2):
        """ K2 is higher than K1 """
        S0, K, T, v , r, q = self.S0, self.K, self.T, self.v, self.r, self.q

        c1 = VanillaEuro(S0, K, T, v, r, q).vanilla_european_call()
        c2 = VanillaEuro(S0, K2, T, v, r, q).vanilla_european_call()

        bull = c2 - c1

        return bull

    def bull_put(self, K2):
        S0, K, T, v, r, q = self.S0, self.K, self.T, self.v, self.r, self.q

        p1 = VanillaEuro(S0, K, T, v, r, q).vanilla_european_put()
        p2 = VanillaEuro(S0, K2, T, v, r, q).vanilla_european_put()

        bull = p2 - p1

        return bull

    def bear_call(self, K2):
        S0, K, T, v, r, q = self.S0, self.K, self.T, self.v, self.r, self.q

        c1 = VanillaEuro(S0, K, T, v, r, q).vanilla_european_call()
        c2 = VanillaEuro(S0, K2, T, v, r, q).vanilla_european_call()

        bear = c1 - c2

        return bear

    def bear_put(self, K2):
        S0, K, T, v, r, q = self.S0, self.K, self.T, self.v, self.r, self.q

        p1 = VanillaEuro(S0, K, T, v, r, q).vanilla_european_put()
        p2 = VanillaEuro(S0, K2, T, v, r, q).vanilla_european_put()

        bull = p1 - p2

        return bull


class BinaryOption(BSmodel):

    def bi_cashcall(self, Q):
        cashcall = Q * np.exp(-1 * self.r * self.T) * ss.norm.cdf(self.d2, 0, 1)

        return cashcall

    def bi_cashput(self, Q):
        cashput = Q * np.exp(-1 * self.r * self.T) * ss.norm.cdf(-1 * self.d2, 0 , 1) * np.exp(-self.q * self.T)

        return cashput

    def bi_assetcall(self):
        assetcall = self.S0 * np.exp(-1 * self.r * self.T) * ss.norm.cdf(self.d1, 0, 1)

        return assetcall

    def bi_assetput(self):
        assetput = self.S0 * np.exp(-1 * self.r * self.T) * ss.norm.cdf(-1 * self.d1, 0, 1) * np.exp(-self.q * self.T)

        return assetput


class BarrierOption(BSmodel):

    def __init__(self, S0, K, T, v, r, q, barrier):
        super().__init__(S0, K, T, v, r, q)
        self.H = barrier
        self.lamb = self._lambda()
        self.y = self._y()
        self.x1 = self._x1()
        self.y1 = self._y1()

    def _lambda(self):
        lamb = (self.r - self.q + self.v ** 2 / 2) / self.v ** 2

        return lamb

    def _y(self):
        y = np.log(self.H ** 2 / (self.S0 * self.K)) / (self.v * np.sqrt(self.T)) + self.lamb * self.v * np.sqrt(self.T)

        return y

    def _x1(self):
        x1 = np.log(self.S0 / self.H) / (self.v * np.sqrt(self.T)) + self.lamb * self.v * np.sqrt(self.T)

        return x1

    def _y1(self):
        y1 = np.log(self.H / self.S0) / (self.v * np.sqrt(self.T)) + self.lamb * self.v * np.sqrt(self.T)

        return y1

    def down_call(self, knock='out'):
        """ knock: 'out' : down-and-out-call, 'in' : down-and-in-call """
        S0, K, T, v , r, q, H, lamb, y, x1, y1 = self.S0, self.K, self.T, self.v, self.r, self.q,\
                                                 self.H, self.lamb, self.y, self.x1, self.y1

        c = VanillaEuro(S0, K, T, v, r, q).vanilla_european_call()
        if K <= H:
            cdi = S0 * np.exp(-q * T) * (H / S0)**(2 * lamb) * ss.norm.cdf(y) \
                  - K * np.exp(-r * T) * (H / S0) ** (2 * lamb - 2) * ss.norm.cdf(y - v * np.sqrt(T))
            cdo = c - cdi

        else:
            cdo = S0 * ss.norm.cdf(x1) * np.exp(-q * T) - K * np.exp(-r * T) * ss.norm.cdf(x1 - v * np.sqrt(T)) \
                  - S0 * np.exp(-q * T) * (H / S0)**(2 * lamb) * ss.norm.cdf(y1) \
                  + K * np.exp(-r * T) * (H / S0)**(2 * lamb - 2) * ss.norm.cdf(y1 - v * np.sqrt(T))
            cdi = c - cdo

        if knock == 'out':
            price = cdo
        elif knock == 'in':
            price = cdi
        else:
            price = -1
            print('please input correct knock type')

        return price

    def up_call(self, knock='out'):
        """ knock: 'out' : up-and-out-call, 'in' : up-and-in-call """
        S0, K, T, v , r, q, H, lamb, y, x1, y1 = self.S0, self.K, self.T, self.v, self.r, self.q,\
                                                 self.H, self.lamb, self.y, self.x1, self.y1
        c = VanillaEuro(S0, K, T, v, r, q).vanilla_european_call()

        if H <= K:
            cuo = 0
            cui = c
        else:
            cui = S0 * ss.norm.cdf(x1) * np.exp(-q * T) - K * np.exp(-r * T) * ss.norm.cdf(x1 - v * np.sqrt(T)) \
                  - S0 * np.exp(-q * T) * (H / S0)**(2 * lamb) * (ss.norm.cdf(-y) - ss.norm.cdf(-y1)) \
                  + K * np.exp(-r * T) * (H / S0)**(2 * lamb - 2) * (ss.norm.cdf(-y + v * np.sqrt(T)) - ss.norm.cdf(-y1 + v * np.sqrt(T)))
            cuo = c - cui

        if knock == 'out':
            price = cuo
        elif knock == 'in':
            price = cui
        else:
            price = -1
            print('please input correct knock type')

        return price

    def up_put(self,  knock='out'):
        """ knock: 'out' : up-and-out-put, 'in' : up-and-in-put """
        S0, K, T, v , r, q, H, lamb, y, x1, y1 = self.S0, self.K, self.T, self.v, self.r, self.q,\
                                                 self.H, self.lamb, self.y, self.x1, self.y1
        p = VanillaEuro(S0, K, T, v, r, q).vanilla_european_put()

        if H >= K:
            pui = -S0 * np.exp(-q * T) *  (H / S0)**(2 * lamb) * ss.norm.cdf(-y)
            + K * np.exp(-r * T) * (H / S0)**(2 * lamb - 2) * ss.norm.cdf(-y + v * np.sqrt(T))
            puo = p - pui
        else:
            puo = -S0 * ss.norm.cdf(-x1) * np.exp(-q * T) + K * np.exp(-r * T) * ss.norm.cdf(-x1 + v * np.sqrt(T)) \
                  + S0 * np.exp(-q * T) * (H / S0)**(2 * lamb) * ss.norm.cdf(-y1) \
                  - K * np.exp(-r * T) * (H / S0)**(2 * lamb - 2) * ss.norm.cdf(-y1 + v * np.sqrt(T))
            pui = p - puo

        if knock == 'out':
            price = puo
        elif knock == 'in':
            price = pui
        else:
            price = -1
            print('please input correct knock type')

        return price

    def down_put(self, knock='out'):
        """ knock: 'out' : down-and-out-put, 'in' : down-and-in-put """
        S0, K, T, v, r, q, H, lamb, y, x1, y1 = self.S0, self.K, self.T, self.v, self.r, self.q, \
                                                self.H, self.lamb, self.y, self.x1, self.y1
        p = VanillaEuro(S0, K, T, v, r, q).vanilla_european_put()

        if H >= K:
            pdo = 0
            pdi = p
        else:
            pdi = -S0 * ss.norm.cdf(-x1) * np.exp(-q * T) + K * np.exp(-r * T) * ss.norm.cdf(-x1 + v * np.sqrt(T)) \
                  + S0 * np.exp(-q * T) * (H / S0)**(2 * lamb) * (ss.norm.cdf(y) - ss.norm.cdf(-y1)) \
                  - K * np.exp(-r * T) * (H / S0)**(2 * lamb - 2) * (ss.norm.cdf(y - v * np.sqrt(T)) - ss.norm.cdf(y1 - v * np.sqrt(T)))
            pdo = p - pdi

        if knock == 'out':
            price = pdo
        elif knock == 'in':
            price = pdi
        else:
            price = -1
            print('please input correct knock type')

        return price

