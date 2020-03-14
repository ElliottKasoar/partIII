#!/usr/bin/env python3
# -*- coding: utf-8 -*-

#Script to makes plots of activation functions
#Currently plots tanh, sigmoid and LeakyReLU

import numpy as np
import matplotlib.pyplot as plt

###############################################################################

x1_max = 6
x1 = np.linspace(-x1_max, x1_max, x1_max * 100 + 1)
y1 = 1 / (1+np.exp(-x1))

#title1 = r'$\sigma(x) = \frac{1}{1 + e^{-x}}$'

title1 = r'$\sigma(x) = (1 + e^{-x})^{-1}$'

fig1, ax1 = plt.subplots()
ax1.cla()
ax1.plot(x1, y1)
ax1.set_xlabel('x', fontsize=13)
ax1.set_ylabel(r'$\sigma (x) $', fontsize=13)
ax1.tick_params(axis="x", labelsize=13)
ax1.tick_params(axis="y", labelsize=13)
leg = ax1.legend([title1], loc='upper left', fontsize=13, handlelength=0)

for item in leg.legendHandles:
    item.set_visible(False)
plt.show()

fig1.savefig("./plots/sigmoid.eps", format='eps', dpi=1000)

###############################################################################
x2_max = 3

x2 = np.linspace(-x2_max, x2_max, x2_max * 100 + 1)
y2 = np.tanh(x2)

title2 = r'$\sigma (x) = tanh(x) $'


fig2, ax2 = plt.subplots()
ax2.cla()
ax2.plot(x2, y2)
ax2.set_xlabel('x', fontsize=13)
ax2.set_ylabel(r'$\sigma (x) $', fontsize=13)
ax2.tick_params(axis="x", labelsize=13)
ax2.tick_params(axis="y", labelsize=13)
leg = ax2.legend([title2], loc='upper left', fontsize=13, handlelength=0)

for item in leg.legendHandles:
    item.set_visible(False)
plt.show()

fig2.savefig("./plots/tanh.eps", format='eps', dpi=1000)

###############################################################################

x3_max = 4
alpha = 0.1

x3 = np.linspace(-x3_max, x3_max, (x3_max * 100 + 1))
y3 = np.zeros(x3_max * 100 + 1)
y3[:(x3_max * 100)//2] = np.multiply(x3[:(x3_max * 100)//2], alpha)
y3[(x3_max * 100)//2:] = x3[(x3_max * 100)//2:]

title3a = r'$\sigma (x) = \alpha x,\ x < 0 $'
title3b = r'$\sigma (x) = x,\ x \geq 0 $'

fig3, ax3 = plt.subplots()
ax3.cla()
ax3.plot(x3[:(x3_max * 100)//2], y3[:(x3_max * 100)//2], 'C0')
ax3.plot(x3[(x3_max * 100)//2:], y3[(x3_max * 100)//2:], 'C0')
ax3.set_ylim([-1,5])
ax3.set_xlabel('x', fontsize=13)
ax3.set_ylabel(r'$\sigma (x) $', fontsize=13)
ax3.tick_params(axis="x", labelsize=13)
ax3.tick_params(axis="y", labelsize=13)
leg = ax3.legend([title3a, title3b], loc='upper left', fontsize=13, handlelength=0)

for item in leg.legendHandles:
    item.set_visible(False)
plt.show()

fig3.savefig("./plots/leakyrelu.eps", format='eps', dpi=1000)


###############################################################################

#fig4, ax4 = plt.subplots()
#ax4.cla()
#ax4.plot(x1,y1)
#ax4.plot(x2,y2)
#ax4.plot(x3,y3)
#ax4.set_xlim([-4,4])
#ax4.set_xlabel('x')
#ax4.set_ylabel('f(x)')

###############################################################################
