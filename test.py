from parsing_utils import *

import os
import matplotlib.pyplot as plt

parser = EpisodeParser('./episodes',time_interval=0)
parser.load_episodes()
#
# parser.plot_episode(-2)
#
parser.plot_training()

#
# def calc_heading(p1, p2):
#     vector = p2[:2] - p1[:2]
#
#     angle = np.arctan2(vector[1], vector[0])
#     angle *= (180 / np.pi)
#     print('output:', angle)
#
#     if angle < 0:
#         return round(angle + 360, 2)
#     else:
#         return round(angle, 2)
#
#
#
# P1 = np.array([1,10])
# P2 = np.array([2,2])
#
# print(calc_heading(P1, P2))
#

