from parsing_utils import *
import os
import matplotlib.pyplot as plt

parser = EpisodeParser('./episodes',time_interval=0)
parser.load_episodes()