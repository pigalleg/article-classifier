"""visualize.py
Plot distributions, heatmaps, bar charts.
"""
import matplotlib.pyplot as plt

def plot_ra_distribution(counts, out_path):
    plt.figure()
    plt.bar(range(len(counts)), counts)
    plt.savefig(out_path)
