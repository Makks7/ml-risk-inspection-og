"""
NAICE-SPE Paper: ML-Enhanced Risk-Based Inspection
Configuration File - Local Execution Setup
"""

import os
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm

# Create local directories
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, 'data')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')
FIGURES_PAPER_DIR = os.path.join(BASE_DIR, 'figures', 'paper')
FIGURES_PRESENTATION_DIR = os.path.join(BASE_DIR, 'figures', 'presentation')

for directory in [DATA_DIR, RESULTS_DIR, FIGURES_PAPER_DIR, FIGURES_PRESENTATION_DIR]:
    os.makedirs(directory, exist_ok=True)

# Publication-quality settings for NAICE-SPE paper
PAPER_STYLE = {
    'figure.dpi': 300,
    'savefig.dpi': 600,
    'font.size': 10,
    'axes.labelsize': 11,
    'axes.titlesize': 12,
    'xtick.labelsize': 9,
    'ytick.labelsize': 9,
    'legend.fontsize': 9,
    'figure.titlesize': 14,
    'lines.linewidth': 1.5,
    'axes.linewidth': 0.8,
    'grid.alpha': 0.3,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.1
}

# Presentation settings (larger fonts, higher contrast)
PRESENTATION_STYLE = {
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'font.size': 14,
    'axes.labelsize': 16,
    'axes.titlesize': 18,
    'xtick.labelsize': 12,
    'ytick.labelsize': 12,
    'legend.fontsize': 12,
    'figure.titlesize': 20,
    'lines.linewidth': 2.5,
    'axes.linewidth': 1.2,
    'grid.alpha': 0.4,
    'figure.facecolor': 'white',
    'axes.facecolor': 'white',
    'savefig.facecolor': 'white',
    'savefig.bbox': 'tight',
    'savefig.pad_inches': 0.15
}

def set_paper_style():
    """Apply publication-quality style for NAICE-SPE paper"""
    plt.rcParams.update(PAPER_STYLE)

def set_presentation_style():
    """Apply presentation-ready style with larger fonts"""
    plt.rcParams.update(PRESENTATION_STYLE)

# API 581 Standard Colors
RISK_COLORS = {
    'Very Low': '#2E7D32',    # Dark Green
    'Low': '#81C784',          # Light Green
    'Medium': '#FFD54F',       # Yellow
    'High': '#FF8A65',         # Orange
    'Very High': '#C62828'     # Dark Red
}

RISK_COLORS_PRES = {
    'Very Low': '#1B5E20',
    'Low': '#4CAF50',
    'Medium': '#FFC107',
    'High': '#FF5722',
    'Very High': '#B71C1C'
}

print("Configuration loaded successfully")
print(f"Base directory: {BASE_DIR}")
print(f"Results will be saved to: {RESULTS_DIR}")
print(f"Paper figures: {FIGURES_PAPER_DIR}")
print(f"Presentation figures: {FIGURES_PRESENTATION_DIR}")