import warnings

warnings.filterwarnings('ignore')

import matplotlib as mpl
from matplotlib import pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.pyplot import GridSpec
from itertools import combinations, islice, takewhile
import seaborn as sns
import json
import mpld3
import numpy as np
import pandas as pd
import scipy.sparse

import os, sys
import warnings
from astropy import units as u
from astropy.coordinates import SkyCoord
import statsmodels.api as sm
sns.set_context('poster', font_scale=1.3)

from ipywidgets import interact, FloatSlider, SelectMultiple, interactive

# "constants"
from scipy.constants import c # speed of light in m/s

from itertools import combinations_with_replacement, permutations, combinations, product

from scipy.optimize import fmin_l_bfgs_b, basinhopping
from scipy.interpolate import LSQUnivariateSpline, splrep, splev, Akima1DInterpolator, interp1d


DIP_RA = 17.3
DIP_RA_ERR = 1.0
DIP_DEC = -61.0
DIP_DEC_ERR = 10.0
DIP_AMPLITUDE = 0.97e-5
DIP_AMPLITUDE_ERR = 0.21e-5 # average of asymmetric errors
DIP_MONOPOLE = -0.178e-5
DIP_MONOPOLE_ERR  = 0.084e-5

dipole = SkyCoord(DIP_RA, DIP_DEC, unit=(u.hourangle, u.deg))

# Data
vltqs = pd.read_csv('../data/vlt-transitions-new.tsv', sep='\t')
vltqs['trans'] = vltqs['transition'] + vltqs.four.astype(str)

keckqs = pd.read_csv("../data/keck-transitions-new.tsv", sep='\t')
keckqs['trans'] = keckqs.transition + keckqs.four.astype(str)

qvals = pd.read_csv("../data/qvalues.txt", sep=' ', index_col=0)

codes = pd.concat([keckqs.set_index('code')[['trans', 'wavelength']],
                   vltqs.set_index('code')[['trans', 'wavelength']]])

all_systems = pd.read_csv("../data/full-parse-new.tsv", sep='\t')

vlt_ccds = pd.read_csv("../data/vlt-ccd.csv", index_col='chip')

vlt = all_systems[all_systems.source.eq('VLT')].copy()
keck = all_systems[all_systems.source.eq('Keck')].copy()