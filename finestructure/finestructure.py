xlabel_wavelength = r"Wavelength $\left[\AA\right]$"

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

fit_dict = {}
key = (0, 0, 1)
knot_positions = [3000.0, 10000.0]


def get_nth_group(grouped, nth):
    """Returns the dataframe of the nth group in a pandas groupby object.
    Not to be confused with the nth per group call.
    """
    return grouped.get_group(list(grouped.groups.keys())[nth])


def observed_shifts(telescope='VLT'):
    waves = []
    shifts = []
    for index, row in all_systems[all_systems.source.eq(telescope)].iterrows():
        for tran in row['transitions'].split():
            rest_wave = qvals.loc[codes.loc[tran].trans].wave
            measured_wave = rest_wave * (1 + row.z_absorption)
            qval = qvals.loc[codes.loc[tran].trans].qval
            waves.append(measured_wave)
            shifts.append(shifted_velocity(row.delta_alpha, qval, rest_wave))
    return np.array(waves), np.array(shifts)


def parse_j2000(name):
    """Takes the J2000 name stored in the results and returns it in a format astropy can understand."""
    return ' '.join([name[1:3], name[3:5], name[5:7], name[7:10], name[10:12], name[12:]])

def j2000_to_theta(name):
    """Returns the angle (degrees) between the position on the sky from 
    a given `name` and the position of the dipole model from 2012, King."""
    c = SkyCoord(parse_j2000(name), unit=(u.hourangle, u.deg))
    return float(c.separation(dipole).to_string(decimal=True))

def dipole_alpha(name):
    """Returns the value of Delta alpha/alpha as given by the best fit 2012 King model for 
    the given name (position).
    """
    theta = j2000_to_theta(name)
    return (DIP_AMPLITUDE * np.cos(np.deg2rad(theta)) + DIP_MONOPOLE) * 1e6


def shifted_velocity(del_alpha, q, lamb):
    # vj =v0 + ∆α xj, xj =−2cqjλ0j,
    x = -2 * c * q * lamb
    return del_alpha * x * 1e-14


def VLT_distortion(measured_wave, 
                   cutoff=10000., 
                   slope1=0.06, 
                   intercept1 = -100.0,
                   slope2 =0.160,
                   intercept2=-1500.0,
                  ):
    """Telescope dependent distortion function for the VLT sample."""
    if measured_wave < cutoff:
        return measured_wave * slope1 + intercept1
    else:
        return measured_wave * slope2 + intercept2

def Keck_distortion(measured_wave, cutoff=10000.):
    """Telescope dependent distortion function for the Keck sample."""
    slope1 = .0600
    intercept1 = -100
    slope2 = .160
    intercept2 = -1500
    if measured_wave < cutoff:
        return measured_wave * slope1 + intercept1
    else:
        return measured_wave * slope2 + intercept2
    
def distorted_velocity(row, measured_wave):
    """Telescope dependent distortion function for the VLT sample."""
    if row.source == "VLT":
        return VLT_distortion(measured_wave)
    elif row.source == "Keck":
        return Keck_distortion(measured_wave)


def generate_sigmas(vshifts):
    return np.random.rand(len(vshifts)) * 30.0 + 50.0

def create_blank_transitions_dataset(source=all_systems):
    df = pd.DataFrame(columns=['system',
                               'J2000',
                               'source',
                               'wavelength',
                               'vshift', 
                               'sigma', 
                               'qval',
                               'rest_wave',
                              ])
    count = 0
    abs_count = 0
    for index, row in source.iterrows():
        waves = []
        rest_waves = []
        vshifts = []
        qvals_list = []
        for tran in row['transitions'].split():
            vshift = 0.0
            rest_wave = qvals.loc[codes.loc[tran].trans].wave
            measured_wave = rest_wave * (1 + row.z_absorption)
            qval = qvals.loc[codes.loc[tran].trans].qval
            waves.append(measured_wave)
            rest_waves.append(rest_wave)
            vshifts.append(vshift)
            qvals_list.append(qval)
        vshifts = np.array(vshifts)
        errors = generate_sigmas(vshifts)
#         vshifts += errors * np.random.randn(len(vshifts))
#         vshifts = vshifts - vshifts[0]
        for single in range(len(vshifts)):
            abs_count += 1
            df.loc[abs_count] = [int(index), #system
                                 row['J2000'],
                                 row.source,
                                 waves[single], #wavelength 
                                 vshifts[single],
                                 errors[single],
                                 qvals_list[single], # qvalues
                                 rest_waves[single],
                                ]
    df['system'] = df.system.astype(int)
    # For numerical stability during fitting, dividing the x-values by 1.0e14
    df['x'] = -2.0 * c * df['qval'] * df['rest_wave'] / 1.0e14
    return df


def fit_alpha(dataframe):
    """Takes a dataframe of transitions and returns a systems dataframe w/ best fit sim_fit_alpha."""
    new_df = all_systems.copy()
    new_df['sim_fit_alpha'] = -1.0
    indices = []
    slopes = []
    for index, dfg in dataframe.groupby('system'):
        design_matrix = sm.add_constant(dfg.x)
        results = sm.WLS(dfg.vshift, design_matrix, weights=1.0/dfg.sigma).fit()
        chisq = np.sum((dfg.vshift - results.fittedvalues)**2.0 / (dfg.sigma) ** 2.0)
        const, slope = results.params
        indices.append(index)
        slopes.append(slope)
    new_df['sim_fit_alpha'].iloc[indices] = slopes
    return new_df



def fit_hypothesis(system, dataframe1, hypothesis):
    """Return the chisq and the fit model object for a given dataframe and hypothesis.
    system=0
    dataframe1=df_a
    hypothesis='x'
    """
    plotdf1 = dataframe1[dataframe1.system == system]
    assert(hypothesis in ['x', 'w'])
    if hypothesis == 'x':
        X = sm.add_constant(plotdf1.x)
    elif hypothesis == 'w':
        X = sm.add_constant(plotdf1.wavelength)
    results = sm.WLS(plotdf1.vshift, X, weights=1.0/plotdf1.sigma).fit()
    chisq = np.sum((plotdf1.vshift - results.fittedvalues)**2.0 / (plotdf1.sigma) ** 2.0)
    return chisq, results

def generate_dataset(gen_dipole_alpha=True,
                     wavelength_distortion=False,
                     seed=228,
                    ):
    df = pd.DataFrame(columns=['system',
                               'J2000',
                               'source',
                               'wavelength',
                               'vshift', 
                               'sigma', 
                               'qval',
                               'rest_wave'
                              ])
    count = 0
    abs_count = 0
    for index, row in all_systems.iterrows():
        waves = []
        rest_waves = []
        vshifts = []
        qvals_list = []
        for tran in row['transitions'].split():
            vshift = 0.0
            rest_wave = qvals.loc[codes.loc[tran].trans].wave
            measured_wave = rest_wave * (1 + row.z_absorption)
            qval = qvals.loc[codes.loc[tran].trans].qval
            if gen_dipole_alpha:
                vshift += shifted_velocity(row['dipole_delta_alpha'],
                                           qval,
                                           rest_wave)
            if wavelength_distortion:
                vshift += distorted_velocity(row, measured_wave)
            waves.append(measured_wave)
            rest_waves.append(rest_wave)
            vshifts.append(vshift)
            qvals_list.append(qval)
        vshifts = np.array(vshifts)
        errors = generate_sigmas(vshifts)
        vshifts += errors * np.random.randn(len(vshifts))
        vshifts = vshifts - vshifts[0]
        for single in range(len(vshifts)):
            abs_count += 1
            df.loc[abs_count] = [int(index), #system
                                 row['J2000'],
                                 row.source,
                                 waves[single], #wavelength 
                                 vshifts[single],
                                 errors[single],
                                 qvals_list[single], # qvalues
                                 rest_waves[single],
                                ]
    df['system'] = df.system.astype(int)
    # For numerical stability during fitting, dividing the x-values by 1.0e14
    df['x'] = -2.0 * c * df['qval'] * df['rest_wave'] / 1.0e14
    return df

def fit_alpha(dataframe):
    design_matrix = sm.add_constant(dataframe['x'])
    vshifts = dataframe['vshift']
    sigmas = dataframe['sigma']
    results = sm.WLS(vshifts, design_matrix, weights=1.0/sigmas).fit()
    chisq = np.sum((vshifts - results.fittedvalues)**2.0 / (sigmas) ** 2.0)
    return (chisq, results)

def read_systematic(infile='../data/run.17.json'):
    """Read systematic error file."""
#     with open('../data/run.17.json', 'w') as fp:
#         json.dump(run17, fp, indent=4, sort_keys=True)
    with open(infile, 'r') as fp:
        temp = dict(json.load(fp))
    return {int(k):v for k, v in temp.items()}


# Plots


def plot_observation(dataframe,
                    color_index=0,
                    fig=None,
                    ax=None,
                   ):
    if fig == None:
        fig = plt.gcf()
    if ax == None:
        ax = plt.gca()
    vshifts = dataframe['vshift']
    sigmas = dataframe['sigma']
    xvals = dataframe['wavelength']
    ax.scatter(xvals, vshifts, color=sns.color_palette()[color_index], label='')
    ax.errorbar(xvals, vshifts, yerr=sigmas, c=sns.color_palette()[color_index],  ls='none', label='')
    ax.hlines(0, 3000, 10000, linestyles=':', lw=1.5, color='k')
    ax.set_xlim(3100, 9800)
    ax.set_ylabel("vshift [m/s]")
    ax.set_xlabel(xlabel_wavelength)


def plot_system_fit(dataframe,
                    color_index=0,
                    fig=None,
                    ax=None,
                   ):
    if fig == None:
        fig = plt.gcf()
    if ax == None:
        ax = plt.gca()

    chisq, results = fit_alpha(dataframe)
    const, slope = results.params
    vshifts = dataframe['vshift']
    sigmas = dataframe['sigma']
    xvals = dataframe['x']
    ax.scatter(xvals, vshifts, color=sns.color_palette()[color_index], label='')
    ax.errorbar(xvals, vshifts, yerr=sigmas, c=sns.color_palette()[color_index],  ls='none', label='')
    xmin = -35.
    xmax = 20.0
    
    ax.hlines(0, xmin, xmax, linestyles=':', lw=1.5, color='k')
    ax.plot(xvals, results.fittedvalues, color='k', label='Fit slope: ' + str(round(slope, 2)) + " ppm")
    ax.set_xlim(xmin, xmax)
    ax.set_ylabel("vshift [m/s]")
    ax.set_xlabel("x")

    ax.legend(loc='best')
    return slope
    
    
def plot_obs_x_zval(dataframe, all_systems, system_index, errorcol='error_delta_alpha', color_index=1):
    fig, axes = plt.subplots(figsize=(12, 12), nrows=3)
    plot_observation(dataframe, ax=axes[0])
    slope = plot_system_fit(dataframe, ax=axes[1])
    x = all_systems.iloc[system_index]['z_absorption']
    y = slope
    e = all_systems.iloc[system_index][errorcol]
    axes[2].scatter(x, (y), c=sns.color_palette()[color_index], label='', s=40)
    axes[2].errorbar(x, (y), yerr=e, c=sns.color_palette()[color_index],  ls='none', label='')
    axes[2].hlines(0, -2, 6, linestyles=':', lw=1.5, color='k')
    axes[2].set_xlim(0.3, 3.9)
    axes[2].set_ylim(-150, 150)
    axes[2].set_ylabel(r"$\Delta \alpha/\alpha$")
    axes[2].set_xlabel(r"Redshift [z]")
    
    fig.suptitle("System: " + str(system_index))
    fig.tight_layout()
    fig.subplots_adjust(top=0.93)

def plot_system_to_ensemble(grouped,
                            system_index,
                            all_systems,
                            errorcol='error_delta_alpha',
                            groupby='system',
                            color_index=1):
    dataframe = get_nth_group(grouped, system_index)
    plot_obs_x_zval(dataframe, all_systems=all_systems, system_index=system_index)

def plot_example_telescope_results(dataframe=keck):
    fig, ax = plt.subplots(figsize=(12, 8))
    x = dataframe.z_absorption
    y = dataframe.delta_alpha
    e = dataframe.error_delta_alpha
    color=0
    label='Slopes'
    ax.scatter(x, y, c=sns.color_palette()[color], label=label)
    ax.errorbar(x, y, yerr=e, c=sns.color_palette()[color],  ls='none', label='')
    ax.hlines(0, -2, 6, linestyles=':', lw=1.5, color='k')
    ax.set_ylabel(r"Slopes per Company ($\Delta \alpha/\alpha$)")
    ax.set_xlabel(r"Decades Ago (Keck redshift [z])")
    ax.legend(loc='best')
    ax.set_xlim(0.3, 3.7)
    ax.set_ylim(-200, 200)
    fig.tight_layout()
    

def plot_example_company(company, # which system to use
                         daa, # ppm of generated slope
                         df_a,
                        ):
    """company=19, daa=2.5, df_a="""
    
    row = all_systems.iloc[company]
    df = df_a[df_a.system==company]
    heights = df.x

    color_index = 0

    waves = []
    rest_waves = []
    vshifts = []
    qvals_list = []
    for tran in row['transitions'].split():
        vshift = 0.0
        rest_wave = qvals.loc[codes.loc[tran].trans].wave
        measured_wave = rest_wave * (1 + row.z_absorption)
        qval = qvals.loc[codes.loc[tran].trans].qval
        vshift += shifted_velocity(daa,
                                   qval,
                                   rest_wave)

        waves.append(measured_wave)
        rest_waves.append(rest_wave)
        vshifts.append(vshift)
        qvals_list.append(qval)

    waves = np.array(waves)
    rest_waves = np.array(rest_waves)
    vshifts = np.array(vshifts)
    qvals_list = np.array(qvals_list)
    sigmas = np.ones_like(waves) * 5.0
    
    vshifts += sigmas * np.random.randn(len(vshifts))
    fig, ax = plt.subplots(figsize=(12, 8))

    design_matrix = sm.add_constant(heights)

    results = sm.WLS(vshifts, design_matrix, weights=1.0/sigmas).fit()
    chisq = np.sum((vshifts - results.fittedvalues)**2.0 / (sigmas) ** 2.0)
    const, slope = results.params

    ax.scatter(heights, vshifts, color=sns.color_palette()[color_index], label='')
    ax.errorbar(heights, vshifts, yerr=sigmas, c=sns.color_palette()[color_index],  ls='none', label='')
    ax.plot(heights, results.fittedvalues, color='k', label='Fit slope: ' + str(round(slope, 2)) + " ppm")
    
    ax.legend(loc='best')
    ax.set_xlabel("Height")
    ax.set_ylabel("Salary")
    ax.set_title("Company: " + str(company) + " Generating slope: " + str(daa))
    fig.tight_layout()


def plot_example_telescope_bins(nbins = 12,
                                alphacol = 'delta_alpha',
                                errorcol = 'error_delta_alpha',
                                binned_lim = 25.0,
                                dataframe=keck,
                               ):
    fig, (ax, ax2) = plt.subplots(figsize=(12, 8), 
                                  nrows=2,
                                  sharex=True,
                                  gridspec_kw={'height_ratios':[2., 1.]},
                                 )


    for index, df in enumerate(np.array_split(dataframe.sort_values('z_absorption'), nbins)):
        color = sns.color_palette(n_colors=13)[index]

        x = df.z_absorption
        y = df[alphacol]
        e = df[errorcol]
        ax.scatter(x, (y), c=color, label='', s=40)
        ax.errorbar(x, (y), yerr=e, c=color,  ls='none', label='')

        x = np.average(df.z_absorption)
        y = np.average(df[alphacol], weights=(1.0 / (df[errorcol] ** 2.0)))
        e = np.sqrt(1.0 / np.sum(1.0 / (df[errorcol] ** 2.0)))
        label=''
        if index == 0:
            label=label
        else:
            label=''
        ax2.scatter(x, y, c=color, label=label)
        ax2.errorbar(x, y, yerr=e, c=color)

    ax.hlines(0, -2, 6, linestyles=':', lw=1.5, color='k')
    ax.hlines(binned_lim, -2, 6, linestyles=':', lw=.5, color='k')
    ax.hlines(-binned_lim, -2, 6, linestyles=':', lw=.5, color='k')
    ax2.hlines(0, -2, 6, linestyles=':', lw=1.5, color='k')

    ax.set_ylabel(r"Slopes per Company ($\Delta \alpha/\alpha$)")
    ax2.set_ylabel(r"Weighted Binneds")
    ax2.set_xlabel(r"Decades Ago (Redshift [z])")
    ax.legend(loc='best')
    ax.set_xlim(0.3, 3.9)
    ax.set_ylim(-150, 150)
    ax2.set_ylim(-binned_lim, binned_lim)
    fig.tight_layout()


def plot_ensemble_distortion(fit_dictionary=fit_dict,
                             key=key,
                             xlabel=xlabel_wavelength,
                             knot_positions=knot_positions,
                             ylabel="Distortion [m/s]",
                             plot_title="Ensemble Wavelength Distortion",
                            ):
    x = np.linspace(np.min(knot_positions), np.max(knot_positions), num=2000)
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.plot(x, fit_dictionary[key]['interpolate'](x))
    ax.set_ylabel(ylabel)
    ax.set_xlabel(xlabel)
    ax.set_title(plot_title)
    fig.tight_layout()

def plot_a_v_z(dataframe, 
               alphacol='delta_alpha', 
               errorcol='extra_error_delta_alpha', 
               nbins=13,
               color=0,
               ylims=(20),
               label='',
               fig=None,
               ax=None,
              ):
    if fig == None:
        fig = plt.gcf()
    if ax == None:
        ax = plt.gca()
    for index, df in enumerate(np.array_split(dataframe.sort_values('z_absorption'), nbins)):
        x = np.average(df.z_absorption)
        y = np.average(df[alphacol], weights=(1.0 / (df[errorcol] ** 2.0)))
        e = np.sqrt(1.0 / np.sum(1.0 / (df[errorcol] ** 2.0)))
        if index == 0:
            label=label
        else:
            label=''
        ax.scatter(x, y, c=sns.color_palette()[color], label=label)
        ax.errorbar(x, y, yerr=e, c=sns.color_palette()[color])
    ax.hlines(0, -2, 6, linestyles=':', lw=1.5, color='k')
    ax.set_ylabel(r"$\Delta \alpha/\alpha$")
    ax.set_xlabel(r"Redshift [z]")
    ax.legend(loc='best')
    ax.set_xlim(0.3, 3.7)
    ax.set_ylim(-ylims, ylims)
    fig.tight_layout()

def plot_a_v_zresid(dataframe, 
                    dataframe2,
                    alphacol='delta_alpha',
                    alphacol2='dipole_delta_alpha',
                    errorcol='extra_error_delta_alpha', color=0, label=''):
    """Measured - model"""
    fig = plt.gcf()
    ax = plt.gca()
    nbins = 13
    for index in range(nbins):
        df = np.array_split(dataframe.sort_values('z_absorption'), nbins)[index]
        x = np.average(df.z_absorption)
        y = np.average(df[alphacol], weights=(1.0 / (df[errorcol] ** 2.0)))
        e = np.sqrt(1.0 / np.sum(1.0 / (df[errorcol] ** 2.0)))
        
        df2 = np.array_split(dataframe2.sort_values('z_absorption'), nbins)[index]
        x2 = np.average(df2.z_absorption)
        y2 = np.average(df2[alphacol2], weights=(1.0 / (df2[errorcol] ** 2.0)))
        e2 = np.sqrt(1.0 / np.sum(1.0 / (df2[errorcol] ** 2.0)))
        if index == 0:
            label=label
        else:
            label=''
        ax.scatter(x, (y - y2), c=sns.color_palette()[color], label=label)
        ax.errorbar(x, (y - y2), yerr=e, c=sns.color_palette()[color])
    ax.hlines(0, -2, 6, linestyles=':', lw=1.5, color='k')
    ax.set_ylabel(r"Residual $\Delta \alpha/\alpha$")
    ax.set_xlabel(r"Redshift [z]")
    ax.legend(loc='best')
    ax.set_xlim(0.3, 3.7)
    ax.set_ylim(-20, 20)
    fig.tight_layout()

def plot_a_v_theta(dataframe,
                   alphacol='delta_alpha',
                   alphacol2='dipole_delta_alpha',
                   nbins=13,
                   errorcol='extra_error_delta_alpha', color=0, label=''):
    """Measured - model"""
    fig = plt.gcf()
    ax = plt.gca()
    for index in range(nbins):
        df = np.array_split(dataframe.sort_values('dipole_angle'), nbins)[index]
        x = np.average(df.dipole_angle)
        y = np.average(df[alphacol], weights=(1.0 / (df[errorcol] ** 2.0)))
        e = np.sqrt(1.0 / np.sum(1.0 / (df[errorcol] ** 2.0)))
        
        if index == 0:
            label=label
        else:
            label=''
        ax.scatter(x, (y), c=sns.color_palette()[color], label=label)
        ax.errorbar(x, (y), yerr=e, c=sns.color_palette()[color])
    ax.hlines(0, -2, 200, linestyles=':', lw=1.5, color='k')
    ax.vlines(90, -30, 30, linestyles=':', lw=1.5, color='k')
    ax.set_ylabel(r"Residual $\Delta \alpha/\alpha$")
    ax.set_xlabel(r"$\Theta$, angle from best-fitting dipole [degrees]")
    ax.legend(loc='best')
    ax.set_xlim(0.0, 180.0)
    ax.set_ylim(-25, 25)
    fig.tight_layout()


def plot_system(system, dataframe):
    plotdf = dataframe[dataframe.system == system]
    fig, (ax1, ax2) = plt.subplots(figsize=(12, 10), nrows=2)
    sns.regplot('wavelength', 'vshift', data=plotdf, ax=ax1)
    sns.regplot('x', 'vshift', data=plotdf, ax=ax2)
    fig.tight_layout()

def plot_hypotheses(system, dataframe1, dataframe2):
    plotdf1 = dataframe1[dataframe1.system == system]
    plotdf2 = dataframe2[dataframe2.system == system]
    fig, ((ax2, ax4), (ax3, ax1)) = plt.subplots(figsize=(14, 10), nrows=2, ncols=2, )
    
    chi_one, mod_one = fit_hypothesis(system=system, dataframe1=dataframe1, hypothesis='w')
    ax1.errorbar(plotdf1.wavelength, plotdf1.vshift, yerr=plotdf1.sigma,  ls='none', 
                 label=r'$\chi^2$ = ' + str(np.round(chi_one, 2)), color=sns.color_palette()[2])
    ax1.scatter(plotdf1.wavelength, plotdf1.vshift, color=sns.color_palette()[2], label='')
    ax1.plot(plotdf1.wavelength, mod_one.fittedvalues, color='k')

    chi_one, mod_one = fit_hypothesis(system=system, dataframe1=dataframe1, hypothesis='x')
    ax3.errorbar(plotdf1.x, plotdf1.vshift, yerr=plotdf1.sigma,  ls='none', 
                 label=r'$\chi^2$ = ' + str(np.round(chi_one, 2)), color=sns.color_palette()[0])
    ax3.scatter(plotdf1.x, plotdf1.vshift, color=sns.color_palette()[0], label='')
    ax3.plot(plotdf1.x, mod_one.fittedvalues, color='k')

    chi_one, mod_one = fit_hypothesis(system=system, dataframe1=dataframe2, hypothesis='w')
    ax2.errorbar(plotdf2.wavelength, plotdf2.vshift, yerr=plotdf2.sigma,  ls='none', 
                 label=r'$\chi^2$ = ' + str(np.round(chi_one, 2)), color=sns.color_palette()[0])
    ax2.scatter(plotdf2.wavelength, plotdf2.vshift, color=sns.color_palette()[0], label='')
    ax2.plot(plotdf2.wavelength, mod_one.fittedvalues, color='k')

    chi_one, mod_one = fit_hypothesis(system=system, dataframe1=dataframe2, hypothesis='x')
    ax4.errorbar(plotdf2.x, plotdf2.vshift, yerr=plotdf2.sigma,  ls='none', 
                 label=r'$\chi^2$ = ' + str(np.round(chi_one, 2)), color=sns.color_palette()[2])
    ax4.scatter(plotdf2.x, plotdf2.vshift, color=sns.color_palette()[2], label='')
    ax4.plot(plotdf2.x, mod_one.fittedvalues, color='k')

    autoAxis = ax2.axis()
    rec = Rectangle((autoAxis[0],
                     autoAxis[2]),
                    (autoAxis[1]-autoAxis[0]),
                    (autoAxis[3]-autoAxis[2]),
                    fill=False,lw=2)
    rec = ax2.add_patch(rec)
    rec.set_clip_on(False)
    
    autoAxis = ax3.axis()
    rec = Rectangle((autoAxis[0],
                     autoAxis[2]),
                    (autoAxis[1]-autoAxis[0]),
                    (autoAxis[3]-autoAxis[2]),
                    fill=False,lw=2)
    rec = ax3.add_patch(rec)
    rec.set_clip_on(False)
    
    ax3.set_title(r"$\alpha$ process $\alpha$ fit")
    ax2.set_title(r"W process W fit")
    ax1.set_title(r"$\alpha$ process W fit")
    ax4.set_title(r"W process $\alpha$ fit")
    ax2.set_ylabel("vshift [m/s]")
    ax3.set_ylabel("vshift [m/s]")
    ax1.set_xlabel("wavelength")
    ax2.set_xlabel("wavelength")
    ax3.set_xlabel("x")
    ax4.set_xlabel("x")
    leg = ax1.legend(handlelength=0, handletextpad=0, fancybox=True, frameon=True, facecolor='white', loc='best')
    for item in leg.legendHandles:
        item.set_visible(False)
    leg = ax2.legend(handlelength=0, handletextpad=0, fancybox=True, frameon=True, facecolor='white', loc='best')
    for item in leg.legendHandles:
        item.set_visible(False)
    leg = ax3.legend(handlelength=0, handletextpad=0, fancybox=True, frameon=True, facecolor='white', loc='best')
    for item in leg.legendHandles:
        item.set_visible(False)
    leg = ax4.legend(handlelength=0, handletextpad=0, fancybox=True, frameon=True, facecolor='white', loc='best')
    for item in leg.legendHandles:
        item.set_visible(False)
    fig.tight_layout()

def plot_example_systematic(region_dictionary):
    color = "black"
    linewidth = 5.0
    fig, ax = plt.subplots(figsize=(12, 8))
    for index in region_dictionary:
        begin = region_dictionary[index]['waves_start']
        end = region_dictionary[index]['waves_end']
        wavelength = np.linspace(begin, end, 50)
        quad = region_dictionary[index]['quad']
        slope = region_dictionary[index]['slope']
        offset = region_dictionary[index]['offset']
        if index == 1:
            color = "blue"
            linestyle = "-"
        elif index == 3:
            color = "red"
            linestyle = '-'
        else:
            color = "black"
            linestyle = '--'
        ax.plot(wavelength, quad * wavelength ** 2.0 + wavelength * slope + offset,
                 color=color, 
                 linewidth=linewidth, 
                 linestyle=linestyle)

    ax.set_xlim(2900, 7500)
    ax.set_xlabel("Hairlength")
    ax.set_ylabel("Systematic Error of Salary")
    fig.tight_layout()

def plot_absorption(specie=['Al', 'Mg'],
                    z=1.3,
                    daa=0.05,
                   ):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hlines(1, 0, 10000)
    for tran, row in qvals[qvals.index.str.startswith(tuple(specie))].iterrows():
        ax.vlines((1.0 + z) * row.wave + row.qval * ((daa + 1)**2.0-1.0), 
                      0, 1, color=plot_colors[tran[:2]])
        ax.vlines((1.0 + z) * row.wave, 
                      0, 1, color=plot_colors[tran[:2]], alpha=0.2)
        
    ax.set_xlim(3000, 8e3)
    ax.set_ylim(0, 1.1)
    ax.set_ylabel("Normalized Flux")
    ax.set_xlabel(r"Wavelength $\AA$")
    
    for spec in specie:
        ax.vlines(-1, -1, 0, color=plot_colors[spec], label=spec)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    # plot_absorption(specie=['Al', 'Mg', 'Fe', 'Ni'], )

def plot_shift(specie=['Al', 'Mg'],
                    z=1.3,
                    daa=0.05,
                   ):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.hlines(0, 0, 10000)
    for tran, row in qvals[qvals.index.str.startswith(tuple(specie))].iterrows():
        ax.scatter((1.0 + z) * row.wave, 
#                    row.qval * ((daa + 1)**2.0-1.0),
                   shifted_velocity(daa, row.qval, (1.0 + z) * row.wave),
                   color=plot_colors[tran[:2]], zorder=3)
        
    ax.set_xlim(3000, 8e3)
    ax.set_ylim(-200, 200)
    ax.set_ylabel("Velocity Shift [m/s]")
    ax.set_xlabel(r"Observed Wavelength $\AA$")
    
    for spec in specie:
        ax.vlines(-1, -1, 0, color=plot_colors[spec], label=spec, zorder=-3)
    box = ax.get_position()
    ax.set_position([box.x0, box.y0, box.width * 0.8, box.height])
    ax.legend(loc='center left', bbox_to_anchor=(1, 0.5))
    fig.tight_layout()
    # plot_shift(specie=['Al', 'Mg', 'Fe', 'Ni'], )