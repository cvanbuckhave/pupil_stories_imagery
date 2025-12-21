# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:54:37 2024

@author: cvanb
"""
# =============================================================================
# Import libraries
# =============================================================================
# Data visualisation
from matplotlib import pyplot as plt
import matplotlib as mpl
from datamatrix.colors.tango import blue, green, gray
import seaborn as sns

# Operations and pupil stuff
import numpy as np
import warnings
import datamatrix
from datamatrix import series as srs, NAN, operations as ops, convert

# Stats
import statsmodels.api as sm
import scipy.stats as stats
import time_series_test as tst
from statsmodels.formula.api import mixedlm
from scipy.stats import spearmanr, shapiro, wilcoxon, mannwhitneyu
from statsmodels.stats.diagnostic import het_white
from scipy.stats.distributions import chi2
from collections import Counter
import pandas as pd
# =============================================================================
# Set ups
# =============================================================================
# Edit the font, font size, grid color, axes width, etc.
plt.rcParams['axes.facecolor'] = 'white'
mpl.rcParams['font.family'] = 'Arial'
plt.rcParams['font.size'] = 30
plt.rcParams['xtick.color'] = 'black'
plt.rcParams['ytick.color'] = 'black'
plt.rcParams['axes.labelcolor'] = 'black'
plt.rcParams['legend.facecolor'] = 'inherit'
plt.rcParams['axes.linewidth'] = 2
plt.rcParams['axes.spines.right'] = False
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.edgecolor'] = 'black'
plt.rcParams['axes.facecolor'] = gray[0]

np.random.seed(123)  # Fix random seed for predictable outcomes
palette = [green[1], blue[1], gray[2], gray[3]]

# =============================================================================
#  Define useful functions
# =============================================================================

def count_nonnan(a):
    return np.sum(~np.isnan(a))

def compute_vars(dm_to_process, dm_bis=None, word=False, dur=True, means=True):
    # Copy of dm to not overwrite it
    warnings.filterwarnings("ignore", category=UserWarning)  
    warnings.filterwarnings("ignore", category=RuntimeWarning)  

    dm_ = dm_to_process
    print(f'Before preprocessing: N = {len(dm_.Subject.unique)} (n = {len(dm_)} trials)')
    
    # Save each trial duration    
    if dur==True:
        dm_.maxdur = NAN
        dm_.samples = NAN
        if word==False:
            for s, p, sdm in ops.split(dm_.Subject, dm_.Paragraph):
                trimmed = srs.trim(sdm.Pupil, value=NAN, end=True, start=False)
                # dm_.maxdur[sdm] = sdm.Timestamp[:, trimmed.depth-1] / 1000 # convert in seconds
                
                start_time = float(sdm.Timestamp[:, 0].unique[0])
                end_time   = float(sdm.Timestamp[:, trimmed.depth-1].unique[0])
                dm_.maxdur[sdm] = (end_time - start_time) / 1000  # in seconds
                dm_.samples[sdm] = trimmed.depth
                dm_.Timestamp[sdm] = sdm.Timestamp - start_time

        else:
            for s, p, w, sdm in ops.split(dm_.Subject, dm_.Paragraph, dm_.Word):
                dm_.maxdur[sdm] = sdm.Timestamp / 1000 # in seconds

    if means == True:
        dm_.Order = NAN
        # Compute the mean pupil size and mean slopes during listening
        if 'Vividness' not in dm_.column_names:
            dm_.Vividness, dm_.Suspense = NAN, NAN
        dm_.mean_pupil = NAN
        for s, par, sdm in ops.split(dm_.Subject, dm_.Paragraph):  
            # Order
            if par.endswith('1'):
                dm_.Order[sdm] = 1
            elif par.endswith('2'):
                dm_.Order[sdm] = 3
            else:
                dm_.Order[sdm] = 2
                
            # Add variables of interest
            dm_bis2 = dm_bis.Subject == s
            if dm_bis2.Vividness[dm_bis2.Paragraph == par].unique != []:
                dm_.Vividness[sdm] = dm_bis2.Vividness[dm_bis2.Paragraph == par].unique[0]
                dm_.Suspense[sdm] = dm_bis2.Suspense[dm_bis2.Paragraph == par].unique[0]
            
            # Take only the real duration of the trace minus 200 ms to compute the mean 
            # to prevent taking into account edge effects                     
            if count_nonnan(sdm.Pupil) > 0:            
                # Pupil-size mean, make sure to only average the pupil in the window of interest
                if len(sdm.Vividness.shape) > 1:
                    start, stop = 0, int(sdm.samples.unique[0] - 60) # Sampling freq = 60 Hz
                    dm_.mean_pupil[sdm] = srs.reduce(sdm.Pupil[:,start:stop], np.nanmean)
                    dm_.Vividness[sdm] = srs.reduce(sdm.Vividness, np.nanmean)
                    dm_.Suspense[sdm] = srs.reduce(sdm.Suspense, np.nanmean)
                else:
                    dm_.mean_pupil[sdm] = np.nanmean(sdm.Pupil)
    
        print('Mean pupil size were computed over the whole reading phase for each Paragraph.')
        
        # Create new variables 
        dm_.pupil_change = NAN
        dm_.mean_vivid, dm_.mean_suspense = NAN, NAN
        for p, s, sdm in ops.split(dm_.Subject, dm_.Story):
            # Compute pupil-size changes as differences in mean pupil size between dark - bright conditions
            dark = sdm.mean_pupil[sdm.Brightness == 'dark'].mean
            bright = sdm.mean_pupil[sdm.Brightness == 'bright'].mean
    
            if dark != NAN and bright != NAN:
                dm_.pupil_change[sdm] = dark - bright
            
            # Mean ratings per subtype (averaged across dark and bright stories)
            if len(sdm.Vividness.shape) > 1:
                dm_.mean_vivid[sdm] = srs.reduce(sdm.Vividness, np.nanmean)
                dm_.mean_suspense[sdm] = srs.reduce(sdm.Suspense, np.nanmean)
            else:
                dm_.mean_vivid[sdm] = np.nanmean(sdm.Vividness)
                dm_.mean_suspense[sdm] = np.nanmean(sdm.Suspense)
            
        print('Pupil-size difference scores were computed for each Story and Participant.')

        # How many trials left per participant?
        print(f'Number of trials left per participant: {np.sort(Counter(dm_.Subject), axis=None)}')
        print(f'How many participants have n number of trials: {Counter(Counter(dm_.Subject).values())}')
        
        print(f'After preprocessing: N = {len(dm_.Subject.unique)} (n = {len(dm_)} trials)')
    
    warnings.filterwarnings("default", category=UserWarning)  
    warnings.filterwarnings("default", category=RuntimeWarning)  
    return dm_

def plot_bars(dm, x=str, y=str, hue=None, hue_order=['bright', 'dark'], order=None, color=None, pal='deep', xlab='Condition', ylab='Mean Pupil Size (a.u.)', title='', fig=True, alpha=1, legend=True):
    """Plot the mean pupil size as bar plots."""
    plt.rcParams['font.size'] = 30
    
    if fig == True:
        plt.figure(figsize=(15,8))
        plt.subplot(1,1,1)
    plt.title(title, fontsize=30)
    sns.barplot(x=x, y=y, hue=hue, data=dm, palette=pal, order=order, hue_order=hue_order, errorbar=('se',1), color=color, alpha=alpha, legend=legend)
    plt.axhline(0, linestyle='solid', color='black')
    plt.xlabel(xlab);plt.ylabel(ylab)
    if fig == True:
        plt.legend(loc='lower right')
        plt.tight_layout()
        plt.show()
        
def quick_visualisation(dm_, trace=True, avg=False, cwd=str):
    """Pupil traces and mean pupil size per cond and per task."""
    dict_hues = {'story1': [green[1], green[0], blue[1]],
                 'story2': [green[1], green[0], blue[1]],
                 'story3': [green[1], blue[1], blue[0]],
                 'story4': [green[1], blue[1], blue[0]]}
    
    # Quick visualisation
    fig, axes = plt.subplots(1, 1, figsize=(28,10))
    plt.subplot(1,1,1)
    tst.plot(dm_, dv='Pupil', hue_factor='Brightness', hues=[green[1], blue[1]], 
             legend_kwargs={'frameon': False, 'loc': 'lower center', 'ncol': 2, 'fontsize': 38,'labels': [f'Bright (N={len(dm_[dm_.Brightness=="bright"])})', f'Dark (N={len(dm_[dm_.Brightness=="dark"])})']},
             annotation_legend_kwargs={'frameon': False, 'loc': 'upper right'}, 
             x0=0, sampling_freq=60)
    plt.xticks(np.arange(0, 90, 5), np.arange(0, 90, 5), fontsize=30)
    plt.xlim([0, 86])
    plt.ylim([-0.4, 0.2])
    plt.ylabel('Baseline-corrected\npupil size (a.u.)', fontsize=45)
    plt.xlabel('Time since story onset (s)', fontsize=40)
    plt.tight_layout()
    plt.savefig(cwd+"/figs/all_traceplot.png", bbox_inches='tight')
    plt.show()
    
    # 2. Mean pupil size per condition and interactions with vividness ratings 
    # Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
    dm_sub0 = convert.to_pandas(dm_)
    fig, axes = plt.subplots(1, 4, figsize=(52,13))
    fig.subplots_adjust(wspace=0.3)
    ax=plt.subplot(1,4,1)
    sns.barplot(x='Brightness', y='mean_pupil', hue=None, data=dm_sub0, palette=palette[0:2], order=['bright', 'dark'], hue_order=None, errorbar=('se',1), alpha=0.7)
    plt.axhline(0, linestyle='solid', color='black')
    plt.ylabel('Pupil-size means (a.u.)')
    handles, labels = ax.get_legend_handles_labels()
    plt.xticks([0, 1], ['Bright', 'Dark']);plt.ylim([-0.14, 0.005])
    ax=plt.subplot(1,4,2)
    dm_sub0 = dm_sub0[dm_sub0.mean_pupil != '']
    sns.barplot(x='mean_vivid', y='mean_pupil', hue='Brightness', data=dm_sub0, palette=palette[0:2], hue_order=['bright', 'dark'], order=None, errorbar=('se',1), alpha=0.7)
    plt.axhline(0, linestyle='solid', color='black');plt.ylim([-0.21, 0.21])
    plt.xlabel('Vividness ratings');plt.ylabel('Pupil-size means (a.u.)')
    handles, labels = ax.get_legend_handles_labels()
    plt.legend(handles=handles, labels=['Bright', 'Dark'], frameon=False)
    ax=plt.subplot(1,4,3)
    sns.barplot(x='mean_vivid', y='pupil_change', hue=None, data=dm_sub0, palette='crest', hue_order=None, order=None, errorbar=('se',1))
    plt.axhline(0, linestyle='solid', color='black');plt.yticks(fontsize=25)
    plt.xlabel('Vividness ratings');plt.ylabel('Pupil-size mean differences (a.u.)')
    ax=plt.subplot(1,4,4)
    test_correlation(dm_, y='mean_vivid', x='pupil_change', alt='greater', fig=True, lab='Vividness', color='green', fs=30)
    test_correlation(dm_, y='mean_suspense', x='pupil_change', alt='greater', fig=True, lab='Suspense', color='violet', fs=30)
    plt.xlabel('Pupil-size mean differences');plt.ylabel('Mean off-line ratings\n');plt.ylim([0.9, 9]);plt.yticks(range(1, 8))
    plt.show()
        
    # Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
    dm_df = convert.to_pandas(dm_)
    
    # Mean pupil sizes per condition (all)
    dm_sub1 = dm_df[dm_df.mean_pupil != '']
    dm_sub1 = dm_sub1[dm_sub1.pupil_change != '']
    plt.rcParams['font.size'] = 30

    fig, axes = plt.subplots(1, 1, figsize=(20,8))
    ax2=plt.subplot(1,1,1)
    sns.barplot(x='Story', y='mean_pupil', hue='Brightness', data=dm_sub1, palette=[green[1], blue[1]], order=['story1', 'story2', 'story3', 'story4'], hue_order=['bright', 'dark'], errorbar=('se',1), alpha=0.7)
    plt.axhline(0, linestyle='solid', color='black')
    handles, labels = ax2.get_legend_handles_labels()
    plt.xticks(range(0, 4), ['N°1', 'N°2', 'N°3', 'N°4'])
    #plt.ylim([-0.2, 0.01])
    plt.xlabel('Story');plt.ylabel('Mean pupil-size changes\nrelative to baseline (a.u.)', color='black')
    plt.legend(handles=handles, labels=['Bright', 'Dark'], frameon=False, title='Condition', loc='lower center', ncols=2)
    plt.tight_layout()
    plt.show()  
    
    fig, axes = plt.subplots(1, 1, figsize=(28,10))
    ax2=plt.subplot(1,1,1)
    sns.barplot(x='Story', y='pupil_change', hue='mean_vivid', data=dm_sub1, palette='crest', order=['story1', 'story2', 'story3', 'story4'], hue_order=None, errorbar=('se',1))
    plt.axhline(0, linestyle='solid', color='black')
    handles, labels = ax2.get_legend_handles_labels()
    plt.xticks(range(0, 4), ['N°1', 'N°2', 'N°3', 'N°4'])
    plt.xlabel('Story');plt.ylabel('Mean pupil-size differences\nbetween Dark and Bright (a.u.)', color='black')
    plt.legend(handles=handles, labels=range(1,8), frameon=False, title='Vividness', ncols=7, loc='upper center')
    plt.tight_layout()
    plt.show()   
  
# Stats
def lm_pupil(dm_tst, formula=False, re_formula="1 + Brightness", pupil_change=False, reml=False, method='Powell'):
    """Test how brightness (dark vs. light) affects the mean pupil size."""    
    # Make sure to filter the 'dynamic' condition
    dm_test = dm_tst.Subject != ''    
    
    # Remove Nans
    dm_valid_data = dm_test.mean_pupil != NAN # remove NaNs (it does it automatically but still)

    if pupil_change == True:
        # Suppress warnings because it's annoying
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        dm_valid_data = ops.group(dm_valid_data, by=[dm_valid_data.Subject, dm_valid_data.Story]) # add dm_sub.response_lang if necessary

        # Make sure to have only unique mean values for each variable per participant 
        for col in dm_valid_data.column_names:
            if type(dm_valid_data[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
                dm_valid_data[col] = srs.reduce(dm_valid_data[col]) # Compute the mean per subtype 
        
        # Unable back the warnings
        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)
        warnings.filterwarnings("default", category=UserWarning)  
        
        dm_valid_data = dm_valid_data.pupil_change != NAN # make sure there's always at least 2 stories to compare

    # The model
    dm_copy = convert.to_pandas(dm_valid_data)

    md = mixedlm(formula, dm_copy, 
                     groups='Subject',
                     re_formula=re_formula, missing='drop')
    
    mdf = md.fit(reml=reml, method='Powell')
        
    return mdf

def print_results(mdf):
    """Print nicely formatted fixed-effect results from a MixedLM fit."""
    summary_table = mdf.summary().tables[1]

    # Case 1: it's a SimpleTable (older versions)
    if hasattr(summary_table, "data"):
        df = pd.DataFrame(summary_table.data[1:], columns=summary_table.data[0])
    # Case 2: already a DataFrame (newer versions)
    elif isinstance(summary_table, pd.DataFrame):
        df = summary_table.copy()
    else:
        raise TypeError(f"Unexpected summary table type: {type(summary_table)}")

    # Clean and rename columns
    df = df.rename(columns={
        'Coef.': 'beta',
        'Std.Err.': 'SE',
        'P>|z|': 'p',
        '[0.025': 'CI_lower',
        '0.975]': 'CI_upper'
    })

    # Convert numeric columns
    numeric_cols = ['beta', 'SE', 'z', 'p', 'CI_lower', 'CI_upper']
    df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')

    # Print results (skip random effects if present)
    for i, row in df.iterrows():
        if "Var" in str(i) or "Cov" in str(i):  # skip variance/covariance terms
            continue
        print(f"{i}: β = {row['beta']:.3f}, SE = {row['SE']:.3f}, "
              f"z = {row['z']:.3f}, p = {row['p']:.3f}, "
              f"95% CI = [{row['CI_lower']:.3f}, {row['CI_upper']:.3f}]")
        
def compare_models(model1, model2, ddf):
    """Null hypothesis: The simpler model is true. 
    Log-likelihood of the model 1 for H0 must be <= LLF of model 2."""
    print(f'Log-likelihood of model 1 <= model 2: {model1.llf <= model2.llf}')
    
    ratio = (model1.llf - model2.llf)*-2
    p = chi2.sf(ratio, ddf) # How many more DoF does M2 has as compared to M1?
    if p >= .05:
        print(f'The simpler model is the better one (Chi({ddf}) = {round(ratio,3)}, p = {round(p,4)}, LLF_M1 = {round(model1.llf,3)}, LLF_M2 = {round(model2.llf,3)})')
    else:
        print(f'The simpler model is not the better one (Chi({ddf}) = {round(ratio,3)}, p = {round(p,4)}, LLF_M1 = {round(model1.llf,3)}, LLF_M2 = {round(model2.llf,3)})')


def check_assumptions(model):
    """Check assumptions for normality of residuals and homoescedasticity.
    Code from: https://www.pythonfordatascience.org/mixed-effects-regression-python/#assumption_check"""
    plt.rcParams['font.size'] = 40
    print('Assumptions check:')
    fig = plt.figure(figsize = (25, 16))
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    #fig.suptitle(f'{suptype}: {model.model.formula} (n = {model.model.n_groups})')
    # Normality of residuals
    ax1 = plt.subplot(2,2,1)
    sns.distplot(model.resid, hist = True, kde_kws = {"fill" : True, "lw": 4}, fit = stats.norm)
    ax1.set_title("KDE Plot of Model Residuals (Red)\nand Normal Distribution (Black)", fontsize=30)
    ax1.set_xlabel("Residuals")
    
    # Q-Q PLot
    ax2 = plt.subplot(2,2,2)
    sm.qqplot(model.resid, dist = stats.norm, line = 's', ax = ax2, alpha=0.5, markerfacecolor='black', markeredgecolor='black')
    ax2.set_title("Q-Q Plot", fontsize=30)
    
    # Shapiro
    labels1 = ["Statistic", "p-value"]
    norm_res = shapiro(model.resid)
    print('Shapir-Wilk test of normality')
    for key, val in dict(zip(labels1, norm_res)).items():
        print(key, val)
    lab1 = f'Shapiro (normality): Statistic = {np.round(norm_res[0],3)}, p = {np.round(norm_res[1],3)}'

    # Homogeneity of variances
    ax3 = plt.subplot(2,2,3)
    sns.scatterplot(y = model.resid, x = model.fittedvalues, alpha=0.8)
    ax3.set_title("RVF Plot", fontsize=30)
    ax3.set_xlabel("Fitted Values")
    ax3.set_ylabel("Residuals")
    
    ax4 = plt.subplot(2,2,4)
    sns.boxplot(x = model.model.groups, y = model.resid)
    plt.xticks(range(0, len(model.model.group_labels)), range(1, len(model.model.group_labels)+1), fontsize=15)
    ax4.set_title("Distribution of Residuals for Weight by Litter", fontsize=30)
    ax4.set_ylabel("Residuals")
    ax4.set_xlabel("Litter")
    
    # White’s Lagrange Multiplier Test for Heteroscedasticity
    print('White’s Lagrange Multiplier Test for Heteroscedasticity')
    het_white_res = het_white(model.resid, model.model.exog)
    labels = ["LM Statistic", "LM-Test p-value", "F-Statistic", "F-Test p-value"]
    for key, val in dict(zip(labels, het_white_res)).items():
        print(key, val)
    lab2 = f'LM Test (homoscedasticity): LM Statistic = {np.round(het_white_res[0],3)}, p = {np.round(het_white_res[1],3)}'
    
    fig.supxlabel(f'{lab1}\n{lab2}')
    plt.tight_layout()
    plt.show()
    
    warnings.filterwarnings("default", category=FutureWarning)
    warnings.filterwarnings("default", category=UserWarning)

def test_correlation(dm_c, x, y, alt='two-sided', pcorr=1, color='red', lab='Vividness', fig=True, fs=30):
    """Test the correlations between pupil measures and questionnaire measures using Spearman's correlation."""
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)

    # Group per participant 
    dm_cor = ops.group(dm_c, by=[dm_c.Subject])

    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_cor.column_names:
        if type(dm_cor[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_cor[col] = srs.reduce(dm_cor[col], operation=np.nanmean)
            
    # The variables to test the correlation
    x, y = dm_cor[x], dm_cor[y]
    
    # Unable back the warnings
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
    # Compute spearman's rank correlation
    cor=spearmanr(x, y, alternative=alt, nan_policy='omit')

    N = len(dm_c.Subject.unique)
    pval = cor.pvalue * pcorr # Apply Bonferroni correction (multiply p-values by the number of tests)
    if pval > 1.0:
        pval = 1.0
    if pval < 0.001:
        pval = '{:.1e}'.format(pval)
    else:
        pval = np.round(pval, 3)
        
    if fig == False:
        res = fr'{chr(961)} = {round(cor.correlation, 3)}, p = {pval}, n = {N}'
    else:
        res = fr'{chr(961)} = {round(cor.correlation, 3)}, p = {pval}'
    
    # Plot the correlations (linear regression model fit)
    plt.rcParams['font.size'] = fs

    if lab != False:
        label = fr'{lab}: {res}'
    else:
        label = ''
    
    print(label)

    if fig == True:
        sns.regplot(data=dm_cor, x=x.name, y=y.name, lowess=False, color=color, label=label, x_jitter=0, y_jitter=0, scatter_kws={'alpha': 0.5, 's': 100}, robust=True)
        plt.legend(frameon=False, markerscale=3, loc='upper center')
        # use statsmodels to estimate a nonparametric lowess model (locally weighted linear regression)
        sns.regplot(data=dm_cor, x=x.name, y=y.name, lowess=True, color=color, label=None, x_jitter=0, y_jitter=0, scatter_kws={'alpha': 0.0}, line_kws={'linestyle': 'dashed', 'alpha':0.6, 'linewidth': 8})
        
    return res


def dist_checks_wilcox(dm_to_check, story):
    """Non-parametric paired t-tests with Wilcoxon signed-rank test."""
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    dm_sub = dm_to_check.Story != story
    dm_sub = ops.group(dm_sub, by=[dm_sub.Subject, dm_sub.Brightness]) 

    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_sub.column_names:
        if type(dm_sub[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_sub[col] = srs.reduce(dm_sub[col]) # Compute the mean per subtype 
        
    print(f"\n{story} (n = {len(dm_sub[dm_sub.Brightness == 'bright'])})")
    
    print('Within participants: (bright vs.dark)')
    #print(f"Condition: {wilcoxon(dm_sub.mean_pupil[dm_sub.Brightness == 'bright'], dm_sub.mean_pupil[dm_sub.Brightness == 'dark'], alternative='less')}")
    print(dm_sub.mean_pupil[dm_sub.Brightness == 'bright'].mean, dm_sub.mean_pupil[dm_sub.Brightness == 'dark'].mean)
    print(dm_sub.mean_pupil[dm_sub.Brightness == 'bright'].std, dm_sub.mean_pupil[dm_sub.Brightness == 'dark'].std)

    # Print descriptives (experiment)
    for sup_, sdm in ops.split(dm_sub.Story):
        print(f'\nSuspense: M = {np.round(sdm.Suspense.mean,3)}, SD = {np.round(sdm.Suspense.std,3)}, n = {len(sdm)}')
        print(f'Vividness: M = {np.round(sdm.Vividness.mean,3)}, SD = {np.round(sdm.Vividness.std,3)}, n = {len(sdm)}')

    print('Between participants:')
    #print(f"Mean vividness per pupil-size changes polarity: {mannwhitneyu(dm_sub.mean_vivid[dm_sub.pupil_change>0], dm_sub.mean_vivid[dm_sub.pupil_change<0], alternative='greater')}")
    print(dm_sub.mean_vivid[dm_sub.pupil_change>0].mean, dm_sub.mean_vivid[dm_sub.pupil_change<0].mean)
    print(dm_sub.mean_vivid[dm_sub.pupil_change>0].std, dm_sub.mean_vivid[dm_sub.pupil_change<0].std)

    # Default back
    warnings.filterwarnings("default", category=FutureWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
