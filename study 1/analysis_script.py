# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:29:48 2024

@author: Claire Vanbuckhave
"""
# =============================================================================
# Set up 
# =============================================================================
# Import Libraries
import os 
from matplotlib import pyplot as plt
from datamatrix import NAN
from datamatrix.colors.tango import blue, gray, green
import numpy as np
import seaborn as sns
from datamatrix import operations as ops
import pandas as pd
from datamatrix import convert
from custom_func import (get_raw, check_blinks, preprocess_dm, supp_plots,
                         lm_pupil, compare_models, lm_slope, plot_traces,
                         test_correlation, check_assumptions,
                         merge_dm_df, plot_dist_scores, plot_correlations,
                         create_control_variables, dist_checks, main_plots)
from collections import Counter

# Define useful variables
cwd=os.getcwd() # auto
cwd='C:/Users/cvanb/Desktop/visual_imagery_pupil/study 1/' # manual
datafolder=cwd+'/data/' # the folder where the EDF files are
outputfolder=cwd+'/output/'
questfile='/results-survey.csv' # the questionnaire data

np.random.seed(111)  # Fix random seed for predictable outcomes
palette = [green[1], blue[1], gray[2], gray[3]]

# =============================================================================
# Prepare the data
# =============================================================================
# Get info about stimuli duration in a dataframe
stim_dur = pd.read_csv(cwd+'/stim_duration.csv',sep=',')

# Get the pupil data in a datamatrix
dm_og = get_raw(datafolder, stim_dur)

# Get questionnaires data in a dataframe
df = pd.read_csv(cwd+questfile,sep=',')

# Check blink rate during listening
dm_og_ = check_blinks(dm_og, new=True)

# Preprocess and apply baseline-correction
dm_ = preprocess_dm(dm_og_)

# Re-check blinks
_ = check_blinks(dm_)

# =============================================================================
# Raw pupil traces
# =============================================================================
# Pupil traces for non-dynamic
legend = plot_traces(dm_og_, plot_type='non-dynamic')
# Pupil traces for dynamic
_ = plot_traces(dm_og_, plot_type='dynamic', slopes=False)
# Per story subtype
_ = plot_traces(dm_og_, plot_type='by-subtype')

# =============================================================================
# Preprocessed pupil traces (baseline corrected)
# =============================================================================
# Add variables from the df to the dm and print descriptives
dm_ = merge_dm_df(dm_, df)

# Pupil traces per type (non-dynamic)
_ = plot_traces(dm_, plot_type='non-dynamic')

# Pupil traces per type and per subtype (non-dynamic)
_ = plot_traces(dm_, plot_type='by-subtype')

# Pupil traces per type (dynamic)
_ = plot_traces(dm_, plot_type='dynamic')

# Group effects (barplots)
main_plots(dm_, 'dynamic')
main_plots(dm_, 'non-dynamic')
main_plots(dm_, 'subtypes')

# =============================================================================
# Statistical analyses
# =============================================================================
# 1. Mixed Linear Models on the mean pupil slopes 
# A. Mean pupil slopes per condition (dynamic condition only)
md1 = lm_slope(dm_, formula='slope_pupil ~ type', re_formula = '1 + type')
check_assumptions(md1, 'M1') # OK

print(md1.summary())
md1.tvalues, md1.pvalues = np.round(md1.tvalues, 3), np.round(md1.pvalues, 3)
print(f'z = {md1.tvalues[1]}, p = {md1.pvalues[1]}, n = {len(md1.random_effects)}')

# B. Test the interaction with the vividness ratings
md2 = lm_slope(dm_, formula='slope_pupil ~ type * response_vivid', re_formula = '1 + type')
check_assumptions(md2, 'M2') # OK

print(md2.summary())
md2.tvalues, md2.pvalues = np.round(md2.tvalues, 3), np.round(md2.pvalues, 3)
print(f'Main effect: z = {md2.tvalues[1]}, p = {md2.pvalues[1]}, n = {len(md2.random_effects)}')
print(f'Interaction: z = {md2.tvalues[3]}, p = {md2.pvalues[3]}, n = {len(md2.random_effects)}')

compare_models(md1, md2, 2) # taking into account the interaction with vividness ratings adds to the model
print(md1.aic < md2.aic)

md3 = lm_slope(dm_, formula='slope_change ~ mean_vivid', re_formula = '1', slope_change=True)
check_assumptions(md3, 'M3') # OK IF WE DON'T PUT RANDOM SLOPES FOR MEAN VIVID 

print(md3.summary())
md3.tvalues, md3.pvalues = np.round(md3.tvalues, 3), np.round(md3.pvalues, 3)
print(f'z = {md3.tvalues[1]}, p = {md3.pvalues[1]}, n = {len(md3.random_effects)}')

# 2. Mixed linear models on the mean pupil sizes
# A. Mean pupil size per condition (except dynamic)
m1 = lm_pupil(dm_, formula = 'mean_pupil ~ type', re_formula = '1 + type')
check_assumptions(m1, 'M1') # OK

print(m1.summary())
m1.tvalues, m1.pvalues = np.round(m1.tvalues, 3), np.round(m1.pvalues, 3)
print(f'z = {m1.tvalues[1]}, p = {m1.pvalues[1]}, n = {len(m1.random_effects)}')

# B. Test the interaction with the vividness ratings
m2 = lm_pupil(dm_, formula = 'mean_pupil ~ type * response_vivid', re_formula = '1 + type')
check_assumptions(m2, 'M2') # OK

print(m2.summary())
m2.tvalues, m2.pvalues = np.round(m2.tvalues, 3), np.round(m2.pvalues, 3)
print(f'Main effect: z = {m2.tvalues[1]}, p = {m2.pvalues[1]}, n = {len(m2.random_effects)}')
print(f'Interaction: z = {m2.tvalues[3]}, p = {m2.pvalues[3]}, n = {len(m2.random_effects)}')

# Compare the models
compare_models(m1, m2, 2) # Adding one more column to the data (one more input variable) would add one more degree of freedom for the model.
print(m1.aic < m2.aic)

# C. Pupil-size changes
m3 = lm_pupil(dm_, formula = 'pupil_change ~ mean_vivid', re_formula = '1 + mean_vivid', pupil_change=True, reml=True)    
check_assumptions(m3, 'M3') # OK

print(m3.summary())
m3.tvalues, m3.pvalues = np.round(m3.tvalues, 3), np.round(m3.pvalues, 3)
print(f'z = {m3.tvalues[1]}, p = {m3.pvalues[1]}, n = {len(m3.random_effects)}')

# D. Different effects per subtype 
# For each subtype
res = []
for subtype, sdm in ops.split(dm_.subtype[dm_.subtype!='dynamic']):
    print(subtype)
    # Test the interaction with the vividness ratings
    m = lm_pupil(sdm, formula = 'pupil_change ~ mean_vivid', re_formula='1 + mean_vivid', pupil_change=True, reml=True)
    
    # Check assumptions
    check_assumptions(m, subtype) 
    
    # Print the summary 
    print(m.summary())
    
    m.tvalues, m.pvalues = np.round(m.tvalues, 3), np.round(m.pvalues, 4)
    resultat = f'z = {m.tvalues[1]}, p = {m.pvalues[1]}, n = {len(m.random_effects)}'
    print(resultat)
    res.append(resultat)
    
# 3. Correlations 
    # Correlations between the mean VVIQ scores and mean SUIS scores
test_correlation(dm_, x='VVIQ', y='SUIS', alt='greater', lab='', plot_=False)
    # Big great plot
plot_correlations(dm_, what='all')
    # Between ratings
test_correlation(dm_, x='mean_vivid', y='mean_effort', alt='less', lab='', plot_=False)
test_correlation(dm_, x='mean_vivid', y='mean_emo', alt='greater', lab='', plot_=False)
test_correlation(dm_, x='mean_vivid', y='mean_val', alt='greater', lab='', plot_=False)
    # Per subtype
plot_correlations(dm_, what='by-subtype')

# =============================================================================
# Exploratory Analyses 
# =============================================================================
# Add control variables
dm_ctrl = create_control_variables(dm_)

# 1. Control for the effect of effort and arousal (emotional intensity) 
    # Pupil mean differences
for ctrl_var in ['effort_changes', 'emo_changes', 'order_changes']: # can also try with 'nonnan_pupil', 'n_blinks'... 
    print(ctrl_var)
    mS1 = lm_pupil(dm_ctrl, formula = f'pupil_change ~ mean_vivid + {ctrl_var}', re_formula='1 + mean_vivid', reml=True, pupil_change=True)
    check_assumptions(mS1, 'Controls')
    
    print(mS1.summary())
    mS1.tvalues, mS1.pvalues = np.round(mS1.tvalues, 3), np.round(mS1.pvalues, 3)
    print(f'Main effect: z = {mS1.tvalues[1]}, p = {mS1.pvalues[1]}, n = {len(mS1.random_effects)}')

# mS2 = lm_pupil(dm_ctrl, formula = 'mean_pupil ~ type * response_vivid + response_effort + emotional_intensity', re_formula='1 + type', reml=True)
# check_assumptions(mS2, 'Controls')

# print(mS2.summary())
# mS2.tvalues, mS2.pvalues = np.round(mS2.tvalues, 3), np.round(mS2.pvalues, 3)
# print(f'Main effect: z = {mS1.tvalues[1]}, p = {mS1.pvalues[1]}, n = {len(mS1.random_effects)}')

# 2. Check similar distributions between...
dist_checks(dm_ctrl, legend)

# =============================================================================
# Supplementary visualisations - /!\ plot land /!\
# =============================================================================
# Plot distributions of subjective measures
plot_dist_scores(dm_ctrl)

# Controls and possible confounds
supp_plots(dm_ctrl, what='order') # order of presentation
supp_plots(dm_ctrl, what='emotional_intensity') 
supp_plots(dm_ctrl, what='effort') 
supp_plots(dm_ctrl, what='aphantasia') 

# =============================================================================
# Visualisation: Individual effects (point plots)
# =============================================================================
# 1. Mean pupil size per condition and per participant (all conditions)
plt.figure(figsize=(30,10))
ax = plt.subplot(1,1,1)
dm = dm_.subtype != 'dynamic'
#dm = dm.pupil_change != NAN
dm.mean_pupil_change = ''
for s, sdm in ops.split(dm.subject_nr):
    dm.mean_pupil_change[sdm] = sdm.pupil_change.mean
dm = ops.sort(dm, by=dm.mean_pupil_change)
list_order1 = list(dict.fromkeys(dm.subject_nr))
dm_sub = convert.to_pandas(dm)
plt.title('All non-dynamic stories')
sns.pointplot(data=dm_sub, x='subject_nr', y='mean_pupil', hue_order=['light', 'dark'], palette=palette[:2], hue='type', order=list_order1, markersize=18, linewidth=5, estimator=np.mean, errorbar=('se',1))
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['Bright', 'Dark'], ncol=2, loc='upper center', frameon=False)
plt.axhline(0, linestyle='solid', color='black')
plt.xlabel('Participants');plt.ylabel('Mean Pupil Size (a.u.)')
#plt.xticks([])
plt.tight_layout()
plt.show()

# 2. Mean pupil size changes per participant
plt.figure(figsize=(20,10))
#plt.title('All non-dynamic stories')
sns.pointplot(data=dm_sub, x='subject_nr', y='pupil_change', hue='category', order=list_order1, markersize=18, linewidth=5, estimator=np.mean, palette=['black'], errorbar=('se',1), legend=False)
plt.axhline(0, linestyle='solid', color='black')
plt.xlabel('Participants');plt.ylabel('Pupil-size mean differences\n (Dark - Bright) (a.u.)')
#plt.legend(ncol=5, title='Aphantasia (VVIQ < 2)', frameon=False, loc='upper center')
plt.xticks([])
plt.ylim([-1500, 1500])
plt.tight_layout()
plt.show()

    # How many participants with at least one positive score?
N = len(dm.subject_nr.unique)
n = len(set(dm.subject_nr[dm.pupil_change > 0]))
print(f'{round(n/N * 100, 2)}% ({n}/{N}) of positive changes')

    # How many positive scores per participant (among those who have at least 1)?
list_p = dm.subject_nr[dm.pupil_change > 0]
n_counts = Counter(list_p)
print(Counter(n_counts.values()))

    # How many negative scores per participant (among those who have at least 1)?
list_n = dm.subject_nr[dm.pupil_change <= 0]
n_counts = Counter(list_n)
print(Counter(n_counts.values()))

    # How many participants have a mean positive score (averaged across all subtypes)?
N = len(dm.subject_nr.unique)
n = len(set(dm.subject_nr[dm.mean_pupil_change > 0]))
print(f'{round(n/N * 100, 2)}% ({n}/{N}) of positive changes')

# 3. Mean pupil size changes per participant per mean vivid
for subtype, name in zip(['happy', 'lotr', 'neutral'], ['Birthday Party', 'Lord of the Rings', 'Neutral']):
    fig = plt.figure(figsize=(40,20))
    i=1;plt.suptitle(name, fontsize=60)
    dm_sub = dm_.subtype != 'dynamic'
    dm_sub = dm_sub.subtype == subtype
    for thresh in np.arange(1, 5, 0.5):
        sdm = dm_sub.mean_vivid >= thresh
        ax = plt.subplot(2,4,i)
        sdm.mean_pupil_change = ''
        for s, sdm_ in ops.split(sdm.subject_nr):
            sdm.mean_pupil_change[sdm_] = sdm_.pupil_change.mean
        sdm = sdm.mean_pupil_change != NAN
        sdm = ops.sort(sdm, by=sdm.mean_pupil_change)
        list_order_ = list(dict.fromkeys(sdm.subject_nr))
        ax.set_title(f'\nmean vividness ≥ {thresh}', fontsize=40)
        sdm_sub = convert.to_pandas(sdm)
        #sns.pointplot(data=sdm_sub, x='subject_nr', y='pupil_change', hue='aphantasia', hue_order=['No', 'Yes'], markersize=15, linewidth=8, order=list_order_, estimator=np.mean, palette=['black', 'red'], errorbar=('se', 1))
        sns.pointplot(data=sdm_sub, x='subject_nr', y='pupil_change', hue='category', hue_order=None, markersize=15, linewidth=8, order=list_order_, estimator=np.mean, color='black', errorbar=('se', 1), legend=False)

        handles, labels = ax.get_legend_handles_labels()
        
        #plt.legend(handles=handles, labels=labels, ncol=1, loc='lower right', title='Aphantasia (VVIQ < 2)')
        plt.axhline(0, color='black', linestyle='solid')
        N = len(sdm.subject_nr.unique)
        n = len(set(sdm.subject_nr[sdm.mean_pupil_change > 0]))
        plt.xlabel(f'{round(n/N * 100, 2)}% ({n}/{N})')
        plt.xticks([], []);plt.ylim([-1700, 1800]);plt.yticks(range(-1700, 1900, 500), range(-1700, 1900, 500))
        plt.xlim([-2, len(sdm.subject_nr.unique)]);i+=1
    fig.supylabel('Pupil Size Changes\n (Dark - Bright) (a.u.)', fontsize=50, ha='center')
    fig.supxlabel('Participants', fontsize=50)
    plt.tight_layout()
    plt.show()

# 4. Plot the mean slopes per condition and per participant (dynamic only)
plt.figure(figsize=(20,10))
ax = plt.subplot(1,1,1)
dm = dm_.subtype == 'dynamic'
dm = dm.slope_change != NAN
dm = ops.sort(dm, by=dm.slope_change)
list_order1 = list(dict.fromkeys(dm.subject_nr))
dm_sub = convert.to_pandas(dm)
plt.title('Dynamic stories only')
sns.pointplot(data=dm_sub, x='subject_nr', y='slope_pupil', hue='type', order=list_order1, markersize=18, linewidth=5, estimator=np.nanmean, hue_order=['light_dark', 'dark_light'], palette=palette[0:2])
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles, ['Bright to dark', 'Dark to bright'], ncol=4, loc='upper center', frameon=False)
plt.axhline(0, linestyle='solid', color='black')
plt.xlabel('Participants');plt.ylabel('Pupil Slope Changes (a.u.)\n (Bright to dark - Dark to bright')
plt.xticks([])
plt.tight_layout()
plt.show()

# 5. Plot the mean slopes per condition per participant (dynamic only) and aphantasia
plt.figure(figsize=(20,10))
plt.title('Dynamic stories only')
sns.pointplot(data=dm_sub, x='subject_nr', y='slope_change', hue='category', order=list_order1, markersize=18, linewidth=5, estimator=np.nanmean, palette=['black'], legend=False)
plt.axhline(0, linestyle='solid', color='black')
plt.xlabel('Participants');plt.ylabel('Mean Slope Changes (a.u.)\n(Bright to dark - Dark to bright)')
#plt.legend(ncol=3, title='Aphantasia (VVIQ < 2)', loc='upper center', frameon=False)
plt.xticks([])
plt.tight_layout()
plt.show()

# How many participants with positive slope difference scores?
N = len(dm.subject_nr.unique)
n = len(set(dm.subject_nr[dm.slope_change > 0]))
print(f'{round(n/N * 100, 2)}% ({n}/{N}) of positive changes')

# 3. Mean pupil slope changes per participant per mean vivid
fig = plt.figure(figsize=(40,20));i=1
dm_sub = dm_.subtype == 'dynamic';plt.suptitle('Dynamic', fontsize=60)
for thresh in np.arange(1, 5, 0.5):
    sdm = dm_sub.mean_vivid >= thresh
    ax = plt.subplot(2,4,i)
    sdm.mean_slope_change = ''
    for s, sdm_ in ops.split(sdm.subject_nr):
        sdm.mean_slope_change[sdm_] = sdm_.slope_change.mean
    sdm = sdm.mean_slope_change != NAN
    sdm = ops.sort(sdm, by=sdm.mean_slope_change)
    list_order_ = list(dict.fromkeys(sdm.subject_nr))
    ax.set_title(f'\nmean vividness ≥ {thresh}', fontsize=40)
    sdm_sub = convert.to_pandas(sdm)
    #sns.pointplot(data=sdm_sub, x='subject_nr', y='slope_change', hue='aphantasia', hue_order=['No', 'Yes'], order=list_order_, markersize=15, linewidth=8, estimator=np.mean, palette=['black', 'red'])
    sns.pointplot(data=sdm_sub, x='subject_nr', y='slope_change', hue='category', hue_order=None, order=list_order_, markersize=15, linewidth=8, estimator=np.mean, palette=['black'], legend=False)

    handles, labels = ax.get_legend_handles_labels()

    #plt.legend(handles=handles, labels=labels, ncol=1, loc='lower right', title='Aphantasia (VVIQ < 2)')
    plt.axhline(0, color='black', linestyle='solid')
    N = len(sdm.subject_nr.unique)
    n = len(set(sdm.subject_nr[sdm.mean_slope_change > 0]))
    plt.xlabel(f'{round(n/N * 100, 2)}% ({n}/{N})')
    plt.xticks([], []);plt.ylim([-0.8, 0.85])
    plt.xlim([-2, len(sdm.subject_nr.unique)]);i+=1
fig.supylabel('Pupil Slope Changes (a.u.)\n (Bright to dark - Dark to bright)', fontsize=50, ha='center')
fig.supxlabel('Participants', fontsize=50)
plt.tight_layout()
plt.show()