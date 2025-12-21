# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:29:48 2024

@author: Claire Vanbuckhave
"""
# =============================================================================
#%% Set up 
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
                         test_correlation, check_assumptions, create_img,
                         merge_dm_df, plot_dist_scores, plot_correlations,
                         create_control_variables, dist_checks, main_plots,
                         dist_checks_wilcox, print_results)
from collections import Counter
from scipy.stats import wilcoxon

# Define useful variables
cwd=os.getcwd() # auto
cwd='D:/data_experiments/visual_imagery_pupil/study_2' # manual
datafolder=cwd+'/data/' # the folder where the EDF files are
outputfolder=cwd+'/output/'
questfile='/results-survey.csv' # the questionnaire data

np.random.seed(111)  # Fix random seed for predictable outcomes
palette = [green[1], blue[1], gray[2], gray[3]]

# =============================================================================
#%% Prepare the data
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
#%% Raw pupil traces
# =============================================================================
# Pupil traces for non-dynamic
legend = plot_traces(dm_og_, plot_type='non-dynamic', cwd=cwd)
# Pupil traces for dynamic
_ = plot_traces(dm_og_, plot_type='dynamic', slopes=False, cwd=cwd)
# Per story subtype
_ = plot_traces(dm_og_, plot_type='by-subtype', cwd=cwd)

# =============================================================================
#%% Preprocessed pupil traces (baseline corrected)
# =============================================================================
# Add variables from the df to the dm and print descriptives
dm_ = merge_dm_df(dm_, df)

# Pupil traces per type (non-dynamic)
_ = plot_traces(dm_, plot_type='non-dynamic', cwd=cwd)

# Pupil traces per type and per subtype (non-dynamic)
_ = plot_traces(dm_, plot_type='by-subtype', cwd=cwd)

# Pupil traces per type (dynamic)
_ = plot_traces(dm_, plot_type='dynamic', cwd=cwd)

# Group effects (barplots)
main_plots(dm_, 'dynamic', cwd)
main_plots(dm_, 'non-dynamic', cwd)
main_plots(dm_, 'subtypes', cwd)

# Save to csv file 
#dm_df = convert.to_pandas(dm_)
#dm_df.to_csv(cwd+'/data_pupil.csv', index=True)
#df.to_csv(cwd+'/data_quest.csv', index=True)
# dm_df_sub = dm_df[['mean_vivid', 'mean_effort', 'mean_emo', 'pupil_change', 'version', 'suptype']]
# dm_df_sub.to_csv(cwd+'/data_grouped.csv', index=True)
# dm_df_sub = np.round(dm_df_sub.groupby(['suptype', 'version']).describe(),3)
# dm_df_sub.to_csv(cwd+'/descriptives_summary.csv', index=True)

# =============================================================================
#%% Statistical analyses
# =============================================================================
# 1. Mixed Linear Models on the mean pupil slopes 
# A. Mean pupil slopes per condition (dynamic condition only)
md1 = lm_slope(dm_, formula='slope_pupil ~ type', re_formula = '1 + type')
check_assumptions(md1, 'M1') # OK
print(md1.summary())
print_results(md1)

# B. Test the interaction with the vividness ratings
md2 = lm_slope(dm_, formula='slope_pupil ~ type * response_vivid', re_formula = '1 + type')
check_assumptions(md2, 'M2') # OK
print(md2.summary())
print_results(md2)

compare_models(md1, md2, 2) # taking into account the interaction with vividness ratings adds to the model
print(md1.aic < md2.aic)

md3 = lm_slope(dm_, formula='slope_change ~ mean_vivid', re_formula = '1', slope_change=True)
check_assumptions(md3, 'M3') # OK IF WE DON'T PUT RANDOM SLOPES FOR MEAN VIVID 
print(md3.summary())
print_results(md3)

#%% 2. Mixed linear models on the mean pupil sizes
# A. Mean pupil size per condition (except dynamic)
m1 = lm_pupil(dm_, formula = 'mean_pupil ~ type', re_formula = '1 + type')
check_assumptions(m1, 'M1') # OK
print(m1.summary())
print_results(m1)

# B. Test the interaction with the vividness ratings
m2 = lm_pupil(dm_, formula = 'mean_pupil ~ type * response_vivid', re_formula = '1 + type')
check_assumptions(m2, 'M2') # OK
print(m2.summary())
print_results(m2)

# Compare the models
compare_models(m1, m2, 2) # Adding one more column to the data (one more input variable) would add one more degree of freedom for the model.
print(m1.aic < m2.aic)

# C. Pupil-size changes
m3 = lm_pupil(dm_, formula = 'pupil_change ~ mean_vivid', re_formula = '1 + mean_vivid', pupil_change=True, reml=True)    
check_assumptions(m3, 'M3') # OK
print(m3.summary())
print_results(m3)

#%% 3. Correlations 
# Correlations between the mean VVIQ scores and mean SUIS scores
test_correlation(dm_, x='VVIQ', y='SUIS', alt='greater', lab='', plot_=False)

# Big great plot
plot_correlations(dm_, what='all')

# Other main correlations
for s in ['dynamic', set(['happy', 'lotr', 'neutral'])]:
    dm_cor = dm_.subtype == s
    if s == 'dynamic':
        dm_cor = dm_cor.slope_change != NAN
        var = 'slope_change'
    else:
        dm_cor = dm_cor.pupil_change != NAN
        var = 'pupil_change'
    print(s)
    test_correlation(dm_cor, x='response_vivid', y='VVIQ', alt='greater', lab='', plot_=False)
    test_correlation(dm_cor, x='response_vivid', y='SUIS', alt='greater', lab='', plot_=False)
    test_correlation(dm_cor, x=var, y='VVIQ', alt='greater', lab='', plot_=False)
    test_correlation(dm_cor, x=var, y='SUIS', alt='greater', lab='', plot_=False)

# Correlations between subjective ratings
# test_correlation(dm_, x='mean_vivid', y='mean_effort', alt='less', lab='', plot_=False)
# test_correlation(dm_, x='mean_vivid', y='mean_emo', alt='greater', lab='', plot_=False)
# test_correlation(dm_, x='mean_vivid', y='mean_val', alt='greater', lab='', plot_=False)

    # Per subtype
#plot_correlations(dm_, what='by-subtype')

# =============================================================================
#%% Control Analyses 
# =============================================================================
# 0. Different effects per subtype?
for subtype, sdm in ops.split(dm_.subtype[dm_.subtype!='dynamic']):
    print(subtype)
    # Test the interaction with the vividness ratings
    m = lm_pupil(sdm, formula = 'pupil_change ~ mean_vivid', re_formula='1 + mean_vivid', pupil_change=True, reml=True)
    #m = lm_pupil(sdm, formula = 'mean_pupil ~ type * response_vivid', re_formula='1 + type', pupil_change=False, reml=True)

    # Check assumptions
    check_assumptions(m, subtype) 
    
    # Print the summary 
    print(m.summary())
    print_results(m)
    
    
# Add control variables
dm_ctrl = create_control_variables(dm_)

# 1. Check similar distributions between conditions with KS test
dist_checks(dm_ctrl, legend)

# 2. Another way to check no differences, with paired Wilcoxon test
dist_checks_wilcox(dm_ctrl)

# 1. Control for the effect of significant differences if any
    # Pupil mean differences
for ctrl_var in ['order_changes']: #or: ['effort_changes', 'emo_changes', 'order_changes', 'n_blinks']: # can also try with 'nonnan_pupil', 'n_blinks'... 
    print(ctrl_var)
    mS1 = lm_pupil(dm_ctrl, formula = f'pupil_change ~ mean_vivid + {ctrl_var}', re_formula='1 + mean_vivid', reml=True, pupil_change=True)
    check_assumptions(mS1, 'Controls')
    print(mS1.summary())
    print_results(mS1)
    
    # Compare the models
    compare_models(m3, mS1, 2) # Adding one more column to the data (one more input variable) would add one more degree of freedom for the model.
    print(m3.aic < mS1.aic)

    
# =============================================================================
#%% Supplementary visualisations - /!\ plot land /!\ - for supp materials (figures)
# =============================================================================
# Plot distributions of subjective measures
plot_dist_scores(dm_ctrl)

# Controls and possible confounds
supp_plots(dm_ctrl, what='order') # order of presentation
supp_plots(dm_ctrl, what='emotional_intensity') 
supp_plots(dm_ctrl, what='effort') 
supp_plots(dm_ctrl, what='aphantasia') 

supp_plots(dm_ctrl, what='effort_vivid') 
supp_plots(dm_ctrl, what='emo_vivid') 
supp_plots(dm_ctrl, what='time') 

# Individual effects (point plots)
# 1. Mean pupil size per condition and per participant (all conditions)
plt.figure(figsize=(30,10))
ax = plt.subplot(1,1,1)
dm = dm_ctrl.subtype != 'dynamic'
dm = dm.pupil_change != NAN
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
plt.title('All non-dynamic stories')
sns.pointplot(data=dm_sub, x='subject_nr', y='pupil_change', hue='category', order=list_order1, markersize=18, linewidth=5, estimator=np.mean, palette=['black'], errorbar=('se',1), legend=False)
plt.axhline(0, linestyle='solid', color='black')
plt.xlabel('Participants');plt.ylabel('Pupil-size mean differences\n (Dark - Bright) (a.u.)')
plt.legend(ncol=5, title='Aphantasia (VVIQ < 2)', frameon=False, loc='upper center')
plt.xticks([])
plt.ylim([-1500, 1500])
plt.tight_layout()
plt.show()

    # How many participants with at least one positive score?
N = len(dm_ctrl.subject_nr.unique)
n = len(set(dm_ctrl.subject_nr[dm_ctrl.pupil_change > 0]))
print(f'{round(n/N * 100, 2)}% ({n}/{N}) of positive changes')

    # How many positive scores per participant (among those who have at least 1)?
list_p = dm_ctrl.subject_nr[dm_ctrl.pupil_change > 0]
n_counts = Counter(list_p)
print(Counter(n_counts.values()))

    # How many negative scores per participant (among those who have at least 1)?
list_n = dm_ctrl.subject_nr[dm_ctrl.pupil_change <= 0]
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
    dm_sub = dm_ctrl.subtype != 'dynamic'
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
dm = dm_ctrl.subtype == 'dynamic'
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

# 6. Mean pupil slope changes per participant per mean vivid
fig = plt.figure(figsize=(40,20));i=1
dm_sub = dm_ctrl.subtype == 'dynamic';plt.suptitle('Dynamic', fontsize=60)
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

############################################################################

