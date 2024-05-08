# -*- coding: utf-8 -*-
"""
Created on Thu Feb 15 13:29:48 2024

@author: cvanb
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
from custom_func import (get_raw, merge_dm_df, check_blinks, preprocess_dm, plot_bars,
                         lm_pupil, compare_models, create_control_variables,
                         test_correlation, check_assumptions,
                         dist_checks_wilcox, individual_profile)
from collections import Counter
import time_series_test as tst

# Define useful variables
cwd=os.getcwd() # auto
cwd='C:/Users/cvanb/Desktop/visual_imagery_pupil/study 2/' # manual
datafolder=cwd+'/data/' # the folder where the EDF files are
questfile='/results-survey.csv' # the questionnaire data

np.random.seed(123)  # Fix random seed for predictable outcomes if any random stuff happening
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

# =============================================================================
# Descriptives
# =============================================================================
# Add questionnaire variables
dm_ = merge_dm_df(dm_, df)

# Add useful variables 
dm_ctrl = create_control_variables(dm_)

# Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
dm_df = convert.to_pandas(dm_ctrl)

# Save to csv file (e.g., to play with the data in Jamovi)
#dm_df.to_csv(cwd+'data_pupil.csv', index=True)
#df.to_csv(cwd+'data_quest.csv', index=True)
# dm_df_sub = dm_df[['mean_vivid', 'mean_effort', 'mean_emo', 'pupil_change', 'version', 'suptype']]
# dm_df_sub.to_csv(cwd+'data_grouped.csv', index=True)
# dm_df_sub = np.round(dm_df_sub.groupby(['suptype', 'version']).describe(),3)
# dm_df_sub.to_csv(cwd+'descriptives_summary.csv', index=True)

# =============================================================================
# Statistical analyses
# =============================================================================
# 1. Mixed linear models on the mean pupil sizes
for s, sdm in ops.split(dm_ctrl.suptype):
    s = 'rabbit'
    print(s)  
    sdm = dm_ctrl.suptype == s

    # A. Main effect of brightness condition 
    m1 = lm_pupil(sdm, formula = 'mean_pupil ~ type', re_formula = '1 + type')
    check_assumptions(m1) # OK
    
    print(m1.summary())
    m1.tvalues, m1.pvalues = np.round(m1.tvalues, 4), np.round(m1.pvalues, 4)
    print(f'z = {m1.tvalues[1]}, p = {m1.pvalues[1]}, n = {len(m1.random_effects)}')

    # B. Interaction with vividness
    m2 = lm_pupil(sdm, formula = 'mean_pupil ~ type * response_vivid', re_formula = '1 + type')
    check_assumptions(m2) # OK
    
    print(m2.summary())
    m2.tvalues, m2.pvalues = np.round(m2.tvalues, 3), np.round(m2.pvalues, 3)
    print(f'z = {m2.tvalues[1]}, p = {m2.pvalues[1]}, n = {len(m2.random_effects)}')
    print(f'z = {m2.tvalues[3]}, p = {m2.pvalues[3]}, n = {len(m2.random_effects)}')

    # Compare the models
    compare_models(m1, m2, 2) 
    print(m1.aic < m2.aic)
    
    # C. Pupil-size changes: Individual effects
    # MLM
    if s == 'rabbit':
        re_formula = '1 + mean_vivid + version'
    else:
        re_formula = '1 + version' # messes with assumption checks if add 'mean_vivid'
    m3 = lm_pupil(sdm, formula = 'pupil_change ~ mean_vivid', re_formula = re_formula, pupil_change=True)    
    check_assumptions(m3) # OK
    
    print(m3.summary())
    m3.tvalues, m3.pvalues = np.round(m3.tvalues, 3), np.round(m3.pvalues, 3)
    print(f'z = {m3.tvalues[1]}, p = {m3.pvalues[1]}, n = {len(m3.random_effects)}')

    # D. Spearman's correlation
    sdm_ = sdm.SUIS != ''
    test_correlation(sdm, x='pupil_change', y='mean_vivid', alt='greater', lab='Ratings', fig=False)
    test_correlation(sdm_, x='pupil_change', y='VVIQ', alt='greater', lab='Ratings', fig=False)
    test_correlation(sdm_, x='pupil_change', y='SUIS', alt='greater', lab='Ratings', fig=False)
    
    # E. Add covariates
    dist_checks_wilcox(sdm, s)

    m4 = lm_pupil(sdm, formula = 'pupil_change ~ vivid + emo_changes', re_formula = re_formula, pupil_change=True)  
    m5 = lm_pupil(sdm, formula = 'pupil_change ~ vivid + effort_changes', re_formula = re_formula, pupil_change=True)  

    check_assumptions(m4) # OK
    check_assumptions(m5) # OK

    print(m4.summary())
    print(m5.summary())

    m4.tvalues, m4.pvalues = np.round(m4.tvalues, 3), np.round(m4.pvalues, 3)
    print(f'z = {m4.tvalues[1]}, p = {m4.pvalues[1]}, n = {len(m4.random_effects)}')
    print(f'z = {m4.tvalues[2]}, p = {m4.pvalues[2]}, n = {len(m4.random_effects)}')
    print(f'z = {m4.tvalues[3]}, p = {m4.pvalues[3]}, n = {len(m4.random_effects)}')

# General correlations with questionnaires
dm_cor = dm_ctrl.SUIS != ''
test_correlation(dm_cor, x='SUIS', y='VVIQ', alt='greater', lab='Ratings', fig=False)
test_correlation(dm_cor, x='mean_vivid', y='VVIQ', alt='greater', lab='Ratings', fig=False)
test_correlation(dm_cor, x='mean_vivid', y='SUIS', alt='greater', lab='Ratings', fig=False)

test_correlation(dm_cor, x='mean_vivid', y='mean_emo', alt='greater', lab='Ratings', fig=False)
test_correlation(dm_cor, x='mean_vivid', y='mean_val', alt='greater', lab='Ratings', fig=False)
test_correlation(dm_cor, x='mean_vivid', y='mean_effort', alt='less', lab='Ratings', fig=False)

# =============================================================================
# Visualisation: Group effects (barplots)
# =============================================================================
# 1. Pupil traces
fig, axes = plt.subplots(1, 1, figsize=(28,10))
plt.subplot(1,1,1)
sdm=dm_ctrl.suptype=='rabbit'
#sdm = sdm.version == 2
tst.plot(sdm, dv='pupil', hue_factor='type', hues=[blue[1], green[1]], 
         legend_kwargs={'frameon': False, 'loc': 'lower center', 'ncol': 2, 'fontsize': 40,'labels': [f'Dark (N={len(sdm[sdm.type=="dark"])})', f'Bright (N={len(sdm[sdm.type=="light"])})']},
         annotation_legend_kwargs={'frameon': False, 'loc': 'upper right'}, 
         x0=0, sampling_freq=1)
plt.xticks(np.arange(0, 11700+500, 500), np.arange(0, int(11700/100)+5, 5))
plt.xlim([0, 11700])
plt.xlabel('Time since story onset (s)', fontsize=50)
plt.ylabel('Baseline-corrected\npupil size (a.u.)', fontsize=50)
plt.text(s='A', x=-1200.0, y=500, fontsize=65)
plt.tight_layout()
plt.show()
    
fig, axes = plt.subplots(1, 1, figsize=(28,10))
plt.subplot(1,1,1)
sdm=dm_ctrl.suptype=='self'
tst.plot(sdm, dv='pupil', hue_factor='type', hues=[blue[1], green[1]], 
         legend_kwargs={'frameon': False, 'loc': 'lower center', 'ncol': 2, 'fontsize': 40,'labels': [f'Dark (N={len(sdm[sdm.type=="dark"])})', f'Bright (N={len(sdm[sdm.type=="light"])})']},
         annotation_legend_kwargs={'frameon': False, 'loc': 'upper right'}, 
         x0=0, sampling_freq=1)
plt.xticks(np.arange(0, 3000+100, 100), np.arange(0, int(3000/100)+1, 1))
plt.xlim([0, 3000])
plt.xlabel('Time since start imagine (s)', fontsize=50)
plt.ylabel('Baseline-corrected\npupil size (a.u.)', fontsize=50)
plt.text(s='A', x=-300.0, y=750, fontsize=65)
plt.tight_layout()
plt.show()

# 2. Mean pupil size per condition and interactions with vividness ratings 
        # RABBIT STORIES
dm_sub0 = dm_df[dm_df.suptype == 'rabbit']
fig, axes = plt.subplots(1, 4, figsize=(47,12))
fig.subplots_adjust(wspace=0.3)
ax=plt.subplot(1,4,1)
plot_bars(dm_sub0, x='type', y='mean_pupil', hue=None, hue_order=None, order=['light', 'dark'], pal=palette[0:2], fig=False, alpha=0.7, ylab='Pupil-size means (a.u.)', xlab='Condition')
handles, labels = ax.get_legend_handles_labels()
plt.text(s='B', x=-1.0, y=30, fontsize=65);plt.xticks(ticks=range(0,2), labels=['Bright', 'Dark'])
ax=plt.subplot(1,4,2)
dm_sub0 = dm_sub0[dm_sub0.mean_pupil != '']
plot_bars(dm_sub0, x='response_vivid', y='mean_pupil', hue='type', hue_order=['light', 'dark'], ylab='Pupil-size means (a.u.)', pal=palette[0:2], xlab='Trial-by-trial vividness ratings', title=None, fig=False, alpha=0.7)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=['Bright', 'Dark'], frameon=False)
plt.text(s='C', x=-1.5, y=70, fontsize=65)
ax=plt.subplot(1,4,3)
plt.text(s='D', x=-2.5, y=1350, fontsize=65);plt.yticks(range(-1250, 1500, 250), fontsize=20)
plot_bars(dm_sub0, x='mean_vivid', y='pupil_change', hue=None, hue_order=None, ylab='Pupil-size mean differences (a.u.)', pal='crest', xlab='Mean vividness ratings', title=None, fig=False)
ax=plt.subplot(1,4,4)
plt.text(s='E', x=-2500, y=7.15, fontsize=65)
dm_cor = dm_ctrl.suptype == 'rabbit'
#test_correlation(dm_cor, x='mean_vivid', y='pupil_change', alt='greater', fig=True, lab='Ratings')
dm_cor = dm_cor.SUIS != ''
test_correlation(dm_cor, y='VVIQ', x='pupil_change', alt='greater', fig=True, lab='VVIQ', color='green', fs=35)
test_correlation(dm_cor, y='SUIS', x='pupil_change', alt='greater', fig=True, lab='SUIS', color='violet', fs=35)
plt.xlabel('Pupil-size mean differences (a.u.)');plt.ylabel('Mean questionnaire scores');plt.ylim([0.9, 7]);plt.yticks(range(1, 6))
plt.show()

# SELF TRIALS
dm_sub0 = dm_df[dm_df.suptype == 'self']
fig, axes = plt.subplots(1, 3, figsize=(35,10))
fig.subplots_adjust(wspace=0.6)
ax=plt.subplot(1,3,1)
plot_bars(dm_sub0, x='type', y='mean_pupil', hue=None, hue_order=None, order=['light', 'dark'], pal=palette[0:2], fig=False, alpha=0.7, ylab='Pupil-size means (a.u.)', xlab='Condition')
handles, labels = ax.get_legend_handles_labels()
plt.text(s='B', x=-1.1, y=192, fontsize=65);plt.xticks(ticks=range(0,2), labels=['Bright', 'Dark'])
ax=plt.subplot(1,3,2)
dm_sub0 = dm_sub0[dm_sub0.mean_pupil != '']
plot_bars(dm_sub0, x='response_vivid', y='mean_pupil', hue='type', hue_order=['light', 'dark'], ylab='Pupil-size means (a.u.)', pal=palette[0:2], xlab='Trial-by-trial vividness\nratings', title=None, fig=False, alpha=0.7)
handles, labels = ax.get_legend_handles_labels()
plt.legend(handles=handles, labels=['Bright', 'Dark'], frameon=False)
plt.text(s='C', x=-2.0, y=560, fontsize=65)
ax=plt.subplot(1,3,3)
plt.text(s='D', x=-3.0, y=1080, fontsize=65)
plot_bars(dm_sub0, x='mean_vivid', y='pupil_change', hue=None, hue_order=None, ylab='Pupil-size mean differences\n(a.u.) (Dark - Bright)', pal='crest', xlab='Mean vividness ratings', title=None, fig=False)
plt.show()

# =============================================================================
# Controls (possible covariates)
# =============================================================================
# Try to find why negative scores
dm_df['is_positive'] = np.where(dm_df['pupil_change']>0, 'Yes', 'No')
dm_df['version_order'] = np.where(dm_df['version']==1, 'Bright first', 'Dark first')

# # Other effects on mean pupil size and slopes (controls)
# for s, dm_sub in ops.split(dm_ctrl.suptype):
#     sub_df = convert.to_pandas(dm_sub)
#     sub_df = sub_df[sub_df.mean_pupil != '']

#     # Emotional intensity (absolute value of valency) and Effort    
#     plt.figure(figsize=(20,40))
#     plt.suptitle(f'{s} (n = {len(dm_sub.subject_nr.unique)})', fontsize=45)
#     plt.subplot(6,2,1);plot_bars(sub_df, x='mean_emo', y='pupil_change', pal='flare', xlab='Mean emotional intensity ratings', ylab='Mean Pupil Changes (a.u.)', fig=False)
#     plt.subplot(6,2,2);plot_bars(sub_df, x='mean_effort', y='pupil_change', pal='YlOrBr', xlab='Mean effort ratings', ylab='Pupil-size mean\ndifferences (a.u.)', fig=False)
#     plt.subplot(6,2,3);plot_bars(sub_df, x='mean_vivid', y='pupil_change', pal='crest', xlab='Mean vividness ratings', ylab='Pupil-size mean\ndifferences (a.u.)', fig=False)
    
#     plt.subplot(6,2,4);plot_bars(sub_df, x='version', y='pupil_change', pal=[gray[2], gray[3]], xlab='Bright/Dark version order', ylab='Pupil-size mean\ndifferences (a.u.)', fig=False)
    
#     plt.subplot(6,2,5);plot_bars(sub_df, x='version', y='response_vivid', hue='type', hue_order=None, pal=palette[0:2], xlab='Bright/Dark version order', ylab='Vividness ratings', fig=False)
#     plt.subplot(6,2,6);plot_bars(sub_df, x='version', y='response_vivid', pal=[gray[2], gray[3]], xlab='Bright/Dark version order', ylab='Vividness ratings', fig=False)
    
#     plt.subplot(6,2,7);plot_bars(sub_df, x='version', y='response_val', hue='type', hue_order=None, pal=palette[0:2], xlab='Bright/Dark version order', ylab='Valence ratings', fig=False)
#     plt.subplot(6,2,8);plot_bars(sub_df, x='version', y='response_val', pal=[gray[2], gray[3]], xlab='Bright/Dark version order', ylab='Valence ratings', fig=False)
    
#     plt.subplot(6,2,9);plot_bars(sub_df, x='version', y='response_effort', hue='type', hue_order=None, pal=palette[0:2], xlab='Bright/Dark version order', ylab='Effort ratings', fig=False)
#     plt.subplot(6,2,10);plot_bars(sub_df, x='version', y='response_effort', pal=[gray[2], gray[3]], xlab='Bright/Dark version order', ylab='Effort ratings', fig=False)
    
#     plt.subplot(6,2,11);plot_bars(sub_df, x='version', y='mean_pupil', hue='type', hue_order=None, pal=palette[0:2], xlab='Bright/Dark version order', ylab='Mean pupil size (a.u.)', fig=False)
#     plt.subplot(6,2,12);plot_bars(sub_df, x='version', y='mean_pupil', pal=[gray[2], gray[3]], xlab='Bright/Dark version order', ylab='Mean pupil size (a.u.)', fig=False)
    
#     plt.tight_layout()
#     plt.show()

# Descriptives 
for type_ in ['rabbit', 'self']:
    plt.figure(figsize=(30,7))
    plt.suptitle(type_, fontsize=40)
    plt.subplot(1,4,1);plot_bars(dm_df[dm_df.suptype == type_], x='version', y='mean_pupil', hue='type', order=[1, 2], color='black', xlab='Order', hue_order=['light', 'dark'], ylab='Mean pupil size (a.u.)', alpha=0.7, fig=False, pal=palette[0:2], legend=True)#;plt.ylim([2800, 5000])
    plt.legend(ncol=2, title='Condition', fontsize=20);plt.xticks(range(0,2), ['Bright first', 'Dark first'])
    plt.subplot(1,4,2);plot_bars(dm_df[dm_df.suptype == type_], x='type', y='response_vivid', hue='version', color='black', xlab='Order', order=['light', 'dark'], hue_order=None, ylab='Vividness', alpha=0.7, fig=False, pal=palette[0:2], legend=True);plt.ylim([1, 5])
    plt.legend(ncol=2, title='Condition', fontsize=20);plt.xticks(range(0,2), ['Bright first', 'Dark first'])
    plt.subplot(1,4,3);plot_bars(dm_df[dm_df.suptype == type_], x='type', y='response_effort', hue='version', color='black', xlab='Condition', order=['light', 'dark'], hue_order=None, ylab='Mental effort', alpha=0.7, fig=False, pal=palette[0:2], legend=True);plt.ylim([-2, 2])
    plt.legend(ncol=2, title='Condition', fontsize=20);plt.xticks(range(0,2), ['Bright first', 'Dark first'])
    plt.subplot(1,4,4);plot_bars(dm_df[dm_df.suptype == type_], x='type', y='emotional_intensity', hue='version', color='black', xlab='Condition', order=['light', 'dark'], hue_order=None, ylab='Emotional intensity', alpha=0.7, fig=False, pal=palette[0:2], legend=True);plt.ylim([0, 3])
    plt.legend(ncol=2, title='Condition', fontsize=20);plt.xticks(range(0,2), ['Bright first', 'Dark first'])
    plt.tight_layout()
    plt.show()

# plt.figure(figsize=(10,8))
# dm_sub = dm_df[dm_df.suptype == 'rabbit']
# plot_bars(dm_sub, x='version_order', y='mean_pupil', hue='order', hue_order=None, fig=False, legend=True, ylab='Vividness', xlab='Presentation order', fs=25)
# plt.legend(ncol=2, title='Order', fontsize=20)
# plt.tight_layout()
# plt.show()

#dm_df = dm_df[dm_df.SUIS != '']
subdm_df = dm_df[dm_df.suptype == 'rabbit']
#subdm_df.drop_duplicates(subset=['subject_nr'], keep='first', inplace=True) 

individual_profile(dm_df, 30) # aphantasia + v1

dm_neg = dm_ctrl.pupil_change < 0
dm_neg = dm_neg.suptype == 'rabbit'
Counter(dm_neg.version)

# =============================================================================
# Aphantasia (descriptives)
# =============================================================================
# plt.figure(figsize=(30,10))
# plt.subplot(1,3,1);plot_bars(dm_df, x='type', y='response_val', hue='aphantasia', hue_order=None, pal=['black', 'orange', 'red'], xlab='Condition', ylab='Emotional Valence', alpha=0.5, fig=False);plt.ylim([-3,3])
# plt.subplot(1,3,2);plot_bars(dm_df, x='type', y='response_effort', hue='aphantasia', hue_order=None, pal=['black', 'orange', 'red'], xlab='Condition', ylab='Effort ratings', alpha=0.5, fig=False);plt.ylim([-2,2])
# plt.subplot(1,3,3);plot_bars(dm_df, x='type', y='response_vivid', hue='aphantasia', hue_order=None, pal=['black', 'orange', 'red'], xlab='Condition', ylab='Vividness ratings', alpha=0.5, fig=False);plt.ylim([0.9,5])
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(30,20))
# dm_sub = dm_df[dm_df.pupil_change != '']
# plt.subplot(2,2,1);plot_bars(dm_sub, x='type', y='mean_effort', order=['light', 'dark'], hue='aphantasia', hue_order=None, pal=['black', 'orange', 'red'], xlab='Condition', ylab='Mean effort ratings', alpha=0.5, fig=False);plt.ylim([-2,2])
# plt.subplot(2,2,2);plot_bars(dm_sub, x='type', y='mean_vivid', order=['light', 'dark'], hue='aphantasia', hue_order=None, pal=['black', 'orange', 'red'], xlab='Condition', ylab='Mean vividness ratings', alpha=0.5, fig=False);plt.ylim([1,5])
# plt.subplot(2,2,3);plot_bars(dm_sub, x='type', y='response_val', order=['light', 'dark'], hue='aphantasia', hue_order=None, pal=['black', 'orange', 'red'], xlab='Condition', ylab='Mean valency ratings', alpha=0.5, fig=False);plt.ylim([-3,3.5])
# plt.subplot(2,2,4);plot_bars(dm_sub, x='aphantasia', y='pupil_change', hue_order=None, pal=['black', 'orange', 'red'], xlab='Aphantasia (mean VVIQ score < 2)', ylab='Pupil-size mean differences\n (Dark - Bright) (a.u.)', alpha=0.5, fig=False)
# plt.tight_layout()
# plt.show()

# plt.figure(figsize=(20,10))
# sdm = dm_df[dm_df.suptype == 'rabbit']
# plt.subplot(1,2,1);plot_bars(sdm, x='version', y='pupil_change', hue='aphantasia', hue_order=None, pal=['black', 'orange', 'red'], xlab='Version', ylab='Pupil-size mean differences\n (Dark - Bright) (a.u.)', alpha=0.5, fig=False);plt.title('Rabbit')
# sdm = dm_df[dm_df.suptype == 'self'] 
# plt.subplot(1,2,2);plot_bars(sdm, x='version', y='pupil_change', hue='aphantasia', hue_order=None, pal=['black', 'orange', 'red'], xlab='Version', ylab='Pupil-size mean differences (a.u.)\n (Dark - Bright)', alpha=0.5, fig=False);plt.title('Self-chosen')
# plt.tight_layout()
# plt.show()

# =============================================================================
# Visualisation: Individual effects (point plots)
# =============================================================================
# 1. Mean pupil size per condition and per participant (all conditions)
for s, subdm in ops.split(dm_ctrl.suptype):

    dm = subdm.pupil_change != NAN
    dm.mean_pupil_change = ''
    for p, sdm in ops.split(dm.subject_nr):
        dm.mean_pupil_change[sdm] = sdm.pupil_change.mean
    dm = ops.sort(dm, by=dm.mean_pupil_change)
    
    # How many participants have a mean positive score (averaged across all subtypes)?
    N = len(dm.subject_nr.unique)
    n = len(set(dm.subject_nr[dm.pupil_change > 0]))
    print(f'{round(n/N * 100, 2)}% ({n}/{N}) of positive changes')

    list_order1 = list(dict.fromkeys(dm.subject_nr))
    dm_sub = convert.to_pandas(dm)
    
    # plt.figure(figsize=(20,10))
    # ax = plt.subplot(1,1,1)
    # sns.pointplot(data=dm_sub, x='subject_nr', y='mean_pupil', hue_order=['light', 'dark'], palette=palette[:2], hue='type', order=list_order1, markersize=18, linewidth=5, estimator=np.mean, errorbar=('se',1))
    # handles, labels = ax.get_legend_handles_labels()
    # plt.legend(handles, ['Bright', 'Dark'], ncol=2, loc='upper center', frameon=False)
    # plt.axhline(0, linestyle='solid', color='black')
    # plt.xlabel('Participants');plt.ylabel('Mean Pupil Size (a.u.)')
    # #plt.xticks([])
    # plt.title(s)
    # plt.tight_layout()
    # plt.show()
    
    # 2. Mean pupil size changes per participant
    plt.figure(figsize=(20,10))
    ax = plt.subplot(1,1,1)
    plt.title(s)
    sns.pointplot(data=dm_sub, x='subject_nr', y='pupil_change', hue='aphantasia', hue_order=['No', 'Mild', 'Yes'], order=list_order1, markersize=18, linewidth=5, estimator=np.mean, palette=['black', 'orange', 'red'], errorbar=('se',1), legend=True)
    handles, labels = ax.get_legend_handles_labels()
    plt.axhline(0, linestyle='solid', color='black')
    plt.xlabel('Participants');plt.ylabel('Pupil-size mean differences\n (Dark - Bright) (a.u.)')
    plt.legend(handles, labels, ncol=5, title='Visual Aphantasia', frameon=False, loc='upper center')
    #plt.xticks([])
    #plt.ylim([-1500, 1500])
    plt.tight_layout()
    plt.show()
    
    # plt.figure(figsize=(20,10))
    # ax = plt.subplot(1,1,1)
    # plt.title(s)
    # sns.pointplot(data=dm_sub, x='subject_nr', y='pupil_change', hue='outlier', hue_order=[0, 1], order=list_order1, markersize=18, linewidth=5, estimator=np.mean, palette=['black', 'red'], errorbar=('se',1), legend=True)
    # handles, labels = ax.get_legend_handles_labels()
    # plt.axhline(0, linestyle='solid', color='black')
    # plt.xlabel('Participants');plt.ylabel('Pupil-size mean differences\n (Dark - Bright) (a.u.)')
    # plt.legend(handles, ['No', 'Yes'], ncol=5, title='Outlier', frameon=False, loc='upper center')
    # #plt.xticks([])
    # #plt.ylim([-1500, 1500])
    # plt.tight_layout()
    # plt.show()
    
    plt.figure(figsize=(20,10))
    ax = plt.subplot(1,1,1)
    plt.title(s)
    sns.pointplot(data=dm_sub, x='subject_nr', y='pupil_change', hue='subject_parity', hue_order=None, order=list_order1, markersize=18, linewidth=5, estimator=np.mean, palette="hls", errorbar=('se',1), legend=True)
    handles, labels = ax.get_legend_handles_labels()
    plt.axhline(0, linestyle='solid', color='black')
    plt.xlabel('Participants');plt.ylabel('Pupil-size mean differences\n (Dark - Bright) (a.u.)')
    plt.legend(handles, labels, ncol=5, title='Participant number', frameon=False, loc='upper center')
    #plt.xticks([])
    #plt.ylim([-1500, 1500])
    plt.tight_layout()
    plt.show()

# # 2. Mean pupil size changes per participant
# dm = dm_ctrl.pupil_change != NAN
# dm.mean_pupil_change = ''
# for s, sdm in ops.split(dm.subject_nr):
#     dm.mean_pupil_change[sdm] = sdm.pupil_change.mean
# dm = ops.sort(dm, by=dm.mean_pupil_change)
# list_order1 = list(dict.fromkeys(dm.subject_nr))
# dm_sub = convert.to_pandas(dm)
# plt.figure(figsize=(20,12))
# ax = plt.subplot(1,1,1)
# #plt.title('All non-dynamic stories')
# sns.pointplot(data=dm_sub[dm_sub.suptype=='rabbit'], x='subject_nr', y='pupil_change', hue='version', order=list_order1, markersize=18, linewidth=5, estimator=np.mean, palette=[red[2], green[2]], errorbar=('se',1), legend=True)
# sns.pointplot(data=dm_sub[dm_sub.suptype=='self'], x='subject_nr', y='pupil_change', hue='version', order=list_order1, markersize=18, linewidth=5, estimator=np.mean, palette=[red[0], green[0]], errorbar=('se',1), legend=True, linestyles='dashed')
# handles, labels = ax.get_legend_handles_labels()
# plt.axhline(0, linestyle='solid', color='black')
# plt.xlabel('Participants');plt.ylabel('Pupil-size mean differences\n (Dark - Bright) (a.u.)')
# plt.legend(handles, ['Rabbit: Bright -> Dark', 'Rabbit: Dark -> Bright', 'Self: Bright -> Dark', 'Self: Dark -> Bright'], ncol=2, title='Stories order', frameon=False, bbox_to_anchor=[0.8, 1.4])
# #plt.xticks([])
# #plt.ylim([-1500, 1500])
# plt.tight_layout()
# plt.show()

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

