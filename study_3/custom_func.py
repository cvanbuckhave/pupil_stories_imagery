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
from datamatrix import plot
import matplotlib as mpl
from datamatrix.colors.tango import blue, green, gray
import seaborn as sns
import pandas as pd

# Operations and pupil stuff
import numpy as np
import warnings
import datamatrix
from datamatrix import series as srs, NAN, operations as ops, convert
from eyelinkparser import parse, defaulttraceprocessor
from datamatrix.multidimensional import reduce

# Stats
import statsmodels.api as sm
import scipy.stats as stats
import pingouin as pg
import time_series_test as tst
from statsmodels.formula.api import mixedlm
from scipy.stats import spearmanr, shapiro, wilcoxon, mannwhitneyu
from statsmodels.stats.diagnostic import het_white
from scipy.stats.distributions import chi2
from collections import Counter
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

# Define dict
response_dict_en = {'No image at all, you only “know” that you were thinking of the scene': 1,
                    'Vague and dim': 2,
                    'Moderately clear and vivid': 3,
                    'Clear and reasonably vivid': 4,
                    'Perfectly clear and vivid as real seeing': 5,
                    'Very negative': -3,
                    'Negative': -2,
                    'Slightly Negative': -1,
                    'Neutral': 0,
                    'Slightly Positive': 1,
                    'Positive': 2,
                    'Very Positive': 3,
                    'Very low effort': -2,
                    'Low effort': -1,
                    'Neither low nor high effort': 0,
                    'High effort': 1,
                    'Very high effort': 2}
                   
# =============================================================================
#  Define useful functions
# =============================================================================
def get_raw(path, dataframe):
    dm = parse(
        maxtracelen=12700,
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,
            downsample=10,
            mode='advanced'),
        folder=path,
        pupil_size=True,
        gaze_pos=False,              # Don't store gaze-position information to save memory
        time_trace=False,            # Don't store absolute timestamps to save memory
        multiprocess=4,
        asc_encoding='cp1252')
    
    print('Blink reconstruction and down sampling (100 Hz) have been applied.')
        
    # Create theoretical (in samples) and true (in seconds) duration columns 
    dm.duration, dm.dur = '', ''
    for scene, sdm in ops.split(dm.scene):
        duration_theo = int(dataframe.samples[dataframe.scene==scene])  # in samples
        duration_true = int(dataframe.duration[dataframe.scene==scene]) # in seconds 

        dm.duration[sdm] = duration_theo
        dm.dur[sdm] = duration_true
    
    # Recode answers
    cols = [dm.response_vivid, dm.response_effort, dm.response_val]
    for var in cols:
        for rep, sdm in ops.split(var):
            var[var == rep] = response_dict_en[rep]
            
    # Assign new variable
        # Emotional intensity (absolute value of valency)
    dm.emotional_intensity = np.abs(dm.response_val)
        # Correct answer
    dm.correct_answer[dm.correct_answer=='yes'] = 1
    dm.correct_answer[dm.correct_answer=='no'] = 0
    dm.correct_answer = dm.correct_answer * 100 # as percentage
        # Create suptype
    dm.suptype = ''
    dm.suptype[dm.subtype == 'self'] = 'self'
    dm.suptype[dm.subtype != 'self'] = 'rabbit'   
    
    # Which version is it? (versio 1: bright->dark; version 2: dark->bright)
    dm.version = ''
    for p, s, sdm in ops.split(dm.subject_nr, dm.scene):
        dm.version[sdm] = int(str(sdm.scene.unique[0])[-1])
        # Keep track of trialid / presentation order
    dm.order = ''
    for p, s, sdm in ops.split(dm.subject_nr, dm.suptype):
        if (sdm.trialid[sdm.type == 'dark'][0] < sdm.trialid[sdm.type == 'light'][0]) == True:
            dm.order[sdm.type == 'dark'] = "first"
            dm.order[sdm.type == 'light'] = "second"
        else:
            dm.order[sdm.type == 'dark'] = "second"
            dm.order[sdm.type == 'light'] = "first"
        
    return dm[dm.ptrace_target, dm.ptrace_fixation, dm.subject_nr,
              dm.response_vivid, dm.response_effort, dm.response_val, dm.suptype,
              dm.correct_answer, dm.emotional_intensity, dm.suptype, dm.dur,
              dm.type, dm.subtype, dm.scene, dm.trialid, dm.duration, dm.stimID, 
              dm.blinkstlist_fixation, dm.blinkstlist_target,
              dm.fixylist_target, dm.fixxlist_target, dm.trace_length_target,
              dm.order, dm.version, dm.trace_length_fixation]

def merge_dm_df(dm_to_merge, df):
    """Compute main variables and add them to the dm."""
    # Copy of dm
    dm = dm_to_merge.subject_nr != ''
    
    # Match participants between dm and df
    new, df['excluded'] = dm.subject_nr.unique, 0
    for i in list(df.index):
        if df.Q00ID[i] in new: 
            df.loc[i, 'excluded'] = 0
        else: 
            df.loc[i, 'excluded'] = 1
    df = df[df.excluded == 0].reset_index()
    print(f'Datamatrix and dataframe match in terms of participants: {list(np.sort(df.Q00ID)) == list(np.sort(new))}')

    # Compute the mean VVIQ score for each participant
    VVIQ_cols = [i for i in df.columns if i.startswith('VVIQQ0')]
    df['VVIQ'] = df[VVIQ_cols].mean(axis=1)
    print(f"Cronbach's alpha for VVIQ: {pg.cronbach_alpha(data=df[VVIQ_cols])}")
    
    # Compute the mean SUIS score for each participant
    SUIS_cols = [i for i in df.columns if i.startswith('SUISQ')]
    df['SUIS'] = df[SUIS_cols].mean(axis=1)
    print(f"Cronbach's alpha for SUIS: {pg.cronbach_alpha(data=df[SUIS_cols])}")
    
    # Recode
    df.Q02LANG = df.Q02LANG.replace({'-oth-': 'Other'})
    df.Q04SEX = df.Q04SEX.replace({1: 'M', 2: 'F'})
    
    # Merge columns of the df to the dm
    dm.VVIQ, dm.SUIS, dm.lang_comp, dm.vision, dm.hearing = '', '', '', '', ''
    dm.age, dm.sex, dm.mothertongue = '', '', ''
    for p, sdm in ops.split(dm.subject_nr): # subset the dm
        if p != 11: # No questionnaire data for participant 11 (limesurvey crashed when they pressed 'submit')
            sub_df = df[df['Q00ID'] == p] # Subset the df too
            dm.VVIQ[sdm] = float(sub_df.VVIQ.iloc[0])
            dm.SUIS[sdm] = float(sub_df.SUIS.iloc[0])
            dm.lang_comp[sdm] = float(sub_df.Q03LANG.iloc[0])
            dm.vision[sdm] = float(sub_df.Q05EYE.iloc[0])
            dm.hearing[sdm] = float(sub_df.Q06EAR.iloc[0])
            dm.age[sdm] = float(sub_df.Q01AGE.iloc[0])
            dm.sex[sdm] = str(sub_df.Q04SEX.iloc[0])
            dm.mothertongue[sdm] = str(sub_df.Q02LANG.iloc[0])
    
    # Create aphant category if any
    dm.aphantasia = ''
    for p, sdm in ops.split(dm.subject_nr):
        if sdm.VVIQ > 2:
            dm.aphantasia[sdm] = 'No'
        elif sdm.VVIQ == 1:
            dm.aphantasia[sdm] = 'Yes'
        elif sdm.VVIQ <= 2:
            dm.aphantasia[sdm] = 'Mild'
    print(f'{len(dm.subject_nr[dm.aphantasia == "Yes"].unique)} participant with VVIQ < 2 {dm.subject_nr[dm.aphantasia == "Yes"].unique}.')
    
    # Print descriptives (experiment)
    for type_, sup_, sdm in ops.split(dm.type, dm.suptype):
        print('\n', type_, sup_)    
        print(f'On average, participants reported that imagining the {type_} stories necessitated "... effort" (M = {np.round(sdm.response_effort.mean,3)}, SD = {np.round(sdm.response_effort.std,3)}, n = {len(sdm)}),')
        print(f'induced "... emotions" (M = {np.round(sdm.emotional_intensity.mean,3)}, SD = {np.round(sdm.emotional_intensity.std,3)}, n = {len(sdm)})')
        print(f'and were imagined as "..." (M = {np.round(sdm.response_vivid.mean,3)}, SD = {np.round(sdm.response_vivid.std,3)}, n = {len(sdm)}).')
        print(f'The mean accuracy for this brightness condition was {np.round(sdm.correct_answer.mean,2)} (SD = {np.round(sdm.correct_answer.std,2)}, n = {len(sdm)}).')
    
    # aggregate data by subject and condition
    pm = ops.group(dm, by=[dm.subject_nr, dm.suptype])
        
    # calculate mean blink rate per condition
    pm.response_effort = srs.reduce(pm.response_effort)
    pm.emotional_intensity = srs.reduce(pm.emotional_intensity)
    pm.response_vivid = srs.reduce(pm.response_vivid)
    pm.correct_answer = srs.reduce(pm.correct_answer)

    # Print descriptives (experiment)
    for sup_, sdm in ops.split(pm.suptype):
        print('\n', sup_)    
        print(f'Effort: M = {np.round(sdm.response_effort.mean,2)}, SD = {np.round(sdm.response_effort.std,2)}, n = {len(sdm)}')
        print(f'Emotional intensity: M = {np.round(sdm.emotional_intensity.mean,2)}, SD = {np.round(sdm.emotional_intensity.std,2)}, n = {len(sdm)}')
        print(f'Vividness: M = {np.round(sdm.response_vivid.mean,2)}, SD = {np.round(sdm.response_vivid.std,2)}, n = {len(sdm)}')
        print(f'Accuracy: {np.round(sdm.correct_answer.mean,2)}, SD = {np.round(sdm.correct_answer.std,2)}, n = {len(sdm)}')

    print('Mean accuracy %:', np.round(pm.correct_answer.mean, 2), np.round(pm.correct_answer.std, 2))
    print(Counter(pm.correct_answer))
    
    # Print descriptives (questionnaire)
    print(df[['VVIQ', 'SUIS']].describe())
    print(df[['Q01AGE', 'Q02LANG', 'Q03LANG', 'Q05EYE', 'Q06EAR']].describe())
    print(df[['Q02LANG']].describe())
    print(df.groupby('Q04SEX').Q01AGE.describe())

    return dm

def count_nonnan(a):
    return np.sum(~np.isnan(a))

def check_blinks(dm_blinks, new=False):
    """Check number of blinks per condition and per participant."""
    if new==True:
        dm_blinks.n_blinks=NAN
        for p, s, sdm in ops.split(dm_blinks.subject_nr, dm_blinks.stimID):
            dm_blinks.n_blinks[sdm] = np.round((reduce(sdm.blinkstlist_target, count_nonnan) / (sdm.trace_length_target / 1000)) * 60)

        dm_blinks.mean_blinks=NAN
        for p, sdm in ops.split(dm_blinks.subject_nr):
            dm_blinks.mean_blinks[sdm] = sdm.n_blinks.mean

    # aggregate data by subject and condition
    pm = ops.group(dm_blinks, by=[dm_blinks.subject_nr, dm_blinks.scene])
        
    # calculate mean blink rate per condition
    pm.mean_blink_rate = srs.reduce(pm.n_blinks)
    
    print("Blinks during listening")
    for identifier in dm_blinks.subject_nr.unique:
        print(f"{identifier}: {round(dm_blinks.n_blinks[dm_blinks.subject_nr == identifier].mean,3)} (Total = {dm_blinks.n_blinks[dm_blinks.subject_nr == identifier].sum})")
    
    df_blinks = convert.to_pandas(dm_blinks)
    df_pm = convert.to_pandas(pm)

    # Plot the mean blink rate as a function of experimental condition and participant
    plt.figure(figsize=(20,8))
    x = sns.pointplot(
        x="scene",
        y="n_blinks",
        hue="subject_nr",
        data=df_blinks,
        ci=None,
        palette=sns.color_palette(['indianred']),
        markers='.')
    plt.setp(x.lines, alpha=.4)
    plt.setp(x.collections, alpha=.4)
    sns.pointplot(
        x="scene",
        y="mean_blink_rate",
        data=df_pm,
        linestyles='solid',
        color='crimson',
        markers='o',
        scale=2)
    plt.xlabel('Story', fontsize=30)
    plt.ylabel('Blink rate\n(number of blinks per minute)', fontsize=30)
    plt.legend([], [], frameon=False)
    plt.xticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    
    # Check fixations 
    plt.figure(figsize=(15,8));plt.suptitle('Gaze / eye position', fontsize=35)
    plt.subplot(1,2,1);plt.title('Bright stories', fontsize=30)
    x = np.array(dm_blinks.fixxlist_target[dm_blinks.type=='light'])
    y = np.array(dm_blinks.fixylist_target[dm_blinks.type=='light'])
    x = x.flatten()
    y = y.flatten()
    plt.hexbin(x, y, gridsize=25)
    plt.axhline(500, color='white');plt.axvline(500, color='white')
    plt.yticks(fontsize=25);plt.xticks(fontsize=25);plt.xlabel('x', fontsize=25);plt.ylabel('y', fontsize=25)
    plt.subplot(1,2,2);plt.title('Dark stories', fontsize=30)
    x = np.array(dm_blinks.fixxlist_target[dm_blinks.type=='dark'])
    y = np.array(dm_blinks.fixylist_target[dm_blinks.type=='dark'])
    x = x.flatten()
    y = y.flatten()
    plt.hexbin(x, y, gridsize=25)
    plt.axhline(500, color='white');plt.axvline(500, color='white')
    plt.yticks(fontsize=25);plt.xticks(fontsize=25);plt.xlabel('x', fontsize=25);plt.ylabel('y', fontsize=25)
    plt.tight_layout()
    plt.show()
    
    return dm_blinks

# Preprocessing and removing outlier trials
def preprocess_dm(dm_to_process):
    """
    Remove trials with baseline pupil sizes under or above 2 SD, apply baseline 
    correction and compute all variables.
    """
    # Copy of dm to not overwrite it
    dm_ = dm_to_process.subject_nr != ''
    
    print(f'Before preprocessing: {len(dm_)} trials.')

    # How many errors?
    error_participants = dm_.subject_nr[dm_.correct_answer == 0]
    errors_list = Counter(error_participants)
    print(f'For participants who had at least one error, how many did they make: {errors_list}')
    print(f'How many participants had n number of errors: {Counter(errors_list.values())}')
    
    # Check blink rate
    plt.figure(figsize=(13,8));warnings.filterwarnings("ignore", category=UserWarning) 
    
    dm_.z_blinks = ops.z(dm_.n_blinks) # z-transform 
    dm_.z_blinks_ = ops.z(dm_.mean_blinks) # z-transform

    sns.distplot(dm_.z_blinks);plt.axvline(2)
    plt.xlabel('Blink rate per minute (z-scored)')
    plt.tight_layout()
    plt.show();warnings.filterwarnings("default", category=UserWarning)  

        # If we don't want to exclude them and add an 'is_outlier' variable in the model
    dm_.is_outlier_trial = 'no'
    dm_.is_outlier_trial[dm_.z_blinks > 2.0] = 'yes'
    
    dm_.is_outlier = 'no'
    dm_.is_outlier[dm_.z_blinks_ > 2.0] = 'yes'
    
    blinky = list(dm_.subject_nr[dm_.z_blinks >= 2.0])
    blinks = set(dm_.n_blinks[dm_.z_blinks >= 2.0])
    print(f'N = {len(blinky)} ({len(set(blinky))} participants) with a lot of blinks ({blinks}; M = {dm_.n_blinks.mean}, STD = {dm_.n_blinks.std})')
    print(blinky)
            
    # Check percentage of non-nan values for each trial
    dm_.nonnan_pupil = ''
    for p, t, sdm in ops.split(dm_.subject_nr, dm_.stimID):
        dm_.nonnan_pupil[sdm] = count_nonnan(sdm.ptrace_target[:, 0:int(sdm.duration.unique[0])]) / int(sdm.duration.unique[0]) * 100 # percentage of non-nan values
    
    plt.figure(figsize=(13,8));warnings.filterwarnings("ignore", category=UserWarning)  
    sns.distplot(dm_.nonnan_pupil)
    plt.xlabel('Percentage of valid samples per trial')
    plt.tight_layout()
    plt.show();warnings.filterwarnings("default", category=UserWarning)  
    print(f'{len(dm_[dm_.nonnan_pupil < 50])} trials with more than 50% of NAN values ({len(dm_.subject_nr[dm_.nonnan_pupil < 50].unique)} participants).')
   
    # The main pupil trace
    dm_.ptrace_fixation.depth = 100 # 1s total duration

    # Interpolate the pupil traces because trial exclusion is costy
    for s, sd, sdm in ops.split(dm_.subject_nr, dm_.trialid):
        dm_.ptrace_fixation[sdm, 0:100] = srs.interpolate(sdm.ptrace_fixation)[:, 0:100]
        dm_.ptrace_target[sdm, 0:int(sdm.duration.unique[0]+100)] = srs.interpolate(sdm.ptrace_target)[:, 0:int(sdm.duration.unique[0]+100)]
    print('Pupil size traces linearly interpolated.')
    
    # Trim tails        
    dm_.ptrace_fixation = srs.trim(dm_.ptrace_fixation, value=NAN, end=True, start=True)
    dm_.ptrace_target = srs.trim(dm_.ptrace_target, value=NAN, end=True, start=True)

    # Smooth the traces to reduce the jitter
    dm_.ptrace_target = srs.smooth(dm_.ptrace_target, 5) 
    print('Pupil size traces smoothed with a Hanning window of size 51.')
    
    # Baseline correction on the 50 ms before story onset with the subtractive method
    dm_.baseline, dm_.pupil = ops.SeriesColumn(depth=100), ops.SeriesColumn(depth=dm_.ptrace_target.depth)
    dm_.baseline = dm_.ptrace_fixation[:, 95:100] # the 50 ms baseline period
    
    for sup, t, sdm in ops.split(dm_.suptype, dm_.trialid):
        if sup == 'rabbit':
            dm_.pupil[sdm] = srs.baseline(sdm.ptrace_target, sdm.baseline, 0, 5, method='subtractive') # baseline correct pupil size
        else:
            dm_.pupil[sdm] = srs.baseline(sdm.ptrace_target, sdm.baseline, 0, 5, method='subtractive') # baseline correct pupil size

        #dm_.ptrace_fixation[sdm] = srs.baseline(sdm.ptrace_fixation, sdm.ptrace_fixation, 0, 5, method='subtractive') # baseline correct pupil size
    print('Baseline-correction applied on the last 50 ms before story onset.')
    
    # Exclude trials with unrealistic baseline-corrected mean pupil size (outliers) 
    print(f'Before trial exclusion (pupil size): {len(dm_)} trials.')
    plt.figure(figsize=(13,8));warnings.filterwarnings("ignore", category=UserWarning)  
    dm_.z_pupil = NAN
    for s, sdm in ops.split(dm_.suptype):
        dm_.z_pupil[sdm] = ops.z(reduce(sdm.pupil))
        sns.distplot(sdm.z_pupil, label=s)
    plt.axvline(-2);plt.axvline(2)
    plt.xlabel('Mean pupil size (z-scored)')
    plt.tight_layout();plt.legend()
    plt.show();warnings.filterwarnings("default", category=UserWarning)  
    
    unrealistic1 = list(np.array(dm_.subject_nr[dm_.z_pupil > 2.0]))
    unrealistic2 = list(np.array(dm_.subject_nr[dm_.z_pupil < -2.0]))
    nan_ = list(np.array(dm_.subject_nr[dm_.z_pupil == NAN]))
    print(unrealistic1, unrealistic2, nan_)
    print(f'{len(unrealistic1)+len(unrealistic2) + len(nan_)} trials with outlier or NAN pupil sizes.')
    
    dm_ = dm_.z_pupil != NAN 
    dm_ = dm_.z_pupil <= 2.0 
    dm_ = dm_.z_pupil >= -2.0 
    print(f'After trial exclusion (pupil size): {len(dm_)} trials.')

    # Compute the mean pupil size and mean slopes during listening
    dm_.mean_pupil, dm_.mean_fixation = NAN, NAN
    dm_.raw_mean = NAN
    for p, trial, sdm in ops.split(dm_.subject_nr, dm_.trialid):
        suptype = sdm.suptype.unique[0]
        if suptype == 'rabbit':
            start, minus = 0, 20
        else:
            start, minus = 0, 20
        # Take only the real duration of the trace minus 200 ms to compute the mean and slopes
        # to prevent taking into account edge effects 
        duration = int(start + sdm.duration.unique[0] - minus)
        pupil = sdm.pupil[:, start:duration] # don't take the first 50 ms neither (baseline)
        pupil_raw = sdm.ptrace_target[:, start:duration] # don't take the first 50 ms neither (baseline)

        if count_nonnan(pupil) > 0:            
            # Pupil-size mean
            dm_.mean_pupil[sdm] = reduce(pupil, np.nanmean)
            dm_.mean_fixation[sdm] = reduce(sdm.baseline, np.nanmean)
            dm_.raw_mean[sdm] = reduce(pupil_raw, np.nanmean)

    print('Mean pupil size and slopes were computed over the whole listening phase for each trial.')
        
    # Create new variables 
    dm_.pupil_change = NAN
    dm_.mean_vivid, dm_.mean_effort, dm_.mean_emo, dm_.mean_val = NAN, NAN, NAN, NAN
    for p, s, sdm in ops.split(dm_.subject_nr, dm_.suptype):

        # Compute pupil-size changes as differences in mean pupil size between dark - bright conditions
        pupil_change = sdm.mean_pupil[sdm.type =='dark'].mean - sdm.mean_pupil[sdm.type =='light'].mean 
        dm_.pupil_change[sdm] = pupil_change
      
        # Mean ratings per subtype (averaged across dark and bright stories)
        dm_.mean_vivid[sdm] = sdm.response_vivid.mean
        dm_.mean_effort[sdm] = sdm.response_effort.mean
        dm_.mean_val[sdm] = sdm.response_val.mean
        dm_.mean_emo[sdm] = sdm.emotional_intensity.mean
    

    #dm_ = dm_.pupil_change != NAN # keep only valid pairs of mean pupil size
    
    # How many trials left per participant?
    print(f'Number of trials left per participant: {np.sort(Counter(dm_.subject_nr), axis=None)}')
    print(f'How many participants have n number of trials: {Counter(Counter(dm_.subject_nr).values())}')
    
    to_exclude = [i for i in dict(Counter(dm_.subject_nr)).keys() if dict(Counter(dm_.subject_nr))[i] < 2]
    
    print(f'{to_exclude} has less than 50% of remaining trials.')
    dm_ = dm_.subject_nr != set(to_exclude)
        
    print(f'After preprocessing: N = {len(dm_.subject_nr.unique)} (n = {len(dm_)} trials)')
    
    print(Counter(dm_.subject_nr[dm_.pupil_change == NAN]))
    print(Counter(dm_.suptype[dm_.pupil_change == NAN]))

    return dm_

def quick_visualisation(dm_to_visualise, raw=False, after_preprocess=True, cwd=str):
    """Pupil traces and mean pupil size per cond and per task."""
    
    # Quick visualisation
    fig, axes = plt.subplots(1, 1, figsize=(28,10))
    plt.subplot(1,1,1)#;plt.title('Rabbit trials')
    sdm=dm_to_visualise.suptype=='rabbit'
    if raw == False:
        tst.plot(sdm, dv='pupil', hue_factor='type', hues=[blue[1], green[1]], 
                 legend_kwargs={'frameon': False, 'loc': 'lower center', 'ncol': 2, 'labels': [f'Dark (N={len(sdm[sdm.type=="dark"])})', f'Bright (N={len(sdm[sdm.type=="light"])})']},
                 annotation_legend_kwargs={'frameon': False, 'loc': 'lower center', 'ncol': 2}, 
                 x0=0, sampling_freq=1)
    else:
        tst.plot(sdm, dv='ptrace_target', hue_factor='type', hues=[blue[1], green[1]], 
                 legend_kwargs={'frameon': False, 'loc': 'lower center', 'ncol': 2, 'labels': [f'Dark (N={len(sdm[sdm.type=="dark"])})', f'Bright (N={len(sdm[sdm.type=="light"])})']},
                 annotation_legend_kwargs={'frameon': False, 'loc': 'lower center', 'ncol': 2}, 
                 x0=0, sampling_freq=1)
    plt.xticks(np.arange(0, 11700+500, 500), np.arange(0, int(11700/100)+5, 5))
    plt.xlim([0, 11700])
    plt.xlabel('Time since story onset (s)', fontsize=40)
    plt.ylabel('Baseline-corrected\npupil size (a.u.)', fontsize=45)
    plt.tight_layout()
    plt.savefig(cwd+"/figs/rabbit_traceplot.png", bbox_inches='tight')
    plt.show()
    
    fig, axes = plt.subplots(1, 1, figsize=(28,10))
    plt.subplot(1,1,1)#;plt.title('Self trials')
    sdm=dm_to_visualise.suptype=='self'
    if raw==False:
        plot.trace(sdm.pupil[sdm.type=='dark'], color=blue[1], label=f'Dark (N = {len(sdm.pupil[sdm.type=="dark"])})')
        plot.trace(sdm.pupil[sdm.type=='light'], color=green[1], label=f'Dark (N = {len(sdm.pupil[sdm.type=="light"])})')
    else:
        plot.trace(sdm.ptrace_target[sdm.type=='dark'], color=blue[1], label=f'Dark (N = {len(sdm.ptrace_target[sdm.type=="dark"])})')
        plot.trace(sdm.ptrace_target[sdm.type=='light'], color=green[1], label=f'Dark (N = {len(sdm.ptrace_target[sdm.type=="light"])})')
    plt.legend(loc='lower center', frameon=True,ncol=2)
    plt.xticks(range(0,3100,100), range(0,31,1))#;plt.ylim([-800,800])
    plt.xlim([0, 3000])
    plt.xlabel('Time since start trial (s)', fontsize=40)
    fig.supylabel('Baseline-corrected\npupil size (a.u.)', fontsize=45)
    plt.tight_layout()
    plt.savefig(cwd+"/figs/self_traceplot.png", bbox_inches='tight')
    plt.show()
    
    if after_preprocess==True:
        # Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
        dm_df = convert.to_pandas(dm_to_visualise)
    
        # Mean pupil sizes and mean slopes per condition (all)
        dm_sub1 = dm_df[dm_df.mean_pupil != '']
        #dm_sub1 = dm_sub1[dm_sub1.pupil_change != '']
        fig, axes = plt.subplots(1, 1, figsize=(18,8))
        ax2=plt.subplot(1,1,1)
        if raw==False:
            plot_bars(dm_sub1, x='suptype', y='mean_pupil', hue='type', hue_order=['light', 'dark'], order=None, pal=[green[1], blue[1]], fig=False, alpha=0.7)
        else:
            plot_bars(dm_sub1, x='suptype', y='raw_mean', hue='type', hue_order=['light', 'dark'], order=None, pal=[green[1], blue[1]], fig=False, alpha=0.7)
        handles, labels = ax2.get_legend_handles_labels()
        plt.xticks(range(0, 2), ['Great Rabbit', 'Self-selected'])
        plt.xlabel('Story content');plt.ylabel('Mean pupil-size changes\nrelative to baseline (a.u.)', color='black')
        plt.legend(handles=handles, labels=['Bright', 'Dark'], frameon=False, title='Version')
        plt.tight_layout()
        plt.savefig(cwd+"/figs/exp3_traceplot.png", bbox_inches='tight')
        plt.show()
        
def create_control_variables(original_dm):
    """All the control variables."""
    # Create copy of dm to not overwrite it
    dm = original_dm.subject_nr != ''
    
    # Effort, valence, etc.
    dm.vivid_changes, dm.val_changes, dm.emo_changes, dm.effort_changes = NAN, NAN, NAN, NAN
    dm.mean_diff = NAN
    for p, s, sdm in ops.split(dm.subject_nr, dm.suptype):
        # Rating differences
        dm.vivid_changes[sdm] = sdm.response_vivid[sdm.type =='dark'].mean - sdm.response_vivid[sdm.type =='light'].mean
        dm.effort_changes[sdm] = sdm.response_effort[sdm.type =='dark'].mean - sdm.response_effort[sdm.type =='light'].mean 
        dm.val_changes[sdm] = sdm.response_val[sdm.type =='dark'].mean - sdm.response_val[sdm.type =='light'].mean 
        dm.emo_changes[sdm] = sdm.emotional_intensity[sdm.type =='dark'].mean - sdm.emotional_intensity[sdm.type =='light'].mean 
        
    # New variables free of variation between types
    dm.vivid = dm.mean_vivid - np.abs(dm.vivid_changes)

    # How many non-nan values in the pupil time series?
    dm.nonnan_pupil = NAN
    for p, t, sdm in ops.split(dm.subject_nr, dm.stimID):
        dm.nonnan_pupil[sdm] = int((count_nonnan(sdm.pupil[:, :int(sdm.duration[0])]) / sdm.duration) * 100)
    print(f'/!\ {len(dm.subject_nr[dm.nonnan_pupil < 50])} trials with very few non-nan values ({list(dm.nonnan_pupil[dm.nonnan_pupil < 50])}% of non-nan values)')

    # Keep track of trialid / presentation order
    dm.order_changes = NAN
    for p, s, sdm in ops.split(dm.subject_nr, dm.suptype):
        if len(sdm) > 1: # make sure that there are indeed 2 trials to compare
            n0_dark = sdm.trialid[sdm.type == 'dark'][0]
            n0_bright = sdm.trialid[sdm.type == 'light'][0]
            
            # the rank difference between the two
            dm.order_changes[sdm] = n0_dark - n0_bright
            # if dark is before bright:
                # pupil size is greater for dark 
                # trialid of dark is smaller
                # so the difference is negative
                # but should have greater pupil differences
    
    
    # Mean of changes that can affect pupil size
    dm.changes = (dm.effort_changes + dm.emo_changes)/2

    # Compute the percentage of trials that are in an order that would facilitate 
    # having smaller pupil sizes for bright stories per participant
    dm.numbers = NAN
    for s, sdm in ops.split(dm.subject_nr):
        if len(sdm) > 0:
            dm.numbers[sdm] = len(sdm[sdm.order_changes < 0]) / len(sdm) * 100
    
    for s, sdm in ops.split(dm.suptype):
        sdm = sdm.pupil_change != NAN
        print(f'{s}: {len(sdm[sdm.order_changes < 0])/len(sdm) * 100} % of trial pairs in the dark->bright order (n = {len(sdm)})')

    return dm

def plot_bars(dm, x=str, y=str, hue=None, hue_order=['light', 'dark', 'light_dark', 'dark_light'], order=None, color=None, pal='deep', xlab='Condition', ylab='Mean Pupil Size (a.u.)', title='', fig=True, alpha=1, legend=True, fs=30):
    """Plot the mean pupil size as bar plots."""
    plt.rcParams['font.size'] = fs
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


# Stats
def lm_pupil(dm_tst, formula=False, re_formula="1 + type", pupil_change=False, reml=False, method='Powell'):
    """Test how brightness (dark vs. light) affects the mean pupil size."""    
    # Make sure to filter the 'dynamic' condition
    dm_test = dm_tst.subtype != ''    
    
    # Remove Nans
    dm_valid_data = dm_test.mean_pupil != NAN # remove NaNs 

    if pupil_change == True:
        # Suppress warnings because it's annoying
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)

        dm_valid_data = ops.group(dm_valid_data, by=[dm_valid_data.subject_nr, dm_valid_data.suptype, dm_valid_data.is_outlier]) # add dm_sub.response_lang if necessary

        # Make sure to have only unique mean values for each variable per participant 
        for col in dm_valid_data.column_names:
            if type(dm_valid_data[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
                dm_valid_data[col] = reduce(dm_valid_data[col]) # Compute the mean per subtype 
        
        # Unable back the warnings
        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)
        warnings.filterwarnings("default", category=UserWarning)  
    
        dm_valid_data.pupil_change = np.round(dm_valid_data.pupil_change, 10) # this is to prevent issues with assumption checks
        dm_valid_data = dm_valid_data.pupil_change != NAN # make sure there's always at least 2 stories to compare

    # The model
    md = mixedlm(formula, dm_valid_data, 
                     groups='subject_nr',
                     re_formula=re_formula)
    
    mdf = md.fit(reml=reml, method=method)
        
    return mdf


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


def check_assumptions(model, suptype):
    """Check assumptions for normality of residuals and homoescedasticity.
    Code from: https://www.pythonfordatascience.org/mixed-effects-regression-python/#assumption_check"""
    plt.rcParams['font.size'] = 40
    print('Assumptions check:')
    fig = plt.figure(figsize = (25, 16))
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    fig.suptitle(f'{suptype}: {model.model.formula} (n = {model.model.n_groups})')
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

def test_correlation(dm_c, x, y, alt='two-sided', pcorr=1, color='red', lab='VVIQ', fig=True, fs=30):
    """Test the correlations between pupil measures and questionnaire measures using Spearman's correlation."""
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Group per participant 
    dm_cor = dm_c.pupil_change != NAN

    dm_cor = ops.group(dm_cor, by=dm_cor.subject_nr)
    
    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_cor.column_names:
        if type(dm_cor[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_cor[col] = reduce(dm_cor[col], operation=np.nanmean)
    
    #dm_cor = dm_cor.mean_pupil != NAN
    
    # The variables to test the correlation
    x, y = dm_cor[x], dm_cor[y]
    
    # Unable back the warnings
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
    # Compute spearman's rank correlation
    cor=spearmanr(x, y, alternative=alt)

    N = len(dm_cor.subject_nr.unique)
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
    print(res)
    
    # Plot the correlations (linear regression model fit)
    plt.rcParams['font.size'] = fs

    if lab != False:
        label = fr'{lab}: {res}'
    else:
        label = ''
    if fig == True:
        sns.regplot(data=dm_cor, x=x.name, y=y.name, lowess=False, color=color, label=label, x_jitter=0, y_jitter=0, scatter_kws={'alpha': 0.5, 's': 100}, robust=True)
        plt.legend(frameon=False, markerscale=3, loc='upper center')
        # use statsmodels to estimate a nonparametric lowess model (locally weighted linear regression)
        sns.regplot(data=dm_cor, x=x.name, y=y.name, lowess=True, color=color, label=None, x_jitter=0, y_jitter=0, scatter_kws={'alpha': 0.0}, line_kws={'linestyle': 'dashed', 'alpha':0.6, 'linewidth': 8})
        
    return res


def dist_checks_wilcox(dm, suptype):
    """Non-parametric paired t-tests with Wilcoxon signed-rank test."""
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    dm_sub = dm.suptype == suptype
    dm_sub = dm_sub.pupil_change != NAN
    dm_sub = ops.group(dm_sub, by=[dm_sub.subject_nr, dm_sub.type, dm_sub.order]) 

    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_sub.column_names:
        if type(dm_sub[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_sub[col] = reduce(dm_sub[col]) # Compute the mean per subtype 
        
    print(f"\n{suptype} (n = {len(dm_sub[dm_sub.type == 'light'])})")
    print('Within participants: (light vs.dark)')
    s, p = wilcoxon(dm_sub.n_blinks[dm_sub.type == 'light'], dm_sub.n_blinks[dm_sub.type == 'dark'], nan_policy='omit')
    print(f"Blinks: light != dark, S = {s:.0f}, p = {np.round(p,3)}")
    print(f"Bright: M = {np.round(dm_sub.n_blinks[dm_sub.type == 'light'].mean, 2)}, SD = {np.round(dm_sub.n_blinks[dm_sub.type == 'light'].std,2)}")
    print(f"Dark: M = {np.round(dm_sub.n_blinks[dm_sub.type == 'dark'].mean, 2)}, SD = {np.round(dm_sub.n_blinks[dm_sub.type == 'dark'].std,2)}")

    s, p = wilcoxon(dm_sub.response_effort[dm_sub.type == 'light'], dm_sub.response_effort[dm_sub.type == 'dark'], nan_policy='omit')
    print(f"Effort: light != dark, S = {s:.0f}, p = {np.round(p,3)}")
    print(f"Bright: M = {np.round(dm_sub.response_effort[dm_sub.type == 'light'].mean, 2)}, SD = {np.round(dm_sub.response_effort[dm_sub.type == 'light'].std,2)}")
    print(f"Dark: M = {np.round(dm_sub.response_effort[dm_sub.type == 'dark'].mean, 2)}, SD = {np.round(dm_sub.response_effort[dm_sub.type == 'dark'].std,2)}")

    s, p = wilcoxon(dm_sub.emotional_intensity[dm_sub.type == 'light'], dm_sub.emotional_intensity[dm_sub.type == 'dark'], nan_policy='omit')
    print(f"Arousal: light != dark, S = {s:.0f}, p = {np.round(p,3)}")
    print(f"Bright: M = {np.round(dm_sub.emotional_intensity[dm_sub.type == 'light'].mean, 2)}, SD = {np.round(dm_sub.emotional_intensity[dm_sub.type == 'light'].std,2)}")
    print(f"Dark: M = {np.round(dm_sub.emotional_intensity[dm_sub.type == 'dark'].mean, 2)}, SD = {np.round(dm_sub.emotional_intensity[dm_sub.type == 'dark'].std,2)}")
    
    s, p = wilcoxon(dm_sub.mean_pupil[dm_sub.order == 'second'], dm_sub.mean_pupil[dm_sub.order == 'first'], nan_policy='omit')
    print(f"Presentation order: first != second, S = {s:.0f}, p = {np.round(p,2)}")
    print(f"First: M = {np.round(dm_sub.mean_pupil[dm_sub.order == 'first'].mean, 2)}, SD = {np.round(dm_sub.mean_pupil[dm_sub.order == 'first'].std,2)}")
    print(f"Second: M = {np.round(dm_sub.mean_pupil[dm_sub.order == 'second'].mean, 2)}, SD = {np.round(dm_sub.mean_pupil[dm_sub.order == 'second'].std,2)}")

    dm_sub = dm.suptype == suptype
    dm_sub = ops.group(dm_sub, by=[dm_sub.subject_nr, dm_sub.version]) 

    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_sub.column_names:
        if type(dm_sub[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_sub[col] = reduce(dm_sub[col]) # Compute the mean per subtype 
    
    # Print descriptives (experiment)
    for sup_, sdm in ops.split(dm_sub.suptype):
        print(f'\nEffort: M = {np.round(sdm.response_effort.mean,3)}, SD = {np.round(sdm.response_effort.std,3)}, n = {len(sdm)}')
        print(f'Emotional intensity: M = {np.round(sdm.emotional_intensity.mean,3)}, SD = {np.round(sdm.emotional_intensity.std,3)}, n = {len(sdm)}')
        print(f'Vividness: M = {np.round(sdm.response_vivid.mean,3)}, SD = {np.round(sdm.response_vivid.std,3)}, n = {len(sdm)}')
        print(f'Accuracy: {np.round(sdm.correct_answer.mean,2)}, SD = {np.round(sdm.correct_answer.std,2)}, n = {len(sdm)}\n')

    # print('Between participants:')
    # print(f"Version: (2 vs. 1)\nPupil size: {mannwhitneyu(dm_sub.mean_pupil[dm_sub.version==1], dm_sub.mean_pupil[dm_sub.version==2])}")
    # print(dm_sub.mean_pupil[dm_sub.version == 2].mean, dm_sub.mean_pupil[dm_sub.version == 1].mean)
    # print(dm_sub.mean_pupil[dm_sub.version == 2].std, dm_sub.mean_pupil[dm_sub.version == 1].std)
    
    # print(f"Vividness: {mannwhitneyu(dm_sub.response_vivid[dm_sub.version==2], dm_sub.response_vivid[dm_sub.version==1])}")
    # print(dm_sub.response_vivid[dm_sub.version == 2].mean, dm_sub.response_vivid[dm_sub.version == 1].mean)
    # print(dm_sub.response_vivid[dm_sub.version == 2].std, dm_sub.response_vivid[dm_sub.version == 1].std)

    # print(f"Effort: {mannwhitneyu(dm_sub.effort_changes[dm_sub.version==2], dm_sub.mean_effort[dm_sub.version==1])}")
    # print(dm_sub.mean_effort[dm_sub.version == 2].mean, dm_sub.mean_effort[dm_sub.version == 1].mean)
    # print(dm_sub.mean_effort[dm_sub.version == 2].std, dm_sub.mean_effort[dm_sub.version == 1].std)

    # print(f"Arousal: {mannwhitneyu(dm_sub.mean_emo[dm_sub.version==2], dm_sub.mean_emo[dm_sub.version==1])}")
    # print(dm_sub.mean_emo[dm_sub.version == 2].mean, dm_sub.mean_emo[dm_sub.version == 1].mean)
    # print(dm_sub.mean_emo[dm_sub.version == 2].std, dm_sub.mean_emo[dm_sub.version == 1].std)
    
    # Default back
    warnings.filterwarnings("default", category=FutureWarning)
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
def individual_profile(df_to_plot, subject_nr, suptype='rabbit'):
    """Plot individual profile to investigate."""
    df = df_to_plot[df_to_plot.suptype == suptype]
    df = df[df.subject_nr == subject_nr]
    # Descriptives 
    for type_ in ['rabbit', 'self']:
        plt.figure(figsize=(32,10))
        plt.suptitle(f"The {suptype} trials: Participant #{subject_nr} (version: {df.version.to_list()[0]})\nVVIQ: {df.VVIQ.to_list()[0]} | SUIS: {df.SUIS.to_list()[0]}\n Age: {df.age.to_list()[0]} | Sex: {df.sex.to_list()[0]}\n Pupil-diff score: {df.pupil_change.to_list()[0]}", fontsize=40)
        plt.subplot(1,6,1);plot_bars(df, x='type', y='mean_pupil', hue='type', color='black', xlab='Condition', order=['light', 'dark'], hue_order=['light', 'dark'], ylab='Mean pupil size (a.u.)', alpha=0.7, fig=False, pal=palette[0:2], legend=False);plt.title('Imagine')
        plt.subplot(1,6,2);plot_bars(df, x='type', y='mean_fixation', hue='type', color='black', xlab='Condition', order=['light', 'dark'], hue_order=['light', 'dark'], ylab='', alpha=0.7, fig=False, pal=palette[0:2], legend=False);plt.title('Fixation')
        plt.subplot(1,6,3);plot_bars(df, x='type', y='n_blinks', hue='type', color='black', xlab='Condition', order=['light', 'dark'], hue_order=['light', 'dark'], ylab='', alpha=0.7, fig=False, pal=palette[0:2], legend=False);plt.title('Blinks')
        plt.subplot(1,6,4);plot_bars(df, x='type', y='response_vivid', hue='type', color='black', xlab='Condition', order=['light', 'dark'], hue_order=['light', 'dark'], ylab='', alpha=0.7, fig=False, pal=palette[0:2], legend=False);plt.title('Vividness')
        plt.subplot(1,6,5);plot_bars(df, x='type', y='response_effort', hue='type', color='black', xlab='Condition', order=['light', 'dark'], hue_order=['light', 'dark'], ylab='', alpha=0.7, fig=False, pal=palette[0:2], legend=False);plt.title('Effort')
        plt.subplot(1,6,6);plot_bars(df, x='type', y='response_val', hue='type', color='black', xlab='Condition', order=['light', 'dark'], hue_order=['light', 'dark'], ylab='', alpha=0.7, fig=False, pal=palette[0:2], legend=False);plt.title('Emotions')
        plt.tight_layout()
        plt.show()

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
        