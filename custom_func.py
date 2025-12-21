# -*- coding: utf-8 -*-
"""
Created on Sat Mar  9 15:54:37 2024

@author: cvanb
"""
# =============================================================================
# Import libraries 
# =============================================================================
# Plots and visualisation
import matplotlib as mpl
import seaborn as sns
from matplotlib import pyplot as plt
from datamatrix import plot
from datamatrix.colors.tango import blue, green, gray
import pandas as pd
# Operations and calculation
import datamatrix
import numpy as np
import warnings
from eyelinkparser import parse, defaulttraceprocessor
from datamatrix import series as srs, NAN, operations as ops, convert
from datamatrix.multidimensional import reduce
# Statistics
import scipy.stats as stats
import statsmodels.api as sm
import pingouin as pg
from statsmodels.formula.api import mixedlm
from scipy.stats import spearmanr, shapiro, kstest, wilcoxon
from statsmodels.stats.diagnostic import het_white
from scipy.stats.distributions import chi2
from collections import Counter

# =============================================================================
# Create necessary variables
# =============================================================================
# Dictionary to recode ratings in numerical values (for English and Dutch versions)
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
                        
response_dict_nl = {'Helemaal geen beeld, ik “weet” alleen dat ik aan de scène dacht': 1,
                    'Vaag beeld': 2,
                    'Matig realistisch en levendig': 3,
                    'Realistisch en redelijk levendig': 4,
                    'Volkomen realistisch, zo levendig als echt zien': 5,
                    'Zeer negatieve': -3,
                    'Negatieve': -2,
                    'Licht negatieve': -1,
                    'Neutrale': 0,
                    'Licht positieve': 1,
                    'Positieve': 2,
                    'Zeer positieve': 3,
                    'Zeer weinig moeite': -2,
                    'Weinig moeite': -1,
                    'Noch weinig noch veel moeite': 0,
                    'Veel moeite': 1, 
                    'Zeer veel moeite': 2}

# Edit the font, font size, grid color, axes width, etc. style
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

palette = [green[1], blue[1], gray[2], gray[3]] # the main color palette for plots
warnings.filterwarnings("ignore", category=FutureWarning) # prevent this because it's annoying

# =============================================================================
# The main functions to be used
# =============================================================================
# Define useful functions
def get_raw(path, dataframe):

    dm = parse(
        maxtracelen=3700,
        traceprocessor=defaulttraceprocessor(
            blinkreconstruct=True,
            downsample=10,
            mode='advanced'),
        folder=path,
        pupil_size=True,
        gaze_pos=False,              # Don't store gaze-position information to save memory
        time_trace=False,            # Don't store absolute timestamps to save memory
        multiprocess=4,
        asc_encoding='cp1252'
        )
    
    print('Blink reconstruction and down sampling (100 Hz) have been applied.')
        
    # Create theoretical (in samples) and true (in seconds) duration columns 
    dm.duration, dm.dur = '', ''
    for lang, scene, sdm in ops.split(dm.response_lang, dm.scene):
        stim_dur_ = dataframe[dataframe.lang==lang]
        duration_theo = int(stim_dur_.samples[stim_dur_.scene==scene])  # in samples
        duration_true = int(stim_dur_.duration[stim_dur_.scene==scene]) # in seconds 

        dm.duration[sdm] = duration_theo
        dm.dur[sdm] = duration_true
    
    # The main pupil trace variable
    dm.pupil = dm.ptrace_target[:,:]

    # Create suptype
    dm.suptype = ''
    dm.suptype[dm.type == 'dark'] = 'dark'
    dm.suptype[dm.type == 'dark_light'] = 'dark'
    dm.suptype[dm.type == 'light'] = 'light'
    dm.suptype[dm.type == 'light_dark'] = 'light'
    
    # Recode answers
    cols = [dm.response_vivid, dm.response_effort, dm.response_val]
    for var in cols:
        for rep, sdm in ops.split(var):
            if dm.response_lang[sdm] == 'English':
                var[var == rep] = response_dict_en[rep]
            else:
                var[var == rep] = response_dict_nl[rep]
     
    # Assign new variable
        # Emotional intensity (absolute value of valency)
    dm.emotional_intensity = np.abs(dm.response_val)
    
    # Correct answer
    dm.correct_answer[dm.correct_answer=='yes'] = 1
    dm.correct_answer[dm.correct_answer=='no'] = 0
    dm.correct_answer = dm.correct_answer * 100 # as percentage
    
    return dm[dm.pupil, dm.ptrace_target, dm.ptrace_fixation, dm.subject_nr,
              dm.response_vivid, dm.response_effort, dm.response_val, 
              dm.correct_answer, dm.emotional_intensity, dm.suptype, dm.dur,
              dm.type, dm.subtype, dm.scene, dm.trialid, dm.duration, dm.stimID, 
              dm.response_lang, dm.blinkstlist_fixation, dm.blinkstlist_target,
              dm.fixylist_target, dm.fixxlist_target, dm.trace_length_target]

def create_control_variables(original_dm):
    """All the control variables."""
    # Create copy of dm to not overwrite it
    dm = original_dm.subject_nr != ''
    
    # Create supcategory
    dm.category = ''
    for s, sdm in ops.split(dm.subtype):
        if s == 'dynamic':
            dm.category[sdm] = 'Dynamic'
        else:
            dm.category[sdm] = 'Non-dynamic'
    
    # Effort, valence, etc.
    dm.vivid_changes, dm.val_changes, dm.emo_changes, dm.effort_changes = NAN, NAN, NAN, NAN
    for p, s, sdm in ops.split(dm.subject_nr, dm.subtype):
        # Rating differences
        dm.vivid_changes[sdm] = sdm.response_vivid[sdm.suptype =='dark'].mean - sdm.response_vivid[sdm.suptype =='light'].mean
        dm.effort_changes[sdm] = sdm.response_effort[sdm.suptype =='dark'].mean - sdm.response_effort[sdm.suptype =='light'].mean
        dm.val_changes[sdm] = sdm.response_val[sdm.suptype =='dark'].mean - sdm.response_val[sdm.suptype =='light'].mean
        dm.emo_changes[sdm] = sdm.emotional_intensity[sdm.suptype =='dark'].mean - sdm.emotional_intensity[sdm.suptype =='light'].mean
    
    # Mean of changes that can affect pupil size
    dm.changes = (dm.effort_changes + dm.emo_changes)/2

    # Keep track of trialid / presentation order
    dm.order_changes = NAN
    for p, s, sdm in ops.split(dm.subject_nr, dm.subtype):
        if len(sdm) > 1: # make sure that there are indeed 2 trials to compare
            n0_dark = sdm.trialid[sdm.suptype == 'dark'][0]
            n0_bright = sdm.trialid[sdm.suptype == 'light'][0]
            
            # the rank difference between the two
            dm.order_changes[sdm] = n0_bright - n0_dark
            # if dark is before bright, then diff is positive, and should have positive pupil differences
    
    # Compute the percentage of trials that are in an order that would facilitate 
    # having smaller pupil sizes for bright stories per participant
    dm.numbers = NAN
    for s, sdm in ops.split(dm.subject_nr):
        if len(sdm) > 0:
            dm.numbers[sdm] = len(sdm[sdm.order_changes > 0]) / len(sdm) * 100
    
    for s, sdm in ops.split(dm.category):
        if s == 'Non-dynamic':
            sdm = sdm.pupil_change != NAN
        else:
            sdm = sdm.slope_change != NAN
        print(f'{s}: {len(sdm[sdm.order_changes > 0])/len(sdm) * 100} % of trial pairs in the dark->bright order (n = {len(sdm)})')

    return dm

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
    print('Datamatrix and dataframe match in terms of participants: {list(np.sort(df.Q00ID)) == list(np.sort(new))}')

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
        elif sdm.VVIQ == 2:
            dm.aphantasia[sdm] = 'Mild'
        elif sdm.VVIQ < 2:
            dm.aphantasia[sdm] = 'Yes'
    print(f'{len(dm.subject_nr[dm.aphantasia == "Yes"].unique)} participant with VVIQ < 2 {dm.subject_nr[dm.aphantasia == "Yes"].unique}.')
    
    # Print descriptives
    print(f'Vividness: M = {np.round(dm.mean_vivid.mean,3)}, SD = {np.round(dm.mean_vivid.std,3)}, n = {len(dm.subject_nr.unique)}.')

    for type_, sdm in ops.split(dm.type):
        print('\n', type_)    
        print(f'On average, participants reported that imagining the {type_} stories necessitated "... effort" (M = {np.round(sdm.response_effort.mean,3)}, SD = {np.round(sdm.response_effort.std,3)}, n = {len(sdm)}),')
        print(f'induced "... emotions" (M = {np.round(sdm.emotional_intensity.mean,3)}, SD = {np.round(sdm.emotional_intensity.std,3)}, n = {len(sdm)})')
        print(f'and were imagined as "..." (M = {np.round(sdm.response_vivid.mean,3)}, SD = {np.round(sdm.response_vivid.std,3)}, n = {len(sdm)}).')
        print(f'The mean accuracy for this brightness condition was {np.round(sdm.correct_answer.mean,2)} (SD = {np.round(sdm.correct_answer.std,2)}, n = {len(sdm)}).')

    print('Mean accuracy %:', np.round(dm.correct_answer.mean, 2), np.round(dm.correct_answer.std, 2))
    print(Counter(Counter(dm.subject_nr[dm.correct_answer == 0]).values()))
    print('Language:', len(dm.subject_nr[dm.response_lang=='English'].unique), len(dm.subject_nr[dm.response_lang=='Dutch'].unique))

    # Print descriptives
    for type_, sdm in ops.split(dm.subtype):
        print('\n', type_)    
        print(f'On average, participants reported that imagining the {type_} stories necessitated "... effort" (M = {np.round(sdm.response_effort.mean,3)}, SD = {np.round(sdm.response_effort.std,3)}, n = {len(sdm)}),')
        print(f'induced "... emotions" (M = {np.round(sdm.emotional_intensity.mean,3)}, SD = {np.round(sdm.emotional_intensity.std,3)}, n = {len(sdm)})')
        print(f'and were imagined as "..." (M = {np.round(sdm.response_vivid.mean,3)}, SD = {np.round(sdm.response_vivid.std,3)}, n = {len(sdm)}).')
        print(f'The mean accuracy for this brightness condition was {np.round(sdm.correct_answer.mean,2)} (SD = {np.round(sdm.correct_answer.std,2)}, n = {len(sdm)}).')

    # Descriptives
    print(df[['VVIQ', 'SUIS']].describe())
    print(df[['Q01AGE', 'Q02LANG', 'Q03LANG', 'Q05EYE', 'Q06EAR']].describe())
    print(df[['Q02LANG']].describe())
    print(df.groupby('Q04SEX').Q01AGE.describe())
    
    return dm

def plot_dist_scores(dm_to_plot):
    """Plot distributions on VVIQ, SUIS, vivid, effort and valence."""
    # Mean vviq scores
    dm_individual = ops.group(dm_to_plot, by=dm_to_plot.subject_nr)
    dm_individual.VVIQ = reduce(dm_individual.VVIQ)
    dm_individual.SUIS = reduce(dm_individual.SUIS)
    dm_individual.response_vivid = reduce(dm_individual.response_vivid)
    dm_individual.response_effort = reduce(dm_individual.response_effort)
    dm_individual.response_val = reduce(dm_individual.response_val)

    plt.figure(figsize=(25,10))
    plt.subplot(1,2,1)
    sns.distplot(dm_individual.VVIQ, label='VVIQ', color='green');plt.xlim([1,5])
    sns.distplot(dm_individual.SUIS, label='SUIS', color='violet');plt.xlim([1,5])
    plt.xlabel('Mean individual scores');plt.ylabel('Density');plt.ylim([0,1.5])
    plt.legend(ncol=1, title='Questionnaires', frameon=False, loc='upper left')
    plt.subplot(1,2,2)
    sns.distplot(dm_individual.response_vivid, label='Vividness', color='red')
    sns.distplot(dm_individual.response_effort, label='Effort', color='blue')
    sns.distplot(dm_individual.response_val, label='Valence', color='orange')
    plt.xlim([-3,5]);plt.xticks(range(-3,6,1), range(-3,6,1));plt.ylim([0,1.5])
    plt.xlabel('Mean individual scores');plt.ylabel('Density')
    plt.legend(ncol=1, title='Ratings', frameon=False, loc='upper left')
    plt.tight_layout()
    plt.show()
    
def count_nonnan(a):
    return np.sum(~np.isnan(a))

def check_blinks(dm_blinks, new=False):
    """Check number of blinks per condition and per participant."""
    if new==True:
        dm_blinks.n_blinks=''
        for p, s, sdm in ops.split(dm_blinks.subject_nr, dm_blinks.stimID):
            dm_blinks.n_blinks[sdm] = (reduce(sdm.blinkstlist_target, count_nonnan) / (sdm.trace_length_target / 1000)) * 60
        
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
    if new == True:
        plt.title('Before preprocessing')
    else:
        plt.title('After preprocessing')
    plt.xticks(fontsize=20)
    plt.tight_layout()
    plt.show()
    
    # Check fixations 
    plt.figure(figsize=(15,8));plt.suptitle('Gaze / eye position', fontsize=35)
    plt.subplot(1,2,1);plt.title('Bright stories', fontsize=30)
    x = np.array(dm_blinks.fixxlist_target[dm_blinks.suptype=='light'])
    y = np.array(dm_blinks.fixylist_target[dm_blinks.suptype=='light'])
    x = x.flatten()
    y = y.flatten()
    plt.hexbin(x, y, gridsize=25)
    plt.axhline(500, color='white');plt.axvline(500, color='white')
    plt.yticks(fontsize=25);plt.xticks(fontsize=25);plt.xlabel('x', fontsize=25);plt.ylabel('y', fontsize=25)
    plt.subplot(1,2,2);plt.title('Dark stories', fontsize=30)
    x = np.array(dm_blinks.fixxlist_target[dm_blinks.suptype=='dark'])
    y = np.array(dm_blinks.fixylist_target[dm_blinks.suptype=='dark'])
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
    
    # How many errors?
    error_participants = dm_.subject_nr[dm_.correct_answer == 0]
    errors_list = Counter(error_participants)
    print(f'For participants who had at least one error, how many did they make: {errors_list}')
    print(f'How many participants had n number of errors: {Counter(errors_list.values())}')
    
    # Preprocess fixation phase
    dm_.ptrace_fixation.depth = 100 # 1s total duration
    dm_.ptrace_fixation = srs.trim(dm_.ptrace_fixation, value=NAN, end=True, start=True)

    # Exclude participants with an abusive number of blinks (+ 2 SD)
    print(f'Before trial exclusion (blinks): N = {len(dm_)} trials.')
    dm_.z_blinks = ops.z(dm_.n_blinks) # z-transform
    thresh = 2 # above 2 SD

    plt.figure(figsize=(13,8));warnings.filterwarnings("ignore", category=UserWarning)  
    sns.distplot(dm_.z_blinks);plt.axvline(thresh)
    plt.xlabel('Blink rate per minute (z-scored)')
    plt.tight_layout()
    plt.show();warnings.filterwarnings("default", category=UserWarning)  

    blinky = list(dm_.subject_nr[dm_.z_blinks > thresh])
    print(f'{len(blinky)} trials ({len(set(blinky))} participants) with a lot of blinks ({dm_.n_blinks[dm_.z_blinks > thresh].unique}; M = {dm_.n_blinks.mean}, STD = {dm_.n_blinks.std})')
    print(blinky)
    
    dm_ = dm_.z_blinks <= thresh # exclude those above
    print(f'After trial exclusion (blinks): N = {len(dm_)} trials.')

    # Trim tails
    dm_.ptrace_target = srs.trim(dm_.ptrace_target, value=NAN, end=True, start=True)
    
    # Check number of nan values per trial
    dm_.nonnan_pupil = ''
    for p, t, sdm in ops.split(dm_.subject_nr, dm_.stimID):
        dm_.nonnan_pupil[sdm] = count_nonnan(sdm.ptrace_target[:, 0:int(sdm.duration.unique[0])]) / int(sdm.duration.unique[0]) * 100 # percentage of non-nan values
    
    plt.figure(figsize=(13,8));warnings.filterwarnings("ignore", category=UserWarning)  
    sns.distplot(dm_.nonnan_pupil)
    plt.xlabel('Percentage of valid samples per trial')
    plt.tight_layout()
    plt.show();warnings.filterwarnings("default", category=UserWarning)  

    print(f'{len(dm_[dm_.nonnan_pupil < 50])} trials with more than 50% of NAN values ({len(dm_.subject_nr[dm_.nonnan_pupil < 50].unique)} participants).')
    
    dm_ = dm_.nonnan_pupil >= 50 # keep relevant trials
    
    # Interpolate the first and last 200 milliseconds of each trace (to not have unreconstructed blinks
    # taken into account when doing the baseline correction and prevent edge effects)
    dm_.ptrace_target[:, 0:20] = srs.interpolate(srs.concatenate(dm_.ptrace_fixation[:,80:100], dm_.ptrace_target[:,0:20]))[:, 20:40]
    dm_.ptrace_target[:, 3380:3420] = srs.interpolate(dm_.ptrace_target[:, 3380:3420])
    print('Pupil size traces linearly interpolated from 0 to 200 ms.')
       
    # Smooth the traces to reduce the jitter
    dm_.pupil = srs.smooth(dm_.ptrace_target, 5) 
    print('Pupil size traces smoothed with a Hanning window of size 51.')
    
    # Exclude trials with unrealistic mean pupil-size (outliers) 
    print(f'Before trial exclusion (pupil size): {len(dm_)} trials.')
    dm_.z_pupil = NAN 
    for p, sdm in ops.split(dm_.subject_nr):
        dm_.z_pupil[sdm] = ops.z(reduce(sdm.pupil, np.nanmean))
    
    plt.figure(figsize=(13,8));warnings.filterwarnings("ignore", category=UserWarning)  
    sns.distplot(dm_.z_pupil);plt.axvline(-2);plt.axvline(2)
    plt.xlabel('Raw pupil-size means (z-scored)')
    plt.tight_layout()
    plt.show();warnings.filterwarnings("default", category=UserWarning)  
    
    print(dm_.subject_nr[dm_.z_pupil == NAN], dm_.subject_nr[dm_.z_pupil > 2.0], dm_.subject_nr[dm_.z_pupil < -2])
    dm_ = dm_.z_pupil != NAN 
    dm_ = dm_.z_pupil <= 2.0 
    dm_ = dm_.z_pupil >= -2.0 
    print(f'After trial exclusion (pupil size): {len(dm_)} trials.')
        
    # Baseline correction on the first 50 ms with the subtractive method
    bl_start, bl_end = 0, 5 # baseline duration (samples)
    #dm_.baseline = dm_.ptrace_fixation[:, 95:100]
    for p, sdm in ops.split(dm_.subject_nr):
        dm_.pupil[sdm] = srs.baseline(sdm.pupil, sdm.pupil, bl_start, bl_end)
    print('Baseline-correction applied on the first 50 ms after story onset.')
    
    # Compute the mean pupil size and mean slopes during listening
    dm_.mean_pupil, dm_.slope_pupil = NAN, NAN
    for p, trial, sdm in ops.split(dm_.subject_nr, dm_.trialid):
        # Take only the real duration of the trace minus 200 ms to compute the mean and slopes
        # to prevent taking into account edge effects 
        duration = int(bl_end + int(sdm.duration.unique[0]) - 20)
        pupil = sdm.pupil[:, bl_end:duration] # don't take the first 50 ms neither (baseline)
        
        if count_nonnan(pupil) > 0:
            time = np.arange(bl_end, duration, 1)
            pupil_ip = np.array(srs.interpolate(pupil))[0] # interpolate because it's going to have to fit a linear regression anyway
            
            if sdm.subtype.unique[0] != 'dynamic':
                # Pupil-size mean
                mean = reduce(pupil, operation=np.nanmean)
                dm_.mean_pupil[sdm] = mean
            else:
                # Pupil-size slopes 
                #slope, intercept, r_value, p_value, std_err = stats.linregress(time, pupil_ip)     
                slope, intercept = np.polyfit(time, pupil_ip, 1) # gives the same slopes as above 
                dm_.slope_pupil[sdm] = np.round(slope, 2)
            
    print('Mean pupil size and slopes were computed over the whole listening phase for each trial.')

    dm_ = ops.sort(dm_, by=dm_.subtype) # sort the datamatrix by subtype

    # Create new variables 
    dm_.pupil_change, dm_.slope_change = NAN, NAN
    dm_.mean_vivid, dm_.mean_effort, dm_.mean_emo, dm_.mean_val = NAN, NAN, NAN, NAN
    for p, s, sdm in ops.split(dm_.subject_nr, dm_.subtype):
        if s != 'dynamic':
            # Compute pupil-size changes as differences in mean pupil size between dark - bright conditions
            pupil_change = sdm.mean_pupil[sdm.type =='dark'].mean - sdm.mean_pupil[sdm.type =='light'].mean
            dm_.pupil_change[sdm] = pupil_change
        else:
            # Compute pupil-slope changes as differences in mean pupil slopes between bright to dak dark - dark to bright conditions
            slope_change = sdm.slope_pupil[sdm.type =='light_dark'].mean - sdm.slope_pupil[sdm.type =='dark_light'].mean
            dm_.slope_change[sdm] = slope_change
                    
        # Mean ratings per subtype (averaged across dark and bright stories)
        dm_.mean_vivid[sdm] = sdm.response_vivid.mean
        dm_.mean_effort[sdm] = sdm.response_effort.mean
        dm_.mean_val[sdm] = sdm.response_val.mean
        dm_.mean_emo[sdm] = sdm.emotional_intensity.mean

    # How many trials left per participant?
    print(f'Number of trials left per participant: {np.sort(Counter(dm_.subject_nr), axis=None)}')
    print(f'How many participants have n number of trials: {Counter(Counter(dm_.subject_nr).values())}')
    
    to_exclude = [i for i in dict(Counter(dm_.subject_nr)).keys() if dict(Counter(dm_.subject_nr))[i] < 4]
    
    print(f'{to_exclude} has less than 50% of remaining trials.')
    dm_ = dm_.subject_nr != set(to_exclude)
    print(f'After preprocessing: N = {len(dm_.subject_nr.unique)} (n = {len(dm_)} trials)')

    return dm_

# Plot traces
def plot_traces(dm, plot_type='non-dynamic', bl=False, slopes=True, cwd=str):
    """Plot the pupil traces.
        plot_type: either 'by-participant', 'by-subtype', 'non-dynamic' or 'dynamic.
        bl: if bl is True, plots only the first 200 ms of the pupil traces for the individual plots.'"""
    if plot_type == 'by-participant':
        # Plot individual traces 
        plt.figure(figsize=(45,25))
        i=1;plt.suptitle(f'N = {len(dm.subject_nr.unique)}; {len(dm)} trials')
        for s, sdm in ops.split(dm.subject_nr):
            plt.subplot(7,5,i)        
            plt.title(s, color='black')
            sdm_light = sdm.suptype == 'light'
            sdm_dark = sdm.suptype == 'dark'
            if len(sdm_light) > 0:
                if bl==True:
                    plt.plot(sdm_light.pupil[:,0:20].plottable, color=green[1])
                else:
                    plt.plot(sdm_light.pupil.plottable, color=green[1])
            if len(sdm_dark) > 0:
                if bl==True:
                    plt.plot(sdm_dark.pupil[:,0:20].plottable, color=blue[1])
                else:
                    plt.plot(sdm_dark.pupil.plottable, color=blue[1])
            i+=1;plt.xticks(fontsize=15);plt.yticks(fontsize=15)
        legend=None
        plt.tight_layout()
        plt.savefig(cwd+f"/figs/{plot_type}_mainplot.png", bbox_inches='tight')
        plt.show()
    
    if plot_type == 'non-dynamic':
        # Pupil traces per type (non-dynamic)
        fig = plt.figure(figsize=(35,10))
        plot.trace(dm.pupil[dm.type=='light'], color=green[1], label=f"Bright (N={len(dm.pupil[dm.type=='light'])})")
        plot.trace(dm.pupil[dm.type=='dark'], color=blue[1], label=f"Dark (N={len(dm.pupil[dm.type=='dark'])})")
        #fig.text(s='A\n', x=0.05, y=0.8, fontsize=75)
        plt.ylabel('Baseline-corrected\npupil size (a.u.)', fontsize=45)
        plt.xlim([0,3400])
        legend = plt.legend(loc='lower left', frameon=False, fontsize=40)
        for legobj in legend.legend_handles:
            legobj.set_linewidth(6)
        plt.xticks(range(0,3500,100), range(0,35), fontsize=22)
        plt.xlabel('Time since story onset (s)', fontsize=45)
        plt.savefig(cwd+f"/figs/{plot_type}_mainplot.png", bbox_inches='tight')
        plt.show()
    
    if plot_type == 'by-subtype':
        # Pupil traces per type and per subtype (non-dynamic)
        for s, sdm in ops.split(dm.subtype[dm.subtype != 'dynamic']):
            plt.figure(figsize=(35,10))
            duration = int(sdm.duration.min)
            plt.title(f'{s}')
            plot.trace(sdm.pupil[sdm.type=='dark'], color=blue[1], label=f"Dark (N={len(sdm.pupil[sdm.type=='dark'])})")
            plot.trace(sdm.pupil[sdm.type=='light'], color=green[1], label=f"Bright (N={len(sdm.pupil[sdm.type=='light'])})")
            #plt.axhline(0, color='black');
            plt.ylabel('Baseline-corrected\npupil size (a.u.)')
            plt.xlim([0,duration])
            # for dur in dm_.duration[dm_.subtype != 'dynamic'].unique:
            #     plt.axvline(dur)
            legend = plt.legend(loc='lower left', frameon=False, fontsize=40)
            for legobj in legend.legend_handles:
                legobj.set_linewidth(6)
            plt.xticks(range(0,duration+100,100), range(0,int(duration/100+1)), fontsize=20);plt.xlabel('Time since story onset (s)', fontsize=40)
            plt.savefig(cwd+f"/figs/{plot_type}_mainplot.png", bbox_inches='tight')
            plt.show()
    
    if plot_type == 'dynamic':
        # Pupil traces per type (dynamic)
        fig = plt.figure(figsize=(35,10))
        dm_light_dark = dm.type =='light_dark'
        dm_dark_light = dm.type =='dark_light'
        plt.subplot(1,1,1)
        plot.trace(srs.interpolate(dm_light_dark.pupil), color=green[1], label=f"Bright to dark (N={len(dm_light_dark)})")
        plot.trace(srs.interpolate(dm_dark_light.pupil), color=blue[1], label=f"Dark to bright (N={len(dm_dark_light)})")
        if slopes == True:
            slope_dark_light = dm_dark_light.slope_pupil.mean
            slope_light_dark = dm_light_dark.slope_pupil.mean
            plt.plot(np.arange(0, 3200+500, 500), np.arange(0, 3200+500, 500) * slope_light_dark, color=green[1], linestyle='dotted', linewidth=8)
            plt.plot(np.arange(0, 3200+500, 500), np.arange(0, 3200+500, 500)* slope_dark_light, color=blue[1], linestyle='dotted', linewidth=8)
        #fig.text(s='A\n', x=0.05, y=0.8, fontsize=75)
        plt.xlim([0,3100])
        plt.ylabel('Baseline-corrected\npupil size (a.u.)', fontsize=45)
        legend = plt.legend(loc='lower left', frameon=False, fontsize=40)
        for legobj in legend.legend_handles:
            legobj.set_linewidth(6)
        plt.xticks(range(0,3200,100), range(0,32), fontsize=22)
        plt.xlabel('Time since story onset (s)', fontsize=45)
        plt.savefig(cwd+f"/figs/{plot_type}_mainplot.png", bbox_inches='tight')
        plt.show()
    
    return legend

# Plot dist
def plot_dist(sub=str, h = None, palette=list, data=None, title='', leg_loc=list):
    if sub=='non-dynamic':
        labs = ['Bright', 'Dark']
        data = data.subtype != 'dynamic'
    elif sub=='dynamic':
        labs = ['Bright to dark', 'Dark to bright']
        data = data.subtype == 'dynamic'
        
    data = convert.to_pandas(data)

    fig = plt.figure(figsize=(28,16))
    fig.suptitle(title, fontsize=60)
    plt.subplot(2,2,1)
    sns.histplot(data=data, x=data.response_vivid, hue='type', stat='count', fill=False, palette=palette, legend=False, binwidth=0.3, linewidth=6)
    plt.xlabel('Trial-by-trial vividness ratings');plt.ylabel('Number of trials')

    plt.subplot(2,2,2)
    sns.histplot(data=data, x=data.response_effort, hue='type', stat='count', fill=False, palette=palette, legend=False, binwidth=0.3, linewidth=6)
    plt.xlabel('Trial-by-trial effort ratings');plt.ylabel('Number of trials')
    
    plt.subplot(2,2,3)
    sns.histplot(data=data, x=data.response_val, hue='type', stat='count', fill=False, palette=palette, legend=False, binwidth=0.3, linewidth=6)
    plt.xticks(range(-3, 4), range(-3, 4))
    plt.xlabel('Trial-by-trial emotional valence ratings');plt.ylabel('Number of trials')
    
    plt.subplot(2,2,4)
    sns.histplot(data=data, x=data.emotional_intensity, hue='type', stat='count', fill=False, palette=palette, legend=False, binwidth=0.3, linewidth=6)
    plt.xlabel('Trial-by-trial emotional intensity ratings');plt.ylabel('Number of trials')
    plt.xticks(range(0, 4), range(0, 4))

    fig.legend(handles=h, labels=labs, frameon=False, bbox_to_anchor = leg_loc, ncols=2, fontsize=50)
    plt.tight_layout()
    plt.show()

def plot_bars(dm, x=str, y=str, hue=None, hue_order=['light', 'dark', 'light_dark', 'dark_light'], order=None, color=None, pal='deep', xlab='Condition', ylab='Mean Pupil Size (a.u.)', title='', fig=True, alpha=1, legend=True):
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
    
# Stats
def lm_pupil(dm_tst, formula=False, re_formula="1 + type", pupil_change=False, reml=False, add_cat=False):
    """Test how brightness (dark vs. light) affects the mean pupil size."""    
    # Make sure to filter the 'dynamic' condition
    dm_test = dm_tst.subtype != 'dynamic'    
    
    # Rename 
    dm_test.subtype[dm_test.subtype == 'happy'] = 'birthday'
    
    # Remove Nans
    dm_valid_data = dm_test.mean_pupil != NAN # remove NaNs 
    
    if pupil_change == True:
        # Suppress warnings because it's annoying
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        if add_cat == 'order_type':
            # Add categorical variables in the list 'by = [ ]', to test them (e.g., by = [..., dm_valid_data.order_type])
            dm_valid_data = ops.group(dm_valid_data, by=[dm_valid_data.subject_nr, dm_valid_data.subtype, dm_valid_data.order_type]) 
        else:
            dm_valid_data = ops.group(dm_valid_data, by=[dm_valid_data.subject_nr, dm_valid_data.subtype]) 

        # Make sure to have only unique mean values for each variable per participant 
        for col in dm_valid_data.column_names:
            if type(dm_valid_data[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
                dm_valid_data[col] = reduce(dm_valid_data[col]) # Compute the mean per subtype 
        
        dm_valid_data.pupil_change = np.round(dm_valid_data.pupil_change, 10) # this is necessary to meet the assumption criterias
            # it does not change the final results (p and z values) 
            
        # Unable back the warnings
        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)
        warnings.filterwarnings("default", category=UserWarning)  

    dm_valid_data = dm_valid_data.pupil_change != NAN # make sure there's always at least 2 stories to compare

    # The model
    md = mixedlm(formula, dm_valid_data, 
                     groups='subject_nr',
                     re_formula=re_formula)
    
    mdf = md.fit(reml=reml, method='Powell')
        
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


def lm_slope(dm_tst, formula='', re_formula='1 + suptype', slope_change=False, reml=False, method=['Powell']):
    """Test how brightness (dark to light vs. light to dark) affects the pupil size slope."""
    # Make sure that it's only the dynamic condition
    dm_tst = dm_tst.subtype == 'dynamic'
    
    # Valid data
    dm_valid_data = dm_tst.slope_pupil != NAN # remove NaNs
    
    if slope_change == True:
        # Suppress warnings because it's annoying
        warnings.filterwarnings("ignore", category=FutureWarning)
        warnings.filterwarnings("ignore", category=RuntimeWarning)
        warnings.filterwarnings("ignore", category=UserWarning)
        
        dm_valid_data = ops.group(dm_valid_data, by=[dm_valid_data.subject_nr]) # add dm_sub.response_lang if necessary
        
        # Make sure to have only unique mean values for each variable per participant 
        for col in dm_valid_data.column_names:
            if type(dm_valid_data[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
                dm_valid_data[col] = reduce(dm_valid_data[col]) # Compute the mean 
                
        # Unable back the warnings
        warnings.filterwarnings("default", category=FutureWarning)
        warnings.filterwarnings("default", category=RuntimeWarning)
        warnings.filterwarnings("default", category=UserWarning)
        
    dm_valid_data = dm_valid_data.slope_change != NAN # make sure there's always at least 2 stories to compare
    
    md = mixedlm(formula, dm_valid_data, 
                      groups='subject_nr',
                      re_formula=re_formula)
    
    mdf = md.fit(reml=reml, method='Powell')
    
    return mdf

def check_assumptions(model, sup_title=''):
    """Check assumptions for normality of residuals and homoescedasticity.
    Code from: https://www.pythonfordatascience.org/mixed-effects-regression-python/#assumption_check"""
    plt.rcParams['font.size'] = 40
    print('Assumptions check:')
    fig = plt.figure(figsize = (25, 16))
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    fig.suptitle(f'{sup_title}\n{model.model.formula} (n = {model.model.n_groups})')
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


def test_correlation(dm_c, x, y, alt='two-sided', pcorr=1, color='red', lab='VVIQ', plot_=True, fig=False, fs=30):
    """Test the correlations between pupil measures and questionnaire measures using Spearman's correlation."""
    # Suppress warnings because it's annoying
    warnings.filterwarnings("ignore", category=RuntimeWarning)
    warnings.filterwarnings("ignore", category=UserWarning)
    
    # Exclude nan pupil-changes (slopes and means) to be consistent with the MLM analyses
    dm_1 = dm_c.pupil_change != NAN
    dm_2 = dm_c.slope_change != NAN
    dm_c = dm_1 << dm_2
    
    # Group per participant 
    dm_cor = ops.group(dm_c, by=[dm_c.subject_nr])
    
    # Make sure to have only unique mean values for each variable per participant 
    for col in dm_cor.column_names:
        if type(dm_cor[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
            dm_cor[col] = reduce(dm_cor[col], operation=np.nanmean)
    
    # The variables to test the correlation
    x_, y_ = dm_cor[x], dm_cor[y]
    
    # Unable back the warnings
    warnings.filterwarnings("default", category=RuntimeWarning)
    warnings.filterwarnings("default", category=UserWarning)
    
    # Compute spearman's rank correlation
    cor=spearmanr(x_, y_, alternative=alt)

    N = len(dm_cor)
    pval = cor.pvalue * pcorr # Apply Bonferroni correction (multiply p-values by the number of tests)
    if pval > 1.0:
        pval = 1.0
    if pval < 0.001:
        pval = '{:.1e}'.format(pval)
    else:
        pval = np.round(pval, 3)
        

    if plot_ == False:
        res = fr'{chr(961)} = {round(cor.correlation, 3)}, p = {pval}, n = {N}'
    else:
        res = fr'{chr(961)} = {round(cor.correlation, 3)}, p = {pval}'
    print(res)
    
    # Plot the correlations (linear regression model fit)
    if lab != False:
        label = fr'{lab}: {res}'
    else:
        label = ''
    
    plt.rcParams['font.size'] = fs

    if plot_ == True:
        if fig == True:
            plt.figure(figsize=(12,10))
            plt.xlabel(x_.name);plt.ylabel(y_.name)
            label=None
            plt.title(res);plt.xticks(range(int(min(x_))-1, int(max(x_))+2))
        sns.regplot(data=dm_cor, x=x_.name, y=y_.name, lowess=False, color=color, label=label, x_jitter=None, y_jitter=None, scatter_kws={'alpha': 0.5, 's': 100}, robust=True)
        plt.legend(frameon=False, markerscale=3, loc='upper center')
        # use statsmodels to estimate a nonparametric lowess model (locally weighted linear regression)
        sns.regplot(data=dm_cor, x=x_.name, y=y_.name, lowess=True, color=color, label=None, x_jitter=None, y_jitter=None, scatter_kws={'alpha': 0.0}, line_kws={'linestyle': 'dashed', 'alpha':0.6, 'linewidth': 8})
        if fig == True:
            plt.tight_layout()
            plt.show()
    return res

def plot_correlations(dm, what='all'):
    """Visualisation of the correlations."""
    if what == 'by-subtype':
        plt.rcParams['font.size'] = 35
        # Correlations between by-trial ratings and pupil measures
        plt.figure(figsize=(60,20))
        plt.subplot(1,4,1)
        dm_sub = dm.subtype == 'happy' # Only Birthday
        dm_sub = dm_sub.pupil_change != NAN # Exclude NAN values
        #test_correlation(dm_sub, x='pupil_change', y='mean_vivid', alt='greater', lab='Stories', color='black')
        test_correlation(dm_sub, y='VVIQ', x='pupil_change', alt='greater', color='green', lab='VVIQ')
        test_correlation(dm_sub, y='SUIS', x='pupil_change', alt='greater', color='violet', lab='SUIS')
        plt.ylabel('Subjective ratings');plt.xlabel('Pupil-size mean differences (a.u.)');
        plt.title("Birthday party", fontsize=35)
        plt.ylim([0.9, 6]);plt.yticks(np.arange(1, 6, 1))
        
        plt.subplot(1,4,2)
        dm_sub = dm.subtype == 'lotr' # Only Lord of the rings
        dm_sub = dm_sub.pupil_change != NAN # Exclude NAN values
        #test_correlation(dm_sub, x='pupil_change', y='mean_vivid', alt='greater', lab='Stories', color='black')
        test_correlation(dm_sub, y='VVIQ', x='pupil_change', alt='greater', color='green', lab='VVIQ')
        test_correlation(dm_sub, y='SUIS', x='pupil_change', alt='greater', color='violet', lab='SUIS')
        plt.ylabel('Subjective ratings');plt.xlabel('Pupil-size mean differences (a.u.)')
        plt.title("Lord of the rings", fontsize=35)
        plt.ylim([0.9, 6]);plt.yticks(np.arange(1, 6, 1))
        
        plt.subplot(1,4,3)
        dm_sub = dm.subtype == 'neutral' # Only Neutral
        dm_sub = dm_sub.pupil_change != NAN # Exclude NAN values
        #test_correlation(dm_sub, x='pupil_change', y='mean_vivid', alt='greater', lab='Stories', color='black')
        test_correlation(dm_sub, y='VVIQ', x='pupil_change', alt='greater', color='green', lab='VVIQ')
        test_correlation(dm_sub, y='SUIS', x='pupil_change', alt='greater', color='violet', lab='SUIS')
        plt.ylabel('Subjective ratings');plt.xlabel('Pupil-size mean differences (a.u.)')
        plt.title("Neutral", fontsize=35)
        plt.ylim([0.9, 6]);plt.yticks(np.arange(1, 6, 1))
        
        plt.subplot(1,4,4)
        dm_sub = dm.subtype == 'dynamic' # Only the Dynamic subtype
        dm_sub = dm_sub.slope_change != NAN # Exclude NAN values
        #test_correlation(dm_sub, x='slope_change', y='mean_vivid', alt='greater', lab='Stories', color='black')
        test_correlation(dm_sub, y='VVIQ', x='slope_change', alt='greater', color='green', lab='VVIQ')
        test_correlation(dm_sub, y='SUIS', x='slope_change', alt='greater', color='violet', lab='SUIS')
        plt.ylabel('Subjective ratings');plt.xlabel('Pupil-size slope differences (a.u.)')
        plt.title("Dynamic", fontsize=35)
        plt.ylim([0.9, 6]);plt.yticks(np.arange(1, 6, 1))
        plt.tight_layout()
        plt.show()
    else:
        plt.rcParams['font.size'] = 40
        
        fig, axes = plt.subplots(1,3, figsize=(40, 15), sharey=False) 
            # Correlations between questionnaires and trial-by-trial vividness ratings
        plt.subplot(1,3,1) 
        test_correlation(dm, y='VVIQ', x='mean_vivid', alt='greater', color='green')
        test_correlation(dm, y='SUIS', x='mean_vivid', alt='greater', color='violet', lab='SUIS')
        plt.ylabel('Mean questionnaire scores');plt.xlabel('Mean trial-by-trial vividness ratings')
        plt.ylim([0.9, 6]);plt.yticks(np.arange(1, 6, 1))#;plt.xlim([1.9, 5]);plt.xticks(np.arange(2, 6, 1))
                # Correlations between pupil means and questionnaires
        dm_sub = dm.subtype != {'dynamic'}
        dm_cor1 = dm_sub.pupil_change != NAN # Exclude NAN values
        plt.subplot(1,3,2)
        test_correlation(dm_cor1, x='pupil_change', y='VVIQ', alt='greater', color='green')
        test_correlation(dm_cor1, x='pupil_change', y='SUIS', alt='greater', color='violet', lab='SUIS')
        #test_correlation(dm_cor1, x='pupil_change', y='mean_vivid', alt='greater', color='black', lab='Stories')
        plt.ylabel('');plt.xlabel('Pupil-size mean differences (a.u.)')
        plt.ylim([0.9, 6]);plt.yticks([])
                # Correlations between pupil slopes and questionnaires
        dm_sub = dm.subtype == 'dynamic'
        dm_cor2 = dm_sub.slope_change != NAN # Exclude NAN values
        plt.subplot(1,3,3)
        test_correlation(dm_cor2, x='slope_change', y='VVIQ', alt='greater', color='green', lab='VVIQ')
        test_correlation(dm_cor2, x='slope_change', y='SUIS', alt='greater', color='violet', lab='SUIS')
        #test_correlation(dm_cor2, x='slope_change', y='mean_vivid', alt='greater', color='black', lab='Stories')
        plt.ylabel('');plt.xlabel('Pupil-size slope differences (a.u.)')
        plt.ylim([0.9, 6]);plt.yticks([])
        plt.grid(visible=True)
        plt.tight_layout()
        plt.show()

# def save_to_csv(dm, outputfolder, name, by='subject_nr'):
#     """Save datamatrix to csv file in a table."""
    
#     if by == 'subject_nr':
#         dm_test = ops.group(dm, by=[dm.subject_nr])
#     if by == 'subtype':
#         dm_test = ops.group(dm, by=[dm.subject_nr, dm.subtype])
#     if by == 'type':
#         dm_test = ops.group(dm, by=[dm.subject_nr, dm.subtype, dm.type])
#     if by == 'all':
#         dm_test = dm.subtype != NAN

#     # Make sure to have only unique mean values for each variable per participant 
#     for col in dm_test.column_names:
#         if type(dm_test[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
#             if dm_test[col].ndim > 1:
#                 dm_test[col] = reduce(dm_test[col], operation=np.nanmean)
#     dm_df = convert.to_pandas(dm_test)
#     dm_df.to_csv(outputfolder+name+'.csv', encoding='utf-8', index=False)
    
#     print(f'Saved to csv file in {outputfolder} as {name+".csv"}')

def dist_checks(dm_to_check, leg):
    """Similar distributions between conditions?"""
    # Kolmogorov-Smirnov test: Are the samples from the same distribution? (H0)
    for cat in ['non-dynamic', 'dynamic']:
        if cat == 'non-dynamic':
            dm_sub = dm_to_check.pupil_change != NAN
            dm_sub = dm_sub.subtype != 'dynamic'
        else:
            dm_sub = dm_to_check.slope_change != NAN
            dm_sub = dm_sub.subtype == 'dynamic'
        sdm_light, sdm_dark  = ops.split(dm_sub.suptype, 'light', 'dark')    
        for dms in zip([sdm_light], [sdm_dark]):
            print(f"{dms[0].type.unique} (n = {len(dms[0])}) vs. {dms[1].type.unique} (n = {len(dms[1])})")
            print(f'Blinks: {kstest(dms[0].n_blinks, dms[1].n_blinks)}')
            print(f'Vividness: {kstest(dms[0].response_vivid, dms[1].response_vivid)}')
            print(f'Effort: {kstest(dms[0].response_effort, dms[1].response_effort)}')
            print(f'Valence: {kstest(dms[0].response_val, dms[1].response_val)}')
            print(f'Emotional intensity: {kstest(dms[0].emotional_intensity, dms[1].emotional_intensity)}')
            
            # Visualize
    plot_dist(sub='non-dynamic', h = leg.legend_handles, palette=[green[1], blue[1]], data=dm_to_check, title='Non-dynamic stories\n', leg_loc=[0.67, 0.94])
    plot_dist(sub='dynamic', h = leg.legend_handles, palette=[green[1], blue[1]], data=dm_to_check, title='Dynamic stories\n', leg_loc=[0.75, 0.94])

def main_plots(dm_to_plot, which, cwd):
    """The main plots to illustrate the results."""
    # Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
    dm_df = convert.to_pandas(dm_to_plot)
    plt.rcParams['font.size'] = 35

    # Slope pupil-size and pupil slope differences for dynamic
    if which == 'dynamic':
        dm_sub0 = dm_df[dm_df.subtype == 'dynamic']
        dm_sub = dm_sub0[dm_sub0.slope_pupil != '']
        dm_sub = dm_sub[dm_sub.slope_change != '']
        fig, axes = plt.subplots(1, 4, figsize=(48,12))
        fig.subplots_adjust(wspace=0.3)
        ax=plt.subplot(1,4,1)
        plot_bars(dm_sub, x='type', y='slope_pupil', hue=None, order=['light_dark', 'dark_light'], hue_order=None, ylab='Pupil-size slopes (a.u.)', pal=palette[0:2], xlab='Condition', title=None, fig=False, alpha=0.7)
        #plt.text(s='B', x=-1.0, y=0.020, fontsize=65)
        plt.xticks(ticks=range(0,2), labels=['Bright to dark', 'Dark to bright'])
        ax=plt.subplot(1,4,2)
        plot_bars(dm_sub, x='response_vivid', y='slope_pupil', hue='type', hue_order=['light_dark', 'dark_light'], ylab='Pupil-size slopes (a.u.)', pal=palette[0:2], xlab='Trial-by-trial vividness ratings', title=None, fig=False, alpha=0.7)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=handles, labels=['Bright to dark', 'Dark to bright'], frameon=False, loc='lower right')
        #plt.text(s='C', x=-1.6, y=0.40, fontsize=65)
        ax=plt.subplot(1,4,3)
        plot_bars(dm_sub, x='mean_vivid', y='slope_change', ylab='Pupil-size slope differences (a.u.)', pal='crest', xlab='Mean vividness ratings', title=None, fig=False)
        #plt.text(s='D', x=-2.5, y=0.80, fontsize=65)
        ax=plt.subplot(1,4,4)
        #plt.text(s='E', x=-1, y=7.60, fontsize=65)
        dm_cor = dm_to_plot.subtype == 'dynamic'
        dm_cor = dm_cor.slope_change != NAN
        test_correlation(dm_cor, y='VVIQ', x='slope_change', alt='greater', lab='VVIQ', color='green', fs=35)
        test_correlation(dm_cor, y='SUIS', x='slope_change', alt='greater', lab='SUIS', color='violet', fs=35)
        plt.xlabel('Pupil-size slope differences');plt.ylabel('Mean questionnaire scores');plt.ylim([0.9, 7]);plt.yticks(range(1, 6))
        plt.savefig(cwd+f"/figs/{which}_mainbarplot.png", bbox_inches='tight')
        plt.show()
        
    # Mean pupil-size and pupil-size differences for all non-dynamic subtypes
    elif which == 'non-dynamic':
        dm_sub = dm_df[dm_df.subtype != 'dynamic']
        fig, axes = plt.subplots(1, 4, figsize=(48,12))
        fig.subplots_adjust(wspace=0.3)
        ax=plt.subplot(1,4,1)
        plot_bars(dm_sub, x='type', y='mean_pupil', hue=None, order=['light', 'dark'], hue_order=None, ylab='Pupil-size means (a.u.)', pal=palette[0:2], xlab='Condition', title=None, fig=False, alpha=0.7)
        #plt.text(s='B', x=-1.0, y=95, fontsize=65)
        plt.xticks(ticks=range(0,2), labels=['Bright', 'Dark'])
        ax=plt.subplot(1,4,2)
        plot_bars(dm_sub, x='response_vivid', y='mean_pupil', hue='type', hue_order=['light', 'dark'], ylab='Pupil-size means (a.u.)', pal=palette[0:2], xlab='Trial-by-trial vividness ratings', title=None, fig=False, alpha=0.7)
        handles, labels = ax.get_legend_handles_labels()
        plt.legend(handles=handles, labels=['Bright', 'Dark'], frameon=False, loc='lower right')
        #plt.text(s='C', x=-1.6, y=850, fontsize=65)
        ax=plt.subplot(1,4,3)
        plot_bars(dm_sub, x='mean_vivid', y='pupil_change', ylab='Pupil-size mean differences (a.u.)', pal='crest', xlab='Mean vividness ratings', title=None, fig=False)
        #plt.text(s='D', x=-2.5, y=500, fontsize=65)
        ax=plt.subplot(1,4,4)
        #plt.text(s='E', x=-1200, y=7.15, fontsize=65)
        dm_cor = dm_to_plot.subtype != 'dynamic'
        dm_cor = dm_cor.pupil_change != NAN
        test_correlation(dm_cor, y='VVIQ', x='pupil_change', alt='greater', lab='VVIQ', color='green', fs=35)
        test_correlation(dm_cor, y='SUIS', x='pupil_change', alt='greater',lab='SUIS', color='violet', fs=35)
        plt.xlabel('Pupil-size mean differences');plt.ylabel('Mean questionnaire scores');plt.ylim([0.9, 7]);plt.yticks(range(1, 6))
        plt.savefig(cwd+f"/figs/{which}_mainbarplot.png", bbox_inches='tight')
        plt.show()
        
        
    # Mean pupil-size and pupil-size differences per subtypes
    elif which == 'subtypes':
        fig = plt.figure(figsize=(32,10));i=1
        titles = ['Birthday Party', 'Lord of the Rings', 'Neutral']
        for subtype, sdm in ops.split(dm_to_plot.subtype[dm_to_plot.subtype!='dynamic']):
            ax=plt.subplot(1,3,i)
            sdm_df = convert.to_pandas(sdm)
            sdm_df = sdm_df[sdm_df.mean_pupil != '']
            sdm_df = sdm_df[sdm_df.pupil_change != '']
            ax.set_ylim([-1000, 1000]);ax.set_yticks(range(-1000,1500,500), range(-1000,1500,500))
            plot_bars(sdm_df, x='mean_vivid', hue='mean_vivid', y='pupil_change', pal='crest', xlab='', title=None, fig=False, ylab='', legend=False)
            if i>1:
                ax.set_yticks([], [])
            plt.ylabel('');plt.xlabel('')
            plt.title(f'{titles[i-1]}', fontsize=45)#;plt.xticks(range(0,5), range(1,6))
            i+=1
        fig.supxlabel('Mean vividness ratings (dark and bright) per subtype and per participant', ha='center', fontsize=50)
        fig.supylabel('Pupil-size mean differences\n(a.u.) (Dark - Bright)', ha='center', fontsize=50)
        plt.tight_layout()
        plt.savefig(cwd+f"/figs/{which}_mainbarplot.png", bbox_inches='tight')
        plt.show()

def supp_plots(dm_to_plot, what):
    """Lots of plots."""
    # Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
    dm_df = convert.to_pandas(dm_to_plot)
    
    # Order of presentation
    if what == 'order':
        # On mean pupil sizes
        fig, axes = plt.subplots(2, 1, figsize=(18,15))
        dm_sub1 = dm_df[dm_df.category != 'Dynamic']
        dm_sub1 = dm_sub1[dm_sub1.pupil_change != '']
        dm_sub1 = dm_sub1[dm_sub1.mean_pupil != '']
        dm_sub1.numbers = np.round(dm_sub1.numbers,2)
        ax1=plt.subplot(2,1,1)
        plot_bars(dm_sub1, x='numbers', y='mean_pupil', hue='type', order=None, hue_order=['light', 'dark'], pal=palette[0:2], fig=False, alpha=0.7)
        handles, labels = ax1.get_legend_handles_labels()
        plt.xlabel('Percentage of trials in the dark->bright order');plt.ylabel('Pupil-size means (a.u.)', color='black')
        plt.legend(handles=handles, labels=['Bright', 'Dark'], frameon=False)
        
        plt.subplot(2,1,2)
        plot_bars(dm_df, x='subtype', y='order_changes', pal='hls', xlab='Story subtype', ylab='Rank differences (Bright - Dark)', fig=False)
        plt.tight_layout()
        plt.show()
        
            # Doesn't really make sense to look for this in the dynamic stories,
            # but if you are curious:
        # fig, axes = plt.subplots(1, 1, figsize=(15,10))
        # ax2=plt.subplot(1,1,1)
        # dm_sub2 = dm_df[dm_df.category == 'Dynamic']
        # dm_sub2 = dm_sub2[dm_sub2.slope_pupil != '']
        # dm_sub2 = dm_sub2[dm_sub2.slope_change != '']
        # dm_sub2.numbers = np.round(dm_sub2.numbers,2)
        # plot_bars(dm_sub2, x='numbers', y='slope_pupil', hue='type', order=None, hue_order=None, ylab='Mean Pupil-size Slopes (a.u.)', pal=palette[0:2], fig=False, alpha=0.7)
        # handles, labels = ax2.get_legend_handles_labels()
        # plt.xlabel('Percentage of trials in the dark-bright -> bright-dark order');plt.ylabel('Pupil-size slopes (normalized)',color='black')
        # plt.legend(handles=handles, labels=['Bright to dark', 'Dark to bright'], frameon=False, loc='lower left')
        # fig.supxlabel('Condition order')
        # plt.tight_layout()
        # plt.show()
    
    # Other effects on mean pupil size and slopes (controls)
    if what == 'language':   # Language
        dm_sub = dm_df[dm_df.mean_pupil != '']
        dm_sub = dm_sub[dm_sub.pupil_change != '']
        plot_bars(dm_sub, x='response_lang', y='mean_pupil', hue = 'type', xlab='Language', pal=palette, hue_order=['light', 'dark'], alpha=0.7)
        dm_sub = dm_df[dm_df.slope_pupil != '']
        dm_sub = dm_sub[dm_sub.slope_change != '']
        plot_bars(dm_sub, x='response_lang', y='slope_pupil', hue = 'type', xlab='Language', ylab='Mean Pupil Slopes (a.u.)', pal=palette, hue_order=['light_dark', 'dark_light'], alpha=0.7)
    
    if what == 'emotional_intensity': # Emotional intensity (absolute value of valency)
        dm_sub = dm_df[dm_df.subtype != 'dynamic']
        dm_sub = dm_sub[dm_sub.pupil_change != '']
        plot_bars(dm_sub, x='mean_emo', y='pupil_change', pal='flare', xlab='Mean emotional intensity ratings', ylab='Pupil-size mean differences\n(a.u.)', title='All non-dynamic conditions')

        dm_sub = dm_df[dm_df.subtype == 'dynamic']
        dm_sub = dm_sub[dm_sub.slope_change != '']
        plot_bars(dm_sub, x='mean_emo', y='slope_change', ylab='Pupil-size slope differences (a.u.)', pal='flare', xlab='Mean emotional intensity ratings', title='Dynamic condition')

        fig = plt.figure(figsize=(32,10));i=1 # Emotional intensity ratings per subtype
        for subtype, sdm in ops.split(dm_to_plot.subtype[dm_to_plot.subtype!='dynamic']):
            ax=plt.subplot(1,3,i)
            sdm_df = convert.to_pandas(sdm)
            sdm_df = sdm_df[sdm_df.pupil_change != '']
            titles = ['Birthday Party', 'Lord of the Rings', 'Neutral']
            ax.set_ylim([-1200, 1000]);ax.set_yticks(range(-1000,1500,500), range(-1000,1500,500))
            plot_bars(sdm_df, x='mean_emo', hue='mean_emo', y='pupil_change', pal='flare', xlab='', title=None, fig=False, ylab='', legend=False)
            if i>1:
                ax.set_yticks([], [])
            plt.title(f'{titles[i-1]}\n', fontsize=40)#;plt.xticks(range(0,5), range(1,6))
            i+=1
        fig.supxlabel('Mean emotional intensity (dark and bright) per subtype and per participant', ha='center', fontsize=50)
        fig.supylabel('Pupil-size mean differences\n(a.u.) (Dark - Bright)', ha='center', fontsize=40)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(25,10)) # descriptives
        plt.subplot(1,3,1);plot_bars(dm_df, x='subtype', y='emotional_intensity', hue='suptype', hue_order=['light', 'dark'], pal=palette[0:2], xlab='Story subtype', ylab='Trial-by-trial emotional intensity', alpha=0.7, fig=False);plt.ylim([0,3])
        plt.subplot(1,3,2);plot_bars(dm_df, x='subtype', y='mean_emo', pal='hls', xlab='Story subtype', ylab='Mean emotional intensity ratings', fig=False);plt.ylim([0,3])
        plt.subplot(1,3,3);plot_bars(dm_df, x='subtype', y='emo_changes', pal='hls', xlab='Story subtype', ylab='Emotional intensity ratings\ndifferences (Dark - Bright)', fig=False);plt.ylim([-2,2])
        plt.tight_layout()
        plt.show()
    
    if what == 'effort':     # Mental effort
        dm_sub = dm_df[dm_df.subtype != 'dynamic']
        dm_sub = dm_sub[dm_sub.pupil_change != '']
        plot_bars(dm_sub, x='mean_effort', y='pupil_change', pal='YlOrBr', xlab='Mean effort ratings', ylab='Pupil-size mean differences\n(a.u.)', title='All non-dynamic conditions')
    
        dm_sub = dm_df[dm_df.subtype == 'dynamic']
        dm_sub = dm_sub[dm_sub.slope_change != '']
        plot_bars(dm_sub, x='mean_effort', y='slope_change', ylab='Mean Slope Changes (a.u.)', pal='YlOrBr', xlab='Mean effort ratings', title='Dynamic condition')
    
        fig = plt.figure(figsize=(32,10));i=1         # Effort ratings per subtype
        for subtype, sdm in ops.split(dm_to_plot.subtype[dm_to_plot.subtype!='dynamic']):
            ax=plt.subplot(1,3,i)
            sdm_df = convert.to_pandas(sdm)
            sdm_df = sdm_df[sdm_df.pupil_change != '']
            titles = ['Birthday Party', 'Lord of the Rings', 'Neutral']
            ax.set_ylim([-1200, 1000]);ax.set_yticks(range(-1000,1500,500), range(-1000,1500,500))
            plot_bars(sdm_df, x='mean_effort', hue='mean_effort', y='pupil_change', pal='YlOrBr', xlab='', title=None, fig=False, ylab='', legend=False)
            if i>1:
                ax.set_yticks([], [])
            plt.title(f'{titles[i-1]}\n', fontsize=40)#;plt.xticks(range(0,5), range(1,6))
            i+=1
        fig.supxlabel('Mean effort ratings (dark and bright) per subtype and per participant', ha='center', fontsize=50)
        fig.supylabel('Pupil-size mean differences\n(a.u.) (Dark - Bright)', ha='center', fontsize=40)
        plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(25,10)) # descriptives
        plt.subplot(1,3,1);plot_bars(dm_df, x='subtype', y='response_effort', hue='suptype', hue_order=['light', 'dark'], pal=palette[0:2], xlab = 'Subtype', ylab='Trial-by-trial effort ratings', alpha=0.7, fig=False);plt.ylim([-2,2])
        plt.subplot(1,3,2);plot_bars(dm_df, x='subtype', y='mean_effort', pal='hls', xlab = 'Subtype', ylab='Mean effort ratings', fig=False);plt.ylim([-2,2])
        plt.subplot(1,3,3);plot_bars(dm_df, x='subtype', y='effort_changes', pal='hls', xlab='Story subtype', ylab='Effort ratings differences\n(Dark - Bright)', fig=False);plt.ylim([-2,2])
        plt.tight_layout()
        plt.show()
        
    if what == 'other': # duration, correct answer %, vividness..
        plt.figure(figsize=(18,10))
        plt.subplot(1,2,1);plot_bars(dm_df, x='subtype', y='dur', hue='suptype', pal=palette[0:2], hue_order=None, xlab='Story Subtype', ylab='Mean story duration (s)', order=['happy', 'lotr', 'neutral', 'dynamic'], alpha=0.7, fig=False)#;plt.ylim([1500,4000])
        plt.subplot(1,2,2);plot_bars(dm_df, x='subtype', y='correct_answer', hue='suptype', pal=palette, xlab='Story Subtype', ylab='Mean accurary %', order=['happy', 'lotr', 'neutral', 'dynamic'], hue_order=['light', 'dark'], alpha=0.7, fig=False);plt.ylim([50,110]);plt.yticks(range(50,110,10))
        plt.axhline(100, linestyle='dashed');plt.tight_layout()
        plt.show()
        
        plt.figure(figsize=(25,10))
        plt.subplot(1,3,1);plot_bars(dm_df, x='subtype', y='response_vivid', hue='suptype', hue_order=['light', 'dark'], pal=palette[0:2], xlab='Subtype', ylab='Trial-by-trial vividness ratings', alpha=0.7, fig=False);plt.ylim([1,5])
        plt.subplot(1,3,2);plot_bars(dm_df, x='subtype', y='mean_vivid', pal='hls', xlab='Subtype', ylab='Mean vividness ratings', order=['happy', 'lotr', 'neutral', 'dynamic'], fig=False);plt.ylim([1,5])
        plt.subplot(1,3,3);plot_bars(dm_df, x='subtype', y='vivid_changes', pal='hls', xlab='Story subtype', ylab='Vividness ratings differences\n(Dark - Bright)', fig=False);plt.ylim([-2,2])
        plt.tight_layout()
        plt.show()
    
    if what == 'aphantasia':
        plt.figure(figsize=(30,10))
        plt.subplot(1,3,1);plot_bars(dm_df, x='type', y='response_val', hue='aphantasia', hue_order=None, pal=['red', 'black'], xlab='Condition', ylab='Emotional Valence', alpha=0.5, fig=False);plt.ylim([-3,3])
        plt.subplot(1,3,2);plot_bars(dm_df, x='type', y='response_effort', hue='aphantasia', hue_order=None, pal=['red', 'black'], xlab='Condition', ylab='Effort ratings', alpha=0.5, fig=False);plt.ylim([-2,2])
        plt.subplot(1,3,3);plot_bars(dm_df, x='type', y='response_vivid', hue='aphantasia', hue_order=None, pal=['red', 'black'], xlab='Condition', ylab='Vividness ratings', alpha=0.5, fig=False);plt.ylim([1,5])
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(30,20))
        dm_sub = dm_df[dm_df.pupil_change != '']
        plt.subplot(2,2,1);plot_bars(dm_sub, x='type', y='mean_effort', order=['light', 'dark'], hue='aphantasia', hue_order=None, pal=['red', 'black'], xlab='Condition', ylab='Mean effort ratings', alpha=0.5, fig=False);plt.ylim([-2,2])
        plt.subplot(2,2,2);plot_bars(dm_sub, x='type', y='mean_vivid', order=['light', 'dark'], hue='aphantasia', hue_order=None, pal=['red', 'black'], xlab='Condition', ylab='Mean vividness ratings', alpha=0.5, fig=False);plt.ylim([1,5])
        plt.subplot(2,2,3);plot_bars(dm_sub, x='type', y='mean_emo', order=['light', 'dark'], hue='aphantasia', hue_order=None, pal=['red', 'black'], xlab='Condition', ylab='Mean emotional intensity ratings', alpha=0.5, fig=False);plt.ylim([-3,3])
        plt.subplot(2,2,4);plot_bars(dm_sub, x='aphantasia', y='pupil_change', hue_order=None, pal=['red', 'black'], xlab='Aphantasia (mean VVIQ score < 2)', ylab='Pupil-size mean differences\n (Dark - Bright) (a.u.)', alpha=0.5, fig=False)
        plt.tight_layout()
        plt.show()

        plt.figure(figsize=(20,10))
        sdm = dm_sub[dm_sub.subtype != 'dynamic']
        plt.subplot(1,2,1);plot_bars(sdm, x='subtype', y='pupil_change', hue='aphantasia', hue_order=None, pal=['red', 'black'], xlab='Aphantasia (VVIQ < 2)', ylab='Pupil-size mean differences\n (Dark - Bright) (a.u.)', alpha=0.5, fig=False)
        sdm = dm_df[dm_df.subtype == 'dynamic'] 
        sdm = sdm[sdm.slope_change != '']
        plt.subplot(1,2,2);plot_bars(sdm, x='subtype', y='slope_change', hue='aphantasia', hue_order=None, pal=['red', 'black'], xlab='Aphantasia (VVIQ < 2)', ylab='Pupil-size slope differences (a.u.)\n (Bright to dark - Bright to dark)', alpha=0.5, fig=False)
        plt.tight_layout()
        plt.show()

    if what == 'effort_vivid':
        fig, axes = plt.subplots(1, 3, figsize=(32,10), sharey=True)
        i=1
        for subtype, sdm in ops.split(dm_to_plot.subtype[dm_to_plot.subtype!='dynamic']):
            ax=plt.subplot(1,3,i)
            sdm_df = convert.to_pandas(sdm)
            sdm_df = sdm_df[sdm_df.mean_pupil != '']
            sdm_df = sdm_df[sdm_df.pupil_change != '']
            titles = ['Birthday Party', 'Lord of the Rings', 'Neutral']
            #ax.set_ylim([-1000, 1000]);ax.set_yticks(range(-1000,1500,500), range(-1000,1500,500))
            plot_bars(sdm_df, x='response_vivid', hue='type', y='response_effort', hue_order=None, pal=palette[0:2], xlab='', title=None, fig=False, ylab='', legend=True)
            if i>1:
                ax.set_yticks([], [])
            plt.ylabel('');plt.xlabel('')
            plt.title(f'{titles[i-1]}', fontsize=45)#;plt.xticks(range(0,5), range(1,6))
            i+=1
        fig.supxlabel('Trial-by-trial vividness ratings per subtype and per participant', ha='center', fontsize=50)
        fig.supylabel('Trial-by-trial effort ratings per\nsubtype and per participant', ha='center', fontsize=50)
        plt.tight_layout()
        plt.show()
        
    if what == 'emo_vivid':
        fig, axes = plt.subplots(1, 3, figsize=(32,10), sharey=True)
        i=1
        for subtype, sdm in ops.split(dm_to_plot.subtype[dm_to_plot.subtype!='dynamic']):
            ax=plt.subplot(1,3,i)
            sdm_df = convert.to_pandas(sdm)
            sdm_df = sdm_df[sdm_df.mean_pupil != '']
            sdm_df = sdm_df[sdm_df.pupil_change != '']
            titles = ['Birthday Party', 'Lord of the Rings', 'Neutral']
            #ax.set_ylim([-1000, 1000]);ax.set_yticks(range(-1000,1500,500), range(-1000,1500,500))
            plot_bars(sdm_df, x='response_vivid', hue='type', y='response_val', hue_order=None, pal=palette[0:2], xlab='', title=None, fig=False, ylab='', legend=True)
            if i>1:
                ax.set_yticks([], [])
            plt.ylabel('');plt.xlabel('')
            plt.title(f'{titles[i-1]}', fontsize=45)#;plt.xticks(range(0,5), range(1,6))
            i+=1
        fig.supxlabel('Trial-by-trial vividness ratings per subtype and per participant', ha='center', fontsize=50)
        fig.supylabel('Trial-by-trial valency ratings per\nsubtype and per participant', ha='center', fontsize=50)
        plt.tight_layout()
        plt.show()
        
    if what == 'changes':
        fig, axes = plt.subplots(1, 3, figsize=(32,10), sharey=True)
        i=1
        for subtype, sdm in ops.split(dm_to_plot.subtype[dm_to_plot.subtype!='dynamic']):
            ax=plt.subplot(1,3,i)
            sdm_df = convert.to_pandas(sdm)
            sdm_df = sdm_df[sdm_df.mean_pupil != '']
            sdm_df = sdm_df[sdm_df.pupil_change != '']
            titles = ['Birthday Party', 'Lord of the Rings', 'Neutral']
            #ax.set_ylim([-1000, 1000]);ax.set_yticks(range(-1000,1500,500), range(-1000,1500,500))
            plot_bars(sdm_df, x='mean_vivid', hue=None, y='changes', hue_order=None, pal='flare', xlab='', title=None, fig=False, ylab='', legend=True)
            if i>1:
                ax.set_yticks([], [])
            plt.ylabel('');plt.xlabel('')
            plt.title(f'{titles[i-1]}', fontsize=45)#;plt.xticks(range(0,5), range(1,6))
            i+=1
        fig.supxlabel('Mean vividness ratings (dark and bright) per subtype and per participant', ha='center', fontsize=50)
        fig.supylabel('Trial-by-trial valency ratings per\nsubtype and per participant', ha='center', fontsize=50)
        plt.tight_layout()
        plt.show()
    
    if what == 'time':
        fig, axes = plt.subplots(1, 1, figsize=(12,8), sharey=True)
        plot_bars(dm_df, x='trialid', y='mean_pupil', hue=None, hue_order=None, pal='flare', xlab='Trial presentation rank', ylab='Mean pupil size (a.u.)', alpha=0.5, fig=False)
        plt.tight_layout()
        plt.show()

def create_img(dm):
    """Test."""
    subdm = dm.subtype != ''
    # Plot individual traces 
    for s, t, sdm in ops.split(subdm.subject_nr, subdm.subtype):
        fig = plt.figure(figsize=(8,5))
        vivid = int(np.round(sdm.mean_vivid.unique[0]))
        sdm_light = sdm.suptype == 'light'
        sdm_dark = sdm.suptype == 'dark'
        if (len(sdm_light) > 0 and len(sdm_dark) > 0):
            plt.plot(sdm_light.pupil.plottable, color=green[1], linewidth=5)
            plt.plot(sdm_dark.pupil.plottable, color=blue[1], linewidth=5)
        plt.xticks([]);plt.yticks([]);plt.tight_layout()
        fig.savefig(f'img/{vivid}/S{s}_{t}_{vivid}.png')
        plt.close()

def dist_checks_wilcox(dm_to_check):
    """Non-parametric paired t-tests with Wilcoxon signed-rank test."""
    # Exclude nan pupil-changes (slopes and means) to be consistent with the MLM analyses
    dm_1 = dm_to_check.pupil_change != NAN
    dm_2 = dm_to_check.slope_change != NAN
    dm = dm_1 << dm_2
    
    for cat, sdm_ctrl in ops.split(dm.category):
        if cat == 'Non-dynamic':
            dm_sub = sdm_ctrl.pupil_change != NAN
            dm_sub = ops.group(dm_sub, by=[dm_sub.subject_nr, dm_sub.suptype]) 

            # Make sure to have only unique mean values for each variable per participant 
            for col in dm_sub.column_names:
                if type(dm_sub[col]) != datamatrix._datamatrix._mixedcolumn.MixedColumn:
                    dm_sub[col] = reduce(dm_sub[col]) # Compute the mean per subtype 
        else:
            dm_sub = sdm_ctrl.slope_change != NAN
        
        print(f"\n{cat} (n = {len(dm_sub[dm_sub.suptype == 'light'])}) (light vs. dark)")
        print(f"Vividness: M = {np.round(dm_sub.mean_vivid[dm_sub.suptype == 'light'].mean,3)}, SD = {np.round(dm_sub.mean_vivid[dm_sub.suptype == 'light'].std,3)}, n = {len(dm_sub[dm_sub.suptype == 'light'])}.")
        
        s, p = wilcoxon(dm_sub.n_blinks[dm_sub.suptype == 'light'], dm_sub.n_blinks[dm_sub.suptype == 'dark'])
        print(f"Blinks: light != dark, S = {s:.0f}, p = {np.round(p,2)}")
        print(f"Bright: M = {np.round(dm_sub.n_blinks[dm_sub.suptype == 'light'].mean, 2)}, SD = {np.round(dm_sub.n_blinks[dm_sub.suptype == 'light'].std,2)}")
        print(f"Dark: M = {np.round(dm_sub.n_blinks[dm_sub.suptype == 'dark'].mean, 2)}, SD = {np.round(dm_sub.n_blinks[dm_sub.suptype == 'dark'].std,2)}")

        s, p = wilcoxon(dm_sub.response_effort[dm_sub.suptype == 'light'], dm_sub.response_effort[dm_sub.suptype == 'dark'])
        print(f"Effort: light != dark, S = {s:.0f}, p = {np.round(p,2)}")
        print(f"Bright: M = {np.round(dm_sub.response_effort[dm_sub.suptype == 'light'].mean, 2)}, SD = {np.round(dm_sub.response_effort[dm_sub.suptype == 'light'].std,2)}")
        print(f"Dark: M = {np.round(dm_sub.response_effort[dm_sub.suptype == 'dark'].mean, 2)}, SD = {np.round(dm_sub.response_effort[dm_sub.suptype == 'dark'].std,2)}")

        s, p = wilcoxon(dm_sub.emotional_intensity[dm_sub.suptype == 'light'], dm_sub.emotional_intensity[dm_sub.suptype == 'dark'])
        print(f"Arousal: light != dark, S = {s:.0f}, p = {np.round(p,2)}")
        print(f"Bright: M = {np.round(dm_sub.emotional_intensity[dm_sub.suptype == 'light'].mean, 2)}, SD = {np.round(dm_sub.emotional_intensity[dm_sub.suptype == 'light'].std,2)}")
        print(f"Dark: M = {np.round(dm_sub.emotional_intensity[dm_sub.suptype == 'dark'].mean, 2)}, SD = {np.round(dm_sub.emotional_intensity[dm_sub.suptype == 'dark'].std,2)}")

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
        