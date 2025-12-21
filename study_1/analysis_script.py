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
import warnings
import datamatrix
from matplotlib import pyplot as plt
from datamatrix import NAN, io, convert, operations as ops, series as srs, FloatColumn
from datamatrix.colors.tango import blue, gray, green
import numpy as np
import seaborn as sns
import glob
from custom_func import (compute_vars, quick_visualisation,
                         lm_pupil, compare_models, print_results,
                         test_correlation, check_assumptions)
from statsmodels.formula.api import mixedlm
import time_series_test as tst
import pandas as pd 
from collections import Counter
from scipy.stats import spearmanr, shapiro, wilcoxon, mannwhitneyu, ttest_rel

# Define useful variables
cwd=os.getcwd() # auto
cwd='D:/data_experiments/visual_imagery_pupil/study_1' # manual
datafolder=cwd+'/csvdata/' # the folder where the EDF files are
#questfile='/results-survey.csv' # the questionnaire data

np.random.seed(123)  # Fix random seed for predictable outcomes if any random stuff happening
palette = [green[1], blue[1], gray[2], gray[3]]

def count_nonnan(a):
    return np.sum(~np.isnan(a))
# ================================================================================
#%% Load data
# =================================================================================
# Path to csv files
files = glob.glob(datafolder + "*.csv")

# Load the CSV file into a DataMatrix
warnings.filterwarnings("ignore", category=UserWarning)  
dm0 = io.readtxt(files[1], default_col_type=FloatColumn) # continuous
warnings.filterwarnings("default", category=UserWarning) 
 
dm1 = io.readtxt(files[1]) # continuous
dm2 = io.readtxt(files[6]) # discrete (only pupil traces by target words)
# Note: dm2 contains the pupil traces after smoothing, interpolating, trial exclusion and baseline correction

# ================================================================================
#%% Preprocess data
# =================================================================================
dm_1 = dm1[dm1.Subject, dm1.Story, dm1.Paragraph, dm1.Brightness, dm1.Timestamp]
dm_1.Pupil = dm0.RPupil # only the right eye, unprocessed

dm_1 = ops.group(dm_1, by=[dm_1.Subject, dm_1.Story, dm_1.Paragraph, dm_1.Brightness])
print('Datamatrix grouped by Subject, Brightness, Story and Paragraph.')

warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", category=RuntimeWarning) 
plt.figure(figsize=(13,8)) 
sns.distplot(dm_1.Pupil)
plt.xlabel('Raw pupil size')
plt.tight_layout()
plt.show()
warnings.filterwarnings("default", category=UserWarning)  
warnings.filterwarnings("default", category=RuntimeWarning) 

# How many Subjects, Stories, Paragraphs?
print(f'n = {len(dm_1.Subject.unique)} participants, {len(dm_1.Paragraph.unique)} paragraphs and {len(dm_1.Story.unique)} stories ({len(dm_1.Brightness.unique)} Brightness levels).')

# Get duration
dm_1 = compute_vars(dm_1, means=False)

# Blink reconstruction (parameters are (hopefully) adapted for 60 Hz)
dm_1.Pupil = srs.blinkreconstruct(
    dm_1.Pupil,
    vt=3,
    vt_start=2,
    vt_end=1,
    maxdur=60,
    margin=2,
    smooth_winlen=1, # First smoothing, during blink reconstruction
    std_thr=3,
    gap_margin=3,
    gap_vt=2,
    mode='advanced')

# Check number of nan values per trial
dm_1.nonnan_pupil = ''
for p, t, sdm in ops.split(dm_1.Subject, dm_1.Paragraph):
    dm_1.nonnan_pupil[sdm] = count_nonnan(sdm.Pupil[:, 0:sdm.samples[0]-1]) / int(sdm.samples.unique[0]) * 100 # percentage of non-nan values

warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", category=RuntimeWarning) 
plt.figure(figsize=(13,8))
sns.distplot(dm_1.nonnan_pupil)
plt.xlabel('Percentage of valid samples per trial')
plt.axvline(50)
plt.tight_layout()
plt.show()  
warnings.filterwarnings("default", category=UserWarning)  
warnings.filterwarnings("default", category=RuntimeWarning) 

print(f'{round(len(dm_1[dm_1.nonnan_pupil < 50])/len(dm_1) * 100, 3)}% trials with more than 50% of NAN values (n = {len(dm_1[dm_1.nonnan_pupil < 50])}/{len(dm_1)} total trials; {len(dm_1.Subject[dm_1.nonnan_pupil < 50].unique)} participants).')

dm_1 = dm_1.nonnan_pupil >= 50 # keep relevant trials only

# Exclude trials with excessive duration
two_std = np.round([dm_1.maxdur.mean - (dm_1.maxdur.std * 2), dm_1.maxdur.mean+(dm_1.maxdur.std * 2)], 3)
very_slow = dict(Counter(dm_1.Subject[dm_1.maxdur > two_std[1]]))
print(very_slow)
too_slow = [i for i in very_slow.keys() if very_slow[i] == 12]
dm_1df = convert.to_pandas(dm_1)

warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", category=RuntimeWarning) 
plt.figure(figsize=(13,8))
for story in list(dm_1.Story.unique): 
    sns.distplot(dm_1.maxdur[dm_1.Story == story])
plt.title(f"-2 STD = {two_std[0]}, 2 STD = {two_std[1]}")
plt.axvline(two_std[0]);plt.axvline(two_std[1])
plt.xlabel('Trial duration (s)')
plt.tight_layout()
plt.show()
warnings.filterwarnings("default", category=UserWarning)  
warnings.filterwarnings("default", category=RuntimeWarning)

print(dm_1df.maxdur.describe())
print(f'{list(dm_1.Subject[dm_1.maxdur < two_std[0]].unique)} participants ({dm_1.maxdur[dm_1.maxdur < two_std[0]]})')
print(f'{list(dm_1.Subject[dm_1.maxdur > two_std[1]].unique)} participants ({dm_1.maxdur[dm_1.maxdur > two_std[1]]})')

dm_1 = dm_1.maxdur >= two_std[0]
#dm_1 = dm_1.maxdur <= two_std[1] # Option 1: exclude trial-wise (that excludes a lot...)
for s in too_slow: # Option 2: only exclude extremely slow readers
    print(dm_1df[dm_1df.Subject == s].maxdur.describe())
    dm_1 = dm_1.Subject != s # really slow reader

print(f'After exclusion for excessive duration: N = {len(dm_1.Subject.unique)} (n = {len(dm_1)} trials)')
print(dm_1df.maxdur.describe())

# Interpolate missing values
for p, t, sdm in ops.split(dm_1.Subject, dm_1.Paragraph):
    dm_1.Pupil[sdm][:, 0:sdm.samples[0]] = srs.interpolate(sdm.Pupil[:, 0:sdm.samples[0]]) # avoid missing values here

# Smooth the traces to reduce the jitter
dm_1.Pupil = srs.smooth(dm_1.Pupil, 3) 
print('Pupil size traces smoothed with a Hanning window of 51 ms.')
 
# Exclude trials with unrealistic mean pupil-size (outliers) 
print(f'Before trial exclusion (pupil size): {len(dm_1)} trials.')
dm_1.z_pupil = NAN 
for p, sdm in ops.split(dm_1.Subject):
    dm_1.z_pupil[sdm] = ops.z(srs.reduce(sdm.Pupil, np.nanmean))

warnings.filterwarnings("ignore", category=UserWarning)  
warnings.filterwarnings("ignore", category=RuntimeWarning) 
plt.figure(figsize=(13,8))  
sns.distplot(dm_1.z_pupil);plt.axvline(-2);plt.axvline(2)
plt.xlabel('Raw pupil-size means (z-scored)')
plt.tight_layout()
plt.show()
warnings.filterwarnings("default", category=UserWarning)  
warnings.filterwarnings("default", category=RuntimeWarning) 

print(dm_1.Subject[dm_1.z_pupil == NAN], dm_1.Subject[dm_1.z_pupil > 2.0], dm_1.Subject[dm_1.z_pupil < -2])
dm_1 = dm_1.z_pupil != NAN 
dm_1 = dm_1.z_pupil <= 2.0 
dm_1 = dm_1.z_pupil >= -2.0 
print(f'After trial exclusion (pupil size): {len(dm_1)} trials.')

 # How many trials left per participant?
print(f'Number of trials left per participant: {np.sort(Counter(dm_1.Subject), axis=None)}')
print(f'How many participants have n number of trials: {Counter(Counter(dm_1.Subject).values())}')

to_exclude = [i for i in dict(Counter(dm_1.Subject)).keys() if dict(Counter(dm_1.Subject))[i] < 6]

print(f'{to_exclude} has less than 50% of remaining trials.')
dm_1 = dm_1.Subject != set(to_exclude)
    
print(f'After preprocessing: N = {len(dm_1.Subject.unique)} (n = {len(dm_1)} trials)')

# Apply baseline correction
dm_1.Pupil = srs.baseline(dm_1.Pupil, dm_1.Pupil, 0, 3, method='subtractive') # baseline correct pupil size

# Add useful variables (mean pupil size, pupil-size changes, etc.)
dm_A = compute_vars(dm_1, dm2, dur=False) 

# =============================================================================
#%% Descriptives and plots
# =============================================================================
# Convert to pandas dataframe because it causes problems with seaborn to use datamatrix
# And save to csv file 
convert.to_pandas(dm_A).describe().to_csv(cwd+'/DESC-data_pupil-all.csv', index=True)
convert.to_pandas(dm2).describe().to_csv(cwd+'/DESC-data_pupil-perword.csv', index=True)

# Questionnaire data
df = pd.read_csv(cwd+'/nettskjema_ratings.csv',sep=',') # continuous

# Consistent with dm
subjects_to_keep = list(dm_A.Subject.unique)

df = df[df["ID"].isin(subjects_to_keep)]

# Extract participant number, sex, and age
df[['Participant', 'Sex', 'Age']] = df['ID'].str.extract(r'P(\d+)([FM])(\d+)')

# Convert Age to numeric 
df['Age'] = df['Age'].astype(int)

print(df['Age'].describe())
print(df.groupby(['Sex'])['Age'].describe())

# Recode answers
df["Memory story1 (Ole) NEI"] = df["Memory story1 (Ole) NEI"].map({"Nei": 1, "Ja": 0})
df["Memory story2 (Per) JA"] = df["Memory story2 (Per) JA"].map({"Nei": 0, "Ja": 1})
df["Memory story3 (Kari) JA"] = df["Memory story3 (Kari) JA"].map({"Nei": 0, "Ja": 1})
df["Memory story4 (Anne) NEI"] = df["Memory story4 (Anne) NEI"].map({"Nei": 1, "Ja": 0})

# Retrieve accuracy
df['Memory_accuracy'] = df[list(df.columns[1:5])].mean(axis=1) * 100
print(df['Memory_accuracy'].describe())
print(Counter(df['Memory_accuracy']))

for col in list(df.columns[1:5]):
    print(df[col].describe())
    print(Counter(df[col]))

dm_A.accuracy = NAN
for s, story, sdm in ops.split(dm_A.Subject, dm_A.Story):  
    # Add variables of interest
    df_sub = df[df["ID"].isin([s])]
    dm_A.accuracy[sdm] = float(df_sub.Memory_accuracy)

# Print descriptives
dm_desc = convert.to_pandas(dm_A)

dm_desc.groupby(by=['Story', 'Brightness'])['Vividness'].describe()
dm_desc.groupby(by=['Story'])['maxdur'].describe()

print(dm_desc.Vividness.describe())
print(dm_desc.Suspense.describe())

dm_desc.Suspense.describe()
print(dm_desc.maxdur.describe())
np.percentile(dm_desc.maxdur, 95)
print(dm_desc.accuracy.describe())

# Visualise preprocessed data
plt.rcParams['font.size'] = 40
quick_visualisation(dm_A, cwd=cwd) # continuous

# =============================================================================
#%% Statistical analyses
# =============================================================================
dm_tst = dm_A.accuracy >= 0 # creates a copy of the datamatrix
dm_tst.Brightness[dm_tst.Brightness == 'light'] = 'bright' # rename

# /!\ This simply helps with convergence warnings but does not change the results /!\
#dm_tst.mean_pupil = ops.z(dm_tst.mean_pupil) 

# The model (no random slopes for Brightness or it creates a convergence warning again)
mdf1 = lm_pupil(dm_tst, formula = 'mean_pupil ~ Brightness', re_formula = '1 + Brightness')
check_assumptions(mdf1) # OK QQplot
print(mdf1.summary())
print_results(mdf1)

mdf2 = lm_pupil(dm_tst, formula = 'mean_pupil ~ Brightness * Vividness', re_formula = '1 + Brightness')
check_assumptions(mdf2) # OK QQplot 
print(mdf2.summary())
print_results(mdf2)

# Compare the models
compare_models(mdf1, mdf2, 2) 
print(mdf1.aic < mdf2.aic)

mdf3 = lm_pupil(dm_tst, formula = 'pupil_change ~ Vividness', re_formula = '1 + Vividness', pupil_change=True, reml=True)
check_assumptions(mdf3) # OK QQplot 
print(mdf3.summary())
print_results(mdf3)

test_correlation(dm_tst, x='mean_vivid', y='mean_suspense', alt='greater', lab='Vividness', fig=False)

test_correlation(dm_tst, x='mean_vivid', y='pupil_change', alt='greater', lab='Vividness', fig=False)
test_correlation(dm_tst, x='mean_suspense', y='pupil_change', alt='greater', lab='Vividness', fig=False)
test_correlation(dm_tst, x='accuracy', y='pupil_change', alt='greater', lab='Vividness', fig=False)

# # Submodels
# for s, subdm in ops.split(dm_tst.Story):
#     # Pupil-size changes: Individual effects
#     # MLM
#     print(s)
#     mdf4 = lm_pupil(subdm, formula = 'pupil_change ~ mean_vivid', re_formula = '1', pupil_change=True, reml=True)
#     check_assumptions(mdf4) 
#     print(mdf4.summary())
    
#%% Controls
mdf2B = lm_pupil(dm_tst, formula = 'mean_pupil ~ Brightness * Vividness + Suspense', re_formula = '1 + Brightness')
check_assumptions(mdf2B) # OK QQplot 
print(mdf2B.summary())
print_results(mdf2B)

# Compare the models
compare_models(mdf2, mdf2B, 1) 
print(mdf2.aic < mdf2B.aic)

mdf2C = lm_pupil(dm_tst, formula = 'mean_pupil ~ Brightness * Vividness + Order', re_formula = '1 + Brightness')
check_assumptions(mdf2C) # OK QQplot 
print(mdf2C.summary())
print_results(mdf2C)

# Compare the models
compare_models(mdf2, mdf2C, 1) 
print(mdf2.aic < mdf2C.aic)
# =============================================================================
#%% Visualisation: Individual effects (point plots)
# =============================================================================
# 1. Mean pupil size per condition and per participant (all conditions)
for s, subdm in ops.split(dm_A.Story):

    subdm = subdm.pupil_change != NAN
    subdm.mean_pupil_change = ''
    for p, sdm in ops.split(subdm.Subject):
        subdm.mean_pupil_change[sdm] = sdm.pupil_change.mean
    subdm = ops.sort(subdm, by=subdm.mean_pupil_change)
    
    # How many participants have a mean positive score (averaged across all subtypes)?
    N = len(subdm.Subject.unique)
    n = len(set(dm_A.Subject[subdm.pupil_change > 0]))
    print(f'{round(n/N * 100, 2)}% ({n}/{N}) of positive changes')

    list_order1 = list(dict.fromkeys(subdm.Subject))
    dm_sub = convert.to_pandas(subdm)
    
    # 2. Mean pupil size changes per participant 
    plt.figure(figsize=(20,10))
    ax = plt.subplot(1,1,1)
    plt.title(str(s)+f' | {round(n/N * 100, 2)}% ({n}/{N}) of positive changes')
    sns.pointplot(data=dm_sub, x='Subject', y='pupil_change', hue='Story', hue_order=None, order=list_order1, markersize=18, linewidth=5, estimator=np.mean, palette=['black'], errorbar=('se',1), legend=False)
    plt.axhline(0, linestyle='solid', color='black')
    plt.xlabel('Participants');plt.ylabel('Pupil-size mean differences\n (Dark - Bright) (a.u.)')
    plt.xticks([])
    plt.tight_layout()
    plt.show()


# How many participants with a mean positive score?
dm_grouped = ops.group(dm_A, by=dm_A.Subject)
dm_grouped.pupil_change = srs.reduce(dm_grouped.pupil_change)
N = len(dm_grouped.Subject.unique)
n = len(set(dm_grouped.Subject[dm_grouped.pupil_change > 0]))
print(f'{round(n/N * 100, 2)}% ({n}/{N}) of positive changes')

subdm = ops.group(dm_A, by=[dm_A.Subject, dm_A.Brightness])
subdm.accuracy = srs.reduce(subdm.accuracy)
subdm = ops.sort(subdm, by=subdm.accuracy)
dm_sub = convert.to_pandas(subdm)

plt.figure(figsize=(20,10))
ax = plt.subplot(1,1,1)
sns.pointplot(data=dm_sub, x='Subject', y='accuracy', hue='Brightness', hue_order=None, markersize=18, linewidth=5, estimator=np.mean, palette=['black'], errorbar=('se',1), legend=False)
plt.axhline(0, linestyle='solid', color='black')
plt.xlabel('Participants');plt.ylabel('Pupil-size mean differences\n (Dark - Bright) (a.u.)')
plt.xticks([])
plt.tight_layout()
plt.show()
