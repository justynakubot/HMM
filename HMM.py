'''
Model assumptions:
* Stock market is driven only by market sentiment
* Market sentiment relies on two factors:
    - most important macroeconomical conditions:
        a. interest rates
        b. Central Banks policy
        c. GDP growth
        d. unemployment rate
        e. geopolitical shocks
        f. black swans
    - previous states of sentiments (macro factors changes relatively slow, do for period
    without any changes we can try to predict daily returns only on the previous sentiment)
    states. For that part we will develop the HMM model of sentiments.
'''

# Define period for which we can consider that data is relevant to current market conditions.
# Proposed: last interest rate increase of Fedeal Reserve
# Project starts 06/04/2024


import yfinance as yf
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from scipy.stats import norm
from sklearn.mixture import GaussianMixture

# Initial input variables

Window_for_training_model = {'start': '2023-03-01', 'end': '2024-03-01'}

# Days when significant macroeconomical data was published and impacts the sentiment
# These need to be excluded from dataset

# FOMC meetings:

FOMC_meeting_days = [
    '2023-03-21',
    '2023-03-22',
    '2023-05-02',
    '2023-05-03',
    '2023-06-13',
    '2023-06-14',
    '2023-07-25',
    '2023-07-26',
    '2023-09-19',
    '2023-09-20',
    '2023-09-20',
    '2023-10-31',
    '2023-11-01',
    '2023-11-12',
    '2023-12-12',
    '2023-12-13',
    '2024-01-19', 
    '2024-01-20',
]

# CPI publications

inflation_report_dates = [
    '2023-03-03',
    '2023-05-10',
    '2023-06-14',
    '2023-07-12',
    '2023-08-11',
    '2023-09-12',
    '2023-10-11',
    '2023-11-14',
    '2023-12-12',
    '2024-01-11',
    '2024-02-13',
]

# Days with unemployment reports

jolts_report_days = [
    '2023-03-28',
    '2023-04-25',
    '2023-05-23',
    '2023-06-27',
    '2023-07-25',
    '2023-08-22',
    '2023-09-26',
    '2023-10-24',
    '2023-11-21',
    '2023-12-19',
    '2024-01-23',
]

days_to_exclude = jolts_report_days + inflation_report_dates + FOMC_meeting_days

# Input data preparation
# Import data of US Stock Market

table = pd.read_html('https://en.wikipedia.org/wiki/List_of_S%26P_500_companies')
tickers = table[0]
stocks = tickers['Symbol'].to_list()
df = yf.download(tickers = stocks, start = Window_for_training_model['start'], 
                 end = Window_for_training_model['end'], interval = '1d')['Adj Close']

# check how many NaN in df
df.isna().sum()
# No annoying NaN for once!!

# calculate % change and transfer to log
df_pct = df.pct_change(axis = 0)
df_pct.dropna(how = 'all', inplace = True)
df_log = np.log(1+df_pct)

# Time to exclude biased days!
df_log_adjusted = df_log.loc[~(df_log.index.isin(days_to_exclude))]


'''
For intraday trading we should focus on stocks with highest volatility of the prices
We can measure this by computing standard deviation of each stock
'''

# 10 Most volatiled stocks in considered period:
most_volatile_10 = df_log_adjusted.std(axis = 0).sort_values().index[:10]


######################################################
#   LET'S TRAIN OUR MODEL!
######################################################

#splitting dataset to training and testing model in proportion 7:3
training_set = df_log_adjusted.iloc[:int(len(df_log_adjusted)*0.7)]
testing_set = df_log_adjusted.iloc[int(len(df_log_adjusted)*0.7):]


'''
Assumptions about sentiment:
* We have three states:
    - Positive sentiment given by distribution with positive expected value
    - Neutral sentiment with EV closest to 0
    - Negative sentiment with negative EV

* All possible future states rely only on our currently observed state.
* Market is effective in the meaning that our current observed state contains all 
possible informations

Each day our model will decide what is the expected value of our next step and will
play long or short. It will close the position at the end of the day and repeat the
procedure.

At the end we will compute return of this algorithm and know if beat the market.
'''

# Check which of the most volatiled stocks fits best for our purposes

number_of_distributions = 3
most_volatile_10

for i in most_volatile_10:
    df_try = training_set[i].copy()

    gmm = GaussianMixture(n_components=number_of_distributions)
    gmm.fit(np.array(df_try.tolist()).reshape(-1,1))

    means = gmm.means_
    covariances = gmm.covariances_

    x = np.linspace(np.array(df_try.tolist()).min(), np.array(df_try.tolist()).max(), 1000)

    for j in range (number_of_distributions):
        plt.plot(x, gmm.weights_[j] * np.exp(-(x - means[j])**2/(2*covariances[j]))[0])
    plt.title(i)
    plt.show()

# After a careful look we can test our model on DUK company

df_try = training_set['DUK'].copy()

gmm = GaussianMixture(n_components=number_of_distributions, covariance_type='full')
gmm.fit(np.array(df_try.tolist()).reshape(-1, 1))

means_cov = pd.DataFrame(data= {'covariance': covariances.T[0][0], 'means': means.T[0]})
means_cov = means_cov.sort_values(by=['means'], ascending=True)

# assign means and variance to negative, positive or neutral state
mean_neg = means_cov['means'].iloc[0].item()
mean_neutral = means_cov['means'].iloc[1].item()
mean_pos = means_cov['means'].iloc[2].item()

variance_neg = means_cov['covariance'].iloc[0].item()
variance_neutral = means_cov['covariance'].iloc[1].item()
variance_pos = means_cov['covariance'].iloc[2].item()

# Create vector of different states
# Calculate points between distributions where we cannot choose in which one we are
df_changes = pd.DataFrame(columns = ['Value', 'Sentiment'])

for i in np.arange(len(df_try)):
    variable_value = df_try.iloc[i]
    likelihood_neg = norm.pdf(variable_value, loc = mean_neg, scale = np.sqrt(variance_neg))
    likelihood_neutral = norm.pdf(variable_value, loc = mean_neutral, scale = np.sqrt(variance_neutral))
    likelihood_pos = norm.pdf(variable_value, loc = mean_pos, scale = np.sqrt(variance_pos))

    max_likelihood = max(likelihood_neg,likelihood_neutral,likelihood_pos)

    if max_likelihood == likelihood_neg:
        most_likely_distribution = 'Negative'
    elif max_likelihood == likelihood_neutral:
        most_likely_distribution = 'Neutral'
    else:
        most_likely_distribution = 'Positive'
    
    df_changes.loc[len(df_changes)] = [variable_value, most_likely_distribution]

# Create matrix of states with their probabilities
states = ['Positive', 'Neutral', 'Negative']

transition_matrix = np.empty((3,3), dtype=object)

for i, from_state in enumerate(states):
    for j, to_state in enumerate(states):
        transition_count = sum(1 for k in range(len(df_changes.Sentiment) - 1)
                               if df_changes.Sentiment[k] == from_state 
                               and df_changes.Sentiment[k+1] == to_state)
        transition_matrix[i][j] = transition_count
transition_matrix = transition_matrix/transition_matrix.sum(axis = 0)
transition_matrix = pd.DataFrame(transition_matrix, index=states, columns=states).round(2)

outcome_matrix = transition_matrix.copy()
outcome_matrix = (outcome_matrix * np.array([mean_pos, mean_neutral, mean_neg]).T)
outcome_matrix.sum()

df_changes_2 = pd.DataFrame(columns=['Value', 'Sentiment'])
for i in np.arange(len(testing_set['DUK'])):
    variable_value = testing_set['DUK'].iloc[i]
    likelihood_neg = norm.pdf(variable_value, loc = mean_neg, scale = np.sqrt(variance_neg))
    likelihood_neutral = norm.pdf(variable_value, loc = mean_neutral, scale = np.sqrt(variance_neutral))
    likelihood_pos = norm.pdf(variable_value, loc = mean_pos, scale = np.sqrt(variance_pos))

    max_likelihood = max(likelihood_neg, likelihood_neutral, likelihood_pos)
    if max_likelihood == likelihood_neg:
        most_likely_distribution = 'Negative'
    elif max_likelihood == likelihood_neutral:
        most_likely_distribution = 'Neutral'
    else:
        most_likely_distribution = 'Positive'

    df_changes_2.loc[len(df_changes_2)] = [variable_value, most_likely_distribution]

sentiment_day_before = df_changes_2['Sentiment'].iloc[:len(df_changes_2) - 1].values.tolist()

# on the training set the last day finished with Neutral sentiment, therefore 
# this is used as the inintial state for testing set
initial_sentiment = ['Neutral']     

list_of_previous_sentiments = initial_sentiment + sentiment_day_before

df_changes_2['Previous_Sentiment'] = pd.Series(list_of_previous_sentiments)
df_changes_2

pos_sum = np.sum(df_changes_2['Value'].loc[(df_changes_2['Previous_Sentiment'] == 'Positive') &
                                           (df_changes_2['Previous_Sentiment']=='Neutral')])
neut_sum = np.sum(df_changes_2['Value'].loc[df_changes_2['Previous_Sentiment']=='Negative']) * -1

model_outcome = neut_sum + pos_sum

np.exp(np.sum(df_changes_2['Value']))-1
    

