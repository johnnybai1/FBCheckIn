# Exploratory Data Analysis for the FB Check-in Training Set

"""
VARIABLE DESCRIPTIONS
Row_id          Unique Row_id           [0, 29,118,020]
x               X coordinate            [0, 10]
y               Y coordinate            [0, 10]
accuracy        Measure of accuracy     [1, 1033]
time            Time of check in        [1, 786239]
place_id        Check in location
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import math

filename = "train.pkl"
df = pd.read_pickle("Data/" + filename) # Training Set data frame

# Initial description of the data set
def describeDF(df):
    print(df.describe().to_string()) # Basic stats for training set
    checkIns = df['place_id'].value_counts()
    print(checkIns.describe().to_string()) # Distribution of place_ids

# Histogram: x is number of check-ins, y is number of place_ids with that number of check-ins
def plotCheckIns(df):
    checkIns = df['place_id'].value_counts()
    plt.figure()
    plt.hist(checkIns.values, bins='fd')
    plt.xlabel("Number of Check-ins")
    plt.ylabel("Number of Place_ids")
    plt.savefig("Figures/CheckIn_count.png")
    plt.close()

# Histogram: x is accuracy, y is number of entries with that accuracy value
def plotAccuracy(df):
    accuracy = df['accuracy']
    plt.figure()
    plt.hist(accuracy, bins='fd')
    plt.xticks(np.arange(0,1400,100)) # x axis ticks from 0 to 1400 by 100
    plt.xlabel("Accuracy")
    plt.ylabel("Number of Entries")
    plt.savefig("Figures/raw_accuracy.png")
    plt.close()

# Histogram: x is time, y is number of entries with that time value
def plotRawTime(df):
    time = df['time']
    plt.figure()
    plt.hist(time, bins='fd')
    plt.xlabel("Time")
    plt.ylabel("Number of Entries")
    plt.savefig("Figures/raw_time.png")
    plt.close()

# Returns a data frame representing the top N checked into place_ids
def getTopNCheckInsDF(df, n=1):
    topN_place_ids = df['place_id'].value_counts().nlargest(n) # index is place_id, value is count
    sub_df = df.loc[df['place_id'].isin(topN_place_ids.index)]
    return sub_df

# Returns a data frame for a particular square block in the FB city
def getSquareBlockDF(df, x, y, size=0.5):
    square = df[df['x'] >= x]
    square = square[square['x'] <= x + size]
    square = square[square['y'] >= y]
    square = square[square['y'] <= y + size]
    return square

# This function aims to elucidate the units 'time' is recorded in. It takes advantage of epoch time.
# We should be able to see a pattern that mimics the real world (e.g. peak hours, closed hours, off
# peak hours.
def timeAnalysis(df):
    top = getTopNCheckInsDF(df) # Analyze the top checked into place only

    plt.figure()
    plt.title("Time attribute for 8772469670")
    plt.subplot(221) # rows cols plot_num
    plt.hist(top['time'], bins='fd')
    plt.title("Raw time value")

    # What if our time was in seconds...?
    t = top['time'].apply(lambda x: pd.Series(time.localtime(x)))
    t = t.rename(index=int,columns={0:"Year", 1:"Month", 2:"Day", 3:"Hour", 4:"Minute",
                                    5:"Seconds", 6:"Weekday", 7:"YearDay"})
    plt.subplot(222) # rows cols plot_num
    plt.hist(t['Hour'], bins=24)
    plt.title('Assumption: Time in Seconds')

    # What if our time was in minutes...?
    t = top['time'].apply(lambda x: pd.Series(time.localtime(x * 60)))
    realTime = t.rename(index=int,columns={0:"Year", 1:"Month", 2:"Day", 3:"Hour", 4:"Minute",
                                    5:"Seconds", 6:"Weekday", 7:"YearDay"})
    plt.subplot(223) # rows cols plot_num
    plt.hist(realTime['Hour'], bins=24)
    plt.title('Assumption: Time in Minutes')

    # What if our time was in hours?
    t = top['time'].apply(lambda x: pd.Series(time.localtime(x*60*24)))
    t = t.rename(index=int,columns={0:"Year", 1:"Month", 2:"Day", 3:"Hour", 4:"Minute",
                                    5:"Seconds", 6:"Weekday", 7:"YearDay"})
    plt.subplot(224) # rows cols plot_num
    plt.hist(t['Hour'], bins=24)
    plt.title('Assumption: Time in Hours')
    plt.save("Figures/Top_Time_Units.png")
    plt.close()

    # below will plot time in terms of days, hours, minutes, dayofweek
    plt.figure()
    plt.subplot(221)
    plt.title("Histogram of Days")
    plt.hist(realTime['Day'], bins=31)
    plt.subplot(222)
    plt.title("Histogram of Hours")
    plt.hist(realTime['Hour'], bins=24)
    plt.subplot(223)
    plt.title("Histogram of Minutes")
    plt.hist(realTime['Minute'], bins=60)
    plt.subplot(224)
    plt.title("Histogram of Day of Week")
    plt.hist(realTime['Weekday'], bins=7)
    plt.xticks(range(0, 7), ["Tue","Wed","Thur","Fri","Sat","Sun","Mon"])
    plt.save("Figures/Top_Time_DayHourMinuteWeekday.png")

def replaceTime(df):
    time_df = df['time'].apply(lambda x: pd.Series(time.localtime(x * 60)))
    time_df = time_df.rename(index=int,columns={0:"Year", 1:"Month", 2:"Day", 3:"Hour", 4:"Minute",
                                    5:"Seconds", 6:"Weekday", 7:"YearDay", 8:"Drop"})
    result = pd.concat([df, time_df], axis=1)
    result.drop('Year', axis=1, inplace=True)
    result.drop('Drop', axis=1, inplace=True)
    minutes = result['Minute']
    result['HourMinute'] = result['Hour'] + minutes.div(60)
    result['HourMinute'] = result['HourMinute'].apply(lambda x: math.sin(math.radians(15 * x)))
    return result

def main():
    # Work with a subset of the data
    square = getSquareBlockDF(df, 4, 6)
    # Extract some meaningful data from the time attribute
    # Weekday = 0 corresponds to Tuesday
    square = replaceTime(square)
    square.to_pickle("Data/train_46.pkl")
    # print(square.describe().to_string())


if __name__ == "__main__":
    main()
