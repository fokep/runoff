# This Python file uses the following encoding: utf-8
import os, sys

 ######################################################################################
##                                                                                    ##
##  Sogndal Valley Catchment Runoff Model                                             ##
##                                                                                    ##
##  Copyright (c) 2018 - Presthaug, E. F.                                             ##
##                                                                                    ##
##  Distributed under the MIT software license, see                                   ##
##     http://www.opensource.org/licenses/mit-license.php.                            ##
##                                                                                    ##
##  Loosely based on (parts of) HBV Model exercises, University of Zurich /           ##
##      Seibert, J., & Vis, M. J. (2012). /                                           ##
##      Teaching hydrological modeling with a user-friendly catchment-runoff-model /  ##
##      software package. /                                                           ##
##      Hydrology and Earth System Sciences, 16, 3315–3325.                           ##
##                                                                                    ##
 ######################################################################################

## Packages required to run
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import click

modelVersion = '0.3.0'

## Model switches
# Run with or without temperature correction
tempCorr    = True

## Model parameters
ptt         = 0.75              # Treshold temperature (degC) below which solid precipitation occurs
pttm        = 0.75              # Treshold temperature (degC) for melting (snow/glacier)
pcfmax      = 2.75              # Degree-day factor for snow melting
pcfr        = 0.05 * pcfmax     # Degree-day for refreezing (5 % of PCFMAX)
pcwh        = 0.1               # Fraction of melt that can be retained in snowpack (max 10 % of SWE)
ddfSnow     = 3.5               # Degree-day factor for glacier melting
ddfIce      = 5.5               # Degree-day factor for glacier melting
lapseRate   = 5 / 1000          # Static lapse rate -5 degrees per km

## Catchment data
catchArea   = 17.8                      # From NEVINA (km^2)
glacierArea = 6.8 * catchArea / 100     # From NEVINA
iceArea     = 0.5 * glacierArea         # 50/50 snow/ice ?
snowArea    = 0.5 * glacierArea         # 50/50 snow/ice ?

# Elevation profile from Nevina-report
hRef        = 421       # Reference elevation (meteo station)
hMin        = 492       # Lowest elevation level in catchment
h10         = 671       # 10 % of elevation is below this elevation
h20         = 811
h30         = 928
h40         = 1030
h50         = 1109
h60         = 1174
h70         = 1243
h80         = 1332
h90         = 1454      # 90 % of area is below this
hMax        = 1606      # Highest elevation level in catchment

## Helper Functions for array operations
refreezeFraction = lambda t: pcfr * pcfmax * (ptt - t)
""" Degree-day function for refreeze fraction
I:  Temperature
IU: degrees Celsius
O:  Potential amount of meltwater that can refreeze and be added back to SWE
OU: mm
"""

melt = lambda t: pcfmax * (t - pttm)
""" Degree-day function for snow melt
I:  Temperature
IU: degrees Celsius
O:  Potential amount of meltwater that is produced from melting of snow (SWE)
OU: mm
"""

glacierMelt = lambda t: t * ((ddfSnow * snowArea) + (ddfIce * iceArea))
""" Degree-day function for glacier melt
I:  Temperature
IU: degrees Celsius
O:  Potential amount of meltwater that is produced from melting of snow- and ice-covered area of glacier
OU: mm * km^2
"""

unitConversion = lambda x: x * 1000 / 24 / 60 / 60 # Change values here if using a different time resolution
""" Unit conversion function to get Q in desired units
I:  Discharge
IU: km^2 * mm / day
O:  Discharge
OU: m^3 / s
"""

## Load dataset 
# Need to have headers "date;temp;precip" or "date;temp:precip;swe"
# date (time unit) have to be formatted according to ISO 8601 (YYYY-MM-DD)
# Load the file containing the meteo data as a dataframe 'df' with pandas module
datasetFilename = 'dataset'
df = pd.read_csv(datasetFilename + '.csv', delimiter=';', parse_dates=['date'])

print('\n')
print('#####################')
print('##      MODEL      ##')
print('##     ', modelVersion, '     ##')
print('#####################')
print('\n')

## START OF MODEL FUNCTION
def model(dataset):
    """ Model
    I:  Dataset as pandas dataframe with "date, temp, precip" columns
    IU: date format: ISO 8601, temperature: degrees Celsius, precipitation: mm
    O:  Adds columns to dataframe, containing simulated SWE and discharge 
    OU: SWE in mm/time, discharge in m^3/s (depending on unitConversion settings)
    """
# Model part 1: Snow, snowmelt, refreeze and precipitation
    for i in range(1, len(dataset)): # Should be rewritten to iterate python style (NTS: kept like this because i-1 becomes an issue)
        """ Loop through model parts from day 1 to end of dataset"""

# Model part 1.1: swe_sim
        if dataset.loc[i, 'temp'] <= ptt:
            """ Execute if temp today is below or equal to freezing threshold"""
            dataset.loc[i, 'swe_sim'] = dataset.loc[i-1, 'swe_sim'] + dataset.loc[i, 'precip']
            """ Add todays precipitation and previous day SWE"""

# Model part 1.1.1: swe_sim refreeze fraction
            if dataset.loc[i-1, 'swe_sim'] > 0.0 and dataset.loc[i-1, 'temp'] >= pttm:
                """ Execute if something melted yesterday"""
                dataset.loc[i, 'swe_sim'] += min((pcwh * dataset.loc[i, 'swe_sim']), max((dataset.loc[i, 'refreezeFraction']), 0.0))
                """ Add refreeze fraction if positive value (but not more than 10 % of SWE), otherwise add 0.0"""

# Model part 1.1.2: swe_sim melt           
        else:
            """ Execute if temp today is NOT below or equal to freezing threshold"""                                                                           
            if (dataset.loc[i-1, 'swe_sim'] - max((dataset.loc[i, 'melt']), 0.0)) < 0.0:
                """ Execute if melt is greater than actual SWE (produces a negative value)"""
                dataset.loc[i, 'swe_sim'] = 0.0
                """ Set to 0.0"""
            else:
                """ Execute if melt it NOT greater than actual SWE (still positive value)""" 
                dataset.loc[i, 'swe_sim'] = dataset.loc[i-1, 'swe_sim'] - max((dataset.loc[i, 'melt']), 0.0)
                """ Subtract melt"""

# Model part 1.2: run_off
            dataset.loc[i, 'run_off'] = dataset.loc[i, 'precip']
            """ Add precipitation"""
            if dataset.loc[i, 'swe_sim'] < dataset.loc[i-1, 'swe_sim']:
                """ Execute if there has been any melting (SWE has become smaller)"""
                dataset.loc[i, 'run_off'] += dataset.loc[i-1, 'swe_sim'] - dataset.loc[i, 'swe_sim']
                """ Add difference in SWE (melt)"""

# Model part 2: Glacier melt and precipitation on day 0
# Model part 2.1: Precip on day 0
    if dataset.loc[0, 'temp'] > ptt:
        """ Execute if temp on day 0 is above freezing threshold"""
        dataset.loc[0, 'run_off'] = dataset.loc[0, 'precip']
        """ Add precipitation"""

# Model part 2.2: Glacier melt
    for i in range(0, len(dataset)):
        """ Loop through model parts from day 0 to end of dataset"""
        if dataset.loc[i, 'temp'] > pttm:
            """ Execute if temp is above melting threshold"""
            dataset.loc[i, 'glacier_run_off'] = dataset.loc[i, 'glacierMelt']
            """ Add melt from glacier (is in volume units)"""

# Model part 3: Unit conversions,  finishing up
    # Multiply with catchment area (convert to volume units)
    dataset['run_off'] = dataset['run_off'].values * catchArea
    # Apply conversion function to have Q in m^3/s
    dataset['run_off'] = unitConversion(dataset['run_off'].values)
    dataset['glacier_run_off'] = unitConversion(dataset['glacier_run_off'].values)
    # Add the sources in new column "discharge"
    dataset['discharge'] = dataset['run_off'].values + dataset['glacier_run_off'].values
## END OF MODEL FUNCTION

## START OF PLOT FUNCTION
def plotFunc():
    """ Set up plots"""
    # Set date values as the dataframe index (replaces actual index)
    df.set_index(df['date'], inplace=True)

    # Setup for 3 row / 2 column subplots with shared X-axis
    fig, axarr = plt.subplots(3, 2, sharex=True, figsize=(20, 10))          

    axarr[0,0].plot(df.index, df['temp'])
    axarr[0,0].set_title('Anestøl Temperature')
    axarr[0,0].set_ylabel('Temperature [°C]')

    axarr[1,0].bar(df.index, df['precip'])
    axarr[1,0].set_title('Selseng Precipitation')
    axarr[1,0].set_ylabel('Precipitation [mm]')

    axarr[2,0].plot(df.index, df['swe_sim'], label='Simulated SWE')
    if 'swe' in df:
        axarr[2,0].plot(df.index, df['swe'], label='Measured SWE')
    axarr[2,0].set_title('Catchment SWE')
    axarr[2,0].set_ylabel('SWE [mm]')
    axarr[2,0].legend(loc='upper left')

    axarr[0,1].plot(df.index, df['run_off'])
    axarr[0,1].set_title('Precipitation and Melt Discharge')
    axarr[0,1].set_ylabel('Discharge [m³/s]')

    axarr[1,1].plot(df.index, df['glacier_run_off'])
    axarr[1,1].set_title('Glacier Discharge')
    axarr[1,1].set_ylabel('Discharge [m³/s]')

    axarr[2,1].plot(df.index, df['discharge'])
    axarr[2,1].set_title('Total Discharge')
    axarr[2,1].set_ylabel('Discharge [m³/s]')

    axarr[2,0].xaxis.set_major_locator(mdates.MonthLocator(interval=3))
    axarr[2,0].xaxis.set_minor_locator(mdates.MonthLocator(interval=1))
    axarr[2,0].xaxis.set_major_formatter(mdates.DateFormatter('%Y %b'))

    fig.align_labels()

    plt.subplots_adjust(
        left    = 0.05,
        bottom  = 0.05,
        right   = 0.95,
        top     = 0.95,
        wspace  = 0.15,
        hspace  = 0.15
        )
## END OF PLOT FUNCTION

## START OF MODEL RUN
if tempCorr == True:
    """ Execute on temperature corrected elevations"""
# Model part 0: Initialization of arrays

    # Setup 2D array with 10 averaged elevation levels
    heights = np.array([[ 
        (hMin+h10)/2,
        (h10+h20)/2,
        (h20+h30)/2,
        (h30+h40)/2,
        (h40+h50)/2,
        (h50+h60)/2,
        (h60+h70)/2,
        (h70+h80)/2,
        (h80+h90)/2,
        (h90+hMax)/2
        ]])

    # Create array with temp_rows X correction(h)_cols
    temps = (np.array([df['temp']]).T) - ((heights - hRef) * lapseRate)

    # Create pandas dataframe for every elevation level temperature
    nr0 = pd.DataFrame(temps[:,0], columns=['temp'])
    nr1 = pd.DataFrame(temps[:,1], columns=['temp'])
    nr2 = pd.DataFrame(temps[:,2], columns=['temp'])
    nr3 = pd.DataFrame(temps[:,3], columns=['temp'])
    nr3 = pd.DataFrame(temps[:,3], columns=['temp'])
    nr4 = pd.DataFrame(temps[:,4], columns=['temp'])
    nr5 = pd.DataFrame(temps[:,5], columns=['temp'])
    nr6 = pd.DataFrame(temps[:,6], columns=['temp'])
    nr7 = pd.DataFrame(temps[:,7], columns=['temp'])
    nr8 = pd.DataFrame(temps[:,8], columns=['temp'])
    nr9 = pd.DataFrame(temps[:,9], columns=['temp'])

    # Make a list of the dataframes to loop on
    dfs = [nr0, nr1, nr2, nr3, nr4, nr5, nr6, nr7, nr8, nr9]

    # Initialize master dataframe output columns
    df['swe_sim']           = 0.0
    df['run_off']           = 0.0
    df['glacier_run_off']   = 0.0
    df['discharge']         = 0.0

    for index, dataset in enumerate(dfs):
        """ Loop through the nested model on every dataframe in list dfs"""

        ## Initialize columns in the dataframe(s)
        # Precipitation from master dataframe
        dataset['precip'] = df['precip']
        # Intermediary calculations
        dataset['refreezeFraction'] = refreezeFraction(dataset['temp'].values)
        dataset['melt'] = melt(dataset['temp'].values)
        dataset['glacierMelt'] = glacierMelt(dataset['temp'].values)
        # Output
        dataset['swe_sim']          = 0.0   # Needs initial value
        dataset['run_off']          = 0.0   # Avoids calculation / plot errors
        dataset['glacier_run_off']  = 0.0   # Avoids calculation / plot errors

        # Print progress to terminal
        print("Crunching datasets:", str(index +1) + "0 %") 

        # Run Model
        model(dataset)

        # Insert weighted values from each elevation
        df['swe_sim']           += 0.1 * dataset['swe_sim']
        df['run_off']           += 0.1 * dataset['run_off']
        df['glacier_run_off']   += 0.1 * dataset['glacier_run_off']
        df['discharge']         += 0.1 * dataset['discharge']

    # Print progress to terminal
    print('\n')
    print('Finishing up simulation...')
    print('\n')

else:
    """ Execute on master dataframe"""
# Model part 0: Initialization of arrays

    ## Initialize columns in the dataframe
    # Intermediary calculations
    df['refreezeFraction']  = refreezeFraction(df['temp'].values)
    df['melt']              = melt(df['temp'].values)
    df['glacierMelt']       = glacierMelt(df['temp'].values)
    # Output
    df['swe_sim']           = 0.0   # Needs initial value
    df['run_off']           = 0.0   # Avoids calculation / plot errors
    df['glacier_run_off']   = 0.0   # Avoids calculation / plot errors

    # Print progress to terminal
    print("Crunching dataset!") 

    # Run Model
    model(df)

    # Print progress to terminal
    print('\n')
    print('Finishing up simulation...')
    print('\n')

    # Remove intermediary columns from dataset
    df = df.drop(columns=['refreezeFraction', 'melt', 'glacierMelt'])
## END OF MODEL RUN

# Print progress to terminal
print('Simulation completed!')
print('\n')

## Program Output
if click.confirm('Would you like to export the data to file?'):
    filename = input("Please enter a filename for storing the model data: ")
    if filename == datasetFilename:
        print('Error: Filename is same as dataset')
    else:
        df.to_csv(filename + '.csv', sep=';')
        print('Model output has been stored to', filename + '.csv')

if click.confirm('Would you like to save a plot to file?'):
    filename = input("Please enter a filename for storing the model plot: ")
    plotFunc()
    plotFuncRun = True
    plt.savefig(filename + '.png', dpi=150)
    print('Model plot has been saved to', filename + '.png')
else:
    plotFuncRun = False

if click.confirm('Would you like to show a plot?'):
    if plotFuncRun is True:
        plt.show()
    else:
        plotFunc()
        plt.show()

## To do/bugs:
# more elegant overwrite protection for output (also for plot)
#
## Wishlist / future development:
# Refactor, refactor, refactor
# Ask user if it should run with elevation correction or maybe run both always?
# Meltwater storage in snow
# Soil Routine
# More complex glacier routine (energy-based?)
# Reservoir contributions
# Decouple to use average temps instead of repeating?
# Maybe use differenct weights instead of average?
# Rewrite model to use python looping styles (no range() stuff)
# Rewrite to use classes
# Utilize optimization algos to calibrate model params. ref https://docs.scipy.org/doc/scipy/reference/tutorial/optimize.html
# Automate catchment profile import from Nevina?? Talk to NVE
# Maybe a GUI-implementation with map (from kartverket) using the NVE API to fetch params?