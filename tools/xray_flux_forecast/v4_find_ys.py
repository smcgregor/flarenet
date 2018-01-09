"""
This is a script for assembling a y file based on the files currently in the training and validation folders.
It assumes you are using version 4 data.
Delays and catches can be specified in the arrays below.
The resulting file has the following columns:
1. Date Stamp
2. Max Xray during catch after delay
3. Current Xray Flux
4. Delta Xray flux (Column 2. - Column 3.)
5. File name
"""

#!usr/env/bin python

import sys
sys.dont_write_bytecode=True
import csv
import datetime
import json
import numpy as np
import glob
import yaml
import re
import os
import os.path

with open("config.yml", "r") as config_file:
    config = yaml.load(config_file)

filePath = config['aia_path']
dependent_variable_path = filePath + "y/"

if not os.path.exists(dependent_variable_path):
    os.makedirs(dependent_variable_path)

#Y_data
Y_file_path = 'tools/xray_flux_forecast/Flux_2010_2017_max.csv'
Y_data = csv.reader(open(Y_file_path,'r'))
Y_data = list(Y_data)
Y_init_date = datetime.datetime.strptime(Y_data[1][0],'%Y-%m-%d %H:%M:%S')


###############################
#                             #
#  Delay and Catch definition #
#                             #
###############################

delays = [60]
catches = [12, 36, 60, 24*60]

for delay in delays:
    for catch in catches:
        
        def get_Y_index(date):
            """Function that finds the index on the reference large y file for a given date
               @param dependent_variable {date} datetime.datetime.strptime object constructed
                    from the file names"""
            delta = date-Y_init_date
            delta = delta.days*24*30 + delta.seconds/120 + 1
            return int(delta)
        
        def get_yval(index, is_current = False):
            """Function that finds the maximum value within the catch window
               @param dependent_variable {index} index within the reference large y file that points
                    at the element of interest"""
            if is_current:
                if Y_data[index][1] == 'NA':
                    return 0.0
                else:
                    return float(Y_data[index][1])

            catch_arr = []
            end_index = index + catch/2
             
            if end_index > len(Y_data):
                end_index = len(Y_data)  
            for j in range(int(index),int(end_index)):
                if Y_data[j][1] != 'NA':
                    catch_arr.append(float(Y_data[j][1]))
                else:
                    catch_arr.append(0.0)
            return max(catch_arr)
        
        #Adding column names
        Y_vals = [] #[[Date, Y-data - 12min/36min/1hr/24hr max, Channel 0,7 coefficients]]
        y_row = ['Date Stamp']
        y_row += ['Catch Flux']
        y_row += ['Current Flux']
        y_row += ['Delta']
        y_row += ['Filename']
        Y_vals.append(y_row)
        
        #Processing flare files
        flare_files = glob.glob(filePath + 'validation/*_AIA*012h.fthr')
        flare_files += glob.glob(filePath + 'training/*_AIA*012h.fthr')
        for f in flare_files:

            inxSlash =  [m.start() for m in re.finditer('/', f)]
            inxSlash = inxSlash[len(inxSlash)-1]
            f = f[inxSlash+1:]           
            date_s = f[9:22]

            #Current Xray flux
            date = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')
            Y_indexC = get_Y_index(date)

            #Future Xray flux after delay
            date += datetime.timedelta(seconds=60*delay)
            Y_indexF = get_Y_index(date)

            #Store values
            if Y_indexF < len(Y_data):
                y_row = [date_s]
                y_row += [get_yval(Y_indexF, False)]  #Future Xray Flux
                y_row += [get_yval(Y_indexC, True)]  #Current Xray Flux
                y_row += [y_row[1]-y_row[2]]  #Delta
                y_row += [f]  #File name
                Y_vals.append(y_row)

        #Processing no flare files
        no_flare_files = glob.glob(filePath + 'validation/noflr_AIA*000m.fthr')
        no_flare_files += glob.glob(filePath + 'training/noflr_AIA*000m.fthr')
        for f in no_flare_files:
            inxSlash =  [m.start() for m in re.finditer('/', f)]
            inxSlash = inxSlash[len(inxSlash)-1]
            f = f[inxSlash+1:]           
            date_s = f[9:22]

            #Current Xray flux
            date = datetime.datetime.strptime(date_s,'%Y%m%d_%H%M')
            Y_indexC = get_Y_index(date)

            #Future Xray flux after delay
            date += datetime.timedelta(seconds=60*delay)
            Y_indexF = get_Y_index(date)

            #Store values
            if Y_indexF < len(Y_data):
                y_row = [date_s]
                y_row += [get_yval(Y_indexF, False)]  #Future Xray Flux
                y_row += [get_yval(Y_indexC, True)]  #Current Xray Flux
                y_row += [y_row[1]-y_row[2]]  #Delta
                y_row += [f]  #File name
                Y_vals.append(y_row)              
        
        if delay >= 60:
            this_delay = '%02dhr'%(delay/60)
        else:
            this_delay = '%02dmin'%(delay)
        if catch >= 60:
            this_dur = '%02dhr'%(catch/60)
        else:
            this_dur = '%02dmin'%(catch)
        
        print(len(Y_vals))
        writer = csv.writer(open(dependent_variable_path + 'All_Ys_%sDelay_%sMax.csv'%(this_delay,this_dur),'w'))
        writer.writerows(Y_vals)