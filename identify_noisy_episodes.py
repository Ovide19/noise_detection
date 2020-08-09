# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import numpy as np
import pandas as pd
from scipy import signal
#from numpy.polynomial.polynomial import polyfit 
import matplotlib.pyplot as plt
import pdb
import pywt
import datetime
import random
import gzip
import shutil
import os
from scipy import stats
import shutil
import matplotlib.dates as mdates
import time
from config_ML import *
import sklearn
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
#Import needed for logistic regression
from sklearn.linear_model import LogisticRegressionCV
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_recall_curve
from sklearn.exceptions import NotFittedError
import unittest
from sklearn import svm
from sklearn.metrics import roc_curve, auc
from sklearn.metrics import roc_auc_score

def nan_helper(y):
        "Taken from https://stackoverflow.com/questions/6518811/interpolate-nan-values-in-a-numpy-array"
        return np.isnan(y), lambda z: z.nonzero()[0]

def printProgressBar(iteration, total, prefix = '', suffix = '', decimals = 1, length = 100, fill = '█', printEnd = "\r"):
    "Taken from https://stackoverflow.com/questions/3173320/text-progress-bar-in-the-console"
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        length      - Optional  : character length of bar (Int)
        fill        - Optional  : bar fill character (Str)
        printEnd    - Optional  : end character (e.g. "\r", "\r\n") (Str)
    """
    percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / float(total)))
    filledLength = int(length * iteration // total)
    bar = fill * filledLength + '-' * (length - filledLength)
#    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix), end = printEnd)
    print('\r%s |%s| %s%% %s' % (prefix, bar, percent, suffix))
    # Print New Line on Complete
    if iteration == total: 
        print()

def create_output_folders_and_subfolders(root_folder,output_folder):
    if not os.path.isdir(str(root_folder)+str(output_folder)):    
        os.mkdir(str(root_folder)+str(output_folder))
        os.mkdir(str(root_folder)+str(output_folder)+'/kak')
        os.mkdir(str(root_folder)+str(output_folder)+'/kny')
        os.mkdir(str(root_folder)+str(output_folder)+'/kak_vs_kny')
    else:
        print("Output folders and subfolders already exist!")
        
def back_up_the_configuration_file(root_folder,output_folder):
    shutil.copy('myconfig.py', str(root_folder)+str(output_folder))
    

def extract_gunzipped_file(gzfilename,secfilename):
    with gzip.open(gzfilename, 'rb') as f_in:
        with open(secfilename, 'wb') as f_out:
            shutil.copyfileobj(f_in, f_out)
    return 


class Tests(unittest.TestCase):
     
     def test_data_folder(self, path):
          self.assertTrue(os.path.isdir(path))
          

class RawData(object):

    def __init__(self):
        self.date = None
        self.year = None
        self.station_id= None
        self.data = None
        self.decimated_data = None
        self.number_of_missing_samples = 0
        self.data_type = None

    def populate(self,station_id,date,data_type):
        self.data_type = data_type
        self.date = date
        self.year = date.year
        self.station_id = station_id
#        print("Loading: "+str(date))
#        gzfilename='E:/usgs_japan/usgs/'+date.strftime('%Y')+'UTC/definitive/second/'+station_id+date.strftime('%Y')+date.strftime('%m')+date.strftime('%d')+'dsec.sec.gz'
#        gzfilename='E:/usgs_japan_post2010/usgs/'+date.strftime('%Y')+'UTC/definitive/second/'+station_id+date.strftime('%Y')+date.strftime('%m')+date.strftime('%d')+'dsec.sec.gz'
        gzfilename=str(root_folder)+str(input_folder)+'/usgs/'+date.strftime('%Y')+'UTC/definitive/second/'+station_id+date.strftime('%Y')+date.strftime('%m')+date.strftime('%d')+'dsec.sec.gz'

#        if os.path.exists(gzfilename):
        secfilename='.'.join(gzfilename.split('.')[:-1])

        if os.path.exists(secfilename):
#            extract_gunzipped_file(gzfilename,secfilename)
            count = len(open(secfilename).readlines(  ))
#            print(count)
            if count > 86419:
                df=pd.read_table(secfilename, header=19, delim_whitespace=True)

                df.eval(station_id.upper()+data_type)[df.eval(station_id.upper()+data_type) == 99999.00]=np.nan
                df.eval(station_id.upper()+data_type)[df.eval(station_id.upper()+data_type) == 88888.00]=np.nan
                self.data=np.asarray(df.eval(station_id.upper()+data_type))

                self.number_of_missing_samples=np.sum(np.isnan(self.data))
            else:
                print('SEC FILE DOES NOT EXIST')
                self.data=np.empty(86400)*np.nan
                self.number_of_missing_samples=86400  
#        else:
#            print('GUNZIPPED FILE DOES NOT EXIST')
#            self.data=np.empty(86400)*np.nan
#            self.number_of_missing_samples=86400
        self.decimated_data=signal.decimate(self.data,5,ftype='iir', axis=-1, zero_phase=True) 
        
    def plot_specgram_of_decimated_data(self, Fs, title='', x_label='', y_label='', fig_size=None): 
        duration=1
        start_time_noise=8
        start_time_clean=17

        
        plt.ioff()
        fig = plt.figure() 
        
        ax0 = fig.add_subplot(3,2,(1,2))
        ax0.set_ylabel('nT',fontsize=20)
        ax0.set_title(title,fontsize=20) 
        ax0.plot(self.decimated_data, color='black')
        ax0.tick_params(labelsize=20)
        ax0.axvline(int(start_time_noise*3600*Fs),color='magenta',linestyle='--')
        ax0.axvline(int((start_time_noise+duration)*3600*Fs),color='magenta',linestyle='--')        
        ax0.axvline(int(start_time_clean*3600*Fs),color='olive',linestyle='--')
        ax0.axvline(int((start_time_clean+duration)*3600*Fs),color='olive',linestyle='--')
        ax0.set_xlim(left=0,right=86400*Fs)
        
        ax1 = fig.add_subplot(3,2,(3,4))
        pxx,  freq, t, cax = plt.specgram(self.decimated_data, Fs=Fs, detrend='mean', cmap='jet', scale='dB')

        ax1.set_ylabel(y_label,fontsize=20)
        ax1.set_ylim([0,0.1])
        ax1.set_xlim([0,t[-1]])
        ax1.set_xticks(np.arange(7)*t[-1]/6) 
        ax1.set_xticklabels(('00:00','04:00','08:00','12:00','16:00','20:00','24:00')) 
        ax1.tick_params(labelsize=20)
        ax1.axvline(int(start_time_noise*3600),color='magenta',linestyle='--')
        ax1.axvline(int((start_time_noise+duration)*3600),color='magenta',linestyle='--')        
        ax1.axvline(int(start_time_clean*3600),color='olive',linestyle='--')
        ax1.axvline(int((start_time_clean+duration)*3600),color='olive',linestyle='--')

#        fig.colorbar(cax).ax.tick_params(labelsize=40)
        
        ax2 = fig.add_subplot(3,2,5)      
        ax2.plot(self.decimated_data[int(start_time_noise*3600*Fs):int((start_time_noise+duration)*3600*Fs)], color='magenta')
        ax2.set_ylabel('nT',fontsize=20)
        ax2.tick_params(labelsize=20)

        ax3 = fig.add_subplot(3,2,6)
        ax3.plot(self.decimated_data[int(start_time_clean*3600*Fs):int((start_time_clean+duration)*3600*Fs)], color='olive')
        ax3.set_ylabel('nT',fontsize=20)
        ax3.tick_params(labelsize=20)
        
#        fig.colorbar(cax).set_label('Intensity (dB)')
#        fig.colorbar(cax).ax.set_label('Intensity [dB]')
        fig.set_size_inches(18.5, 10.5) 
#        if not os.path.exists(str(root_folder)+str(output_folder)+'/'+str(self.station_id)):
#            os.mkdir(str(root_folder)+str(output_folder)+'/'+str(self.station_id))
#        plt.savefig(str(root_folder)+str(output_folder)+'/'+str(self.station_id)+'/SPEGRAM_'+self.station_id+'_'+self.data_type+'_'+self.date.isoformat()[:10]+'.png') 
        plt.savefig(self.date.strftime('%Y%m%d')+'.png')
        plt.close(fig)       


#    def select_data(self, start_time, duration_in_hours):
#        selected_data=rd.data[int(start_time*3600):int((start_time+duration)*3600)]
#        return selected_data
   
     
    def select_decimate_and_detrend_data(self, start_time, duration_in_hours, decimation_level):
        selected_data=rd.data[int(start_time*3600):int((start_time+duration_in_hours)*3600)]
        selected_decimated_and_detrended_data=signal.detrend(signal.decimate(selected_data,decimation_level,ftype='iir', axis=-1, zero_phase=True))
        return selected_decimated_and_detrended_data     
     



class ValidationData(RawData):


    def plot_prediction(self, prediction, prediction_proba, Fs, title='', x_label='', y_label='', fig_size=None): 

         
        classes = {0: 'noisy',
           1: 'clean'}

        
        plt.ioff()
        fig = plt.figure() 
        
        ax0 = fig.add_subplot(2,2,(1,2))
        ax0.set_ylabel('nT',fontsize=20)
        ax0.set_title(title,fontsize=20) 
        ax0.plot(self.data, color='black')
        ax0.tick_params(labelsize=20)
        for wdx in np.arange(48):
             if str(classes[prediction[wdx]])=='noisy':     
                  ax0.axvline(int(wdx*1800*Fs),color='red',linestyle='--', label='test')
                  ax0.axvspan(int(wdx*1800*Fs), int(wdx*1800*Fs+1800*Fs), alpha=0.5, color='red')
                  ax0.text(int(wdx*1800*Fs),np.max(self.decimated_data),str(classes[prediction[wdx]]),rotation=90, color='red')
             else:
                  ax0.axvspan(int(wdx*1800*Fs), int(wdx*1800*Fs+1800*Fs), alpha=0.5, color='blue')
#                  ax0.axvline(int(wdx*1800*Fs),color='black',linestyle='--', label='test')
                  ax0.text(int(wdx*1800*Fs),np.max(self.decimated_data),str(classes[prediction[wdx]]),rotation=90, color='black')
                  
        ax0.set_xlim(left=0,right=86400*Fs)
        ax1 = ax0.twinx()
        ax1.scatter(np.arange(48)*1800*Fs+900*Fs, prediction_proba[:,0], color='black')
        ax1.set_ylabel("Probability",fontsize=20)
        ax1.tick_params(labelsize=20)

        ax2 = fig.add_subplot(2,2,(3,4))
        pxx,  freq, t, cax = plt.specgram(self.decimated_data, Fs=Fs, detrend='mean', cmap='jet', scale='dB')

        ax2.set_ylabel(y_label,fontsize=20)
        ax2.set_ylim([0,0.1])
        ax2.set_xlim([0,t[-1]])
        ax2.set_xticks(np.arange(7)*t[-1]/6) 
        ax2.set_xticklabels(('00:00','04:00','08:00','12:00','16:00','20:00','24:00')) 
        ax2.tick_params(labelsize=20)
        fig.set_size_inches(18.5, 10.5) 
        plt.savefig(self.date.strftime('%Y%m%d')+'.png')
        plt.close(fig)      


         
def plot_selected_data(first_chunk, second_chunk):   
   fig = plt.figure() 
   ax0 = fig.add_subplot(1,2,1)
   ax0.plot(first_chunk, color='magenta')
   ax1 = fig.add_subplot(1,2,2)
   ax1.plot(second_chunk, color='olive')
   fig.set_size_inches(18.5, 5) 
   plt.savefig(str(idx)+'.png') 
      

def plot_selected_time_series(first_chunk, second_chunk):   
   fig = plt.figure() 
   ax0 = fig.add_subplot(1,2,1)
   ax0.plot(first_chunk, color='magenta')
   ax1 = fig.add_subplot(1,2,2)
   ax1.plot(second_chunk, color='olive')
   fig.set_size_inches(18.5, 5) 
   plt.savefig(str(idx)+'_time_series.png')    

def plot_selected_spectra(first_Pxx, second_Pxx):   
   fig = plt.figure() 
   ax0 = fig.add_subplot(1,2,1)
   ax0.plot(first_Pxx[1],np.log(first_Pxx[0]), color='magenta')
   ax1 = fig.add_subplot(1,2,2)
   ax1.plot(second_Pxx[1],np.log(second_Pxx[0]), color='olive')
   fig.set_size_inches(18.5, 5) 
   plt.savefig(str(idx)+'_spectra.png')    
   
   
def show_psd(X, y, idx) :
  classes = {0: 'noisy',
           1: 'clean'}
  colors = {0: 'magenta',
           1: 'olive'}
  plt.figure(figsize=(4,2))
  plt.plot(X[idx,:],color=colors[y[idx]])
  plt.title("This is a {}".format(classes[y[idx]])+" interval")
  plt.show()   

def analyze_validation_data(station_id, validation_date, data_type):
     duration=0.5
     decimation_level=5
     decimation_frequency=1/decimation_level   
     vd=ValidationData()

     vd.populate(station_id,validation_date,data_type)
     for wdx in np.arange(48):
          selected_data=vd.data[int(wdx*1800):int((wdx+1)*1800)]
          validation_data=signal.detrend(signal.decimate(selected_data,decimation_level,ftype='iir', axis=-1, zero_phase=True))
          Pxx, freqs = plt.psd(validation_data, NFFT=1024, Fs=decimation_frequency, detrend='mean',scale_by_freq=True)
          if wdx==0:
              X_val=Pxx[0:100]/np.max(Pxx[0:100])    
          else:
              X_val=np.vstack((X_val,Pxx[0:100]/np.max(Pxx[0:100])))

     X_validation = preprocessing.scale(X_val,0)     
#     X_validation = X_val    

     prediction=clf.predict(X_validation)

     prediction_proba=clf.predict_proba(X_validation)

     vd.plot_prediction(prediction, prediction_proba, Fs=decimation_frequency,title=str(data_type)+'_'+str(validation_date.date()), x_label='UTC (hh:mm)', y_label='Frequency (Hz)')
     classes = {0: 'noisy', 1: 'clean'}
     colors = {0: 'red', 1: 'blue'}
     fig, axes = plt.subplots(8,6)
     fig.subplots_adjust(hspace=1)     
     for ax, wdx in zip(axes.flatten(), np.arange(48)):
          ax.plot(freqs[0:100], X_validation[wdx], colors[prediction[wdx]])
          ax.set(title=','.join((classes[prediction[wdx]],str(wdx))).upper())
     fig.set_size_inches(37, 10) 
     plt.savefig(validation_date.strftime('%Y%m%d')+'_windows.png') 

   
if __name__=='__main__':
     
     if not os.path.isdir('./training_data'):
          os.mkdir('./training_data')
          
     decimation_level=5
     decimated_frequency=1/decimation_level
     delta=end_date-start_date
     number_of_days=delta.days
     strings = [root_folder, input_folder]
     path_to_data=''.join(strings)
     

     mytests=Tests()
     mytests.test_data_folder(path_to_data)

     try:
          clf
     except NameError:
          print("clf model does not exist yet")
          
          for idx in np.arange(number_of_days):
               date=start_date+datetime.timedelta(int(idx))
               print(date)
               rd=RawData() 
               rd.populate(station_id,date,data_type)
               
               noisy_data = rd.select_decimate_and_detrend_data(start_time_noise, duration, decimation_level)
               clean_data = rd.select_decimate_and_detrend_data(start_time_clean, duration, decimation_level)

      
               Nxx=None
               Nxx, freqs=plt.psd(noisy_data, NFFT=1024, Fs=decimated_frequency, detrend='mean',scale_by_freq=True)
               Cxx=None
               Cxx, freqs=plt.psd(clean_data, NFFT=1024, Fs=decimated_frequency, detrend='mean',scale_by_freq=True) 
               
               plt.clf()
               plt.plot(Nxx[0:100]/np.max(Nxx[0:100]), color='red', label='noisy')
               plt.plot(Cxx[0:100]/np.max(Cxx[0:100]), color='blue', label='clean')
               plt.legend()
               plt.savefig('./training_data/'+str(idx)+'.png')
#               

               if idx==0:
                    X_noisy=Nxx[0:100]/np.max(Nxx[0:100])
                    X_clean=Cxx[0:100]/np.max(Cxx[0:100])
               else:
                    X_noisy=np.vstack((X_noisy,Nxx[0:100]/np.max(Nxx[0:100])))
                    X_clean=np.vstack((X_clean,Cxx[0:100]/np.max(Cxx[0:100])))
                    
          indices_to_reject=[0,2,7,8,18,19,22,23,25,28,30,31,37,38,39,43,46,47,49,50,54,58,60,66,70,73,79,82,83,84,87,96,98,99,102,111,113,120,121,122,128,134,135,138,142,147,152,156,157,160,161,162,170,174,176,177,179,184,187,188,190,191,193,194,201,203,204,211,212,216,217,218,220,229,231,235,237,239,244,252,255,257,258,260,261,272,283,292,294,295,298,320,324,328,329,332,335,343,345,352,354,366,369,370,374,375,386,388,389,392,390,398,404,408,411,412,422,426,427,430,435,439,440,453,455,456,459,460,461,462,464,466,467,471,472,473,475,476,477,478,482,487,489,490,496,497,498,500,505,506,509,511,513,514,515,516,519,520,521,528,529,530,536,541,543,545,549,553,558,560,561,564,569,570,573,574,575,576,580,581,582,585,586,587,594,596,604,607,609,613,615,619,620,626,627,634,635,640,642,649,652,654,656,664,668,669,680,683,688,693,695,696,702,704,705,712,713,714,727]
          X_noisy=np.delete(X_noisy, indices_to_reject, 0)
          X_clean=np.delete(X_clean, indices_to_reject, 0)

          X=np.vstack((X_noisy,X_clean))
          y_noisy=np.zeros((X_noisy.shape[0]))
          y_clean=np.ones((X_clean.shape[0]))
          y=np.hstack((y_noisy,y_clean))
          X_scaled = preprocessing.scale(X)
#          X_scaled = X

          X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2)
               

     
          clf = LogisticRegression(C=0.9)
#          clf = svm.SVC(C=0.9)

          clf.fit(X_train, y_train)
          print("Model accuracy: {:.2f}%".format(clf.score(X_test, y_test)*100))
          
          
          station='kak'
          validation_date=datetime.datetime(2008,12,4)
          data_type='Z'
          analyze_validation_data(station, validation_date, data_type)

          
     else:
          station='kak'
          validation_date=datetime.datetime(2008,12,4)
          data_type='Z'
          analyze_validation_data(station, validation_date, data_type)
