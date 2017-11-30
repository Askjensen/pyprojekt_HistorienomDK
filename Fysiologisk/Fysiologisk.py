#!/usr/bin/env python
# coding=utf-8
######## Ask Emil Løvschall-Jensen, July 2016 ## DR Audience Research ##############
# The script reads in datafiles combining GSR (scin-conductance) data and eye-tracking (pupil-dillation) and calculates agregated distributions.
# Functions are defined first in the script and the 'main' afterwards. The CERN statistical frameworks ROOT and spectral analysis algoritms
# from TSpectrum are used to model data. TSpectrum is used to find the phasic and Tonic component of data as well as phasic peak positions.
# Normalised distributions of these are combined.
# The scripts give the following output under the out folder:
#folder:        content:
#peaksEDA       Individual respondent distrubtions of Scin-conductance data for all sequnces and for full overview
#peaksPD        Individual respondent distrubtions of Pupil-dillation data for all sequnces and for full overview
#phasicpeaks    The same plots as in the above folders but from the phasic component of data alone, so peaks are found from phasic data.
#results        numberofpeaks* plots for number of peaks per sequence and tonic and phasic component
#results        timedistributionsof* plots for number of peaks VS time for each sequence in comparison_list and tonic and phasic component
#output.txt     log-file with all outputs from running the script - including som standards parameters for distributions - not very tidy!


## Define which plots to generate - set to True for plots to be made
#Create distributions for each respondent with EDA data
peakseda = False
#Create distributions for each respondent with pupil-dillation data
peakspd = False
#Create distributions for each respondent with normalized phasic component of data
phasic = False
### Create comparision plots for eda and/or PD data for sequences in Comparison_list
rawedapeaks = True
rawpdpeaks = False

#plot mean values for sequences:
meanraw = True

#do further overvieplots:
dooverview = True


import matplotlib.pyplot as plt
import numpy as np
import time
# from bin.BH import *
# ROOT is a statistical tool from CERN - it can be downloaded here: https://root.cern.ch/content/release-53436
# release: root_v5.34.36.win32.vc12.exe (windows release, but available for MAC and most linux distributions)) (requires VC++ 12.0)
# import ROOT
import sys
import pandas as pd

# sys.path.append('/Applications/root_v5.34.36/lib')
from ROOT import TH1F, TSpectrum, TFile, TCanvas, TLegend, TStyle, gStyle, gROOT, TLatex

#plot styling can be defined here:
#import myrootstyle
# -*- coding: utf-8 -*-


#
from matplotlib.font_manager import FontProperties
import os
import pyvttbl as pt  # Repeated measures anova
from collections import namedtuple  # Dependencies of pyvttbl
from scipy.stats import ttest_ind

gStyle.SetOptTitle(0)
gStyle.SetOptStat(0)
# output to file:
# f = open(os.path.join(os.getcwd(), '../out/output.txt'), 'w')
f = open('../out/output.txt', 'w')
# ASSUMPTIONS #
# timesteps of 32ms so 48*32/1000 s = 1,536 s
# 5 * 1000 / 32 = 156,25 ~ 156
# 10 * 1000 / 32 ~ 312
rolling_average_window = 156  # number of data points to include within a rolling avg window (p=0.3 pearson's corr between original EDA data and normalized phasic data)
peaks_window = 60  # 60 bins of 1 sec. = 1 min windows for peak finding

timewindow = 30 # 10 sek bins for peak finding and mean EDA
binscale = 1./1000. #32. / 60000.  # 1 sek/bin
sigmapeaksinterval = 4 # the significance required above background for peaks
peakamplitude = 0.05
endbuffer = 0

#### datafile tag definitions - these match naming scheme of Biometric Software Suite output-files
sync_pos = 'position'
eda_data = 'EDA'
pupil_data = ['PupilLeft','PupilRight']
event_data = 'tag__info_StudioEventData'
delimiter = ';'

## List of events in the data - these are the names defined in tobii and should be given as a list
Events_list = ('Baseline.avi','01_Indledning.avi','02_Absalon_og_Valdemar.avi','03_Absalon_og_Valdemar.avi','04_Absalon_og_Valdemar.avi',
'05_Absalon_og_Valdemar.avi','06_Saxo.avi','07_Ingeborg.avi','08_Ingeborg.avi','09_Ingeborg.avi','10_Ingeborg.avi','11_Valdemar_Sejr.avi',
'12_Valdemar_Sejr.avi','13_Valdemar_Sejr.avi','14_Ærkebispen_af_Lund.avi','15_Erik_Klipping.avi','16_Erik_Klipping.avi','17_Dronning_Agnes_og_Marsk_Stig.avi',
'18_Afslutning.avi')

# A list of sequences to be directly compared
Comparison_list = {'1':['01_Indledning.avi','01_Indledning.avi'],
                   '2':['02_Absalon_og_Valdemar.avi','02_Absalon_og_Valdemar.avi'],
                   '3':['03_Absalon_og_Valdemar.avi','03_Absalon_og_Valdemar.avi'],
                   '4':['04_Absalon_og_Valdemar.avi','04_Absalon_og_Valdemar.avi'],
                   '5':['05_Absalon_og_Valdemar.avi','05_Absalon_og_Valdemar.avi'],
                   '6':['06_Saxo.avi','06_Saxo.avi'],
                   '7':['07_Ingeborg.avi','07_Ingeborg.avi'],
                   '8':['08_Ingeborg.avi','08_Ingeborg.avi'],
                   '9':['17_Dronning_Agnes_og_Marsk_Stig.avi','17_Dronning_Agnes_og_Marsk_Stig.avi'],
                   '10':['18_Afslutning.avi','18_Afslutning.avi']}

#Path to the directory containing datafiles (output ASCII txt-files from Biometric Software Suite)
#folder_path = 'C:/Data/HistorienOmDK/test/'
#folder_path = 'C:/Data/HistorienOmDK/alle/'
folder_path = 'C:/Data/HistorienOmDK/rest/'
filename_ext = '.txt'


## FUNCTIONS ##
def dataextract(files, label):
    fhandle = open(folder_path + files, 'r')

    temp_data = list()
    n = 0

    for p in fhandle:

        p1 = p.split(delimiter)

        if n == 0:
            label_lookup = p1.index(label)

        elif n != 0:

            if p1[label_lookup] != '':
                temp_data.append(float(p1[label_lookup].replace(',', '.')))

            elif p1[label_lookup] == '':
                if(len(temp_data)==0):
                    temp_data.append(0)
                else:
                    temp_data.append(temp_data[len(temp_data) - 1])

        n = n + 1

    fhandle.close()
    return (temp_data)

#create dir if not exist
def ensure_dir(f):
    d = os.path.dirname(f)
    if not os.path.exists(d):
        os.makedirs(d)

# returns number of peaks in a given subset (window) of the data.
# TSpectrum finder (gausiske) peaks ud fra krav om standardafvigelse fra baggrund og et threshold. (peaks med amplitude under threshold*hoejeste_peak forkastes 0<threshold<1).
# TSpectrum.Search(hist=input data, s= antal standardafvigelser (1 sigma), threshold=20%) https://root.cern.ch/doc/master/classTSpectrum.html
def npeaks(dataset_index_sub, dataarray,name):
    nbins = (len(dataset_event_names[0]) / 2)
    peakshist = TH1F("peaksperseq", "peaks per sequence", nbins, 0, nbins)
    #phasichist = TH1F("phasicsperseq", "phasic per sequence", nbins, 0, nbins)
    #tonichist = TH1F("tonicsperseq", "tonic per sequence", nbins, 0, nbins)

    ibin = 0
    labels = sorted(dataset_event_names[0])
    for bin in range(0, nbins * 2):
        if bin % 2 == 0:
            ibin += 1
            peakshist.GetXaxis().SetBinLabel(ibin, labels[bin])
            #phasichist.GetXaxis().SetBinLabel(ibin, labels[bin])
            #tonichist.GetXaxis().SetBinLabel(ibin, labels[bin])

    for respondent in dataset_index_sub:
        respdataarray = dataarray.dropna(axis=0).loc[respondent]

        hist = TH1F(str(respondent) + "hist", str(respondent) + "hist", int((respdataarray.index[-1]*binscale)),
                    respdataarray.index[0], respdataarray.index[-1])

        for i in range(0, len(respdataarray) - 1):
            hist.Fill(respdataarray.index[i], respdataarray.values[i])
        orig_integral = hist.Integral()
        # loop over series in timeunit of size: peaks_window
        #hist.GetXaxis().SetTitle('Time-position [ms]')
        #hist.GetYaxis().SetTitle('EDA')
        # for sigma in range(1,5):
        #     for thres in [0.05]:  # [0.01,0.05,0.10]:
        #         for res in [1]:  # range(1,5)
        #             s = TSpectrum()
        #             s.SetResolution(res)
        #             cvs = TCanvas("c", "c", 1200, 800)
        #             cvs.cd()
        #             hist.Draw()
        #             cvs.Update()
        #             hist.Write()
        #             s.Search(hist, sigma, "noMarkov nodraw", thres)
        #             cvs.Update()
        #             bg = s.Background(hist, 20, "Compton nodraw")
        #             currentsettings = respondent + 'sigma' + str(sigma) + 'thres' + str(thres) + 'res' + str(res)
        #             cvs.Update()
        #             cvs.SaveAs(os.path.join(os.getcwd(), '../out/peaks/noMarkov' + currentsettings + '.png'))
        #             cvs.Close()
        # 
        #             cmark = TCanvas("c", "c", 1200, 800)
        #             cmark.cd()
        #             hist.Draw()
        #             s.Search(hist, sigma, "nobackground nodraw", thres)
        #             cmark.Update()
        #             bg = s.Background(hist, 20, "nodraw nosmoothing")
        #             cmark.Update()
        #             cmark.SaveAs(os.path.join(os.getcwd(), '../out/peaks/nobackground_' + currentsettings + '.png'))
        #             cmark.Close()

        ## First define peaks for full sequence from baseline untill last sequence
        hist.GetXaxis().SetRangeUser(EventBinsPos[respondent][0] + (1 / binscale),
                                     EventBinsPos[respondent][len(EventBinsPos[respondent])-1] - (1 / binscale))
        # hist.Scale(1./hist.Integral(hist.GetXaxis().FindBin(EventBinsPos[respondent][clipindex]),
        # hist.GetXaxis().FindBin(EventBinsPos[respondent][clipindex + 1])))
        cmark = TCanvas("c", "c", 1200, 800)
        cmark.cd()
        hist.Draw()
        s = TSpectrum()
        np = s.Search(hist, sigmapeaksinterval, "noMarkov same", peakamplitude)
        bg = s.Background(hist, 20, "Compton same")
        cmark.Update()
        dir = os.path.join(os.getcwd(), '../out/peaks' + name + '/overview/')
        ensure_dir(dir)
        cmark.SaveAs(dir + 'OverviewOf'+name+'Peaks_' + respondent + '.png')
        cmark.Close()

        ## then for individual "clips"
        for clipindex in range(0, len(EventBinsPos[respondent]) - 1):
            if (clipindex % 2 == 0):

                hist.GetXaxis().SetRangeUser(EventBinsPos[respondent][clipindex]+(1/binscale),
                                         EventBinsPos[respondent][clipindex + 1]-(1/binscale))
                #hist.Scale(1./hist.Integral(hist.GetXaxis().FindBin(EventBinsPos[respondent][clipindex]),
                                            #hist.GetXaxis().FindBin(EventBinsPos[respondent][clipindex + 1])))
                cmark = TCanvas("c", "c", 1200, 800)
                cmark.cd()
                hist.Draw()
                s = TSpectrum()
                np = s.Search(hist, sigmapeaksinterval, "noMarkov same", peakamplitude)
                bg = s.Background(hist, 20, "Compton same")
                cmark.Update()
                dir = os.path.join(os.getcwd(), '../out/peaks'+name+'/'+EventBinsNames[respondent][clipindex]+'/')
                ensure_dir(dir)
                cmark.SaveAs(dir+'sequencepeaks_'+ respondent + '.png')
                cmark.Close()
                #test = EventBinsNames[respondent][clipindex]
                peakshist.Fill(EventBinsNames[respondent][clipindex], np)
                #phasichist.Fill(EventBinsNames[respondent][clipindex], np)
                #tonichist.Fill(EventBinsNames[respondent][clipindex], np)
                #hist.Scale(orig_integral/hist.Integral())
        # while (j < nbins):
        #     # create windows of peaks_window size (60 sec)
        #     hist.GetXaxis().SetRange(j, j + peaks_window)
        #     # values found to optimal from above plots
        #     peaskpertime1sig.append(s.Search(hist, 4, "noMarkov nodraw", peakamplitude))
        #     j += peaks_window
        #     indices.append(respondent)
        # values1.extend(peaskpertime1sig)
        hist.Delete()
    return peakshist
    # return pd.Series(values1, index=indices)

def meaneda(dataset_index_sub, dataarray):
    nbins = (len(dataset_event_names[0]) / 2)
    values1 = TH1F("meanEDAperseq", "#mu_{EDA} per sequence", nbins, 0, nbins)
    values1.SetStats(False)
    ibin = 0
    labels = sorted(dataset_event_names[0])
    for bin in range(0, len(dataset_event_names[0])):
        if bin % 2 == 0:
            ibin += 1
            values1.GetXaxis().SetBinLabel(ibin, labels[bin])

    for respondent in dataset_index_sub:
        respdataarray = dataarray.dropna(axis=0).loc[respondent]
        # loop over series in timeunit of size: peaks_window
        ## First define peaks per minute for full sequence
        for clipindex in range(0, len(EventBinsPos[respondent]) - 1):
            if (clipindex % 2 == 0):
                hist = TH1F(str(respondent) + str(clipindex) +"hist", str(respondent) + str(clipindex) + "hist", int(len(respdataarray.values)),respdataarray.values.min(), respdataarray.values.max())
                for i in range(0,len(respdataarray.values)):
                    if respdataarray.index[i] > EventBinsPos[respondent][clipindex]:
                        if respdataarray.index[i] <= EventBinsPos[respondent][clipindex+1]:
                            hist.Fill(respdataarray.values[i])
                meanfullrange = hist.GetMean()
                values1.Fill(EventBinsNames[respondent][clipindex], meanfullrange)
                hist.Delete()
    return values1

# def meaninterval2(dataset_index_sub, dataarray,timewindow):
#     values1 = TH1F("meanedapertime", "mean eda per respondent", 12000/timewindow, 0, 120000)
#     values1.SetStats(False)
#     for respondent in dataset_index_sub:
#         respdataarray = dataarray.dropna(axis=0).loc[respondent]
#         # loop over series in timeunit of size: peaks_window
#         ## First define peaks per minute for full sequence
#         for clipindex in range(0, len(EventBinsPos[respondent]) - 1):
#             if (clipindex % 2 == 0):
#                 a=EventBinsPos[respondent][clipindex]*binscale
#                 b=EventBinsPos[respondent][clipindex+1]*binscale
#                 for interval in range(int(round(a/timewindow)),int(round(b/timewindow))):
#                     hist = TH1F(str(respondent) + clipindex + "hist", str(respondent) + "hist",int(len(respdataarray.values)),respdataarray.values.min(),respdataarray.values.max())
#                     for i in range(0,len(respdataarray.values)):
#                         if respdataarray.index[i]*binscale > a+interval*timewindow:
#                             if respdataarray.index[i]*binscale <= a+interval*timewindow+timewindow:
#                                 hist.Fill(respdataarray.values[i])
#                     meaninterval = hist.GetMean()
#                     values1.Fill(a+interval*timewindow, meaninterval)
#                     hist.Delete()
#     return values1

# saves a histogram with the events specified in comparisonlist plotted
def meaninterval(dataset_index_sub, dataarray, comparisonlist,timewindow,name):
    timeinterval=12000 # total of 12000/60 = 200 minutes now - corrected later on
    for ikey in comparisonlist.keys():
        canvas = TCanvas("c", "mead EDA per respondent", 1200, 800)
        canvas.cd()
        leg = TLegend(0.5, 0.67, 0.88, 0.88)
        ievent = 0
        histogram = TH1F("meanedapertime", "meaneda per respondent", (timeinterval+timewindow)/timewindow, 0, timeinterval)
        histogram.GetYaxis().SetTitle('Mean EDA of ' + name + 'data  / '+ str(timewindow) + ' sec.')
        histogram.GetXaxis().SetTitle('Time [s]')
        histogram.GetYaxis().SetTitleOffset(1.4)
        histogram.SetStats(False)
        histogram.Draw()
        canvas.Update()
        histarray = []
        maxvalx = 0
        maxvaly = 0
        for event in comparisonlist[ikey]:
            tmphist = TH1F("peakspermin" + event, event, (timeinterval+timewindow)/timewindow, 0, timeinterval)
            tmphist.SetStats(False)
            tmphist.SetFillStyle(3002)
            histarray.append(tmphist)

        for event in comparisonlist[ikey]:
            for respondent in dataset_index_sub:
                respdataarray = dataarray.dropna(axis=0).loc[respondent]

                for clipindex in range(0, len(EventBinsNames[respondent]) - 1):
                    if EventBinsNames[respondent][clipindex] != event: continue
                    if (clipindex % 2 == 0):
                        a = EventBinsPos[respondent][clipindex] * binscale
                        b = EventBinsPos[respondent][clipindex + 1] * binscale
                        for interval in range(0, int(round( (b-a)/ timewindow))):
                            hist = TH1F(str(respondent) + str(clipindex) + "hist", str(respondent) + "hist",
                                        int(len(respdataarray.values)), respdataarray.values.min(),
                                        respdataarray.values.max())
                            for i in range(0, len(respdataarray.values)):
                                if respdataarray.index[i] * binscale > a + interval * timewindow:
                                    if respdataarray.index[i] * binscale <= a + interval * timewindow + timewindow:
                                        hist.Fill(respdataarray.values[i])
                            meaninterval = hist.GetMean()
                            histarray[ievent].Fill(interval * timewindow, meaninterval)
                            if (interval * timewindow) > maxvalx: maxvalx = interval * timewindow
            histarray[ievent].SetLineColor(ievent+1)
            histarray[ievent].SetFillColor(ievent+1)
            histarray[ievent].SetFillStyle(3003+ievent)
            leg.AddEntry(histarray[ievent], event, "l")
            canvas.Update()
            if(histarray[ievent].GetMaximum()>maxvaly): maxvaly = histarray[ievent].GetMaximum()
            ievent += 1
        canvas.cd()
        histogram.GetYaxis().SetRangeUser(0, maxvaly*1.3)
        histogram.GetXaxis().SetRangeUser(0, maxvalx)
        histogram.Draw()
        for ievent in range(0,len(comparisonlist[ikey])):
            histarray[ievent].Draw("same")
        leg.Draw()
        canvas.Update()
        histogram.SaveAs('../out/rootfiles/compinterval'+name+ikey+'.root')
        canvas.SaveAs('../out/results/compinterval'+name+ikey+'.png')
        canvas.Close()

# saves a histogram with the events specified in  plotted
def npeaksspecific(dataset_index_sub, dataarray, comparisonlist,name):
    for ikey in comparisonlist.keys():
        canvas = TCanvas("c", "Count of peaks per respondent", 1200, 800)
        canvas.cd()
        leg = TLegend(0.5, 0.67, 0.88, 0.88)
        ievent = 0
        histogram = TH1F("peaksperres", "peaks per respondent", 50, 0, 50)
        histogram.GetXaxis().SetTitle('Number of peaks in sequence')
        histogram.GetYaxis().SetTitle('Count')
        histogram.GetYaxis().SetTitleOffset(1.4)
        histogram.SetStats(False)
        histogram.Draw()
        canvas.Update()
        histarray = []
        maxval = 0
        for event in comparisonlist[ikey]:
            tmphist = TH1F("peaksperres" + event, event, 50, 0, 50)
            tmphist.GetXaxis().SetTitle('Number of peaks in sequence')
            tmphist.GetYaxis().SetTitle('Count')
            tmphist.SetStats(False)
            tmphist.SetFillStyle(3002)
            histarray.append(tmphist)

        for event in comparisonlist[ikey]:
            for respondent in dataset_index_sub:
                try:
                    test = dataarray.dropna(axis=0).loc[respondent]
                except Exception:
                    print >> f, '\n failed with respondent: '+respondent+' and name/key: '+ name+'/'+ikey
                    pass
                if(event == '01_Indledning.avi'):
                    test1 = len(dataarray.dropna(axis=0).loc[respondent])
                respdataarray = dataarray.dropna(axis=0).loc[respondent]
                hist = TH1F(str(respondent) + "hist", str(respondent) + "hist",
                            int((respdataarray.index[-1]-respdataarray.index[0])*binscale),
                            respdataarray.index[0], respdataarray.index[-1])
                for i in range(0, len(respdataarray) - 1):
                    hist.Fill(respdataarray.index[i], respdataarray.values[i])
                for clipindex in range(0, len(EventBinsPos[respondent]) - 1):
                    if EventBinsNames[respondent][clipindex] != event: continue
                    if (clipindex % 2 == 0):
                        s = TSpectrum()
                        #canvas1 = TCanvas("c1", "Count of peaks per respondent", 1200, 800)
                        #canvas1.cd()
                        hist.GetXaxis().SetRangeUser(EventBinsPos[respondent][clipindex],EventBinsPos[respondent][clipindex + 1])
                        #hist.Draw()
                        #canvas1.Update()
                        np = s.Search(hist, sigmapeaksinterval, "noMarkov same", peakamplitude)
                        histarray[ievent].Fill(np)
                        #canvas1.Delete()
            canvas.cd()
            histarray[ievent].Draw("same")
            histarray[ievent].SetLineColor(ievent+1)
            histarray[ievent].SetFillColor(ievent+1)
            histarray[ievent].SetFillStyle(3003+ievent)
            leg.AddEntry(histarray[ievent], event, "l")
            canvas.Update()
            if(histarray[ievent].GetMaximum()>maxval): maxval = histarray[ievent].GetMaximum()
            ievent += 1
        canvas.cd()

        histogram.GetYaxis().SetRangeUser(0, maxval)
        canvas.Update()
        #histogram.Write()
        leg.Draw()
        #canvas.Draw()
        # print >> f, "\n KS probability including normalisation of " + str(comparisonlist[ikey][0]) + " and " + str(
        #     comparisonlist[ikey][1])
        # print >> f, histarray[0].KolmogorovTest(histarray[1], "OUN")
        # print >> f, "\n KS Comparison of " + str(comparisonlist[ikey][0]) + " and " + str(comparisonlist[ikey][1])
        # print >> f, histarray[0].KolmogorovTest(histarray[1], "OU")
        # print >> f, "\n Chi2 probability Comparison of " + str(comparisonlist[ikey][0]) + " and " + str(
        #     comparisonlist[ikey][1])
        # print >> f, histarray[0].Chi2Test(histarray[1], "UU")
        # print >> f, "\n mean " + str(comparisonlist[ikey][0]) + ": " + str(histarray[0].GetMean()) + "standard deviation: " + str(histarray[0].GetRMS())
        # print >> f, "mean " + str(comparisonlist[ikey][1]) + ": " + str(histarray[1].GetMean()) + "standard deviation: " + str(histarray[1].GetRMS())
        canvas.SaveAs('../out/rootfiles/numberof'+name+'inseq'+ikey+'.root')
        histogram.SaveAs('../out/rootfiles/numberof'+name+'inseq'+ikey+'.root')
        canvas.SaveAs('../out/results/numberof'+name+'inseq'+ikey+'.png')
        histogram.Delete()
        canvas.Close()

# saves a histogram with the events specified in comparisonlist plotted
def npeaksspecificminutes(dataset_index_sub, dataarray, comparisonlist,timewindow,name):
    timeinterval=12000 # total of 12000/60 = 200 minutes now - corrected later on
    histtonicfull = TH1F("tonicsperminfull", "Tonic Component", (timeinterval + timewindow) / timewindow, 0,
                          timeinterval)
    histtonicfull.GetYaxis().SetTitle('Arousal  / ' + str(timewindow) + ' sec.')
    histtonicfull.GetXaxis().SetTitle('Time [s]')
    histtonicfull.GetYaxis().SetTitleOffset(1.4)
    histtonicfull.SetStats(False)
    histphasicfull = TH1F("phasicperminfull", "Phasic component", (timeinterval + timewindow) / timewindow, 0,
                         timeinterval)
    histphasicfull.GetYaxis().SetTitle('Arousal  / ' + str(timewindow) + ' sec.')
    histphasicfull.GetXaxis().SetTitle('Time [s]')
    histphasicfull.GetYaxis().SetTitleOffset(1.4)
    histphasicfull.SetStats(False)
    histpeaksfull = TH1F("peaksperminfull", "Peaks in arousel", (timeinterval + timewindow) / timewindow, 0,
                          timeinterval)
    histpeaksfull.GetYaxis().SetTitle('Number of peaks  / ' + str(timewindow) + ' sec.')
    histpeaksfull.GetXaxis().SetTitle('Time [s]')
    histpeaksfull.GetYaxis().SetTitleOffset(1.4)
    histpeaksfull.SetStats(False)

    for ikey in comparisonlist.keys():
        canvas = TCanvas("c", "Count of peaks per respondent", 1200, 800)
        canvas.cd()
        leg = TLegend(0.5, 0.67, 0.88, 0.88)
        ievent = 0
        histpeaks = TH1F("peakspermin", "peaks per respondent", (timeinterval+timewindow)/timewindow, 0, timeinterval)
        histpeaks.GetYaxis().SetTitle('Number of peaks  / '+ str(timewindow) + ' sec.')
        histpeaks.GetXaxis().SetTitle('Time [s]')
        histpeaks.GetYaxis().SetTitleOffset(1.4)
        histpeaks.SetStats(False)
        histphasic = TH1F("phasicspermin", "Phasic per respondent", (timeinterval + timewindow) / timewindow, 0,
                         timeinterval)
        histphasic.GetYaxis().SetTitle('Phasic component  / ' + str(timewindow) + ' sec.')
        histphasic.GetXaxis().SetTitle('Time [s]')
        histphasic.GetYaxis().SetTitleOffset(1.4)
        histphasic.SetStats(False)
        histtonic = TH1F("tonicspermin", "Tonic per respondent", (timeinterval + timewindow) / timewindow, 0,
                          timeinterval)
        histtonic.GetYaxis().SetTitle('Tonic component  / ' + str(timewindow) + ' sec.')
        histtonic.GetXaxis().SetTitle('Time [s]')
        histtonic.GetYaxis().SetTitleOffset(1.4)
        histtonic.SetStats(False)
        peakarray = []
        phasicarray = []
        tonicarray = []
        maxvalx = 0
        maxvaly = 0
        for event in comparisonlist[ikey]:
            tmphist = TH1F("peakspermin" + event, event, (timeinterval+timewindow)/timewindow, 0, timeinterval)
            tmphist.GetXaxis().SetTitle('Time [s]')
            tmphist.GetYaxis().SetTitle('Number of peaks / '+ str(timewindow) + ' sec.')
            tmphist.SetStats(False)
            peakarray.append(tmphist)
            ph = TH1F("ph" + event, event, (timeinterval + timewindow) / timewindow, 0, timeinterval)
            phasicarray.append(ph)
            th = TH1F("th" + event, event, (timeinterval + timewindow) / timewindow, 0, timeinterval)
            tonicarray.append(th)

        for event in comparisonlist[ikey]:
            for respondent in dataset_index_sub:
                respdataarray = dataarray.dropna(axis=0).loc[respondent]

                for clipindex in range(0, len(EventBinsNames[respondent]) - 1):
                    if EventBinsNames[respondent][clipindex] != event: continue
                    if (clipindex % 2 == 0):
                        canvas1 = TCanvas("c1", "Count of peaks per respondent", 1200, 800)
                        canvas1.cd()
                        s = TSpectrum()
                        hist = TH1F("hist" + respondent+event, "hist" + respondent + event,
                                    int(len(respdataarray.index)/20),
                                    respdataarray.index[0], respdataarray.index[-1])
                        a = EventBinsPos[respondent][clipindex] + (1/binscale)
                        b = EventBinsPos[respondent][clipindex + 1] - (1/binscale)
                        nbins = (hist.FindBin(b)-hist.FindBin(a))
                        #phasic = TH1F("histphasic" + respondent + event, "hist" + respondent + event,0,b-a, nbins)
                        for i in range(0, len(respdataarray) - 1):
                            hist.Fill(respdataarray.index[i], respdataarray.values[i])
                            #if respdataarray.index[i]>= a:
                                #if respdataarray.index[i]<=b:
                                    #phasic.Fill(respdataarray.index[i], respdataarray.values[i])
                        hist.GetXaxis().SetRangeUser(a, b)
                        if(hist.Integral(hist.FindBin(a), hist.FindBin(b)!=0)):
                            hist.Scale(1./hist.Integral(hist.FindBin(a), hist.FindBin(b)))
                        #phasic.GetXaxis().SetRangeUser(a, b)
                        hist.Draw()
                        canvas1.Update()
                        npeaks = s.Search(hist, sigmapeaksinterval, "noMarkov same", peakamplitude)
                        bg = s.Background(hist, 20, "Compton same")
                        canvas1.Update()
                        hist.Add(bg,-1)
                        for x in range(0,npeaks):
                            peakx = s.GetPositionX()[x]-a
                            peakarray[ievent].Fill(peakx*binscale,npeaks)
                            if((b-a)*binscale>maxvalx): maxvalx = (b-a)*binscale
                        for ibin in range(bg.FindBin(a),bg.FindBin(b)):
                            binx = bg.GetBinCenter(ibin)-a
                            tonicarray[ievent].Fill(binx*binscale,bg.GetBinContent(ibin))
                            phasicarray[ievent].Fill(binx*binscale,hist.GetBinContent(ibin))
            canvas.cd()
            peakarray[ievent].SetLineColor(ievent+1)
            peakarray[ievent].SetFillColor(ievent+1)
            peakarray[ievent].SetFillStyle(3003+ievent)
            tonicarray[ievent].SetLineColor(ievent + 1)
            tonicarray[ievent].SetFillColor(ievent + 1)
            tonicarray[ievent].SetFillStyle(3003 + ievent)
            phasicarray[ievent].SetLineColor(ievent + 1)
            phasicarray[ievent].SetFillColor(ievent + 1)
            phasicarray[ievent].SetFillStyle(3003 + ievent)

            leg.AddEntry(peakarray[ievent], event, "l")
            canvas.Update()
            if(peakarray[ievent].GetMaximum()>maxvaly): maxvaly = peakarray[ievent].GetMaximum()
            ievent += 1
        canvas.cd()
        histpeaks.GetYaxis().SetRangeUser(0, maxvaly*1.3)
        histpeaks.GetXaxis().SetRangeUser(0, maxvalx)
        histpeaks.Draw()
        for ievent in range(len(comparisonlist[ikey])-1,-1,-1):
            peakarray[ievent].Draw("same")
        leg.Draw()
        canvas.Update()
        canvas.SaveAs('../out/rootfiles/timedistributionof' + name + 'inseq'+ikey + '.root')
        canvas.SaveAs('../out/results/timedistributionof' + name + 'inseq'+ikey + '.png')

        histphasic.GetXaxis().SetRangeUser(0, maxvalx)
        histphasic.Draw()
        maxvaly=0
        for ievent in range(len(comparisonlist[ikey])-1,-1,-1):
            if maxvaly <= phasicarray[ievent].GetMaximum():
                    maxvaly = phasicarray[ievent].GetMaximum()
            phasicarray[ievent].Draw("same")
        histphasic.GetYaxis().SetRangeUser(0, maxvaly*1.3)
        leg.Draw()
        canvas.Update()
        canvas.SaveAs('../out/rootfiles/timedistributionofphasic' + name + 'inseq'+ikey + '.root')
        canvas.SaveAs('../out/results/timedistributionofphasic' + name + 'inseq'+ikey + '.png')

        maxvaly=0
        histtonic.GetXaxis().SetRangeUser(0, maxvalx)
        histtonic.Draw()
        for ievent in range(len(comparisonlist[ikey])-1,-1,-1):
            if maxvaly <= tonicarray[ievent].GetMaximum():
                maxvaly = tonicarray[ievent].GetMaximum()
            tonicarray[ievent].Draw("same")
        histtonic.GetYaxis().SetRangeUser(0, maxvaly*1.3)
        leg.Draw()
        canvas.Update()
        canvas.SaveAs('../out/rootfiles/timedistributionoftonic' + name + 'inseq'+ikey + '.root')
        canvas.SaveAs('../out/results/timedistributionoftonic' + name + 'inseq'+ikey + '.png')
        # print >> f, "\n KS probability including normalisation of " + str(comparisonlist[ikey][0]) + " and " + str(comparisonlist[ikey][1])
        # print >> f, histarray[0].KolmogorovTest(histarray[1], "OUN")
        # print >> f, "\n KS Comparison of " + str(comparisonlist[ikey][0]) + " and " + str(comparisonlist[ikey][1])
        # print >> f, histarray[0].KolmogorovTest(histarray[1], "OU")
        # print >> f,"\n Chi2 probability Comparison of " + str(comparisonlist[ikey][0]) + " and " + str(comparisonlist[ikey][1])
        # print >> f, histarray[0].Chi2Test(histarray[1],"UU")
        # print >> f, "\n mean " + str(comparisonlist[ikey][0])+": " + str(histarray[0].GetMean())
        # print >> f, "mean " + str(comparisonlist[ikey][1])+": " + str(histarray[0].GetMean())
        canvas.Close()


def createplotsFullRange(dataset_index_sub, dataarray,timewindow,name):
    timeinterval=12000 # total of 12000/60 = 200 minutes now - corrected later on
    histtonicfull = TH1F("tonicsperminfull", "Tonic per respondent", (timeinterval + timewindow) / timewindow, 0,
                          timeinterval)
    histtonicfull.GetYaxis().SetTitle('Tonic component  / ' + str(timewindow) + ' sec.')
    histtonicfull.GetXaxis().SetTitle('Time [s]')
    histtonicfull.GetYaxis().SetTitleOffset(1.4)
    histtonicfull.SetStats(False)
    histphasicfull = TH1F("tonicsperminfull", "Tonic per respondent", (timeinterval + timewindow) / timewindow, 0,
                         timeinterval)
    histphasicfull.GetYaxis().SetTitle('Tonic component  / ' + str(timewindow) + ' sec.')
    histphasicfull.GetXaxis().SetTitle('Time [s]')
    histphasicfull.GetYaxis().SetTitleOffset(1.4)
    histphasicfull.SetStats(False)
    histpeaksfull = TH1F("tonicsperminfull", "Tonic per respondent", (timeinterval + timewindow) / timewindow, 0,
                          timeinterval)
    histpeaksfull.GetYaxis().SetTitle('Tonic component  / ' + str(timewindow) + ' sec.')
    histpeaksfull.GetXaxis().SetTitle('Time [s]')
    histpeaksfull.GetYaxis().SetTitleOffset(1.4)
    histpeaksfull.SetStats(False)
    canvas = TCanvas("c", "Sum over respondents", 1200, 800)
    maxvalx = 0
    maxvaly = 0
    s = TSpectrum()
    for respondent in dataset_index_sub:
        canvas1 = TCanvas("c1", "Count of peaks per respondent", 1200, 800)
        canvas1.cd()
        respdataarray = dataarray.dropna(axis=0).loc[respondent]
        hist = TH1F("hist" + respondent, "hist" + respondent,
                                    int(len(respdataarray.index)/20),
                                    respdataarray.index[0], respdataarray.index[-1])
        a = EventBinsPos[respondent][0] + (1/binscale)
        b = EventBinsPos[respondent][len(EventBinsPos[respondent])-1] - (1/binscale)
        for i in range(0, len(respdataarray) - 1):
            hist.Fill(respdataarray.index[i], respdataarray.values[i])
        hist.GetXaxis().SetRangeUser(a, b)
        hist.Scale(1./hist.Integral())
        hist.Draw()
        canvas1.Update()
        npeaks = s.Search(hist, sigmapeaksinterval, "noMarkov same", peakamplitude)
        bg = s.Background(hist, 20, "Compton same")
        canvas1.Update()
        hist.Add(bg,-1)
        for x in range(0,npeaks):
            peakx = s.GetPositionX()[x]-a
            histpeaksfull.Fill(peakx*binscale,npeaks)
            if((b-a)*binscale>maxvalx): maxvalx = (b-a)*binscale
        for ibin in range(bg.FindBin(a),bg.FindBin(b)):
            binx = bg.GetBinCenter(ibin)-a
            histtonicfull.Fill(binx*binscale,bg.GetBinContent(ibin))
            histphasicfull.Fill(binx*binscale,hist.GetBinContent(ibin))
    #canvas.cd()
    # histpeaksfull.SetLineColor(1)
    # histpeaksfull.SetFillColor(1)
    # histpeaksfull.SetFillStyle(3003)
    # histtonicfull.SetLineColor(1)
    # histtonicfull.SetFillColor(1)
    # histtonicfull.SetFillStyle(3003)
    # histphasicfull.SetLineColor(1)
    # histphasicfull.SetFillColor(1)
    # histphasicfull.SetFillStyle(3003)
    #
    # leg.AddEntry(histpeaksfull, 1, "l")
    #canvas.Update()
    if(histpeaksfull.GetMaximum()>maxvaly): maxvaly = histpeaksfull.GetMaximum()
    canvas.cd()
    histpeaksfull.GetYaxis().SetRangeUser(0, maxvaly*1.3)
    histpeaksfull.GetXaxis().SetRangeUser(0, maxvalx)
    histpeaksfull.Draw()
    #leg.Draw()
    canvas.Update()
    histpeaksfull.SaveAs('../out/rootfiles/timedistributionof' + name + 'overview' + '.root')
    canvas.SaveAs('../out/results/timedistributionof_' + name + 'overview' + '.png')

    histphasicfull.GetXaxis().SetRangeUser(0, maxvalx)
    maxvaly=histphasicfull.GetMaximum()
    histphasicfull.GetYaxis().SetRangeUser(0, maxvaly*1.3)
    histphasicfull.Draw()
    #leg.Draw()
    canvas.Update()
    histphasicfull.SaveAs('../out/rootfiles/timedistributionofphasic' + name + 'overview' + '.root')
    canvas.SaveAs('../out/results/timedistributionof_phasic' + name + 'overview' + '.png')

    maxvaly=histtonicfull.GetMaximum()
    histtonicfull.GetXaxis().SetRangeUser(0, maxvalx)
    histtonicfull.GetYaxis().SetRangeUser(0, maxvaly*1.3)
    histtonicfull.Draw()
    #leg.Draw()
    canvas.Update()
    histtonicfull.SaveAs('../out/rootfiles/timedistributionoftonic' + name + 'overview' + '.root')
    canvas.SaveAs('../out/results/timedistributionof_tonic' + name + 'overview' + '.png')
    canvas.Close()

def phasic_component(dataset_index_sub, dataarray, window):
    values = list()

    for p in dataset_index_sub:
        tonic_temp = (pd.rolling_mean(dataarray.loc[p], window))

        returndata1 = abs(((pd.Series.subtract(tonic_temp, dataarray.loc[p])) * (
            -1.0)))  # ABS: Negative peaks transformed into positive peaks

        values.extend(returndata1.tolist())

    returndata = pd.Series(values, index=dataarray.index)

    return (returndata)


def tonic_component(dataset_index_sub, dataarray, window):
    values = list()

    for p in dataset_index_sub:
        tonic_temp = (pd.rolling_mean(dataarray.loc[p], window))
        values.extend(tonic_temp)

    return pd.Series(values, index=dataarray.index)


def normalize_series(dataset_index_sub, dataseries):
    values = list()

    for p in dataset_index_sub:
        dataseries1 = (
            (dataseries.loc[p] - dataseries.loc[p].min()) / (dataseries.loc[p].max() - dataseries.loc[p].min()))
        values.extend(dataseries1.tolist())

    return pd.Series(values, index=dataseries.index)


def event_hz_pointers(files, events, event_data_sub, sync_pos_sub):
    fhandle = open(folder_path + files, 'r')

    n = 0
    event_temp_list = list()
    event_hz_markers = list()
    previous = ""

    for p in fhandle:

        p1 = p.split(delimiter)
        value_to_add = 0
        if n == 0:
            label_lookup_event = p1.index(event_data_sub)
            label_lookup_position = p1.index(sync_pos_sub)

        if n != 0:
            if len(p1) < label_lookup_event:
                print >> f, "Error. Unknown error occured when slicing events from dataset"
                print >> f, "A discrepancy between lenght of dataset and event position. Event position exceeded the lenght of dataset."
                print >> f, "Lenght of dataset: ", len(p1)
                print >> f, "Event label position: ", label_lookup_event
                print >> f, "track back error halts script from continuing."
            #print len(p1)
            #print files
            if p1[label_lookup_event] in events:
                event_temp_list.append(p1[label_lookup_event])

                try:
                    value_to_add = int((p1[label_lookup_position]))
                except:
                    print >> f, "Error. Blank cell or string data detected where integer data was expected."
                    print >> f, "This will result in failure to slice events. Dataset should be corrected before continuing"
                    print >> f, "Stop script execution and remove or correct dataset."
                #Add a buffer to the end of the sequence to catch EDA reactions with delay.
                if(p1[label_lookup_event]==previous):
                    event_hz_markers.append(value_to_add+endbuffer)
                else:
                    event_hz_markers.append(value_to_add)
                previous = p1[label_lookup_event]
        n = n + 1

    fhandle.close()

    event_temp_list1, event_hz_markers = (list(x) for x in
                                         zip(*sorted(zip(event_temp_list, event_hz_markers), key=lambda pair: pair[0])))

    return (event_temp_list, event_hz_markers)


def eventmeans(name, sub_event_names, sub_event_datapoints, df):
    temp_mean = list()
    temp_std = list()

    number_of_events = len(sub_event_names)
    n = 0

    while n < number_of_events:
        get_slice_start = int(sub_event_datapoints[n])
        get_slice_stop = int(sub_event_datapoints[n + 1])

        event_arousal_mean = df.loc[name].loc[get_slice_start:get_slice_stop].mean()
        event_arousal_std = df.loc[name].loc[get_slice_start:get_slice_stop].std()

        temp_mean.append(event_arousal_mean.values)
        temp_std.append(event_arousal_std.values)

        n = n + 2

    return (temp_mean, temp_std)


def suplabel(axis, label, label_prop=None,
             labelpad=5,
             ha='center', va='center'):
    ''' Add super ylabel or xlabel to the figure
    Similar to matplotlib.suptitle
    axis       - string: "x" or "y"
    label      - string
    label_prop - keyword dictionary for Text
    labelpad   - padding from the axis (default: 5)
    ha         - horizontal alignment (default: "center")
    va         - vertical alignment (default: "center")
    '''
    fig = plt.gcf()
    xmin = []
    ymin = []
    for ax in fig.axes:
        xmin.append(ax.get_position().xmin)
        ymin.append(ax.get_position().ymin)
    xmin, ymin = min(xmin), min(ymin)
    dpi = fig.dpi
    if axis.lower() == "y":
        rotation = 90.
        x = xmin - float(labelpad) / dpi
        y = 0.5
    elif axis.lower() == 'x':
        rotation = 0.
        x = 0.5
        y = ymin - float(labelpad) / dpi
    else:
        raise Exception("Unexpected axis: x or y")
    if label_prop is None:
        label_prop = dict()
    plt.text(x, y, label, rotation=rotation,
             transform=fig.transFigure,
             ha=ha, va=va,
             **label_prop)


### PROGRAM ###

print >> f, "Script to perform automated data analysis."
print >> f, "Jacob Lyng Wieland, November/December 2015 - rewritten by Ask E. Loevschall-Jensen 2016"
print >> f, "---------------------------------------------------------------------------"
print >> f, "Each dataset will now be read..."





# Initialize global lists
index1_names = list()
index2_syncpos = list()
eda_values_list = list()
pupil_diameter_values_list = list()
filelist = list()
dataset_index = list()
dataset_event_names = list()
dataset_event_datapoints = list()
EventBinsPos = {}
EventBinsNames = {}


# Walk through folder creating file list
for roots, dirs, files in os.walk(folder_path):
    for files_to_compute in files:
        if files_to_compute.endswith(filename_ext):
            filelist.append(files_to_compute)
print >> f, "Number of datasets to compute: ", len(filelist)

# Extract data from files
for masterfile in filelist:
    print >> f, "Processing dataset: ", masterfile
    index_list = dataextract(masterfile, sync_pos)
    eda_data_list = dataextract(masterfile, eda_data)
    pupil_data_array = []
    pupil_data_list = []
    for i in range(0,len(pupil_data)):
        pupil_data_array.append(dataextract(masterfile, pupil_data[i]))
    for i in range(0,len(pupil_data_array[0])):
        pupil_data_list.append((pupil_data_array[0][i] + pupil_data_array[1][i])/2.0)

    event_names, event_datapoints = event_hz_pointers(masterfile, Events_list, event_data,
                                                      sync_pos)  # syntax: file, list of events to look for, data tag in data set describing events, sync positions, delimiter
    id_name = (masterfile.split('.'))[0]

    if len(index_list) != len(eda_data_list):
        "WARNING! in file", id_name, " a difference between index lenght and value lenght has been encounted."
    if len(index_list) != len(pupil_data_list):
        "WARNING! in file", id_name, " a difference between index lenght and pupil diameter lenght has been encounted."
    n1 = 0
    id_name_list = list()

    while (len(index_list)) > n1:
        id_name_list.append(id_name)
        n1 = n1 + 1

    # Clean data focusing solely on defined events. Other data = NaN.

    xn = 0
    xn1 = 0
    xn2 = 0

    eda_data1 = list()
    pupil_data1 = list()
    event_hz_markers1 = event_datapoints
    event_hz_markers1.sort()

    for xp in index_list:

        if len(event_hz_markers1) != xn and event_hz_markers1[xn] == xp:
            # Switching - when in event data array add data otherwise add numpy.nan
            xn = xn + 1
            xn1 = 0 if xn1 else 1

        if xn1 == 0:
            eda_data1.append(np.nan)  # Fill NaN when outside event boundaries
            pupil_data1.append(np.nan)

        elif xn1 == 1:
            eda_data1.append(eda_data_list[xn2])
            pupil_data1.append(pupil_data_list[xn2])

        xn2 = xn2 + 1

    # All data are extended to three lists, essentially creating a MultiIndex (index1+2) and referring values
    index1_names.extend(id_name_list)
    index2_syncpos.extend(index_list)
    eda_values_list.extend(eda_data1)
    pupil_diameter_values_list.extend(pupil_data1)

    # This list keeps track of all data sets in order to compute event means
    dataset_index.append(id_name)
    dataset_event_names.append(event_names)
    dataset_event_datapoints.append(event_datapoints)
    EventBinsPos[id_name] = event_datapoints
    EventBinsNames[id_name] = event_names

print >> f, "Creating dataframe..."
# Make MultiIndex a Tuple zipping two lists together
index3 = list(zip(index1_names, index2_syncpos))

# Create 2D MultiIndex
index4 = pd.MultiIndex.from_tuples(index3, names=['Names', 'Syncpos'])

# Parse the Index to create a new dataframe named DF
df = pd.DataFrame(index=index4)

# Create a new series with the original data
eda_data_series = pd.Series(eda_values_list, index=index4)
pupil_data_series = pd.Series(pupil_diameter_values_list, index=index4)

print >> f, eda_data_series.dropna(axis=0)
# Tonic/phasic computations
# Window size = number of datapoint!

print >> f, "Computing number of peaks in time-window"

rootfile = TFile(os.path.join(os.getcwd(), '../out/rootfiles/histos.root'), "RECREATE")

if(peakseda):
    ################ find peaks, phasic and tonic distributions in the data ######################
    ################ all distributions are normalised to  integral 1 #############################
    peaksperminute_full_range = npeaks(dataset_index, eda_data_series,'EDA')
    #peaksperminute_full_range.GetXaxis().SetTitle('Time [minutes]')
    peaksperminute_full_range.GetYaxis().SetTitle('Peaks per sequence')
    peaksperminute_full_range.GetYaxis().SetTitleOffset(1.4)
    peaksperminute_full_range.SetStats(False)
    peaksperminute_full_range.SetFillStyle(3002)
    peaksperminute_full_range.Write()
    c = TCanvas("c", "c", 1200, 800)
    c.cd()
    peaksperminute_full_range.Draw()
    c.Update()
    c.SaveAs(os.path.join(os.getcwd(), '../out/results/EDApeaksperseq_full_range.png'))

if(peakspd):
    peaksperminute_full_range_PD = npeaks(dataset_index, pupil_data_series,'PD')
    #peaksperminute_full_range_PD.GetXaxis().SetTitle('Time [minutes]')
    peaksperminute_full_range_PD.GetYaxis().SetTitle('Peaks per sequence')
    peaksperminute_full_range_PD.GetYaxis().SetTitleOffset(1.4)
    peaksperminute_full_range_PD.SetStats(False)
    peaksperminute_full_range_PD.SetFillStyle(3002)
    peaksperminute_full_range_PD.Write()
    c = TCanvas("c", "c", 1200, 800)
    c.cd()
    peaksperminute_full_range_PD.Draw()
    c.Update()
    c.SaveAs(os.path.join(os.getcwd(), '../out/results/PDpeaksperseq_full_range.png'))



fontP = FontProperties()
fontP.set_size('small')


### Create comparision plots for eda and/or PD data for sequences in Comparison_list
if(rawedapeaks):
    # plot specific events in same hist
    #npeaksseqtotal= npeaksspecific(dataset_index, eda_data_series, Comparison_list,'rawEDApeaks')
    #npeaksseqminuts = npeaksspecificminutes(dataset_index, eda_data_series, Comparison_list,timewindow,'rawEDApeaks')
    npeaksfullrange = createplotsFullRange(dataset_index, eda_data_series,timewindow,'rawEDApeaks')

if(rawpdpeaks):
    #npeaksseqtotal= npeaksspecific(dataset_index, pupil_data_series, Comparison_list,'rawPDpeaks')
    #npeaksseqminuts = npeaksspecificminutes(dataset_index, pupil_data_series, Comparison_list,timewindow,'rawPDpeaks')
    npeaksfullrange = createplotsFullRange(dataset_index, pupil_data_series,timewindow,'rawPDpeaks')


###########################################################################
###########################################################################

# peaksinwindow_series1 = npeaks(dataset_index, eda_data_series)


# a plot peaks in the full datarange in 1 minute bins summed over all respondents.
# peaksperminute_full_range = TH1F("peaksperminute_full_range", "Peaks per minute with 4 #sigma significance",
#                                 len(peaksinwindow_series1), 0, len(peaksinwindow_series1))

# plot peaks for the full datarange and for events defined in Comparison_list
# for i in range(0, len(peaksinwindow_series1)):
#    peaksperminute_full_range.Fill(i, peaksinwindow_series1.values[i])
# eda_data_series.dropna(axis=0).loc['02418'].index[-1] = 2.629.585
# for event in Comparison_list:
#    for pos in range(int(EventBinsDict['start_'+event]/(1000. * 60.)),int(EventBinsDict['end_'+event]/(1000. * 60.))):
#        int(EventBinsDict['start_MF vaccine med case.avi'] / (1000. * 60.))

# For specific events of interest in same histogram
# peaksperminute_compared = TH1F("peaksperminute_compared", "Peaks per minute with 4 #sigma significance",
#                               len(peaksinwindow_series1), 0, len(peaksinwindow_series1))


####################### find phasic component of EDA ######################
###########################################################################
# timescale for plots
timescaling = 1 / (1000. * 1)  #

if dooverview:
    print >> f, "Computing phasic component of data..."
    phasic_data_series = phasic_component(dataset_index, eda_data_series, rolling_average_window)
    tonic_data_series = tonic_component(dataset_index, eda_data_series, rolling_average_window)
    # print >> f, phasic_data_series.dropna(axis=0)
    # t=phasic_data_series.index
    for respondent in dataset_index:
        # plot raw EDA for first case:
        tmpraw = eda_data_series.dropna(axis=0).loc[respondent]
        tmptonic = tonic_data_series.dropna(axis=0).loc[respondent]
        # Two subplots, the axes array is 1-d
        sp1 = plt.subplot(2, 1, 1)
        plt.plot(tmpraw.index * timescaling, tmpraw.values, label='Raw EDA')
        plt.plot(tmptonic.index * timescaling, tmptonic.values, label='tonic (time avaraged EDA)')
        sp1.set_title(respondent)
        # plt.ylabel('Scin conductance (EDA)')
        plt.legend(bbox_to_anchor=(1.1, 1.2), prop=fontP)
        # phasic part
        plt.subplot(2, 1, 2)
        tmpphasic = phasic_data_series.dropna(axis=0).loc[respondent]
        plt.plot(tmpphasic.index * timescaling, tmpphasic.values,
                 label='phasic rest after subtraction of tonic time avarage')
        plt.legend(bbox_to_anchor=(1.1, 0.15), prop=fontP)
        plt.grid(True)
        plt.xlabel('time [s]')
        suplabel('y', 'Skin conductance (EDA)')
        plt.savefig("../out/respondents/" + respondent + ".png")
        plt.close()
        # plt.show()

# Normalize series output
print >> f, "Normalizing phasic data..."
raw_normalized = normalize_series(dataset_index, eda_data_series)
# phasic_outlier_cleaned = noisereduce(dataset_index, eda_data_series)

# timeindex = []
# for index in range[1:len(eda_data_series)]:
#     timeindex.append[time.strftime('%H:%M:%S', time.gmtime(eda_data_series.index[index]))]


if (meanraw):
    ######################### mean EDA on normalized EDA #########################
    ##############################################################################
    #find mean eda for each sequence
    meaneda_full_range = meaneda(dataset_index, raw_normalized)
    meaneda_full_range.GetYaxis().SetTitle('Mean EDA per sequence')
    meaneda_full_range.GetYaxis().SetTitleOffset(1.4)
    meaneda_full_range.SetStats(False)
    #meaneda_full_range.Write()
    c = TCanvas("c", "c", 1200, 800)
    c.cd()
    meaneda_full_range.Draw()
    c.Update()
    c.SaveAs(os.path.join(os.getcwd(), '../out/results/meaneda_full_range.png'))

    # find mean per interval for each sequence - histogram as for peaks
    meanedainterval = meaninterval(dataset_index, raw_normalized,Comparison_list,timewindow,'rawmean')
    #meanedainterval.GetYaxis().SetTitle('Mean EDA per timeinterval')
    #meanedainterval.GetYaxis().SetTitleOffset(1.4)
    #meanedainterval.SetStats(False)
    #meaneda_full_range.Write()
    #c = TCanvas("c", "c", 1200, 800)
    #c.cd()
    #meanedainterval.Draw()
    #c.Update()
    #c.SaveAs(os.path.join(os.getcwd(), '../out/results/meaneda_intervals.png'))
    ###########################################################################
    ###########################################################################

if(phasic):
    phasic_normalized = normalize_series(dataset_index, phasic_data_series)
    ################ mean EDA based on phasic component of data ###############
    ###########################################################################
    #find mean eda for each sequence of phasic data
    meanphasic_full_range = meaneda(dataset_index, phasic_normalized)
    meanphasic_full_range.GetYaxis().SetTitle('Mean Phasic per sequence')
    meanphasic_full_range.GetYaxis().SetTitleOffset(1.4)
    meanphasic_full_range.SetStats(False)
    #meaneda_full_range.Write()
    c = TCanvas("c", "c", 1200, 800)
    c.cd()
    meanphasic_full_range.Draw()
    c.Update()
    c.SaveAs(os.path.join(os.getcwd(), '../out/results/meanphasic_full_range.png'))

    # find phasic mean per interval for each sequence - histogram as for peaks
    meanphasicinterval = meaninterval(dataset_index, phasic_normalized,Comparison_list,timewindow,'phasicmean')
    #eanphasicinterval.GetYaxis().SetTitle('Mean Phasic per interval')
    #meanphasicinterval.GetYaxis().SetTitleOffset(1.4)
    #meanphasicinterval.SetStats(False)
    #meaneda_full_range.Write()
    #c = TCanvas("c", "c", 1200, 800)
    #c.cd()
    #meanphasicinterval.Draw()
    #c.Update()
    #c.SaveAs(os.path.join(os.getcwd(), '../out/results/meanphasic_intervals.png'))
    ###########################################################################
    ###########################################################################


if(phasic):
    ############################### peaks phasic ##############################
    ###########################################################################
    ##find and plot histograms for peaks in phasic data
    npeaksseqphasic= npeaks(dataset_index, phasic_normalized,'phasicEDA')
    npeaksseqphasic.GetYaxis().SetTitle('Peaks per sequence')
    npeaksseqphasic.GetYaxis().SetTitleOffset(1.4)
    npeaksseqphasic.SetStats(False)
    npeaksseqphasic.SetFillStyle(3002)
    c = TCanvas("c", "c", 1200, 800)
    c.cd()
    npeaksseqphasic.Draw()
    c.Update()
    c.SaveAs(os.path.join(os.getcwd(), '../out/results/peaksperseqphasic_full_range.png'))
    #npeaksphasicseqtotal = npeaksspecific(dataset_index, phasic_normalized, Comparison_list,'phasicEDApeaks')
    #npeaksphasicseqminuts = npeaksspecificminutes(dataset_index, phasic_normalized, Comparison_list,timewindow,'phasicEDApeaks')

    ###########################################################################
    ###########################################################################



if dooverview:
    phasic_data_series = phasic_component(dataset_index, eda_data_series, rolling_average_window)
    phasic_normalized = normalize_series(dataset_index, phasic_data_series)
    tonic_normalized = normalize_series(dataset_index, tonic_data_series)
    ######################### test of individuals #########################
    # plot normalized for short range
    for respondent in dataset_index:
        # plot raw EDA for first case:
        tmpraw = raw_normalized.dropna(axis=0).loc[respondent]
        tmptonic = tonic_normalized.dropna(axis=0).loc[respondent]
        # Two subplots, the axes array is 1-d
        sp1 = plt.subplot(2, 1, 1)
        plt.plot(tmpraw.index[1000:3000] * timescaling, tmpraw.values[1000:3000], label='Normalized EDA')
        plt.plot((tmptonic.index[1000:3000]) * timescaling, tmptonic.values[1000:3000],
                 label='Normalized tonic (time avaraged EDA)')
        sp1.set_title(respondent)
        # plt.ylabel('Scin conductance (EDA)')
        plt.legend(bbox_to_anchor=(1.1, 1.2), prop=fontP)
        # phasic part
        plt.subplot(2, 1, 2)
        tmpphasic = phasic_normalized.dropna(axis=0).loc[respondent]
        plt.plot(tmpphasic.index[1000:3000] * timescaling, tmpphasic.values[1000:3000], label='Normalized phasic EDA')
        plt.legend(bbox_to_anchor=(1.1, 0.15), prop=fontP)
        plt.grid(True)
        plt.xlabel('time [s]')
        suplabel('y', 'Normalized skin conductance (EDA)')
        plt.savefig("../out/respondents/" + respondent + "_normalized.png")
        plt.close()
        # plt.show()

#
# if(True):
#     # add column to existing DataFrame:
#     df[df.index[1], 'EDA'] = phasic_normalized
#
#     # Data validation: Test for equal number of events per dataset,
#     print >> f, "Validating events for each dataset... Number of events and their names must be exactly the same across all datasets"
#
#     n = 0
#     data_events_validating = len(dataset_index[0])
#     while (len(dataset_index) - 1) > n:
#         if (len(dataset_event_names[n])) != (len(dataset_event_names[n + 1])):
#             print >> f, "DATA FAILURE... Unequal number of events detected!! Means may not be valid!"
#
#         n = n + 1
#
#     # Initalize new dataframe - dataframe for means #
#     df1 = pd.DataFrame(index=dataset_event_names[0][0:(len(dataset_event_names[0])):2])
#
#     n = 0
#
#     for p in dataset_index:
#         event_means, event_std = eventmeans(p, dataset_event_names[n], dataset_event_datapoints[n], df)
#         n = n + 1
#
#         df1[p] = event_means
#
#     # Output event with arousal mean and standard deviation
#     print >> f, "Output: event name, arousal mean for event and standard deviation for mean."
#
#     # asktodo - plot this:
#     for p in dataset_event_names[0][0:(len(dataset_event_names[0])):2]:
#         print >> f, p + "," + str((df1.loc[p].mean())[0]) + "," + str(df1.loc[p].std())
#
#     # Repeated Measures ANOVA
#     print >> f, "Performing a repeated measures ANOVA..."
#
#     df_anova = pt.DataFrame()
#
#     headers = namedtuple('headers', ['subject', 'event', 'mean'])
#
#     an = 0
#
#     for p in dataset_event_names[0][0:(len(dataset_event_names[0])):2]:
#
#         for p1 in dataset_index:
#             p2 = float(df1.loc[p].loc[p1])
#             df_anova.insert(headers(p1, an, p2)._asdict())
#
#         an = an + 1
#
#     anova = df_anova.anova(dv='mean', sub='subject', wfactors=['event'])
#
#     print >> f, (anova)
#
#     # Concluding the script doing a T-test to test for sig. diff. means between two predefined means
#     for ikey in Comparison_list.keys():
#         t_test_list1 = list()
#         t_test_list2 = list()
#         print >> f, "Testing for significance between: " + Comparison_list[ikey][0] + " and " + Comparison_list[ikey][1]
#         t_temp_list1 = df1.loc[ Comparison_list[ikey][0]].values
#         t_temp_list2 = df1.loc[ Comparison_list[ikey][1]].values
#
#         for p in t_temp_list1:
#             t_test_list1.append(float(p))
#
#         for p1 in t_temp_list2:
#             t_test_list2.append(float(p1))
#
#         print >> f, ttest_ind(t_test_list1, t_test_list2)

rootfile.Write()
rootfile.Close()
