import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import analysisPlotFunctions as plf

# Add correct path to csv's here.
# Also, savefig is currently hardcoded to my desktop. You should add a valid path here if you want to save figs (which it does by default)
dfGer = pd.read_csv("Ger/VGG16/VGG16_results.csv")
dfEur = pd.read_csv("Eur/VGG16ADDEUR/VGG16ADDEUR_results.csv")
label = 'Pred_Occl_25_Label' #Class ID you want to inspect. Examples are Pred_Label, Orig_Label, Pred_Blurred_x_Label, Pred_Bright_x_Label,
# Pred_Dark_x_Label and Pred_Occl_x_Label
id = 11 # Current class object that you want to inspect misclassifcations
# Plots the class accuracy for this label, i.e. the accuracy per class on the occluded_25 dataset.
plf.plotClassAccuracy(dfGer, dfEur, label)
# Plots the count of predictions for this label. Interesting to see which classes get more predictions, which get less.
plf.plotBothFeat(dfGer, dfEur, label)

# Finds the corresponding test ids where Ger and/or Eur misclassified this class id
print(dfGer.loc[(dfGer['Orig_Label'] == id) & (dfGer[label] != id)][['Orig_Label', label, 'Pred_Label']])
print(dfEur.loc[(dfEur['Orig_Label'] == id) & (dfEur[label] != id)][['Orig_Label', label, 'Pred_Label']])
# Prints the misclassifications as, for this class id. (Found this easier than looking at the confmat)
print(dfGer.loc[dfGer['Orig_Label'] == id][label].value_counts())
print(dfEur.loc[dfEur['Orig_Label'] == id][label].value_counts())
