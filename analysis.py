import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure
import analysisPlotFunctions as plf

dfEur = pd.read_csv("output/VGG16ADDEUR/VGG16ADDEUR_results.csv")

#IMPORTANT: ADD THE CORRECT CSV YOU'RE INSPECTING HERE
dfAug = pd.read_csv("output/VGG16ADDOCC/VGG16ADDOCCL_results.csv")
label = 'Pred_Occl_25_Label' #Class ID you want to inspect. Examples are Pred_Label, Orig_Label, Pred_Blurred_x_Label, Pred_Bright_x_Label,
# Pred_Dark_x_Label and Pred_Occl_x_Label
id = 9 # Current class object that you want to inspect misclassifcations
# Plots the class accuracy for this label, i.e. the accuracy per class on the occluded_25 dataset.
plf.plotClassAccuracy(dfEur, dfAug, label)
# Plots the count of predictions for this label. Interesting to see which classes get more predictions, which get less.
plf.plotBothFeat(dfEur, dfAug, label)

# Finds the corresponding test ids where Ger and/or Eur misclassified this class id
print(dfEur.loc[(dfEur['Orig_Label'] == id) & (dfEur[label] != id)][['Orig_Label', label, 'Pred_Label']])
print(dfAug.loc[(dfAug['Orig_Label'] == id) & (dfAug[label] == 28)][['Orig_Label', label, 'Pred_Label']])
# Prints the misclassifications as, for this class id. (Found this easier than looking at the confmat)
print(dfEur.loc[dfEur['Orig_Label'] == id][label].value_counts())
print(dfAug.loc[dfAug['Orig_Label'] == id][label].value_counts())

# plf.plotAccDiff()
