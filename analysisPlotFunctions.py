import numpy as np
import matplotlib.pyplot as plt
from matplotlib.pyplot import figure

def plotClassAccuracy(df1, df2, label):
    "Plot Accuracy per class for given test-set (e.g. Pred_Blurred_15_Label), for both German (df1) and Eur (df2) datasets,"
    "as well as the original label and original test-set predictions without augmentation for comparison"
    numClasses = 43
    priorClass1 = [len(df1.loc[df1['Orig_Label'] == i]) for i in range(43)]
    priorClass2 = [len(df1.loc[df2['Orig_Label'] == i]) for i in range(43)]
    origAccGer = np.zeros(numClasses)
    origAccEur = np.zeros(numClasses)
    accuracies1 = np.zeros(numClasses)
    accuracies2 = np.zeros(numClasses)
    for idx,row in df1.iterrows():
        key = row['Orig_Label']
        if row[label] == key:
            accuracies1[key] += 1/priorClass2[key]
        if row['Pred_Label'] == key:
            origAccGer[key] += 1/priorClass1[key]
    for idx,row in df2.iterrows():
        key = row['Orig_Label']
        if row[label] == key:
            accuracies2[key] += 1/priorClass2[key]
        if row['Pred_Label'] == key:
            origAccEur[key] += 1/priorClass2[key]
    figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    w=0.2
    plt.bar(np.arange(0,43,1)-1.5*w, origAccGer, width=w, label = "Orig Ger", color = 'blue')
    plt.bar(np.arange(0,43,1)-0.5*w, origAccEur, width=w, label = "Orig Eur", color = 'red')
    plt.bar(np.arange(0,43,1)+0.5*w, accuracies1, width=w, label = "Ger", color = 'orange')
    plt.bar(np.arange(0,43,1)+1.5*w, accuracies2, width=w, label = "Eur", color = 'g')
    plt.xticks(np.arange(0,43,1))
    plt.xlabel("Class", size =14)
    plt.ylabel("Accuracy", size= 14)
    plt.legend(prop={'size': 14})
    plt.title("Class ~ Accuracy {}".format(label), size= 14)
    fig = plt.gcf()
    fig.savefig("C:/Users/alexm/Desktop/AnalysisFigs/classErr_{}.png".format(label), quality = 95, bbox_inches = 'tight')
    plt.show()
    # print(accuracies[0])

def plotBothFeat(df1, df2, feat):
    "Plots count of prediction per class, for a given feature (e.g. Pred_Blurred_15_Label), as well as the"
    "True Label, Original prediction for both German (df1) and European (df2)"
    numClasses = 43
    orig = np.zeros(numClasses)
    predOrigGer = np.zeros(numClasses)
    predOrigEur = np.zeros(numClasses)
    predGer = np.zeros(numClasses)
    predEur = np.zeros(numClasses)
    for i,row in df1.iterrows():
        orig[row['Orig_Label']] += 1
        predGer[row[feat]] += 1
        predOrigGer[row['Pred_Label']] += 1
    for i,row in df2.iterrows():
        predEur[row[feat]] += 1
        predOrigEur[row['Pred_Label']] += 1
    x = np.arange(0,43,1)
    w = 0.15
    figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.bar(x-2*w, orig, width =w, label = "Original")
    plt.bar(x-1*w, predOrigGer, width =w, label = "Pred_Ger")
    plt.bar(x, predOrigEur, width =w, label = "Pred_Eur")
    plt.bar(x+w, predGer, width=w, label = "{} pred Ger".format(feat))
    plt.bar(x+2*w, predEur, width=w, label = "{} pred Eur".format(feat))
    plt.xticks(np.arange(0,43,1))
    plt.xlabel("Class", size =14)
    plt.ylabel("Prediction count", size= 14)
    plt.legend(prop={'size': 14})
    plt.title("Class ~ {} Predictions".format(feat), size= 14)
    fig = plt.gcf()
    fig.savefig("C:/Users/alexm/Desktop/AnalysisFigs/{}_err.png".format(feat), quality = 95, bbox_inches = 'tight')
    plt.show()


def plotPredFeat(df, feat, lab1, lab2, lab3, levels):
    "Not currently called from plots.py. I used this initially  for one dataframe."
    "You can use this to plot 3 features (e.g. Pred_Blurred_15_Label, Pred_Blurred_10_Label and Pred_Blurred_5_Label)"
    "for a single dataframe (e.g. German). Specify label (Blurred here), the 3 features (as above), as well as their levels [5,10,15] in this case"
    numClasses = 43
    orig = np.zeros(numClasses)
    pred = np.zeros(numClasses)
    predb5 = np.zeros(numClasses)
    predb10 = np.zeros(numClasses)
    predb15 = np.zeros(numClasses)

    for i,row in df.iterrows():
        orig[row['Orig_Label']] += 1
        pred[row['Pred_Label']] += 1
        predb5[row[lab1]] += 1
        predb10[row[lab2]] += 1
        predb15[row[lab3]] += 1

    x = np.arange(0,43,1)
    w = 0.15
    figure(num=None, figsize=(20, 10), dpi=80, facecolor='w', edgecolor='k')
    plt.bar(x-2*w, orig, width=w, color='b', align='center', label = "True Label")
    plt.bar(x-w, pred, width=w, color='g', align='center', label = "Original Prediction")
    plt.bar(x, predb5, width=w, color='r', align='center', label = "{} {} Prediction".format(feat, levels[0]))
    plt.bar(x+w, predb10, width=w, color='y', align='center', label = "{} {} Prediction".format(feat, levels[1]))
    plt.bar(x+2*w, predb15, width=w, color='m', align='center', label = "{} {} Prediction".format(feat, levels[2]))
    plt.xticks(x)
    plt.xlabel("Class", size = 14)
    plt.ylabel("Prediction Count", size = 14)
    plt.title("Class predictions for various {} levels.".format(feat), size = 14)
    plt.legend(prop={'size': 14})
    fig = plt.gcf()
    fig.savefig("C:/Users/alexm/Desktop/AnalysisFigs/{}_predictions.png".format(feat), quality = 95, bbox_inches = 'tight')
    plt.show()
