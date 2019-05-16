import csv
import os

RESULTS = [
    ['apple','cherry','orange','pineapple','strawberry']
]
with open("output.csv",'a', newline = '') as resultFile:
    wr = csv.writer(resultFile)
    wr.writerows(RESULTS)