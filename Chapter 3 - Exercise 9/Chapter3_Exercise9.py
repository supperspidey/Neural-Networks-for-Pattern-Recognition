import numpy as np
import csv

###############################################################################

# Load data from file
X = []
t = []
with open('data.csv', 'rb') as csvfile:
    reader = csv.reader(csvfile, delimiter=',')
    for row in reader:
        if row[0] == "x1" and row[1] == "x2" and row[2] == "label":
            continue
        else:
            X.append([float(row[0]), float(row[1])])
            t.append(float(row[2]))

X = np.array(X)
t = np.array(t)

###############################################################################
