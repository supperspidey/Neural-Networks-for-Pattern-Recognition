import csv
import numpy as np

# Define class A and class B
mean_A = [0, 0]
cov_A = [[3, 0], [0, 3]]

mean_B = [10, 10]
cov_B = [[3, 0], [0, 3]]

# Generate random data
data_A = np.random.multivariate_normal(mean_A, cov_A, 100)
label_A = np.ones(len(data_A))
data_B = np.random.multivariate_normal(mean_B, cov_B, 100)
label_B = np.zeros(len(data_B))

# Write to file
with open('data.csv', 'wb') as csvfile:
    writer = csv.writer(csvfile, delimiter=',')
    writer.writerow(['x1', 'x2', 'label'])

    for row in range(0, len(data_A)):
        writer.writerow([data_A[row][0], data_A[row][1], label_A[row]])

    for row in range(0, len(data_B)):
        writer.writerow([data_B[row][0], data_B[row][1], label_B[row]])
