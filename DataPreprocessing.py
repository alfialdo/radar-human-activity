'''
Code for processing the RADAR data
==================================
Krishna
'''

# import libraries
import numpy as np
import glob

# Set the dataset path and list all .dat files
dataset_path = "./dataset/1December2017Dataset"
file_paths = sorted(glob.glob(dataset_path + '/*.dat'))

print(len(file_paths), file_paths[0])

# Loop reading each line of .dat file
# Index 0-3 is float, contains the sensor information
# Index 4:end is complex number, representing the RADAR data
reads_file = []
# len(file_paths)
for i in range(len(file_paths)):
    print("Reading file number: ", i)

    reads_line = []
    with open(file_paths[i], 'r') as file:
        # content = file.read()
        j = 0
        for line in file:
            if j < 4:
                reads_line.append(float(line.strip()))
            else:
                complex_num = line.strip().replace('i', 'j')
                complex_num = complex(complex_num)
                reads_line.append(complex_num)
            j += 1
    
    print('i, len line', i, len(reads_line))
    reads_file.append(reads_line)
    print('i, len file', i, len(reads_file))

print(len(reads_file), len(reads_line))

fc_prev = 0
for j in range(len(reads_file)):
    fc = reads_file[j][0]
    if fc != fc_prev:
        print('not equal: ', fc, fc_prev)
        fc_prev = fc


fc = reads_file[0][0] # center frequency
Tsweep = reads_file[0][1]
Tsweep /= 1000
NTS = reads_file[0][2]
Bw = reads_file[0][3]
# data = reads_file
data = np.asanyarray(reads_file, dtype="object")

print(data.shape)