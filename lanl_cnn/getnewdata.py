
import numpy as np
import glob

import sys


# Number of points per sequence collected
no_points = 500

# Insert the directory where the data is saved here
dir = 'test/test_data'
dir1 = 'mafaulda/data'

# File navigation
files_no = glob.glob(dir1 + '/normal/*.csv')
folders_im = glob.glob(dir1 + '/imbalance/*/*.csv')
folders_hm = glob.glob(dir1 + '/horizontal-misalignment/*')
folders_oh = glob.glob(dir1 + '/overhang/ball_fault/*') + glob.glob(dir1 + '/overhang/cage_fault/*')+ glob.glob(dir1 + '/overhang/outer_race/*')
folders_uh = glob.glob(dir1 + '/underhang/ball_fault/*') + glob.glob(dir1 + '/underhang/cage_fault/*')+ glob.glob(dir1 + '/underhang/outer_race/*')
folders_vm = glob.glob(dir1 + '/vertical-misalignment/*')
train_data = np.empty((0, no_points,8), float)
test_data = np.empty((0, no_points,8), float)
normal_data = np.empty((0, no_points,8), float)
imbalance_data = np.empty((0, no_points,8), float)
hm_data = np.empty((0, no_points,8), float)
vm_data = np.empty((0, no_points,8), float)
oh_data = np.empty((0, no_points,8), float)
uh_data = np.empty((0, no_points,8), float)



train_label = []
test_label = []

r = 15

# from keras.utils import to_categorical
# # Normal data generation
# # Take 7 times more data to balance the number of normal and imbalanced points
# i=0
# for f_on in files_no:
#     source_data = np.loadtxt(f_on, delimiter=",")
#     for j in range(r):
#         A = source_data[2000*j:2000*j+no_points,:]
#         normal_data = np.append(normal_data,A.reshape(1, A.shape[0],A.shape[1]),axis=0)
#     i=i+1
#     print(i)

# normal_data = np.reshape(normal_data,(-1,8))
# np.savetxt(dir + "/normal_data.txt", normal_data)

# # Imbalanced data generation
# for folder in folders_im:
#     i=0
#     files_im = glob.glob( folder +'/*.csv')
#     for f_im in files_im:
#         source_data = np.loadtxt(f_im, delimiter=",")
#         for j in range(r):
#             A = source_data[2000*j:2000*j+no_points,:]
#             imbalance_data = np.append(imbalance_data,A.reshape(1, A.shape[0],A.shape[1]),axis=0)
#         i=i+1
#         print(i)

# imbalance_data = np.reshape(imbalance_data,(-1,8))
# np.savetxt(dir + "/imbalance_data.txt", imbalance_data)

# # Horizontal-mis data generation
# for folder in folders_hm:
#     i=0
#     files_hm = glob.glob( folder +'/*')
#     for f_hm in files_hm:
#         source_data = np.loadtxt(f_hm, delimiter=",")
#         for j in range(r):
#             A = source_data[2000*j:2000*j+no_points,:]
#             hm_data = np.append(hm_data,A.reshape(1, A.shape[0],A.shape[1]),axis=0)

#         i=i+1
#         print(i)

# hm_data = np.reshape(hm_data,(-1,8))
# np.savetxt(dir + "/hm_data.txt", hm_data)

# # verital-mis data generation
# for folder in folders_vm:
#     i=0
#     files_vm = glob.glob( folder +'/*')
#     for f_vm in files_vm:
#         source_data = np.loadtxt(f_vm, delimiter=",")
#         for j in range(r):
#             A = source_data[2000*j:2000*j+no_points,:]
#             vm_data = np.append(vm_data,A.reshape(1, A.shape[0],A.shape[1]),axis=0)
#         i=i+1
#         print(i)

# vm_data = np.reshape(vm_data,(-1,8))
# np.savetxt(dir + "/vm_data.txt", vm_data)

# # Overhang data generation
# for folder in folders_oh:
#     i=0
#     files_im = glob.glob( folder +'/*.csv')
#     for f_im in files_im:
#         source_data = np.loadtxt(f_im, delimiter=",")
#         for j in range(r):
#             A = source_data[2000*j:2000*j+no_points,:]
#             oh_data = np.append(oh_data,A.reshape(1, A.shape[0],A.shape[1]),axis=0)
#         i=i+1
#         print(i)

# oh_data = np.reshape(oh_data,(-1,8))
# np.savetxt(dir + "/oh_data.txt", oh_data)

#underhang
for folder in folders_uh:
    i=0
    files_im = glob.glob( folder +'/*.csv')
    for f_im in files_im:
        source_data = np.loadtxt(f_im, delimiter=",")
        for j in range(r):
            A = source_data[2000*j:2000*j+no_points,:]
            uh_data = np.append(uh_data,A.reshape(1, A.shape[0],A.shape[1]),axis=0)
        i=i+1
        print(i)

uh_data = np.reshape(uh_data,(-1,8))
np.savetxt(dir + "/uh_data.txt", uh_data)


print("Finished parsing the files")