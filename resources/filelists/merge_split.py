import os

total_lines = []
training = []
testing = []
val = []
f_s = open("final_siwis_train.txt", "r")
data_siwis_train = f_s.readlines()

f_s = open("final_siwis_test.txt", "r")
data_siwis_test = f_s.readlines()

f_s = open("final_siwis_val.txt", "r")
data_siwis_val = f_s.readlines()


f_l = open("final_ljs_train.txt", "r")
data_lj_train = f_l.readlines()

f_l = open("final_ljs_test.txt", "r")
data_lj_test = f_l.readlines()

f_l = open("final_ljs_val.txt", "r")
data_lj_val = f_l.readlines()

f_v = open("final_vctk_train.txt", "r")
data_vctk_train = f_v.readlines()

import pdb
pdb.set_trace()

w_s = len(data_siwis_train) + len(data_siwis_val) + len(data_siwis_test)
w_l = len(data_lj_train) + len(data_lj_val) + len(data_lj_test)
w_v = len(data_vctk_train)
total = w_s + w_l+ w_v
w_s, w_l, w_v = w_s/total, w_l/total, w_v/total 

total_test = 500
total_val = 100
n_t_s, n_t_l, n_t_v = round(w_s * total_test), round(w_l * total_test), round(w_v * total_test)
n_v_s, n_v_l, n_v_v =  round(w_s * total_val), round(w_l * total_val), round(w_v * total_val)

test_data = []
val_data = []
train_data = []


train_data = data_siwis_train + data_lj_train + data_siwis_val[n_v_s:] + data_lj_val[n_v_l:] + data_siwis_test[n_t_s:] + data_lj_test[n_t_l:] + data_vctk_train[n_v_v+n_t_v:]
val_data = data_siwis_val[:n_v_s] + data_lj_val[:n_v_l] + data_vctk_train[:n_v_v]
test_data = data_siwis_test[:n_t_s] + data_lj_test[:n_t_l] + data_vctk_train[n_v_v:n_v_v+n_t_v]


with open("final_merged_train.txt","w") as f:
    for each in train_data:
        f.write(each)
with open("final_merged_test.txt","w") as f:
    for each in test_data:
        f.write(each)

with open("final_merged_val.txt","w") as f:
    for each in val_data:
        f.write(each)




