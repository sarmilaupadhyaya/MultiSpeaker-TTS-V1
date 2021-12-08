import os

total_lines = []
training = []
testing = []
val = []
f_s = open("final3_siwis_train.txt", "r")
data_siwis_train = f_s.readlines()

f_s = open("final3_siwis_test.txt", "r")
data_siwis_test = f_s.readlines()

f_s = open("final3_siwis_val.txt", "r")
data_siwis_val = f_s.readlines()


f_l = open("final3_ljs_train.txt", "r")
data_lj_train = f_l.readlines()

f_l = open("final3_ljs_test.txt", "r")
data_lj_test = f_l.readlines()

f_l = open("final3_ljs_val.txt", "r")
data_lj_val = f_l.readlines()

f_v = open("final3_VCTK_data_2.txt", "r")
data_vctk_train2 = f_v.readlines()

f_v = open("final3_VCTK_data_3.txt", "r")
data_vctk_train3 = f_v.readlines()

f_v = open("final3_VCTK_data_4.txt", "r")
data_vctk_train4 = f_v.readlines()

f_v = open("final3_VCTK_data_5.txt", "r")
data_vctk_train5 = f_v.readlines()

f_v = open("final3_VCTK_data_6.txt", "r")
data_vctk_train6 = f_v.readlines()

f_v = open("final3_text_chevalier.txt", "r")
data_chevalier_train7 = f_v.readlines()

f_v = open("final3_text_hugo.txt", "r")
data_hugo_train8 = f_v.readlines()

f_v = open("final3_text_sue.txt", "r")
data_sue_train9 = f_v.readlines()

f_v = open("final3_text_tundra.txt", "r")
data_tundra_train10 = f_v.readlines()


f_v = open("final3_text_zola.txt", "r")
data_zola_train11 = f_v.readlines()



w_s = len(data_siwis_train) + len(data_siwis_val) + len(data_siwis_test)
w_l = len(data_lj_train) + len(data_lj_val) + len(data_lj_test)
w_v2 = len(data_vctk_train2)
w_v3 = len(data_vctk_train3)
w_v4 = len(data_vctk_train4)
w_v5 = len(data_vctk_train5)
w_v6 = len(data_vctk_train6)
w_v7 = len(data_chevalier_train7)
w_v8 = len(data_hugo_train8)
w_v9= len(data_sue_train9)
w_v10 = len(data_tundra_train10)
w_v11 = len(data_zola_train11)

total = w_s + w_l+ w_v2 + w_v3 +w_v4 +w_v5 +w_v6 + w_v7 + w_v8 +w_v9 +w_v10 +w_v11
w_s, w_l, w_v2, w_v3, w_v4,w_v5, w_v6, w_v7, w_v8, w_v9,w_v10,w_v11   = w_s/total, w_l/total, w_v2/total , w_v3/total, w_v4/total , w_v5/total,w_v6/total, w_v7/total , w_v8/total , w_v9/total ,w_v10/total ,w_v11/total

total_test = 500
total_val = 100

n_t_s, n_t_l, n_t_v2, n_t_v3, n_t_v4, n_t_v5, n_t_v6, n_t_v7, n_t_v8, n_t_v9, n_t_v10, n_t_v11 = round(w_s * total_test), round(w_l * total_test), round(w_v2 * total_test), round(w_v3 * total_test), round(w_v4 * total_test), round(w_v5 * total_test), round(w_v6 * total_test), round(w_v7 * total_test), round(w_v8 * total_test), round(w_v9 * total_test), round(w_v10 * total_test), round(w_v11 * total_test)

n_v_s, n_v_l, n_v_v2, n_v_v3, n_v_v4, n_v_v5, n_v_v6, n_v_v7, n_v_v8, n_v_v9, n_v_v10, n_v_v11 =  round(w_s * total_val), round(w_l * total_val), round(w_v2 * total_val), round(w_v3 * total_val), round(w_v4 * total_val), round(w_v5 * total_val), round(w_v6 * total_val), round(w_v7 * total_val), round(w_v8 * total_val), round(w_v9 * total_val), round(w_v10 * total_val), round(w_v11 * total_val)

test_data = []
val_data = []
train_data = []


train_data = data_siwis_train + data_lj_train + data_siwis_val[n_v_s:] + data_lj_val[n_v_l:] + data_siwis_test[n_t_s:] + data_lj_test[n_t_l:] + data_vctk_train2[n_v_v2+n_t_v2:]+ data_vctk_train3[n_v_v3+n_t_v3:]+ data_vctk_train4[n_v_v4+n_t_v4:]+data_vctk_train5[n_v_v5+n_t_v5:]+data_vctk_train6[n_v_v6+n_t_v6:] +data_chevalier_train7[n_v_v7+n_t_v7:]+ data_hugo_train8[n_v_v8+n_t_v8:]+ data_sue_train9[n_v_v9+n_t_v9:]+data_tundra_train10[n_v_v10+n_t_v10:]+data_zola_train11[n_v_v11+n_t_v11:]


val_data = data_siwis_val[:n_v_s] + data_lj_val[:n_v_l] + data_vctk_train2[:n_v_v2] + data_vctk_train3[:n_v_v3] +data_vctk_train4[:n_v_v4]+data_vctk_train5[:n_v_v5]+data_vctk_train6[:n_v_v6] + data_chevalier_train7[:n_v_v7] + data_hugo_train8[:n_v_v8] +data_sue_train9[:n_v_v9]+data_tundra_train10[:n_v_v10]+data_zola_train11[:n_v_v11]


test_data = data_siwis_test[:n_t_s] + data_lj_test[:n_t_l] + data_vctk_train2[n_v_v2:n_v_v2+n_t_v2] + data_vctk_train3[n_v_v3:n_v_v3+n_t_v3] + data_vctk_train4[n_v_v4:n_v_v4+n_t_v4]+data_vctk_train5[n_v_v5:n_v_v5+n_t_v5]+data_vctk_train6[n_v_v6:n_v_v6+n_t_v6]+ data_chevalier_train7[n_v_v2:n_v_v2+n_t_v2] + data_hugo_train8[n_v_v3:n_v_v3+n_t_v3] + data_sue_train9[n_v_v4:n_v_v4+n_t_v4]+data_tundra_train10[n_v_v5:n_v_v5+n_t_v5]+data_zola_train11[n_v_v11:n_v_v11+n_t_v11]



with open("final_merged_train.txt","w") as f:
    for each in train_data:
        f.write(each)
with open("final_merged_test.txt","w") as f:
    for each in test_data:
        f.write(each)

with open("final_merged_val.txt","w") as f:
    for each in val_data:
        f.write(each)




