import numpy as np

data_latih = [
    [5.1, 3.5, 1.4, 0.2], 
    [6.2, 2.2, 4.5, 1.5], 
    [7.7, 2.8, 6.7, 2.0]
]

target_data_latih = [
    [1, 0, 0], #iris-setosa
    [0, 1, 0], #iris-versicolor
    [0, 0, 1]  #iris-virginica
]

v_i_j = [
    [-0.1, 0.3, 0.4, 0.2],
    [0.2, -0.1, -0.2, 0.3]
]

bias_hidden = [0.25, 0.45]

w = [
    [-0.1, 0.2],
    [0.3, -0.2],
    [-0.4, 0.1]
]

bias_output = [-0.4, 0.2, 0.1]

alpha = 0.1

max_epoch = 1100

####################################################################################################

def fungsi_aktivasi(x) :
    delimiter = 1 + np.exp(-x)
    return 1/delimiter

def fungsi_aktivasi_turunan(x) :
    pengkali = 1 - fungsi_aktivasi(x)
    return fungsi_aktivasi(x)*pengkali

def bobot_nguyen_widrow(n, p, v_i_j) :
    root_value = 1/n
    β = 0.7 * (p ** root_value)
    len_v = []
    for i in v_i_j :
        sum = 0
        for j in i :
            sum += j ** 2
        len_v.append(sum ** 0.5)
    v_i_j_baru = []
    for i in range(len(v_i_j)) :
        each = []
        for j in v_i_j[i] :
            value = j * β / len_v[i]
            each.append(value)
        v_i_j_baru.append(each)
    return v_i_j_baru

def each_feedforward(v_i_j_baru, each_data_latih, bias_hidden, w, bias_output) :
    # perhitungan z
    z_in = []
    for i in range(len(v_i_j_baru)) :
        each_z_in = 0
        for j in range(len(v_i_j_baru[0])) : 
            multi = v_i_j_baru[i][j] * each_data_latih[j]
            each_z_in += multi
        each_z_in += bias_hidden[i]
        z_in.append(each_z_in)
    z = [fungsi_aktivasi(i) for i in z_in]
    # perhitungan y
    y_in = []
    for i in range(len(w)) :
        each_y_in = 0
        for j in range(len(w[0])) :
            multi = w[i][j] * z[j]
            each_y_in += multi
        each_y_in += bias_output[i]
        y_in.append(each_y_in)
    y = [fungsi_aktivasi(i) for i in y_in]
    return z, z_in, y, y_in

def each_backpropagation(each_data_latih, each_target_data_latih, w, alpha, z, z_in, y, y_in) :
    ### perhitungan bobot output layer
    # 1. hitung t-y
    t_min_y = []
    for i in range(len(y)) :
        t_min_y.append(each_target_data_latih[i] - y[i])
    turunan_y_in = [fungsi_aktivasi_turunan(i) for i in y_in]

    # 1. faktor error output
    δk = [t_min_y[i] * turunan_y_in[i] for i in range(len(t_min_y))]

    # 2. ubah bobot output
    Δw_j_k = []
    for i in range(len(δk)) :
        δk_mul_z = []
        for j in range(len(z)) :
            multi = δk[i] * z[j] * alpha
            δk_mul_z.append(multi)
        Δw_j_k.append(δk_mul_z)
        
    # 3. perhitungan bias output
    Δw_0_k = [alpha * i for i in δk]
    
    ### perhitungan bobot hidden layer
    # do transpose with w untuk mempermudah perhitungan
    w_transpose = np.transpose(w)

    # 1. perhitungan delta kecil in j (seperti t-y tapi di hidden layer)
    δin_j = []
    for i in range(len(w_transpose)) :
        each = 0
        for j in range(len(w_transpose[0])) :
            each += w_transpose[i][j] * δk[j]
        δin_j.append(each)

    # 1. faktor error hidden 
    turunan_z_in = [fungsi_aktivasi_turunan(i) for i in z_in]
    δj = [δin_j[i] * turunan_z_in[i] for i in range(len(δin_j))]

    # 2. ubah bobot hidden
    Δv_i_j = []
    for i in range(len(δj)) :
        δz = []
        for j in range(len(each_data_latih)) :
            multi = δj[i] * each_data_latih[j] * alpha
            δz.append(multi)
        Δv_i_j.append(δz)
    
    # 3. perhitungan bias hidden
    Δv_0_j = [alpha * i for i in δj]

    # return np.round_(Δw_j_k, 5), np.round_(Δw_0_k, 5), np.round_(Δv_i_j, 5), np.round_(Δv_0_j, 5)
    return Δw_j_k, Δw_0_k, Δv_i_j, Δv_0_j

def each_update_bobot(w, bias_output, v_i_j_baru, bias_hidden, Δw_j_k, Δw_0_k, Δv_i_j, Δv_0_j) :
    w_baru = update_bobot_func(w, Δw_j_k)
    bias_output_baru = update_bobot_func(bias_output, Δw_0_k)
    v_i_j_baru = update_bobot_func(v_i_j_baru, Δv_i_j)
    bias_hidden_baru = update_bobot_func(bias_hidden, Δv_0_j)

    return w_baru, bias_output_baru, v_i_j_baru, bias_hidden_baru

def update_bobot_func(matrix1, matrix2) :
    row_len = len(matrix1)
    if type(matrix1[0]) is type(matrix1) :
        col_len = len(matrix1[0])
        result = np.zeros((row_len, col_len))
        for i in range(row_len) :
            for j in range(col_len) :
                result[i][j] = matrix1[i][j] + matrix2[i][j]
    else :
        result = np.zeros(row_len)
        for i in range(row_len) :
            result[i] = matrix1[i] + matrix2[i]
    return result

def pengujian(v_i_j_baru, data_latih, target_data_latih, bias_hidden, w, bias_output, alpha, max_epoch) :
    epoch = 0
    while (epoch < max_epoch) :
        epoch += 1
        z, z_in, y, y_in = each_feedforward(v_i_j_baru, data_latih[0], bias_hidden, w, bias_output)
        Δw_j_k, Δw_0_k, Δv_i_j, Δv_0_j = each_backpropagation(data_latih[0], target_data_latih[0], w, alpha, z, z_in, y, y_in)
        w, bias_output, v_i_j_baru, bias_hidden = each_update_bobot(w, bias_output, v_i_j_baru, bias_hidden, Δw_j_k, Δw_0_k, Δv_i_j, Δv_0_j)
        print(epoch)
    return w, bias_output, v_i_j_baru, bias_hidden

if __name__ == "__main__" :

    v_i_j_baru = bobot_nguyen_widrow(len(data_latih[0]), len(target_data_latih), v_i_j)

    w, bias_output, v_i_j_baru, bias_hidden = pengujian(v_i_j_baru, data_latih, target_data_latih, bias_hidden, w, bias_output, alpha, max_epoch)

    print('bobot output = ', w, '\n')
    print('bias output = ', bias_output, '\n')
    print('bobot hidden = ', v_i_j_baru, '\n')
    print('bias hidden = ', bias_hidden, '\n')