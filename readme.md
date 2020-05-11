# Backpropagation Neural Network with Iris Dataset

## Iris Dataset
|  Petal Length |  Petal Width |  Sepal Length |  Sepal Width |      Class      |
|:-------------:|:------------:|:-------------:|:------------:|:---------------:|
|      5.1      |      3.5     |      1.4      |      0.2     |   iris-setosa   |
|      6.2      |      2.2     |      4.5      |      1.5     | iris-versicolor |
|      7.7      |      2.8     |      6.7      |      2.0     |  iris-virginica |

## Activation Function
Using binary sigmoid function

![\f(x) = \frac{1}{1+e^-x}](https://latex.codecogs.com/svg.latex?f%28x%29%20%3D%20%5Cfrac%7B1%7D%7B1&plus;e%5E-x%7D)

```python
def fungsi_aktivasi(x) :
    delimiter = 1 + np.exp(-x)
    return 1/delimiter
```

and its derivative 

![\[f'(x) = f(x)*[1-f(x)]](https://latex.codecogs.com/svg.latex?f%27%28x%29%20%3D%20f%28x%29*%5B1-f%28x%29%5D)

```python
def fungsi_aktivasi_turunan(x) :
    pengkali = 1 - fungsi_aktivasi(x)
    return fungsi_aktivasi(x)*pengkali
```

## Stop Condition
Using epoch. Max epoch = 1100

```python
while (epoch < max_epoch) :
```

## How to Run
Install Python 3.x -> Install Numpy at Python -> Use whatever IDE then RUN main.py

## Tahap Perhitungan Algoritma Pengujian
0. Inisialisasi data latih, bobot awal, bias awal semua layer semua neuron
1. Inisialisasi bobot Nguyen-Widrow (untuk bobot hidden saja)
2. Feed forward dari masing-masing row data latih
3. Fase distribusi error
4. Fase update bobot
5. Lakukan iterasi sampai memenuhi syarat tertentu (max epoch)

## LEGENDS
### 0 dan 1
n           = panjang kolom data latih (banyak fitur)
p           = panjang row target data latih (banyak data latih)
x           = data latih
t           = target data latih
v_i_j       = bobot awal untuk hidden neuron (acak)
v_i_j_baru  = bobot baru untuk hidden neuron (Nguyen-Widrow)
v_0_j       = bias awal untuk hidden neuron (acak)
w_j_k       = bobot awal untuk output neuron (acak)
w_0_k       = bias awal untuk output neuron (acak)

### 2
z_in         = hasil perhitungan input pada hidden neuron
z           = hasil perhitungan output (z_in) hidden neuron dengan f(x)
y_in        = hasil perhitungan input pada output neuron
y           = hasil perhitungan output (y_in) hidden neuron dengan f(x)

### 3
δk          = faktor error pada output layer
δin_j       = hasil jumlah delta error dari output neuron (menggunakan δk)
δj          = faktor error pada hidden layer

### 4
Δw_j_k      = perubahan bobot pada output layer (menggunakan δk)
Δw_0_k      = perubahan bias pada output layer (menggunakan δk)
Δv_i_j      = perubahan bobot pada hidden layer (menggunakan δj)
Δv_0_j      = perubahan bias pada hidden layer (menggunakan δj)

### 5
epoch       = genap 1 kali iterasi dari semua data latih