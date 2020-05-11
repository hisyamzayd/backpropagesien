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