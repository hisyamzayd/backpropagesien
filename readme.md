# Backpropagation Neural Network with Iris Dataset

## Iris Dataset

|  Petal Length |  Petal Width |  Sepal Length |  Sepal Width |      Class      |
|:-------------:|:------------:|:-------------:|:------------:|:---------------:|
|      5.1      |      3.5     |      1.4      |      0.2     |   iris-setosa   |
|      6.2      |      2.2     |      4.5      |      1.5     | iris-versicolor |
|      7.7      |      2.8     |      6.7      |      2.0     |  iris-virginica |

## Activation Function

Using binary sigmoid function

<img src="https://latex.codecogs.com/gif.latex?f(x)&space;=&space;\frac{1}{1&plus;e^-x}" title="f(x) = \frac{1}{1+e^-x}" />

```python
def fungsi_aktivasi(x) :
    delimiter = 1 + np.exp(-x)
    return 1/delimiter
```

and its derivative 

<img src="https://latex.codecogs.com/gif.latex?f'(x)&space;=&space;f(x)*[1-f(x)]" title="f'(x) = f(x)*[1-f(x)]" />

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
