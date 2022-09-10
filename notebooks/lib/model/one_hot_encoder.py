# one_hot_encoder.py
#Programmer: Tim Tyree
#Date: 1.15.2022
import numpy as np
def one_hot_encoding_simple(y):
    """returns Y as a one-hot encoding of input, y.
    y is a 1D numpy array with 1 unique value for each unique class.

    Example Usage:
y=np.array([0, 1, 1, 0, 1])
Y=one_hot_encoding_simple(y)
print(f"{Y=}")
    """
    classes=np.unique(y)
    #define an output Y matrix of the correct shape
    Y=np.zeros(shape=(y.shape[0],classes.shape[0]),dtype=int)
    #turn on entries that match
    for n,c in enumerate(classes):
        boo=y==c
        Y[boo,n]=1
    return Y

if __name__=="__main__":
    y=np.array([0, 1, 1, 0, 1])
    Y=one_hot_encoding_simple(y)
    print(f"one_hot_encoding_simple applied to {y=} returned")
    print(f"{Y=}")
