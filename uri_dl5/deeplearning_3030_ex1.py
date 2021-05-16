import timeit
class MyVector():
    def __init__(self,size,is_col=True,fill=0,init_values=None):
        self.vector = []
        self.size = size
        self.is_col = is_col
        if init_values != None:
            if size > len(init_values):
                self.vector.extend(init_values[0:len(init_values)])
                self.vector.extend(init_values[0:size - len(init_values)])
            else:
                self.vector = init_values

        else:
            for i in range(size):
                self.vector.append(fill)

    def __str__(self):
        s = "["
        if self.is_col:
            for i in range(len(self.vector)):
                s+="{0}, \n".format(self.vector[i])
            return s + "]"
        else:
            return str(self.vector)
        # Check for validity of self and other.  If scalar - will broadcast to
        # a vector
    # ---------------------------------------------------------------
    def __check_other(self, other):
        if not isinstance(other,MyVector):
            if (type(other) in [int, float]):
                other = MyVector(self.size, True, fill = other)
            else:
                raise ValueError("* Wrong type of parameter")
        if (self.is_col == False or other.is_col == False):
            raise ValueError("* both vectors must be column vectors")
        if (self.size != other.size):
            raise ValueError("* vectors must be of same size")
        return other    

    # ---------------------------------------------------------------
    # ADD vectors
    # ---------------------------------------------------------------
    def __add__(self,w):
        w = self.__check_other(w)        
        res = []
        for i in range(self.size):
            res.append(self.vector[i] + w.vector[i])
        return (MyVector(self.size, True, fill = 0, init_values=res))

    def __mul__(self,w):
        w = self.__check_other(w)        
        res = []
        for i in range(self.size):
            res.append(self.vector[i] * w.vector[i])
        return (MyVector(self.size, True, fill = 0, init_values=res))
    def __truediv__(self,w):
        w = self.__check_other(w)        
        res = []
        for i in range(self.size):
            res.append(self.vector[i] / w.vector[i])
        return (MyVector(self.size, True, fill = 0, init_values=res))

    def __sub__(self,w):
        w = self.__check_other(w)        
        res = []
        for i in range(self.size):
            res.append(self.vector[i] - w.vector[i])
        return (MyVector(self.size, True, fill = 0, init_values=res))
    def __getitem__(self, key):
        return self.vector[key]
    def __setitem__(self, key,item):
         self.vector[key]=item
    def transpose(self):
        return MyVector(self.size,not self.is_col,init_values=self.vector)
    def __radd__(self,w):
        return self.__add__(w)
    def __rmul__(self,w):
        return self.__mul__(w)
    def __rsub__(self,w):
         a=self.__check_other(w)
         return  a.__sub__(self)

    def __ne__(self,w):
        w=self.__check_other(w)
        r=MyVector(self.size)
        for i in range(r.size):
            if self.vector[i]!=w.vector[i]:
                r[i]=1
        return r

    def __lt__(self,w):
        w=self.__check_other(w)
        r=MyVector(self.size)
        for i in range(r.size):
            if self.vector[i]<w.vector[i]:
                r[i]=1
        return r

    def __le__(self,w):
        w=self.__check_other(w)
        r=MyVector(self.size)
        for i in range(r.size):
            if self.vector[i]<=w.vector[i]:
                r[i]=1
        return r

    def __eq__(self,w):
        w=self.__check_other(w)
        r=MyVector(self.size)
        for i in range(r.size):
            if self.vector[i]==w.vector[i]:
                r[i]=1
        return r
    def __gt__(self,w):
        w=self.__check_other(w)
        r=MyVector(self.size)
        for i in range(r.size):
            if self.vector[i]>w.vector[i]:
                r[i]=1
        return r
    def __ge__(self,w):
        w=self.__check_other(w)
        r=MyVector(self.size)
        for i in range(r.size):
            if self.vector[i]>=w.vector[i]:
                r[i]=1
        return r
    def dot(self,w):
        if not w.is_col or self.is_col:
            raise  ValueError("first vector must be row and second column")
        if w.size!=self.size:
            raise ValueError("vectors must be of same size")
        r=0
        for i in range(self.size):
            r+=self[i]*w[i]
        return r



import matplotlib.pyplot as plt
import numpy as np
import random
import unit10.b_utils as u10




def main():
    try:
        print(MyVector(3,is_col=False,fill=1).dot(MyVector(4,fill=2)))
    except ValueError as err1:
        print("Exception:",err1)
        try:
            print(MyVector(4,is_col=True,fill=1).dot(MyVector(4,fill=2)))
        except ValueError as err2:
            print("Exception:",err2)
            v1,v2 = MyVector(3,is_col=False,fill=4), MyVector(3,fill=2)
            print(v1.dot(v2),v1,v2)





### exeution
### ========
if __name__ == '__main__':
	main()


