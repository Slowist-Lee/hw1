"""Operator implementations."""

from numbers import Number
from typing import Optional, List, Tuple, Union

from ..autograd import NDArray
from ..autograd import Op, Tensor, Value, TensorOp
from ..autograd import TensorTuple, TensorTupleOp
import numpy

# NOTE: we will import numpy as the array_api
# as the backend for our computations, this line will change in later homeworks

BACKEND = "np"
import numpy as array_api

class EWiseAdd(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a + b

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad, out_grad


def add(a, b):
    return EWiseAdd()(a, b)


class AddScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a + self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad


def add_scalar(a, scalar):
    return AddScalar(scalar)(a)


class EWiseMul(TensorOp):
    def compute(self, a: NDArray, b: NDArray):
        return a * b

    def gradient(self, out_grad: Tensor, node: Tensor):
        lhs, rhs = node.inputs
        return out_grad*rhs, out_grad*lhs


def multiply(a, b):
    return EWiseMul()(a, b)


class MulScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a: NDArray):
        return a * self.scalar

    def gradient(self, out_grad: Tensor, node: Tensor):
        return out_grad*self.scalar


def mul_scalar(a, scalar):
    return MulScalar(scalar)(a)


class EWisePow(TensorOp):
    """Op to element-wise raise a tensor to a power."""

    def compute(self, a: NDArray, b: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,b)
        ### END YOUR SOLUTION
        
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        lhs,rhs=node.inputs
        return out_grad * rhs * lhs**(rhs-1),out_grad *lhs**rhs*log(lhs)
        ### END YOUR SOLUTION

def power(a, b):
    return EWisePow()(a, b)


class PowerScalar(TensorOp):
    """Op raise a tensor to an (integer) power."""

    def __init__(self, scalar: int):
        self.scalar = scalar

    def compute(self, a: NDArray) -> NDArray:
        ### BEGIN YOUR SOLUTION
        return array_api.power(a,self.scalar)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        x=node.inputs[0]
        return out_grad*self.scalar*x**(self.scalar-1)
        ### END YOUR SOLUTION


def power_scalar(a, scalar):
    return PowerScalar(scalar)(a)


class EWiseDiv(TensorOp):
    """Op to element-wise divide two nodes."""

    def compute(self, a, b):
        ### BEGIN YOUR SOLUTION
        return array_api.divide(a,b)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a,b=node.inputs
        return out_grad*1/b,out_grad*(-a)/(b**2)
        ### END YOUR SOLUTION


def divide(a, b):
    return EWiseDiv()(a, b)


class DivScalar(TensorOp):
    def __init__(self, scalar):
        self.scalar = scalar

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return a/self.scalar
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad/self.scalar
        ### END YOUR SOLUTION


def divide_scalar(a, scalar):
    return DivScalar(scalar)(a)


class Transpose(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        new_dim=list(range(a.ndim))
        if self.axes == None:
            new_dim[-1],new_dim[-2]=new_dim[-2],new_dim[-1]
        else:
            new_dim[self.axes[0]],new_dim[self.axes[1]]=new_dim[self.axes[1]],new_dim[self.axes[0]]
        return array_api.transpose(a,new_dim)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return Transpose(self.axes)(out_grad)
        ### END YOUR SOLUTION


def transpose(a, axes=None):
    return Transpose(axes)(a)


class Reshape(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.reshape(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return reshape(out_grad,node.inputs[0].shape)
        ### END YOUR SOLUTION


def reshape(a, shape):
    return Reshape(shape)(a)


class BroadcastTo(TensorOp):
    def __init__(self, shape):
        self.shape = shape

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.broadcast_to(a,self.shape)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        new_axes=[]
        padded_len=len(self.shape)-len(a.shape)
        new_shape=[1] * padded_len + list(a.shape)
        for i in range(len(new_shape)):
            if new_shape[i]==1 and self.shape[i]>1:
                new_axes.append(i)
        return reshape(summation(out_grad,tuple(new_axes)),a.shape)
        ### END YOUR SOLUTION


def broadcast_to(a, shape):
    return BroadcastTo(shape)(a)


class Summation(TensorOp):
    def __init__(self, axes: Optional[tuple] = None):
        self.axes = axes

    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.sum(a,self.axes)
        ### END YOUR SOLUTION
    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a = node.inputs[0]
        # 需要先将 out_grad reshape 到正确的形状，然后再 broadcast
        new_shape = list(a.shape)
        if self.axes is None:
            # 如果 axes 为 None，表示对所有维度求和，输出是标量
            new_shape = [1] * len(a. shape)
        else:
            # 对指定的 axes 求和
            axes = self.axes if isinstance(self.axes, tuple) else (self.axes,)
            for axis in axes:
                new_shape[axis] = 1
        
        # 先 reshape，再 broadcast
        out_grad_reshaped = reshape(out_grad, new_shape)
        return broadcast_to(out_grad_reshaped, a.shape)
        ### END YOUR SOLUTION
    # def gradient(self, out_grad, node):
    #     ### BEGIN YOUR SOLUTION
    #     a=node.inputs[0]
    #     # new_shape=list(out_grad.shape)
    #     # if self.axes==None:
    #     #     axes=range(len(a.shape))
    #     # else:
    #     #     axes=self.axes
    #     # for i in axes:
    #     #     new_shape[i]=1
    #     return broadcast_to(out_grad,a.shape)
    #     ### END YOUR SOLUTION


def summation(a, axes=None):
    return Summation(axes)(a)


# class MatMul(TensorOp):
#     def compute(self, a, b):
#         ### BEGIN YOUR SOLUTION
#         return array_api.matmul(a,b)
#         ### END YOUR SOLUTION
#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         a=node.inputs[0]
#         b=node.inputs[1]

#         grad_a=matmul(out_grad,transpose(b))
#         grad_b=matmul(transpose(a),out_grad)

#         # broadcast
#         diff_a=len(grad_a.shape)-len(a.shape)
#         diff_b=len(grad_b.shape)-len(b.shape)

#         pad_a_shape=[1]*diff_a+list(a.shape)
#         pad_b_shape=[1]*diff_b+list(b.shape)

#         axes_a=list(range(diff_a))
#         axes_b=list(range(diff_b))

#         for i in range(len(pad_a_shape)):
#             if pad_a_shape[i]!=grad_a.shape[i] and i not in axes_a:
#                 axes_a.append(i)

#         for i in range(len(pad_b_shape)):
#             if pad_b_shape[i] != grad_b.shape[i] and i not in axes_b:
#                 axes_b.append(i)

#         if axes_a:
#             grad_a=summation(grad_a,tuple(axes_a))
#         if axes_b:
#             grad_b=summation(grad_b,tuple(axes_b))

#         grad_a=grad_a.reshape(a.shape)
#         grad_b=grad_b.reshape(b.shape)
#         return grad_a, grad_b
#         ### END YOUR SOLUTION


class MatMul(TensorOp):
    def compute(self, a, b):
        return a@b

    def gradient(self, out_grad, node):
        a, b = node.inputs
        adjoint1 = out_grad @ transpose(b)
        adjoint2 = transpose(a) @ out_grad
        adjoint1 = summation(adjoint1, axes=tuple(range(len(adjoint1.shape) - len(a.shape))))
        adjoint2 = summation(adjoint2, axes=tuple(range(len(adjoint2.shape) - len(b.shape))))
        return adjoint1, adjoint2



def matmul(a, b):
    return MatMul()(a, b)


class Negate(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.zeros(a.shape)-a
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return -out_grad
        ### END YOUR SOLUTION


def negate(a):
    return Negate()(a)


class Log(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.log(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad*1/node.inputs[0]
        ### END YOUR SOLUTION


def log(a):
    return Log()(a)


class Exp(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.exp(a)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        a=node.inputs[0]
        return out_grad*exp(a)
        ### END YOUR SOLUTION


def exp(a):
    return Exp()(a)

# # RELU: max(0,x)
# class ReLU(TensorOp):
#     def compute(self, a):
#         ### BEGIN YOUR SOLUTION
#         # 前向计算不需要cache，已经是实际数据了
#         mask = (a>0)
#         return a*mask
#         ### END YOUR SOLUTION

#     def gradient(self, out_grad, node):
#         ### BEGIN YOUR SOLUTION
#         # node.inputs[0] 是一个对象，并不是一个具体的值，所以只能realize_cached_data
#         mask = (node.inputs[0].realize_cached_data()>0)
#         return out_grad*mask
#         ### END YOUR SOLUTION

class ReLU(TensorOp):
    def compute(self, a):
        ### BEGIN YOUR SOLUTION
        return array_api.maximum(a, 0)
        ### END YOUR SOLUTION

    def gradient(self, out_grad, node):
        ### BEGIN YOUR SOLUTION
        return out_grad * Tensor(array_api.greater(node.inputs[0].realize_cached_data(), 0))
        ### END YOUR SOLUTION

def relu(a):
    return ReLU()(a)

