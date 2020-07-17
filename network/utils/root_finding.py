import torch
import torch.nn as nn
from torch.autograd import Variable

def newton(func, guess, threshold = 1e-7, max_iteration = None, sor_omega = 1):
    """
    Newton-optimization, finds a root of 'func'.

    Parameters:
     - func: function that takes a scalar tensor as input and returns a scalar tensor
     - guess: the initial guess of the input

    Returns:
     - the optimized input value
     - the number of iterations
    """

    if not isinstance(guess, torch.Tensor):
        guess = torch.tensor(guess)
    assert len(guess.shape)==0 # 0D-scalar

    guess = Variable(guess, requires_grad=True)
    value = func(guess)
    i = 0
    with torch.autograd.enable_grad():
        while abs(value.item()) > threshold:
            value = func(guess)
            value.backward()
            guess.data = guess.data - sor_omega * (value / guess.grad).data
            guess.grad.data.zero_()
            i += 1
            if max_iteration is not None and i >= max_iteration:
                break;
    return guess.data, i

def bisection(func, guess, step=1, threshold=1e-5, max_iteration = None):
    """
    Bisection algorithm, fins a root of 'func'.
    Assumes func to be monotonic.

     Parameters:
     - func: function that takes a scalar tensor as input and returns a scalar tensor
     - guess: the initial guess of the input

    Returns:
     - the optimized input value
     - the number of iterations
    """
    
    if max_iteration is None:
        max_iteration = 1000

    value = func(guess).item()
    #print("guess=%.5f -> value=%.5f"%(guess, value))
    if value > 0:
        right = guess
        right_value = value
        left = guess - step
        left_value = value
        while max_iteration>0:
            max_iteration -= 1
            value = func(left).item()
            assert value < left_value, "function is not monotonic increasing"
            left_value = value
            #print("left=%.5f -> left_value=%.5f"%(left, left_value))
            if left_value < 0:
                break
            step = step * 2
            left = guess - step
    else:
        left = guess
        left_value = value
        right = guess + step
        right_value = value
        while max_iteration>0:
            max_iteration -= 1
            value = func(right).item()
            assert value > right_value, "function is not monotonic increasing"
            right_value = value
            #print("right=%.5f -> right_value=%.5f"%(right, right_value))
            if right_value > 0:
                break
            step = step * 2
            right = guess + step
            
    i = 0
    while abs(value) > threshold and max_iteration>0:
        max_iteration -= 1
        # false position method
        guess = (left * right_value - right * left_value) / (right_value - left_value)
        value = func(guess).item()
        #print("guess=%.5f -> value=%.5f"%(guess, value))
        # keep left and right value
        if value > 0:
            right = guess
            right_value = value
        else:
            left = guess
            left_value = value
        i += 1
    return guess, i