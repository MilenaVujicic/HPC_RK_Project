from mpi4py import MPI
import numpy as np
import matplotlib.pyplot as plt

    
def calculate_new_y(f, x, h, y):

    k1 = f(x, y)
    k2 = f(x+h/2, y+h*k1/2)
    k3 = f(x+h, y-h*k1+2*k2*h)
    y = y + h*(k1+4*k2+k3)/6
    return y

def rk3_parallelized(f, xb, yb, h, xe, comm):
    rank = comm.Get_rank()
    size = comm.Get_size()
    x_vals = []

    if rank == 0:
        x=xb
        while x < xe:
            x_vals.append(x)
            x +=h
        comm.bcast(x_vals, root=0)
    
        return [],[]
    else:
        x_recv = []
        x_recv = comm.bcast(x_vals, root=0)
        step = int(len(x_recv) / (size-1))
        y_temp = []
        if rank != (size-1):
            b_idx = (rank-1)*step
            e_idx = rank*step
            for x in x_recv[b_idx:e_idx]:
                y = calculate_new_y(f, x, yb, h)
                y_temp.append(y)
            return x_recv[b_idx: e_idx], y_temp
        
        else:
            b_idx = (rank-1)*step
            e_idx = len(x_recv)
            for x in x_recv[b_idx:e_idx]:
                y = calculate_new_y(f, x, yb, h)
                y_temp.append(y)
              
            return x_recv[b_idx: e_idx], y_temp
