import numpy as np
import matplotlib.pyplot as plt
import time
import rk3
import rk4
import functions as f
from mpi4py import MPI


def main():

    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    size = comm.Get_size()

    x = 0
    y = 0.1
    h= 0.001
    x_end = 5

    x3 = float(x)
    y3 = float(y)
    h3 = float(h)
    x_end3 = float(x_end)

    x4 = float(x)
    y4 = float(y)
    h4 = float(h)
    x_end4 = float(x_end)
    
    for i in range(size):
        if rank == i:
            start3 = time.time()
            x_rest, y_rest = rk3.rk3_parallelized(f.func, x3, y3, h3, x_end3, comm)
            end3 = time.time()

            if rank == 0:
                pack = []
                pack.append((rank, x_rest, y_rest))
               
                comm.send(pack, dest=rank+1, tag=0)
                print("Rank", rank, "sending")
            elif rank != size - 1:
                pack = comm.recv(source=rank-1, tag=0)
                pack.append((rank, x_rest, y_rest))
                comm.send(pack, dest=rank+1, tag=0)
                print("Rank", rank, "sending")

            else:
                pack = comm.recv(source=rank-1, tag=0)
                pack.append((rank, x_rest, y_rest))
                print(pack[1][0])
                for i in range(1, size):
                    plt.title("RK3")
                    plt.plot(np.array(pack[i][1]), np.array(pack[i][2]), color='red', linestyle='-')
                print("\nTotal time is: ", end3-start3)
    

    
    for i in range(size):
        if rank == i:
            start4 = time.time()
            x_rest, y_rest = rk4.rk4_parallelized(f.func, x4, y4, h4, x_end4, comm)
            end4 = time.time()

            if rank == 0:
                pack = []
                pack.append((rank, x_rest, y_rest))
               
                comm.send(pack, dest=rank+1, tag=0)
                print("Rank", rank, "sending")
            elif rank != size - 1:
                pack = comm.recv(source=rank-1, tag=0)
                pack.append((rank, x_rest, y_rest))
                comm.send(pack, dest=rank+1, tag=0)
                print("Rank", rank, "sending")

            else:
                pack = comm.recv(source=rank-1, tag=0)
                pack.append((rank, x_rest, y_rest))
                print(pack[1][0])
                for i in range(1, size):
                    plt.title("RK4")
                    plt.plot(np.array(pack[i][1]), np.array(pack[i][2]), color='green', linestyle='-')
                print("\nTotal time is: ", end4-start4)
        
    plt.show()

 

if __name__ == "__main__":
    main()