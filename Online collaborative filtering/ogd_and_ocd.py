import numpy as np
import scipy.io
import time
from scipy.sparse.linalg import svds  # Keep svds for OCG
from scipy.linalg import svd  # Use full svd for OGD
import matplotlib.pyplot as plt

# Load the .mat file
data = scipy.io.loadmat('./movie_lens_3.mat')['data']

# Extracting max values for matrix dimensions
m = int(np.max(data[:, 0]))  # Row number
n = int(np.max(data[:, 1]))  # Column number
num_data = len(data)  # Number of data points
print(num_data)

# Hyperparameters
bs = 100  # Batch size
T = int(num_data / bs)
T = 1000 # Total steps
sss = 0.1
ss = sss / np.sqrt(3*T)  # Selected by Grid Search 

maxNorm = 5000
ccc = 1e4
eta_c = ccc  # Selected by Grid Search

# Parameters for OCG-algorithm 
N = 1
tau = np.zeros(N)
p = np.zeros(N)
X = np.zeros((m, n))

for i in range(N):
    tau[i] = 2**i

loss_meta = np.zeros(T)
loss_ogd = np.zeros(T)
time_meta = np.zeros(T)
time_ogd = np.zeros(T)


K = 1
eta = np.zeros(N)
for i in range(N):
    eta[i] = eta_c / np.sqrt(tau[i])

XS = np.zeros((m, n, N))
G = np.zeros((m, n, N))
W_ogd = np.zeros((m, n))  # Weights for OGD

# Run OCG using svds
start_time_meta = time.time()
for i in range(T):
    print(f"Meta Algorithm Iteration {i+1}/{T}")
    flag = np.mod(i, tau) == 0
    flag[0] = True
    G[:, :, flag] = 0
    XS[:, :, flag] = 0

    # Generate a decision of OCG-algorithm
    X = XS

    # Observe loss and update parameters for OCG Algorithm
    f_meta = 0
    f_subs = np.zeros(N)

    for j in range(i * bs, (i + 1) * bs):
        index_m = int(data[j, 0]) - 1  # Adjusting to 0-based index 
        index_n = int(data[j, 1]) - 1  # Adjusting to 0-based index
        v = data[j, 2]

        f_meta += np.abs(X[index_m, index_n] - v)        #updating the loss

        for jj in range(N):
            f_subs[jj] += np.abs(XS[index_m, index_n, jj] - v)
            G[index_m, index_n, jj] += np.sign(XS[index_m, index_n, jj] - v)

    loss_meta[i] = f_meta

    # OCG-Algorithm Update using svds (Sparse SVD)
    Xj = XS[:, :, 0]
    DF = eta[0] * G[:, :, 0] + 2 * Xj
    u, s, v = svds(-DF, k=1)  # Use sparse SVD here
    V = maxNorm * np.outer(u, v)
    Delta = V - Xj

    flag2 = np.sum(-DF * Delta)
    if flag2 <= 1e-3:
        break

    sigma = 0.5 * flag2 / np.sum(Delta * Delta)
    sigma = np.clip(sigma, 0, 1)

    Xj += sigma * Delta
    XS[:, :, 0] = Xj

    time_meta[i] = time.time() - start_time_meta  # Track time for OCG-algorithm

# Run Online Gradient Descent (OGD) using svd
start_time_ogd = time.time()
for i in range(T):
    # Observe loss and update parameters for OGD
    print("Ogd",i)
    f_ogd = 0
    gradient_matrix = np.zeros_like(W_ogd)  # Gradient matrix to accumulate updates

    for j in range(i * bs, (i + 1) * bs):
        index_m = int(data[j, 0]) - 1  # Adjusting to 0-based index
        index_n = int(data[j, 1]) - 1  # Adjusting to 0-based index
        v = data[j, 2]

        prediction_ogd = W_ogd[index_m, index_n]
        f_ogd += np.abs(prediction_ogd - v)

        # Update the gradient matrix
        gradient_ogd = np.sign(prediction_ogd - v)
        gradient_matrix[index_m, index_n] = gradient_ogd

    # Perform SVD-based update for OGD
    u, s, vt = svd(W_ogd - ss * gradient_matrix, full_matrices=False)  # Full SVD
    W_ogd = maxNorm * np.outer(u[:, 0], vt[0, :])  # Update weights using rank-1 approximation

    loss_ogd[i] = f_ogd
    time_ogd[i] = time.time() - start_time_ogd  # Track time for OGD

# Save results for both algorithms
#np.savez('./Result/Multi_IOCG_OGD_separate.npz', loss_meta=loss_meta, time_meta=time_meta, loss_ogd=loss_ogd, time_ogd=time_ogd)

# Assuming 'loss_meta' and 'time_meta' are the results from the algorithm
# Slice the data for the first 10 iterations
num_iterations = T

loss_meta_10 = np.cumsum(loss_meta)
time_meta_10 = np.cumsum(time_meta)
loss_meta_10 = loss_meta_10[:num_iterations]
time_meta_10 = time_meta_10[:num_iterations]

loss_ogd_10 = np.cumsum(loss_ogd)
time_ogd_10 = np.cumsum(time_ogd)
loss_ogd_10 = loss_ogd_10[:num_iterations]
time_ogd_10 = time_ogd_10[:num_iterations]

# Plot 1: Cumulative Loss Over First 1000 Iterations for Meta-Algorithm and OGD
plt.figure(figsize=(10, 6))
plt.plot(loss_meta_10, label='Meta Algorithm Loss (First 1000)', color='b')
plt.plot(loss_ogd_10, label='OGD Loss (First 1000)', color='r')
plt.xlabel('Iterations')
plt.ylabel('Cumulative Loss')
plt.title('Cumulative Loss Over First 1000 Iterations')
plt.legend()
plt.grid(True)
plt.savefig('./cumulative_loss_meta_ogd_1000_iterations.png')
plt.show()

# Plot 2: Time Per Iteration (First 1000 Iterations) for Meta-Algorithm and OGD
plt.figure(figsize=(10, 6))
time_meta_per_iteration_10 = np.diff(time_meta_10, prepend=0)
time_ogd_per_iteration_10 = np.diff(time_ogd_10, prepend=0)

plt.plot(time_meta_per_iteration_10, label='Meta Algorithm Time (First 1000)', color='b')
plt.plot(time_ogd_per_iteration_10, label='OGD Time (First 1000)', color='r')
plt.xlabel('Iterations')
plt.ylabel('Time (seconds)')
plt.title('Time Per Iteration (First 1000 Iterations)')
plt.legend()
plt.grid(True)
plt.savefig('./time_per_iteration_meta_ogd_1000_iterations.png')
plt.show()

