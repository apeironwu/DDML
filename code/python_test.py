import numpy as np

np.random.seed(128)

K = 3
n = 5
p = 2

covmat_U_eps = np.fromfunction(lambda i, j: np.power(0.99, np.abs(i - j)), (2, 2))

# print(covmat_U_eps)


# arr_U_eps = np.random.multivariate_normal(np.zeros(2), covmat_U_eps, K * n).reshape(K, n, 2).transpose(2, 0, 1)
# mat_U = arr_U_eps[0]
# mat_eps = arr_U_eps[1]


covmat_U_eps = np.fromfunction(lambda i, j: np.power(0.99, np.abs(i - j)), (2, 2))
arr_U_eps = np.random.multivariate_normal(np.zeros(2), covmat_U_eps, K * n).T.reshape(2, n, K)

mat_U = arr_U_eps[0] 
mat_eps = arr_U_eps[1]

print(arr_U_eps)
print("====mat_U: \n",  mat_U)
print("==== mat_eps: \n", mat_eps)