import numpy as np; np.random.seed(318375); n=10; B=np.random.randint(-5, 6, size=(n, n)); A=B@B.T+n*np.eye(n); G=np.linalg.cholesky(A); print('Residual 2-norm:', np.linalg.norm(A - G@G.T, 2))
