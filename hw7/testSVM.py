import numpy as np
from numpy import linalg
import cvxopt
import cvxopt.solvers

def svm(X, y):
	n_samples, n_features = X.shape
	print n_samples

	# Gram matrix
	K = np.zeros((n_samples, n_samples))
	for i in range(n_samples):
    		for j in range(n_samples):
        			K[i,j] = np.dot(X[i], X[j])

	P = cvxopt.matrix(np.outer(y,y) * K)
	q = cvxopt.matrix(np.ones(n_samples) * -1)
	A = cvxopt.matrix(y, (1,n_samples))
	b = cvxopt.matrix(0.0)

	G = cvxopt.matrix(np.diag(np.ones(n_samples) * -1))
	h = cvxopt.matrix(np.zeros(n_samples))

	# solve QP problem
	solution = cvxopt.solvers.qp(P, q, G, h, A, b)

	print solution

	# Lagrange multipliers
	a = np.ravel(solution['x'])

	print "a", a

	# Support vectors have non zero lagrange multipliers
	ssv = a > 1e-5
	ind = np.arange(len(a))[ssv]
	a = a[ssv]
	sv = X[ssv]
	sv_y = y[ssv]
	print "%d support vectors out of %d points" % (len(a), n_samples)
	print "sv", sv
	print "sv_y", sv_y

	# Intercept
	b = 0
	for n in range(len(a)):
    		b += sv_y[n]
    		b -= np.sum(a * sv_y * K[ind[n],ssv])
	b /= len(a)

	# Weight vector
	w = np.zeros(n_features)
	for n in range(len(a)):
    		w += a[n] * sv_y[n] * sv[n]

	print "a", a
	return w, b, sv_y, sv, a

if __name__ == "__main__":

	def test():
    		X = np.array([[3.,3.],[4.,4.],[7.,7.],[8.,8.]])
    		y = np.array([1.,1.,-1.,-1.])
    		w, b, sv_y, sv, a = svm(X, y)
    		print "w", w
    		print "b", b
    		print 'test points'
    		print np.dot([2.,2.], w) + b # > 1
    		print np.dot([9.,9.], w) + b # < -1              
	test() 
