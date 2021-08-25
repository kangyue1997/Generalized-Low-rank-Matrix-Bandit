import numpy as np
from scipy import linalg
import math

class reward_model():
    def __init__(self):
        pass
    def logistic(self,z):
        return np.log(1 + np.exp(z))
    def mu_logistic(self,z):
        return 1 / (1 + np.exp(-z))
    def d_logistic(self,z):
        return np.exp(-z) / np.square(1 + np.exp(-z))
    def poisson(self,z):
        return np.exp(z)
    def mu_poisson(self,z):
        return self.poisson(z)
    def d_poisson(self,z):
        return self.poisson(z)

def soft_thres(s,lambd):
    for i in range(s.shape[0]):
        if s[i] > lambd:
            s[i] = s[i] - lambd
        elif s[i] < -lambd:
            s[i] = s[i] + lambd
        else:
            s[i] = 0
    return s

def soft_thres_mat(X,lambd):
    u, s, vh = np.linalg.svd(X, full_matrices=False)
    return u.dot(np.diag(soft_thres(s,lambd))).dot(vh)



def collection(r,n_vec,d1,d2, seed_val = 0, seed = None):
    if seed is None:
        X = np.random.normal(0,1, size = [n_vec,d1*d2])
        for i in range(n_vec):
            X[i,:] /= np.linalg.norm(X[i,:], ord=2)
        return X*math.sqrt(r)
    else:
        np.random.seed(seed_val)
        X = np.random.normal(0,1, size=[n_vec, d1 * d2])
        for i in range(n_vec):
            X[i,:] /= np.linalg.norm(X[i, :], ord=2)
        return X*math.sqrt(r)


class stage1_pg(object):
    def __init__(self, n_vec, d1, d2, x, model, Theta, lambd, beta, step, error_bar, max_iter, n, context, seed_start = 0, r_pro = 1):
        self.x = x
        """ data """
        self.n_vec = n_vec
        self.d1 = d1
        self.d2 = d2
        self.context = context
        """whether it is contextual bandit or not"""
        if self.context:
            self.opt_arm = np.zeros(n, dtype=int)
            self.opt_arm_reward = np.zeros(n)
#            self.exp_reward = np.zeros(n, n_vec)
        else:
            self.opt_arm = 0
            self.opt_arm_reward = 0
#            self.exp_reward = np.zeros(n_vec)
        self.exp_reward = np.zeros(n_vec)
        self.model = model
        """generalized linear bandit type"""
        self.Theta = Theta
        """true Theta"""
        self.r = np.linalg.matrix_rank(Theta)
        """rank of Theta"""
        self.lambd = lambd
        """penalization parameter"""
        self.beta = beta
        """backtracking line search beta"""
        self.step = step
        """backtracking line search original stepsize"""
        self.error_bar = error_bar
        self.max_iter = max_iter
        self.n = n
        """# draws in stage 1"""
        self.regret = np.zeros(n)
        self.cum_regret = np.zeros(n)
        self.theta_est = 0
        self.U_hat, self.U_hat_perp = 0,0
        self.V_hat, self.V_hat_perp = 0,0
        self.S_hat = 0
        self.xnew = np.empty(shape=[0,d1,d2])
        self.ynew = 0
        self.bfunc = getattr(reward_model(), model)
        self.mufunc = getattr(reward_model(), 'mu_' + model)
        self.dfunc = getattr(reward_model(), 'd_' + model)
        self.t = 0
        self.arm = np.zeros(n, dtype=int)
        self.observed = np.zeros(n)
        self.Theta_est = 0
        self.seed_start = seed_start
        self.r_pro = r_pro

    def get_opt_arm(self):
        if self.context:
            for i in range(self.n):
                ins_x = collection(self.r_pro,self.n_vec,self.d1,self.d2,seed_val=self.seed_start+i, seed = True)
                ins_x = ins_x.reshape(self.n_vec, self.d1, self.d2)
                for j in range(self.n_vec):
                    self.exp_reward[j] = self.mufunc(np.sum(np.multiply(self.Theta,ins_x[j,:,:])))
                self.opt_arm[i] = np.argmax(self.exp_reward)
                self.opt_arm_reward[i] = np.max(self.exp_reward)
                arm = np.random.choice(self.n_vec)
                instant_reward = self.exp_reward[arm]
                self.regret[i] = self.opt_arm_reward[i] - instant_reward
                self.cum_regret[i] = np.sum(self.regret[:(i + 1)])
                self.arm[i] = arm
                if self.model == 'logistic':
                    self.observed[i] = np.random.binomial(1, instant_reward)
                if self.model == 'poisson':
                    self.observed[i] = np.random.poisson(instant_reward)
                self.xnew = np.vstack((self.xnew, ins_x[arm,:,:][np.newaxis,:,:]))
        else:
            for i in range(self.n_vec):
                self.exp_reward[i] = self.mufunc(np.sum(np.multiply(self.Theta,self.x[i,:,:])))
            self.opt_arm = np.argmax(self.exp_reward)
            self.opt_arm_reward = np.max(self.exp_reward)


    def pull(self):
        arm = np.random.choice(self.n_vec)
        instant_reward = self.exp_reward[arm]
        self.regret[self.t] = self.opt_arm_reward - instant_reward
        self.cum_regret[self.t] = np.sum(self.regret[:(self.t+1)])
        self.arm[self.t] = arm
        if self.model == 'logistic':
            self.observed[self.t] = np.random.binomial(1,instant_reward)
        if self.model == 'poisson':
            self.observed[self.t] = np.random.poisson(instant_reward)
        self.t += 1

    def run_xy(self):
        if self.context:
            self.get_opt_arm()
        else:
            self.get_opt_arm()
            for _ in range(self.n):
                self.pull()
            self.xnew = self.x[self.arm, :, :]
        self.ynew = self.observed
        return self.xnew, self.ynew

    def run(self):
        x = self.xnew
        y = self.ynew
        iter = 1
        Theta_old = np.random.normal(0, 1, size=(self.d1, self.d2))
        error = float('inf')
        while iter <= self.max_iter and error > self.error_bar:
            iter2 = 1
            t = self.step
            grad = np.sum((self.mufunc(x.reshape(x.shape[0],-1).dot(Theta_old.reshape(-1,1))).ravel() - y.ravel())[:,np.newaxis,np.newaxis]*x, axis = 0)/self.n
            prox = soft_thres_mat(Theta_old - t*grad, self.lambd*t)
            G = (Theta_old - prox)/t
            l1 = np.average(self.bfunc(x.reshape(x.shape[0],-1).dot(prox.reshape(-1,1))).ravel() - x.reshape(x.shape[0],-1).dot(prox.reshape(-1,1)).ravel()*y)
            r1 = np.average(self.bfunc(x.reshape(x.shape[0],-1).dot(Theta_old.reshape(-1,1))).ravel() - x.reshape(x.shape[0],-1).dot(Theta_old.reshape(-1,1)).ravel()*y)
            r2 = -t * np.sum(np.multiply(grad,G)) + (t/2)*np.linalg.norm(G)**2
            while iter2 <= self.max_iter and l1 > r1 + r2:
                iter2 += 1
                t = t * self.beta
                grad = np.sum((self.mufunc(x.reshape(x.shape[0], -1).dot(Theta_old.reshape(-1, 1))).ravel() - y.ravel())[:, np.newaxis, np.newaxis] * x, axis=0)/self.n
                prox = soft_thres_mat(Theta_old - t * grad, self.lambd * t)
                G = (Theta_old - prox) / t
                l1 = np.average(self.bfunc(x.reshape(x.shape[0], -1).dot(prox.reshape(-1, 1))).ravel() - x.reshape(x.shape[0],-1).dot(prox.reshape(-1, 1)).ravel() * y)
                r1 = np.average(self.bfunc(x.reshape(x.shape[0], -1).dot(Theta_old.reshape(-1, 1))).ravel() - x.reshape(x.shape[0], -1).dot(Theta_old.reshape(-1, 1)).ravel() * y)
                r2 = -t * np.sum(np.multiply(grad, G)) + (t / 2) * np.linalg.norm(G) ** 2
            Theta_new = prox
            error = np.linalg.norm(Theta_old - Theta_new)
            Theta_old = Theta_new
            iter += 1
#            print("Iteration number: {0}".format(iter))
        self.Theta_est = Theta_new
        U, S, V = linalg.svd(self.Theta_est, full_matrices = True)
        self.U_hat, self.U_hat_perp = U[:,:self.r], U[:,self.r:]
        self.S_hat = S[:self.r]
        self.V_hat, self.V_hat_perp = V.transpose()[:,:self.r], V.transpose()[:,self.r:]


class get_tilde(object):
    def __init__(self,d1, d2, model = 'logistic'):
        self.Theta = np.zeros([d1, d2])
        self.bfunc = getattr(reward_model(), model)
        self.mufunc = getattr(reward_model(), 'mu_' + model)
        self.dfunc = getattr(reward_model(), 'd_' + model)
        self.d1 = d1
        self.d2 = d2

    def reset(self):
        self.Theta = np.zeros([self.d1, self.d2])

    def tilde(self, x, y):
        tilde_x = np.empty(shape = [0,x.shape[1]*x.shape[2]])
        tilde_y = np.array([])
        for i in range(x.shape[0]):
            val = np.sqrt(self.dfunc(np.sum(np.multiply(self.Theta,x[i,:,:]))))
            tilde_x = np.vstack((tilde_x, val * x[i,:,:].reshape(1,-1)))
            tilde_y = np.append(tilde_y, (y[i]- self.mufunc(np.sum(np.multiply(self.Theta, x[i,:,:]))))/val)
        return tilde_x, tilde_y

def solve_theta_peaceman(lambd, get_tilde, x, y, alpha = 0.9, beta = 1, error_bar = 10**(-4), max_iter = 500):
    tilde_X, tilde_y = get_tilde.tilde(x,y)
    Theta = get_tilde.Theta
    d = x.shape[1]*x.shape[2]
    thetax = np.zeros(shape=[d])
    thetay = np.zeros(shape=[d])
    error = float('inf')
    iter = 1
    rho = np.zeros(shape=[d])
    while iter <= max_iter and error > error_bar:
        thetax_new = np.matmul(np.linalg.inv(2*tilde_X.transpose().dot(tilde_X) + beta*np.identity(d)),(beta*thetay- beta*Theta.ravel()+ rho + (2*tilde_X.transpose().dot(tilde_y)/x.shape[0])).reshape(-1,1)).ravel()
        rho_12 = rho - alpha*beta*(thetax_new -thetay + Theta.ravel())
        thetay_new = soft_thres_mat((thetax_new+Theta.ravel()-rho_12/beta).reshape(x.shape[1],x.shape[2]), 2*lambd/beta).reshape(1,-1)
        rho_new = rho_12 - alpha*beta*(thetax_new - thetay_new + Theta.ravel())
        error = np.max([np.linalg.norm(thetay - thetay_new), np.linalg.norm(thetax - thetax_new)])
        iter += 1
        thetay = thetay_new
        thetax = thetax_new
        rho = rho_new
    return thetay.reshape(x.shape[1],x.shape[2])

def peaceman(lambd,r,d1,d2, x,y,max_iter=500, error_bar = 10**(-4), theta_start = None):
    model = get_tilde(d1,d2)
    losses = []
    epoch = max_iter
    Theta_old = theta_start
    if theta_start is None:
        Theta_old = np.zeros([x.shape[1], x.shape[2]])
    for i in range(epoch):
        Theta_new = solve_theta_peaceman(lambd, model,x,y, error_bar= error_bar, max_iter=max_iter)
        loss = np.linalg.norm(Theta_new - Theta_old)
#        print("Iteration number: {0}".format(i+1))
        if loss < error_bar:
            break
        Theta_old, model.Theta = Theta_new, Theta_new
        losses.append(loss)
    U, S, V = linalg.svd(Theta_old, full_matrices=True)
    U_hat, U_hat_perp = U[:, :r], U[:, r:]
    S_hat = S[:r]
    V_hat, V_hat_perp = V.transpose()[:, :r], V.transpose()[:, r:]
    return Theta_old, U_hat, U_hat_perp, S_hat, V_hat, V_hat_perp



