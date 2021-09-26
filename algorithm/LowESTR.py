import numpy as np
import math
from scipy import linalg
from stage1 import *

# adapted from the codes by Yangyi Lu, Amirhossein Meisami, Ambuj Tewari in their work "Low-Rank Generalized Linear Bandit Problems"
# explore the low rank subspace
class explore:
    def __init__(self, k, d1, d2, iters, theta, X, sigma, lam, gamma, r, model):
        # number of arms
        self.k = k
        # feature dimensions
        self.d1 = d1
        self.d2 = d2
        # number of iterations
        self.iters = iters
        # noise sd
        # step count
        self.t = 0
        self.sigma = sigma
        # linear coefficient
        self.theta = theta
        # rank of linear coefficient
        self.r = r
        # feature for each arm: k by d1d2 matrix
        self.X = X
        # random reward and expected reward
        self.reward = np.zeros(iters)
        self.exp_reward = np.zeros(iters)
        # expected regret, cumulative regret
        self.regret = np.zeros(iters)
        self.cum_regret = np.zeros(iters)
        # arm pulled
        self.arm = np.zeros(iters)
        # opt arm
        self.opt_arm = 0
        self.opt_exp_reward = 0
        # squared loss
        self.loss = 0
        # ridge regression parameter
        self.lam = lam
        # GD parameter
        self.gamma = gamma
        # estimated subspaces and theta
        self.theta_est = 0
        self.U_hat, self.U_hat_perp = 0,0
        self.V_hat, self.V_hat_perp = 0,0
        self.S_hat = 0
        self.model = model
        self.bfunc = getattr(reward_model(), model)
        self.mufunc = getattr(reward_model(), 'mu_' + model)
        self.dfunc = getattr(reward_model(), 'd_' + model)

    def get_opt_arm(self):
        exp_reward = np.zeros(self.k)
        for i in range(self.k):
            exp_reward[i] = np.inner(self.theta.flatten(),self.X[i,:])
        self.opt_arm = np.argmax(exp_reward)
        self.opt_exp_reward = self.mufunc(exp_reward[self.opt_arm])

    def pull(self):
        # select arm
        self.arm[self.t] = np.random.choice(self.k)
        # update expected reward and random reward
        self.exp_reward[self.t] = np.inner(self.theta.flatten(), self.X[self.arm[self.t].astype(int), :])
        if self.model == 'logistic':
            self.reward[self.t] = np.random.binomial(1,self.mufunc(self.exp_reward[self.t]))
        if self.model == 'poisson':
            self.reward[self.t] = np.random.poisson(self.mufunc(self.exp_reward[self.t]))
        self.regret[self.t] = self.opt_exp_reward - self.mufunc(self.exp_reward[self.t])
        self.cum_regret[self.t] = sum(self.regret[:(self.t+1)])
        self.t += 1

    def run(self):
        self.get_opt_arm()
        for i in range(self.iters):
            self.pull()
        # proximal GD: solve an estimate for theta using self.reward, self.arm and self.iters
        theta = np.random.normal(0, 1, size=(self.d1, self.d2))
        gap = 1
        #for j in range(10000):
        while gap>1e-3:
#           print(np.linalg.norm(theta-self.theta, ord='fro'))
            exp_reward = np.matmul(self.X[self.arm.astype(int), :], theta.reshape((self.d1*self.d2, 1))).flatten()
            diff = exp_reward-self.reward
            # proximal GD
            sq_grad_theta = (1.0/self.iters)*np.matmul(diff.reshape((1,self.iters)), self.X[self.arm.astype(int), :])
            sq_grad_theta = sq_grad_theta.reshape((self.d1, self.d2))
            theta_new = theta - self.gamma*sq_grad_theta
            # SVD for theta_new
            P,D,Q = linalg.svd(theta_new, full_matrices=True)
            D_new = [d-self.lam if d >= self.lam else 0 for d in D]
            theta_new = np.dot(P, np.dot(np.diag(D_new),Q))
            gap = np.linalg.norm(theta-theta_new, ord='fro')
            theta = theta_new
        self.theta_est = theta
        # SVD
        U_hat_full,S_hat_full,V_hat_full = linalg.svd(self.theta_est, full_matrices=True)
        # low rank approx
        self.U_hat, self.U_hat_perp = U_hat_full[:,:self.r], U_hat_full[:,self.r:]
        self.V_hat, self.V_hat_perp = np.transpose(V_hat_full[:self.r,:]), np.transpose(V_hat_full[self.r:,:])
        self.S_hat = S_hat_full[:self.r]


class LowOFUL:
    def __init__(self, k, d1, d2, r, theta, X, sigma, U_hat, V_hat, U_hat_perp, V_hat_perp, T2, delta, lam, lam_perp, B, B_perp, model):
        # number of arms
        self.k = k
        # feature dimensions
        self.d1 = d1
        self.d2 = d2
        # linear coefficient and its rank
        self.theta = theta
        self.r = r
        # arm set
        self.X = X
        # noise sd
        self.sigma = sigma
        # estimated U, V, U_perp, V_perp
        self.U_hat = U_hat
        self.V_hat = V_hat
        self.U_hat_perp = U_hat_perp
        self.V_hat_perp = V_hat_perp
        # number of steps
        self.T2 = T2
        # low dim k
        self.low_k = (self.d1+self.d2)*self.r-self.r**2
        # failure rate
        self.delta = delta
        # penalty parameters
        self.lam = lam
        self.lam_perp = lam_perp
        self.B = B
        self.B_perp = B_perp
        # rotated arms and coefficients
        self.X_rot = np.zeros(shape=(self.k, self.d1*self.d2))
        self.theta_rot = np.zeros(shape=(self.d1*self.d2,))
        # random reward
        self.reward = []
        # opt arm
        self.opt_arm = 0
        self.opt_exp_reward = 0
        # expected regret, cumulative regret
        self.regret = []
        self.cum_regret = []
        self.model = model
        self.bfunc = getattr(reward_model(), model)
        self.mufunc = getattr(reward_model(), 'mu_' + model)
        self.dfunc = getattr(reward_model(), 'd_' + model)


    def get_opt_arm(self):
        exp_reward = np.zeros(self.k)
        for i in range(self.k):
            exp_reward[i] = np.inner(self.theta.flatten(),self.X[i,:])
        self.opt_arm = np.argmax(exp_reward)
        self.opt_exp_reward = self.mufunc(exp_reward[self.opt_arm])

    def rotate(self):
        U_full = np.hstack((self.U_hat, self.U_hat_perp))
        V_full = np.hstack((self.V_hat, self.V_hat_perp))
        theta_mat_rot = np.matmul(np.matmul(np.transpose(U_full),self.theta),V_full)
        self.theta_rot = np.concatenate((theta_mat_rot[:self.r, :self.r].flatten(), theta_mat_rot[self.r:, :self.r].flatten(), theta_mat_rot[:self.r, self.r:].flatten(), theta_mat_rot[self.r:, self.r:]), axis=None)
        for i in range(self.k):
            X_i_mat = self.X[i,:].reshape((self.d1, self.d2))
            X_i_mat_rot = np.matmul(np.matmul(np.transpose(U_full),X_i_mat),V_full)
            self.X_rot[i,:] = np.concatenate((X_i_mat_rot[:self.r, :self.r].flatten(), X_i_mat_rot[self.r:, :self.r].flatten(), X_i_mat_rot[:self.r, self.r:].flatten(), X_i_mat_rot[self.r:, self.r:]), axis=None)

    def run(self):
        self.get_opt_arm()
        self.rotate()
        Lam_vec = np.concatenate((np.full((self.low_k,),self.lam),np.full((self.d1*self.d2-self.low_k,),self.lam_perp)),axis=None)
        Lam = np.diag(Lam_vec)
        V, theta_hat, g = Lam, np.zeros(shape=(self.d1*self.d2,1)), np.zeros(shape=(self.d1*self.d2,1))
        beta_sq = self.sigma*math.sqrt(2*math.log(1/self.delta))+math.sqrt(self.lam)*self.B+math.sqrt(self.lam_perp)*self.B_perp
        UCB_a = [0]*self.k
        for t in range(self.T2):
            for i in range(self.k):
                bonus = beta_sq*np.inner(self.X_rot[i,:],np.matmul(np.linalg.inv(V),self.X_rot[i,:].reshape((self.d1*self.d2,1))).flatten())
                UCB_a[i] = np.inner(theta_hat.flatten(),self.X_rot[i,:]) + bonus
            i_pull = np.argmax(UCB_a)
            # if t%10==0:
            #     print("t")
            #     print(t)
            #     print("i pull")
            #     print(i_pull)
            #     print("opt arm")
            #     print(self.opt_arm)
            # arm to be pulled: d1d2 by 1
            X_pull = self.X_rot[i_pull,:].reshape(self.d1*self.d2,1)
            # store random reward
            curr_exp_reward = np.inner(self.X_rot[i_pull,:],self.theta_rot)
            if self.model == 'logistic':
                v = np.random.binomial(1, self.mufunc(curr_exp_reward))
            if self.model == 'poisson':
                v = np.random.poisson(self.mufunc(curr_exp_reward))
            self.reward.append(v)
            self.regret.append(self.opt_exp_reward-self.mufunc(curr_exp_reward))
#           print(self.opt_exp_reward-self.mufunc(curr_exp_reward))
            self.cum_regret.append(sum(self.regret))
#           print(self.cum_regret[-1])
            # update confidence interval of true vector theta
            V = V + np.matmul(X_pull, np.transpose(X_pull))
            g = g + X_pull*self.reward[-1]
            theta_hat = np.matmul(np.linalg.inv(V),g)
            beta_sq = self.sigma*math.sqrt(math.log(np.linalg.det(V)/np.linalg.det(Lam)/(self.delta**2)))+math.sqrt(self.lam)*self.B+math.sqrt(self.lam_perp)*self.B_perp

