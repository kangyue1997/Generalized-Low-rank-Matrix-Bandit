from stage1 import *
from sklearn.linear_model import LogisticRegression
import math

def g_function(X,theta,model,lambd2,lambd2_perp,k):
    g1 = getattr(reward_model(), 'mu_' + model)
    p = X.shape[1]
    Lambd = np.diag(np.concatenate((np.full((k,),lambd2),np.full((p-k,),lambd2_perp)), axis = None))
    return g1(X.dot(theta)).reshape(1,-1).dot(X).ravel()+ Lambd.dot(theta)


def g_function_dev(X,theta,model,lambd2,lambd2_perp,k):
    """X is two dim matrix"""
    g1 = getattr(reward_model(), 'd_' + model)
    p = X.shape[1]
    Lambd = np.diag(np.concatenate((np.full((k,),lambd2),np.full((p-k,),lambd2_perp)), axis = None))
    return (g1(X.dot(theta.reshape(-1, 1))).ravel()[np.newaxis,:]*X.T).dot(X) + Lambd


def Projgd(theta_hat, M_inv, X, model, lambd2, lambd2_perp, k, step, max_iter = 5000, error_bar = 10**(-4)):
    theta_old = theta_hat - theta_hat
    error = float('inf')
    iter = 1
    while iter <= max_iter and error > error_bar:
        z = theta_old - step*g_function_dev(X,theta_old,model,lambd2,lambd2_perp,k).dot(M_inv).dot(g_function(X,theta_old,model, lambd2, lambd2_perp,k)-g_function(X,theta_hat,model, lambd2, lambd2_perp,k))
        theta_new = z/np.max((1, np.linalg.norm(z)))
        error = np.linalg.norm(theta_new-theta_old)
        theta_old = theta_new
        iter += 1
    return theta_new


def FindProject(theta_hat, M_inv, X,model,lambd2,lambd2_perp,k, col):
    The = col
    n_vec = col.shape[0]
    val = np.zeros(n_vec)
    for i in range(n_vec):
        v = g_function(X,The[i,:],model,lambd2,lambd2_perp,k) - g_function(X,theta_hat,model,lambd2,lambd2_perp,k)
        val[i] = v.T.dot(M_inv).dot(v).ravel()[0]
    return The[np.argmin(val),:]

class LowGLMUCB(object):
    def __init__(self,n, n_vec, d1, d2, x, model, Theta, lambd2, lambd2_perp, delta, S, S_perp, context,
                 U_hat, U_hat_perp, V_hat, V_hat_perp, c_mu, r_pro, k_mu, ymax, seed_start = 0, projgd = False, step = 0.1, history = False, hisx = 0, hisy = 0, rho = 1):
        self.col = collection(r_pro,2500,d1,d2)
        self.n = n
        self.x = x
        self.n_vec = n_vec
        self.d1 = d1
        self.d2 = d2
        self.Theta = Theta
        self.lambd2 = lambd2
        self.lambd2_perp = lambd2_perp
        self.r = np.linalg.matrix_rank(Theta)
        self.delta = delta
        self.S = S
        self.S_perp = S_perp
        self.context = context
        '''whether the contextual information varies'''
        self.model = model
        self.U_hat = U_hat
        self.U_hat_perp = U_hat_perp
        self.V_hat = V_hat
        self.V_hat_perp = V_hat_perp
        self.r = np.linalg.matrix_rank(Theta)
        self.k = self.r*(d1+d2-self.r)
        self.p = d1*d2
        self.Theta_rot = Theta
        self.x_rot = np.zeros(shape = [n_vec, d1*d2])
        self.U_full = 0
        self.V_full = 0
        self.c_mu = c_mu
        self.opt_arm = 0
        self.opt_arm_reward = 0
        self.bfunc = getattr(reward_model(), model)
        self.mufunc = getattr(reward_model(), 'mu_' + model)
        self.dfunc = getattr(reward_model(), 'd_' + model)
        self.exp_reward = np.zeros(n_vec)
        self.t = 0
        self.regret = np.zeros(n)
        self.cum_regret = np.zeros(n)
        self.r_pro = r_pro
        self.k_mu = k_mu
        self.ymax = ymax
        self.seed_start = seed_start
        self.projgd = projgd
        '''How to find the projected parameter value. T: projected gradient descent, F: discretization search'''
        self.step = step
        self.history = history
        '''Whether to use the information in Stage 1'''
        self.hisx = hisx
        self.hisy = hisy
        self.rho = rho

    def rotate_Theta(self):
        self.U_full = np.hstack((self.U_hat,self.U_hat_perp))
        self.V_full = np.hstack((self.V_hat,self.V_hat_perp))
        Theta_mat_rot = np.matmul(np.matmul(np.transpose(self.U_full), self.Theta), self.V_full)
        self.Theta_rot = np.concatenate((Theta_mat_rot[:self.r, :self.r].ravel(),Theta_mat_rot[self.r:, :self.r].ravel(),Theta_mat_rot[:self.r, self.r:].ravel(), Theta_mat_rot[self.r:, self.r:].ravel()),axis=None)

    def rotate_x(self):
        for i in range(self.n_vec):
            x_mat_rot = np.matmul(np.matmul(np.transpose(self.U_full), self.x[i,:,:]), self.V_full)
            self.x_rot[i,:] = np.concatenate((x_mat_rot[:self.r, :self.r].ravel(),x_mat_rot[self.r:, :self.r].ravel(),x_mat_rot[:self.r, self.r:].ravel(),x_mat_rot[self.r:, self.r:].ravel()), axis=None)

    def get_opt_arm(self):
        for i in range(self.n_vec):
            self.exp_reward[i] = self.mufunc(np.sum(np.multiply(self.Theta, self.x[i, :, :])))
        self.opt_arm = np.argmax(self.exp_reward)
        self.opt_arm_reward = np.max(self.exp_reward)


    def run(self):
        self.rotate_Theta()
        Lam_vec = np.concatenate(
            (np.full((self.k,), self.lambd2), np.full((self.d1 * self.d2 - self.k,), self.lambd2_perp)), axis=None)
        Lam, M, M_inv = np.diag(Lam_vec), np.diag(Lam_vec) / self.c_mu, self.c_mu * np.diag(1 / Lam_vec)
        X = np.empty([0, self.d1 * self.d2])
        y = []
        if self.history:
            for _ in range(self.hisx.shape[0]):
                x_mat_rot = np.matmul(np.matmul(np.transpose(self.U_full), self.hisx[self.t, :, :]), self.V_full)
                x_rot = np.concatenate((x_mat_rot[:self.r, :self.r].ravel(),
                                                   x_mat_rot[self.r:, :self.r].ravel(),
                                                   x_mat_rot[:self.r, self.r:].ravel(),
                                                   x_mat_rot[self.r:, self.r:].ravel()), axis=None)
                v = x_rot.reshape(self.d1 * self.d2, 1)
                M = M + np.matmul(v, v.T)
                M_inv = M_inv - M_inv.dot(np.matmul(v, v.T)).dot(M_inv) / (1 + v.T.dot(M_inv).dot(v).ravel()[0])
                X = np.vstack((X, x_rot))
                X[self.t, self.k:] *= math.sqrt(self.lambd2 / self.lambd2_perp)
                self.t += 1
            y = np.append(y, self.hisy)
            if y[0] == y[1]:
                y[1] = 1 - y[1]
            if self.context:
                for t in range(self.n):
                    self.x = collection(self.r_pro, self.n_vec, self.d1, self.d2, seed = 0, seed_val=self.seed_start+self.t).reshape(self.n_vec, self.d1, self.d2)
                    self.rotate_x()
                    self.get_opt_arm()
                    clf = LogisticRegression(penalty='l2', C=1/(2*self.lambd2), fit_intercept=False,
                                             solver='saga').fit(X, y)
                    theta_hat = clf.coef_[0]
                    theta_hat[self.k:] *= math.sqrt(self.lambd2/self.lambd2_perp)
                    val = np.zeros(self.n_vec)
                    rho = self.k_mu / self.c_mu * (0.01*math.sqrt(self.k * math.log(1 + self.c_mu * self.S ** 2 * (self.t + 1) / (self.k * self.lambd2)) + self.c_mu * self.S ** 2 * (self.t + 1) / self.lambd2_perp - 2 * math.log(self.delta / 2)) + math.sqrt(self.c_mu) * (math.sqrt(self.lambd2) * self.S + math.sqrt(self.lambd2_perp) * self.S_perp))
                    rho *= self.rho
                    for i in range(self.n_vec):
                        val[i] = self.mufunc(np.inner(theta_hat, self.x_rot[i, :])) + rho * math.sqrt(
                            self.x_rot[i, :].reshape(1, self.d1 * self.d2).dot(M_inv).dot(
                                self.x_rot[i, :].reshape(self.d1 * self.d2, 1)).ravel()[0])
                    arm = np.argmax(val)
                    instant_reward = self.exp_reward[arm]
                    self.regret[t] = self.opt_arm_reward - instant_reward
                    self.cum_regret[t] = np.sum(self.regret[:(t+1)])
                    v = self.x_rot[arm, :].reshape(self.d1 * self.d2, 1)
                    M = M + np.matmul(v, v.T)
                    M_inv = M_inv - M_inv.dot(np.matmul(v, v.T)).dot(M_inv) / (1 + v.T.dot(M_inv).dot(v).ravel()[0])
                    X = np.vstack((X, self.x_rot[arm, :]))
                    X[self.t, self.k:] *= math.sqrt(self.lambd2 / self.lambd2_perp)
                    self.t += 1
                    if self.model == 'logistic':
                        y = np.append(y, np.random.binomial(1, instant_reward))
                    if self.model == 'poisson':
                        y = np.append(y, np.random.poisson(instant_reward))
            else:
                self.rotate_x()
                self.get_opt_arm()
                for t in range(self.n):
                    clf = LogisticRegression(penalty='l2', C = 1/(2*self.lambd2), fit_intercept=False, solver='saga').fit(X, y)
                    theta_hat = clf.coef_[0]
                    theta_hat[self.k:] *= math.sqrt(self.lambd2/self.lambd2_perp)
                    val = np.zeros(self.n_vec)
                    rho = self.k_mu/self.c_mu*(0.01*math.sqrt(self.k * math.log(1 + self.c_mu * self.S ** 2 * (self.t + 1) / (self.k*self.lambd2)) + self.c_mu*self.S**2*(self.t + 1)/self.lambd2_perp - 2*math.log(self.delta/2)) + math.sqrt(self.c_mu) *(math.sqrt(self.lambd2)*self.S + math.sqrt(self.lambd2_perp)*self.S_perp))
                    rho *= self.rho
                    for i in range(self.n_vec):
                        val[i] = self.mufunc(np.inner(theta_hat, self.x_rot[i,:])) + rho*math.sqrt(self.x_rot[i,:].reshape(1,self.d1*self.d2).dot(M_inv).dot(self.x_rot[i,:].reshape(self.d1*self.d2,1)).ravel()[0])
                    arm = np.argmax(val)
                    instant_reward = self.exp_reward[arm]
                    self.regret[t] = self.opt_arm_reward - instant_reward
                    self.cum_regret[t] = np.sum(self.regret[:(t+1)])
                    v = self.x_rot[arm, :].reshape(self.d1 * self.d2, 1)
                    M = M + np.matmul(v, v.T)
                    M_inv = M_inv - M_inv.dot(np.matmul(v, v.T)).dot(M_inv) / (1 + v.T.dot(M_inv).dot(v).ravel()[0])
                    X = np.vstack((X, self.x_rot[arm, :]))
                    X[self.t,self.k:] *= math.sqrt(self.lambd2 / self.lambd2_perp)
                    self.t += 1
                    if self.model == 'logistic':
                        y = np.append(y,np.random.binomial(1, instant_reward))
                    if self.model == 'poisson':
                        y = np.append(y,np.random.poisson(instant_reward))


        else:
            if self.context:
                for _ in range(5):
                    self.x = collection(1, self.n_vec, self.d1, self.d2, seed=0 ,seed_val= self.seed_start + self.t).reshape(self.n_vec,self.d1,self.d2)
                    self.rotate_x()
                    self.get_opt_arm()
                    arm = np.random.choice(self.n_vec)
                    instant_reward = self.exp_reward[arm]
                    self.regret[self.t] = self.opt_arm_reward - instant_reward
                    self.cum_regret[self.t] = np.sum(self.regret[:(self.t + 1)])
                    X = np.vstack((X,self.x_rot[arm,:]))
                    X[self.t,self.k:] *= math.sqrt(self.lambd2/self.lambd2_perp)
                    v = self.x_rot[arm, :].reshape(self.d1 * self.d2, 1)
                    M = M + np.matmul(v, v.T)
                    M_inv = M_inv - M_inv.dot(np.matmul(v, v.T)).dot(M_inv) / (1 + v.T.dot(M_inv).dot(v).ravel()[0])
                    if self.model == 'logistic':
                        y = np.append(y,np.random.binomial(1, instant_reward))
                    if self.model == 'poisson':
                        y = np.append(y,np.random.poisson(instant_reward))
                    self.t += 1
                if y[0] == y[1]:
                    y[1] = 1 - y[1]


                for _ in range(5, self.n):
                    self.x = collection(self.r_pro, self.n_vec, self.d1, self.d2, seed = 0, seed_val=self.seed_start+self.t).reshape(self.n_vec, self.d1, self.d2)
                    self.rotate_x()
                    self.get_opt_arm()
                    clf = LogisticRegression(penalty='l2', C=1/(2*self.lambd2), fit_intercept=False,
                                             solver='saga').fit(X, y)
                    theta_hat = clf.coef_[0]
                    theta_hat[self.k:] *= math.sqrt(self.lambd2/self.lambd2_perp)
                    if np.linalg.norm(theta_hat, ord=2) > self.r_pro and not self.projgd:
                        theta_hat = FindProject(theta_hat, M_inv, X, self.model, self.lambd2,self.lambd2_perp, self.k, self.col)
                    if np.linalg.norm(theta_hat, ord=2) > self.r_pro and self.projgd:
                        theta_hat = Projgd(theta_hat, M_inv, X, self.model, self.lambd2, self.lambd2_perp, self.k, step = self.step)
                    val = np.zeros(self.n_vec)
                    rho = self.k_mu / self.c_mu * (0.01*math.sqrt(self.k * math.log(1 + self.c_mu * self.S ** 2 * (self.t + 1) / (self.k * self.lambd2)) + self.c_mu * self.S ** 2 * (self.t + 1) / self.lambd2_perp - 2 * math.log(self.delta / 2)) + math.sqrt(self.c_mu) * (math.sqrt(self.lambd2) * self.S + math.sqrt(self.lambd2_perp) * self.S_perp))
                    rho *= self.rho
                    for i in range(self.n_vec):
                        val[i] = self.mufunc(np.inner(theta_hat, self.x_rot[i, :])) + rho * math.sqrt(
                            self.x_rot[i, :].reshape(1, self.d1 * self.d2).dot(M_inv).dot(
                                self.x_rot[i, :].reshape(self.d1 * self.d2, 1)).ravel()[0])
                    arm = np.argmax(val)
                    instant_reward = self.exp_reward[arm]
                    self.regret[self.t] = self.opt_arm_reward - instant_reward
                    self.cum_regret[self.t] = np.sum(self.regret[:(self.t + 1)])
                    v = self.x_rot[arm, :].reshape(self.d1 * self.d2, 1)
                    M = M + np.matmul(v, v.T)
                    M_inv = M_inv - M_inv.dot(np.matmul(v, v.T)).dot(M_inv) / (1 + v.T.dot(M_inv).dot(v).ravel()[0])
                    X = np.vstack((X, self.x_rot[arm, :]))
                    X[self.t, self.k:] *= math.sqrt(self.lambd2 / self.lambd2_perp)
                    self.t += 1
                    if self.model == 'logistic':
                        y = np.append(y, np.random.binomial(1, instant_reward))
                    if self.model == 'poisson':
                        y = np.append(y, np.random.poisson(instant_reward))
            else:
                self.rotate_x()
                self.get_opt_arm()
                Lam_vec = np.concatenate((np.full((self.k,),self.lambd2),np.full((self.d1*self.d2-self.k,),self.lambd2_perp)), axis = None)
                Lam, M, M_inv = np.diag(Lam_vec), np.diag(Lam_vec)/self.c_mu, self.c_mu * np.diag(1/Lam_vec)
                X = np.empty([0,self.d1*self.d2])
                y = []
                for _ in range(5):
                    arm = np.random.choice(self.n_vec)
                    instant_reward = self.exp_reward[arm]
                    self.regret[self.t] = self.opt_arm_reward - instant_reward
                    self.cum_regret[self.t] = np.sum(self.regret[:(self.t+1)])
                    X = np.vstack((X,self.x_rot[arm,:]))
                    X[self.t,self.k:] *= math.sqrt(self.lambd2/self.lambd2_perp)
                    v = self.x_rot[arm, :].reshape(self.d1 * self.d2, 1)
                    M = M + np.matmul(v, v.T)
                    M_inv = M_inv - M_inv.dot(np.matmul(v, v.T)).dot(M_inv) / (1 + v.T.dot(M_inv).dot(v).ravel()[0])
                    if self.model == 'logistic':
                        y = np.append(y,np.random.binomial(1, instant_reward))
                    if self.model == 'poisson':
                        y = np.append(y,np.random.poisson(instant_reward))
                    self.t += 1
                if y[0] == y[1]:
                    y[1] = 1 - y[1]
                for _ in range(5,self.n):
                    clf = LogisticRegression(penalty='l2', C = 1/(2*self.lambd2), fit_intercept=False, solver='saga').fit(X, y)
                    theta_hat = clf.coef_[0]
                    theta_hat[self.k:] *= math.sqrt(self.lambd2/self.lambd2_perp)
                    if np.linalg.norm(theta_hat, ord=2) > self.r_pro and not self.projgd:
                        theta_hat = FindProject(theta_hat, M_inv, X, self.model, self.lambd2,self.lambd2_perp, self.k, self.col)
                    if np.linalg.norm(theta_hat, ord=2) > self.r_pro and self.projgd:
                        theta_hat = Projgd(theta_hat, M_inv, X, self.model, self.lambd2, self.lambd2_perp, self.k, step = self.step)
                    val = np.zeros(self.n_vec)
                    rho = self.k_mu/self.c_mu*(0.01*math.sqrt(self.k * math.log(1 + self.c_mu * self.S ** 2 * (self.t + 1) / (self.k*self.lambd2)) + self.c_mu*self.S**2*(self.t + 1)/self.lambd2_perp - 2*math.log(self.delta/2)) + math.sqrt(self.c_mu) *(math.sqrt(self.lambd2)*self.S + math.sqrt(self.lambd2_perp)*self.S_perp))
                    rho *= self.rho
                    for i in range(self.n_vec):
                        val[i] = self.mufunc(np.inner(theta_hat, self.x_rot[i,:])) + rho*math.sqrt(self.x_rot[i,:].reshape(1,self.d1*self.d2).dot(M_inv).dot(self.x_rot[i,:].reshape(self.d1*self.d2,1)).ravel()[0])
                    arm = np.argmax(val)
                    instant_reward = self.exp_reward[arm]
                    self.regret[self.t] = self.opt_arm_reward - instant_reward
                    self.cum_regret[self.t] = np.sum(self.regret[:(self.t + 1)])
                    v = self.x_rot[arm, :].reshape(self.d1 * self.d2, 1)
                    M = M + np.matmul(v, v.T)
                    M_inv = M_inv - M_inv.dot(np.matmul(v, v.T)).dot(M_inv) / (1 + v.T.dot(M_inv).dot(v).ravel()[0])
                    X = np.vstack((X, self.x_rot[arm, :]))
                    X[self.t,self.k:] *= math.sqrt(self.lambd2 / self.lambd2_perp)
                    self.t += 1
                    if self.model == 'logistic':
                        y = np.append(y,np.random.binomial(1, instant_reward))
                    if self.model == 'poisson':
                        y = np.append(y,np.random.poisson(instant_reward))


def tune_lowGLMUCB(paras,n, n_vec, d1, d2, x, model, Theta, lambd2, lambd2_perp, delta, S, S_perp, context,
                 U_hat, U_hat_perp, V_hat, V_hat_perp, c_mu, r_pro, k_mu, ymax, seed_start, projgd, history, hisx, hisy):
    if projgd:
        best = float('Inf')
        for rho in paras['rho']:
            mod = LowGLMUCB(n, n_vec, d1, d2, x, model, Theta, lambd2, lambd2_perp, delta, S, S_perp, context,
                            U_hat, U_hat_perp, V_hat, V_hat_perp, c_mu, r_pro, k_mu, ymax, seed_start, projgd = projgd, step = st, history = history, hisx = hisx, hisy = hisy, rho= rho)
            mod.run()
            tmp = mod.cum_regret
            if tmp[-1] < best:
                best = tmp[-1]
                reg = tmp
        return reg
    else:
        best = float('Inf')
        for rho in paras['rho']:
            mod = LowGLMUCB(n, n_vec, d1, d2, x, model, Theta, lambd2, lambd2_perp, delta, S, S_perp, context,
                        U_hat, U_hat_perp, V_hat, V_hat_perp, c_mu, r_pro, k_mu, ymax, seed_start, history = history, hisx = hisx, hisy = hisy, rho = rho)
            mod.run()
            tmp = mod.cum_regret
            if tmp[-1] < best:
                best = tmp[-1]
                reg = tmp
        return reg