from sklearn.linear_model import LogisticRegression
from stage1 import *
'''G-ESTS adapted to SGD-TS in stage 2'''

class Gradient(object):
    def __init__(self):
        pass

    def logistic(self, x, y, theta, lamda=0):
        return x * (-y + 1 / (1 + np.exp(-x.dot(theta)))) + 2 * lamda * theta

class gcontext(object):
    def __init__(self, K, r, true_theta, seed_value, U_full = 0, V_full = 0, x = None, rot = True):
        self.K = K
        self.r = r  #rank of the matrix
        self.reward = 0
        self.optimal = 0
        self.theta = true_theta
        self.d1 = true_theta.shape[0]
        self.d2 = true_theta.shape[1]
        self.U_full = U_full
        self.V_full = V_full
        self.d = (self.d1+self.d2-self.r)*self.r
        """x is none implies contextual bandit"""
        if x is None:
            x = collection(1, self.K, self.d1, self.d2, seed_val=seed_value, seed = True)
        else:
            x = x.reshape(x.shape[0],-1)
            self.K = x.shape[0]
        self.fv = x
        self.fv_rot = np.zeros(shape=[self.K, self.d])
        self.rot = rot
        if self.rot:
            Theta_mat_rot = np.matmul(np.matmul(np.transpose(U_full), self.theta), V_full)
            self.theta_rot = np.concatenate((Theta_mat_rot[:self.r, :self.r].ravel(),Theta_mat_rot[self.r:, :self.r].ravel(),Theta_mat_rot[:self.r, self.r:].ravel()),axis=None)
            for i in range(self.K):
                x_mat_rot = np.matmul(np.matmul(np.transpose(U_full), self.fv[i,:].reshape(self.d1,self.d2)), V_full)
                self.fv_rot[i,:] = np.concatenate((x_mat_rot[:self.r, :self.r].ravel(),x_mat_rot[self.r:, :self.r].ravel(),x_mat_rot[:self.r, self.r:].ravel()), axis=None)
        else:
            self.fv_rot = self.fv
            self.d = self.d1*self.d2


    def build_bandit(self, model):
        f = getattr(reward_model(), 'mu_' + model)
        self.reward = [f(self.fv[i].dot(self.theta.ravel())) for i in range(self.K)]
        self.optimal = max(self.reward)  # max reward

    def ber(self, i):
        return np.random.binomial(1, self.reward[i])



class SGD_TS(object):
    def __init__(self, class_context, model, T, context, r, true_theta, dist = 'ber', seed_start = 0, rot = True):
        self.data = class_context
        self.data.build_bandit(model)
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist)
        self.grad = getattr(Gradient(), model)
        self.context = context
        self.seed_start = seed_start
        self.r = r
        self.theta = true_theta
        self.model = model
        self.rot = rot
        self.U_full = self.data.U_full
        self.V_full = self.data.V_full
        self.dist = dist

    def glm(self, eta0, tau, g1, g2):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        feature = self.data.fv_rot
        for t in range(tau):
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.random_sample(pull)
            regret[t] = regret[t-1] + self.data.optimal - self.data.reward[pull]
#            print(regret[t])
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            if self.context:
                self.data = gcontext(K, self.r, true_theta=self.theta, seed_value = self.seed_start+t+1, U_full=self.U_full, V_full=self.V_full, x = None, rot=self.rot)
                self.data.build_bandit(self.model)
                self.random_sample = getattr(self.data, self.dist)
                feature = self.data.fv_rot
        if y[0] == y[1]:
            y[1] = 1-y[0]
        clf = LogisticRegression(penalty = 'none', fit_intercept = False, solver = 'saga').fit(X, y)
        theta_hat = clf.coef_[0]
        grad = np.zeros(d)
        theta_tilde = np.zeros(d)
        theta_tilde[:] = theta_hat[:]
        theta_bar = np.zeros(d)
        for t in range(tau, T):
            K = len(feature)
            ts_idx = [0]*K
            if t%tau == 0:
                j = t//tau
                cov = (2*g1**2 + 2*g2**2) * np.identity(d) / j
                eta = eta0/j
                theta_tilde -= eta*grad
                distance = np.linalg.norm(theta_tilde-theta_hat)
                if distance > 2:
                    theta_tilde = theta_hat + 2*(theta_tilde-theta_hat)/distance
                grad = np.zeros(d)
                theta_bar = (theta_bar * (j-1) + theta_tilde) / j
                theta_ts = np.random.multivariate_normal(theta_bar, cov)
            for arm in range(K):
                ts_idx[arm] = feature[arm].dot(theta_ts)
            pull = np.argmax(ts_idx)
            observe_r = self.random_sample(pull)
            grad += self.grad(feature[pull], observe_r, theta_tilde, 0)
            regret[t] = regret[t-1] + self.data.optimal - self.data.reward[pull]
#            print(regret[t])
            if self.context:
                self.data = gcontext(K, self.r, true_theta=self.theta, seed_value=self.seed_start+t+1, U_full=self.U_full, V_full=self.V_full, x=None, rot = self.rot)
                self.data.build_bandit(self.model)
                self.random_sample = getattr(self.data, self.dist)
                feature = self.data.fv_rot
        return regret


def tune_lowsgdts(bandit, dist, T, d, model, context, true_theta, seed_start, paras, r, rot = True):
    sgd_ts = SGD_TS(bandit, model, T, context, r, true_theta, dist, seed_start, rot)
    best = float('Inf')
    for C in paras['C']:
        tau = int(max(d, math.log(T)) * C)
        for eta0 in paras['step_size']:
            for g1 in paras['explore']:
                for g2 in paras['explore']:
                    tmp = sgd_ts.glm(eta0, tau, g1, g2)
                    if tmp[-1] < best:
                        reg = tmp
                        best = tmp[-1]
    return reg