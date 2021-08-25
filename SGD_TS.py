from run_simulation import *

from sklearn.linear_model import LogisticRegression
from stage1 import *
from run_simulation import *

class Gradient(object):
    def __init__(self):
        pass

    def logistic(self, x, y, theta, lamda=0):
        return x * (-y + 1 / (1 + np.exp(-x.dot(theta)))) + 2 * lamda * theta

class context(object):
    def __init__(self, K, r, T, true_theta, seed_value, x = None):
        self.K = K
        self.r = r
        self.reward = 0
        self.optimal = 0
        self.theta = true_theta
        self.d1 = true_theta[0]
        self.d2 = true_theta[1]
        self.d = self.d1*self.d2
        self.T = T
        if x is None:
            np.random.seed(seed_value)
            x = np.random.normal(0, 1, (K, self.d1*self.d2))
        else:
            x = x.reshape(x.shape[0],-1)
            self.K = x.shape[0]
        self.fv = x

    def build_bandit(self, model):
        f = getattr(reward_model(), 'mu_' + model)
        self.reward = [f(self.fv[i].dot(self.theta.ravel())) for i in range(self.K)]
        self.optimal = max(self.reward)  # max reward

    def ber(self, i):
        return np.random.binomial(1, self.reward[i])


# bandit = context(K, r, d, true_theta = Theta, seed_value = 0, x = None)
# bandit.build_bandit(model)

class SGD_TS(object):
    def __init__(self, class_context, model, T, context, dist = 'ber'):
        self.data = class_context
        self.T = T
        self.d = self.data.d
        self.random_sample = getattr(self.data, dist)
        self.grad = getattr(Gradient(), model)
        self.model = getattr(reward_model(), 'mu_' + model)
        self.context = context
    def glm(self, eta0, tau, g1, g2):
        T = self.T
        d = self.data.d
        regret = np.zeros(self.T)
        y = np.array([])
        y = y.astype('int')
        X = np.empty([0, d])
        feature = self.data.fv
        for t in range(tau):
            K = len(feature)
            pull = np.random.choice(K)
            observe_r = self.random_sample(pull)
            regret[t] = regret[t-1] + self.data.optimal - self.data.reward[pull]
            y = np.concatenate((y, [observe_r]), axis = 0)
            X = np.concatenate((X, [feature[pull]]), axis = 0)
            if self.context:
                self.data = context(K, r, d, true_theta = Theta, seed_value = t+1, x = None)
                self.data.build_bandit(model)
                feature = self.data.fv
        if y[0] == y[1]:
            y[1] = 1-y[0]
        clf = LogisticRegression(penalty = 'none', fit_intercept = False, solver = 'lbfgs').fit(X, y)
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
            if self.context:
                self.data = context(K, r, d, true_theta=Theta, seed_value=t+1, x=None)
                self.data.build_bandit(model)
                feature = self.data.fv
        return regret

