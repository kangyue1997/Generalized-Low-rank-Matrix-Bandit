from LowESTR import *
from stage2_GESTS import *
from stage2_GESTT import *
import warnings
import concurrent.futures
warnings.filterwarnings("ignore")

model = 'logistic'
n_sim = 100
n_vec = 2000
d1, d2, r = 10, 10, 2
d = 10
p = d1*d2
dom_d = r*(d1+d2-r)
Theta = np.zeros([d1,d2])
v1 = np.random.normal(0,1,[d1,1])
v1 /= np.linalg.norm(v1)
v2 = np.random.normal(0,1,[d1,1])
v2 = v2 - np.sum(v2*v1)*v1
v2 /= np.linalg.norm(v2)
Theta = 3*v1.dot(v1.T) + 3*v2.dot(v2.T)
Wr = 3
T = 45000
T1 = 1200
T2 = T-T1
context = False
lambd = 0.0069  # need more experiments !!!!
lambd_oful = 0.01*math.sqrt(1/T1)
IsPeaceman = True
delta = 0.01
sigma = 0.01
S = 1
S2 = 1
c_mu = math.exp(5)/(1+math.exp(5))**2
k_mu = 0.25
S_perp = k_mu*d1**3*S2*r/T1/Wr**2/c_mu**2
print("S_perp = k_mu*d1**3*S2*r/T1/Wr**2/c_mu**2")
S_perp_oful = (sigma**2)*(d1+d2)**3*r/T1/(Wr**2) #oful
ymax = 1
lambd2 = 1
lambd2_perp = c_mu*S2**2*T2/(dom_d*math.log(1+c_mu*S**2*T2/(dom_d*lambd2)))
lambd2_oful = 1  #oful
lambd2_perp_oful = T2/dom_d/math.log(1+T2/lambd2)  #oful
print("lambda: {0}".format(lambd))
history = True


parameters = {
        'step_size': [0.01,0.1,1,10], #[0.01, 0.05, 0.1, 0.5, 1, 5, 10], # total 7
        'explore': [0.1,1,5], #[0.01, 0.1, 1, 5, 10], # total 5
        'C': [1,3,5,7], #list(range(1,11)),
        'rho': [0.2,1,5]
    }

def para_func_fix(nn):
    #   print("Total of replication: {0}".format(nn))
    cum_reg_lowsgdts = np.zeros(shape=[T])
    seed_start1 = nn * T
    seed_start2 = seed_start1 + T1
    np.random.seed(nn+100000000)
    ux = np.random.normal(0, 1, size=[n_vec, d1, d2])
    for i in range(n_vec):
        ux[i, :, :] /= np.linalg.norm(ux[i, :, :])
    stage1 = stage1_pg(n_vec, d1, d2, ux, model='logistic', Theta=Theta, lambd=lambd, beta=0.9, step=1,
                       error_bar=10 ** (-4), max_iter=5000, n=T1, context=context, seed_start=seed_start1)
    "stage 1 pgd"
    x, y = stage1.run_xy()
    if not IsPeaceman:
        stage1.run()
        Theta_est, U_hat, U_hat_perp, S_hat, V_hat, V_hat_perp = stage1.Theta_est, stage1.U_hat, stage1.U_hat_perp, stage1.S_hat, stage1.V_hat, stage1.V_hat_perp
        print("pgd: {0}".format(np.linalg.norm(Theta_est - Theta)))
    else:
        Theta_est, U_hat, U_hat_perp, S_hat, V_hat, V_hat_perp = peaceman(lambd=stage1.lambd, r=stage1.r, d1=stage1.d1,d2=stage1.d2, x=x, y=y, error_bar=10 ** (-4))
        print("peaceman: {0}".format(np.linalg.norm(Theta_est - Theta)))
    cum_reg_lowsgdts[:T1] = stage1.cum_regret
    U_full = np.hstack((U_hat, U_hat_perp))
    V_full = np.hstack((V_hat, V_hat_perp))
    if context:
        ux = None
    bandit = gcontext(n_vec, r, Theta, seed_value = seed_start2, U_full = U_full, V_full = V_full, x = ux)
    cum_reg_lowsgdts[T1:] = stage1.cum_regret[-1]+tune_lowsgdts(bandit, dist='ber', T = T2, d = bandit.d, model = model, context = context, true_theta= Theta, seed_start=seed_start2, paras= parameters, r = r)
    bandit1 = gcontext(n_vec, r, Theta, seed_value = seed_start1, U_full = U_full, V_full = V_full, x = ux, rot = False)
    cum_reg_sgdts = tune_lowsgdts(bandit1, dist='ber', T = T, d = bandit1.d, model = model, context = context, true_theta= Theta, seed_start=seed_start1, paras= parameters, r = r, rot= False)
    reg_lowglmucb_proj = tune_lowGLMUCB(parameters, T2, n_vec, d1, d2, ux, model, Theta, lambd2, lambd2_perp, delta, S,
                                    S_perp, context, U_hat, U_hat_perp, V_hat,
                                    V_hat_perp, c_mu, S, k_mu, ymax, seed_start=seed_start2, projgd=True,
                                    history=history, hisx=x, hisy=y)
    cum_reg_lowglmucb_proj = np.concatenate((stage1.cum_regret, [a + stage1.cum_regret[-1] for a in reg_lowglmucb_proj]),
                                        axis=None)
    if context:
        return cum_reg_lowsgdts, cum_reg_sgdts, cum_reg_lowglmucb_proj
    else:
        stage_1 = explore(n_vec, d1, d2, T1, Theta, ux.reshape(n_vec, -1), sigma, lambd_oful, gamma=0.1, r=r, model=model)
        stage_1.run()
        stage_2 = LowOFUL(n_vec, d1, d2, r, Theta, ux.reshape(n_vec, -1), sigma, stage_1.U_hat, stage_1.V_hat,
                          stage_1.U_hat_perp, stage_1.V_hat_perp,
                          T2, delta=delta, lam=lambd2_oful, lam_perp=lambd2_perp_oful, B=S, B_perp=S_perp, model=model)
        stage_2.run()
        cum_reg_oful = np.concatenate((stage_1.cum_regret, [a + stage_1.cum_regret[-1] for a in stage_2.cum_regret]),
                                      axis=None)
        return cum_reg_lowsgdts, cum_reg_sgdts, cum_reg_oful, cum_reg_lowglmucb_proj



if context:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [nn for nn in range(n_sim)]
        results = executor.map(para_func_fix, secs)
        results = list(results)
    res1 = [i[0] for i in list(results)]
    res2 = [i[1] for i in list(results)]
    res3 = [i[2] for i in list(results)]
    res4 = [i[3] for i in list(results)]
    #
    print('finish')
    print('Context=F, G-ESTS: {0}'.format(sum(res1) / n_sim))
    print('Context=F, SGD-TS: {0}'.format(sum(res2) / n_sim))
    print('Context=F, LowESTR: {0}'.format(sum(res3) / n_sim))
    print('Context=F, G-ESTT: {0}'.format(sum(res4) / n_sim))
    print('\n')
else:
    with concurrent.futures.ProcessPoolExecutor() as executor:
        secs = [nn for nn in range(n_sim)]
        results = executor.map(para_func_fix, secs)
        results = list(results)
    res1 = [i[0] for i in list(results)]
    res2 = [i[1] for i in list(results)]
    res3 = [i[2] for i in list(results)]
    #
    print('finish')
    print('Context=F, G-ESTS: {0}'.format(sum(res1) / n_sim))
    print('Context=F, SGD-TS: {0}'.format(sum(res2) / n_sim))
    print('Context=F, G-ESTT: {0}'.format(sum(res3) / n_sim))
    print('\n')








