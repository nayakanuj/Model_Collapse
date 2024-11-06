import numpy as np
from matplotlib import cm
import matplotlib.pyplot as plt
from scipy.stats import *
import scipy as scp
from modules_core import *
from modules_dbg import *

FONT_SIZE = 12
plt.rcParams['mathtext.fontset'] = 'cm'
plt.rcParams['font.family'] = 'STIXGeneral'
plt.rcParams["font.size"] = str(FONT_SIZE)
plt.rcParams['figure.figsize'] = 8, 6


if __name__ == "__main__":
    
    # dt, eps, eps_str = 4, 0.9, '0pt9'
    #dt, eps, eps_str = 2, 0.9, '0pt9'
    #dt, eps, eps_str = 3, 0.5, '0pt5'
    # dt, eps, eps_str = 4, 0.8, '0pt8'
    # dt, eps, eps_str = 4, 0.5, '0pt5'
    dt, eps, eps_str = 6, 0.5, '0pt5'
    # dt, eps, eps_str = 7, 0.5, '0pt5'
    # dt, eps, eps_str = 10, 0.5, '0pt5'
    degree_dist = 'binomial_binomial'
    closed_form = False
    mult_fact = 6
    
    # varsigma, tau = 1e7, 1e7
    varsigma, tau = 2e5, 8e5       ## dt, eps, eps_str = 6, 0.5, '0pt5'
    
    # min_scale = 0.5
    # max_scale = 0.8

    # min_scale = 0.1
    # max_scale = 0.9

    # min_scale = 0.1
    # max_scale = 0.9
    # min_scale = 0.001
    
    # min_scale = 0.02
    # max_scale = 0.5
     
    # min_scale = 0.15 # dt, eps, eps_str = 6, 0.5, '0pt5'
    # max_scale = 0.49    

    min_scale = 0.15 # dt, eps, eps_str = 6, 0.5, '0pt5'
    max_scale = 0.49       
    ind_flops_min, ind_flops_max, ind_flops_step = 40, 100, 10
    n_flops = int(np.floor((ind_flops_max-ind_flops_min)/ind_flops_step))
    # decay_vec = 0.45**(np.linspace(2,5,n_flops))
    decay_vec = 0.45**(np.linspace(3,6,n_flops))


    if closed_form:
        filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts50_closed_form'    
    else:
        # filename = 'data_dir/'+degree_dist+'_dt'+str(dt)+'_eps'+eps_str+'_numpts50' 
        # filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts100'
        filename = 'data_dir/'+degree_dist+'_biterr_dt'+str(dt)+'_eps'+eps_str+'_numpts200'
    
    flops_slearnt_dict = np.load(filename+'.npy', allow_pickle='TRUE').item()

    # min_FLOPS = 1e5
    # max_FLOPS = 1e6
    # num_FLOPS = 5
    # # FLOPS_vec = np.logspace(np.log10(min_FLOPS), np.log10(max_FLOPS), num_FLOPS)    
    # FLOPS_vec = np.linspace(min_FLOPS, max_FLOPS, num_FLOPS)    

    # FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[10:40:4]
    # FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[10::4]
    # FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[15::5]
    # FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[30:60:5]
    FLOPS_vec = np.array([float(x) for x in list(flops_slearnt_dict.keys())])[40:100:10]
    FLOPS_vec_scaled = FLOPS_vec*varsigma*tau*mult_fact  

    fig, ax = plt.subplots(1, 2, layout='constrained')
    fig.set_size_inches(7, 3.5)
    lines = [0]*len(FLOPS_vec)

    for ind_FLOPS, FLOPS in enumerate(FLOPS_vec):
        s_low = max(1.0, np.sqrt(FLOPS)*(min_scale*(1-decay_vec[ind_FLOPS])))
        s_high = np.sqrt(FLOPS)*max_scale*(1+decay_vec[ind_FLOPS])        
        # args = (dt, FLOPS, eps)
        # s_opt = my_minimizer(num_skills_learnt, args, s_low, s_high, num_pts=30)
        print(f"ind_FLOPs={ind_FLOPS}, FLOPs = {FLOPS}")
        
        if closed_form:
            # biterr [TODO]
            plot_slearnt_vs_s_closed_form(s_low, s_high, FLOPS, eps, dt, fig_num=0)
        else:
            line1, _ = plot_slearnt_vs_s_biterr(s_low, s_high, FLOPS, eps, dt, fig, ax)
            lines[ind_FLOPS] = line1


        brkpnt1 = 1
        
    
    PB_vec = (1-np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_learnt'] for x in range(len(FLOPS_vec))])/np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_opt'] for x in range(len(FLOPS_vec))]))
    epsBP_vec = np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['eps_BP'] for x in range(len(FLOPS_vec))])
    s_opt_vec = np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_opt'] for x in range(len(FLOPS_vec))])
    s_learnt_opt_vec = np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_learnt'] for x in range(len(FLOPS_vec))])
    t_opt_vec = FLOPS_vec/np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['s_opt'] for x in range(len(FLOPS_vec))])
    alpha_vec = np.array([flops_slearnt_dict[str(FLOPS_vec[x])]['alpha'] for x in range(len(FLOPS_vec))])

    D_opt_vec = t_opt_vec*tau
    N_opt_vec = s_opt_vec*varsigma

    flops_base_vec = [f"{FLOPS:0.2e}" for FLOPS in FLOPS_vec]
    flops_exponent_vec = ["10^{"+str(int(np.log10(FLOPS)))+"}" for FLOPS in FLOPS_vec]
    legends = [f"FLOPs=$"+flops_base_vec[ind1][0:4]+"\\times"+flops_exponent_vec[ind1]+"$" for ind1 in range(len(flops_base_vec))]
    plot_slearnt_vs_s_postproc(s_opt_vec, s_learnt_opt_vec, epsBP_vec, fig, ax, lines, legends)

    # plt.figure(1)
    # plt.plot(s_opt_vec, epsBP_vec - eps)
    # plt.xlabel("$R^*$")
    # plt.ylabel("$\epsilon^* - \epsilon$")
    # plt.xscale("log")
    # plt.yscale("log")    

    # plt.figure(1)    
    # plt.subplot(1,2,1)
    # plt.xlim((1e2, 7e4))
    # plt.subplot(1,2,2)
    # plt.xlim((1e2, 7e4))
    # plt.ylim((0.49, 0.6))
    # fig = plt.gcf()
    # fig.set_size_inches(8, 4)


    brkpnt1 = 1

        

    
  