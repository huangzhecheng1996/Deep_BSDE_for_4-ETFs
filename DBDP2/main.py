import numpy as np
import torch
import math

from StochasticVolatilityModel import StochasticVolatilityModel
from LambdaNetwork import LambdaNetwork
from lnSRE_Equation import lnSRE_Equation
from DBDPLnSRESolver import DBDPLnSRESolver

def main():

    #Market parameter from 2015-2019 for ETFs (a: XLK, b: XLF, c: XLE, d: XLV, M: S&P 500)
 
    learned_params={'r': 0.02,
     'alpha_M': np.float64(0.31758716229363354),
     'beta_M': np.float64(12.865293073450562),
     'sigma_M': np.float64(0.5567144599143198),
     'V_M_0': np.float64(0.0189888392640686),
    
     'm_WM': np.float64(3.354523485100627),
     'rho_M_own': np.float64(-0.7642250277122952),
     'alpha_a': np.float64(0.07727275369083576),
     'beta_a': np.float64(6.487735427328434),
     'sigma_a': np.float64(0.23066160515955714),
     'V_a_0': np.float64(0.0012848933407741759),
     'gamma_aM': np.float64(0.10409094494463005),
     'delta_a': np.float64(1.2913818216054989),
     'rho_a_own': np.float64(-0.14001667734830328),
     'rho_aM': np.float64(-0.23725634573966328),
     'm_Wa': np.float64(9.431835928740613),
                    
     'alpha_b': np.float64(0.12318490140550321),
     'beta_b': np.float64(5.403004898619638),
     'sigma_b': np.float64(0.3138066810004761),
     'V_b_0': np.float64(0.005771490909311954),
     'gamma_bM': np.float64(0.1295705460845422),
     'delta_b': np.float64(1.3909018688410821),
     'rho_b_own': np.float64(0.13820211398549068),
     'rho_bM': np.float64(-0.18857046827588872),
     'm_Wb': np.float64(1.3015027352197628),
                    
     'alpha_c': np.float64(0.3131507669534341),
     'beta_c': np.float64(4.518314982255215),
     'sigma_c': np.float64(0.4862757188147902),
     'V_c_0': np.float64(0.021239945146650497),
     'gamma_cM': np.float64(0.11340752901865712),
     'delta_c': np.float64(1.5260306838894788),
     'rho_c_own': np.float64(0.1459288721760854),
     'rho_cM': np.float64(-0.09135745509553944),
     'm_Wc': np.float64(-1.687284055716071),
                    
     'alpha_d': np.float64(0.06143494062158897),
     'beta_d': np.float64(4.301198282673263),
     'sigma_d': np.float64(0.23794924421903255),
     'V_d_0': np.float64(0.0022944511885847176),
     'gamma_dM': np.float64(0.094963395374469),
     'delta_d': np.float64(1.0238773268690975),
     'rho_d_own': np.float64(-0.17585205340095938),
     'rho_dM': np.float64(-0.17265007426401133),
     'm_Wd': np.float64(2.4033245803872934),
     
     # Set to 0: orthogonal volatility provides no premium
     'ell_a': 0.0, 'ell_b': 0.0, 'ell_c': 0.0, 'ell_d': 0.0,
     'm_ZaM': 0.0, 'm_ZbM': 0.0, 'm_ZcM': 0.0, 'm_ZdM': 0.0,
     'ell_ZM': 0.0}

    T_HORIZON = 1.0
    DT = 1 / 252
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    simulator = StochasticVolatilityModel(learned_params)

    value_net = ValueNetwork(
        num_steps=int(T_HORIZON / DT),
        state_dim=5,
        hidden_layers=[64, 64] 
    ).to(DEVICE)

    sre_eq = lnSRE_Equation(learned_params, device=DEVICE)

    solver = DBDPLnSRESolver(
        params=learned_params,
        value_net=value_net,
        sre_equation=sre_eq,
        dt=DT,
        T=T_HORIZON,
        device=DEVICE
    )

    N = solver.N_steps
    massive_epochs_list = [700 for i in range(N)]

    final_log_p0 = solver.train_backward(
        simulator=simulator,
        target_mse=5e-7,
        max_epochs_list=massive_epochs_list,
        lr_start=1e-3,                                
        lr_patience=20,                    
        min_lr=5e-7,
        batch_size=32768,
        save_interval=50,
        resume_ckpt_path=None,
        global_warm_start=False
    )

    print(f"P(0) = {math.exp(final_log_p0):.6f}")

    solver.save_model(file_path='sre_dbdp2_value_model_step1.pt')

if __name__ == "__main__":
    main()