
import jax
import jax.numpy as jnp
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import jax.numpy as jnp
import jax
import numpy as np
from scipy.stats import qmc


def simulation_alpha_stable_rqmc(alpha: float, beta: float, gamma: float, delta: float, n: int, seed : int = None):
    """
    Generate n samples from an alpha-stable distribution using an RQMC sequence.
    
    Parameters:
        alpha (float): Stability parameter (0 < alpha <= 2).
        beta (float): Skewness parameter (-1 <= beta <= 1).
        gamma (float): Scale parameter.
        delta (float): Location parameter.
        n (int): Number of samples to generate.
    
    Returns:
        jnp.ndarray: A vector of n samples from the specified alpha-stable distribution.
    """
    # 1. Generate a 2D Sobol sequence with scrambling (RQMC)
    sampler = qmc.Sobol(d=2, scramble=True)
    # Generate n points in [0,1]^2
    points = sampler.random(n)
    
    # 2. Transform the first coordinate to [-pi/2, pi/2] for u
    half_pi = jnp.pi / 2
    # Note: Convert the numpy array to a jax array if necessary
    u = jnp.array(points[:, 0]) * jnp.pi - half_pi
    
    # 3. Transform the second coordinate via the inverse CDF to get exponential variables
    v = jnp.array(points[:, 1])
    # Avoid issues with logarithm of zero by adding a small epsilon if needed.
    epsilon = 1e-10
    w = -jnp.log(v + epsilon)
    
    # 4. Calculate y using the Chambers–Mallows–Stuck method
    tol = 1e-6  # tolerance for checking if alpha ~ 1
    
    def branch_alpha_eq_1():
        return (2 / jnp.pi) * (
            (half_pi + beta * u) * jnp.tan(u) -
            beta * jnp.log((half_pi * w * jnp.cos(u)) / (half_pi + beta * u))
        )

    def branch_alpha_neq_1():
        S = (1 + beta**2 * jnp.tan(half_pi * alpha)**2) ** (1 / (2 * alpha))
        B = (1 / alpha) * jnp.arctan(beta * jnp.tan(half_pi * alpha))
        return S * (jnp.sin(alpha * (u + B)) / (jnp.cos(u)) ** (1/alpha)) * (
            (jnp.cos(u - alpha * (u + B))) / w
        ) ** ((1 - alpha) / alpha)
    
    y = jax.lax.cond(jnp.abs(alpha - 1) < tol, branch_alpha_eq_1, branch_alpha_neq_1)
    
    # 5. Apply the scale and location transformation.
    return gamma * y + delta




#--------------------------------------------------------Statistics summary start-----------------------------------


#some functions
def log_abs_phi(t,x):
    """log |Phi(t)| for a=1 and a!=1, 
    abs value of a complex exponential | exp(z)| , where z=a+ib ->> | exp(z)|= exp(real(z))=exp(a)

    +we take log so all this expression (3.1) simplifies to :

    log |Phi(t)|= - (gamma**alpha) * abs(t)**alpha for a=1 & a!=1
    but problem: we can't use it, so we use another formula of :::

    phi_t = sum(np.exp(1j * t * x)) / len(x)
    log_abs_phi_hat = np.log(abs(phi_t)) (as in s4 took in abs and log)
    
    """
    phi_t = np.mean(np.exp(1j * t * x))   
    log_abs_phi_hat = np.log(abs(phi_t))  
    return   log_abs_phi_hat

def u_hat(t, x):
    num = np.mean(np.sin(t * x))                
    denum = np.mean(np.cos(t * x))
    return np.arctan(num / denum)#typo in pdf!! as sum sin/sum cos= True!!

def summary_stat_2(X, ksi =  0.25):
    """
    Calcule la statistique résumée pour x
    Grâce à la transformation de Zolotarev.
    """
    Z = np.array([X[3*i-2] - ksi * X[3*i-1] - (1-ksi) * X[3*i] for i in range(1, len(X)//3 + 1)]) 

    V = np.log(np.abs(Z)
               ) 
    U = np.sign(X)

    S_V = np.var(V
                 )
    S_U = np.var(U)

    eta_hat = np.mean(U)

    teta_hat = np.mean(V)
    
    v_tilde = 6/(np.pi)**2 * S_V - 1.5 * S_U + 1

    v_hat = max(v_tilde, (1+eta_hat)**2 * 0.25)
    delta_hat = np.mean(X)

    return np.array([v_hat , eta_hat, teta_hat, float(delta_hat)])
#s3    
def s3(x):
    t1, t2, t3, t4 = 0.2, 0.8, 0.1, 0.4
    # 1. α̂ -----------------------------------------------------------------------
    alpha_hat = np.log( 
        log_abs_phi(t1, x) / log_abs_phi(t2, x)
        
    ) / np.log(abs(t1 / t2))

    # 2. γ̂ ---------------------------------------------------------------------
    log_gamma_hat = (
        np.log(abs(t1)) * np.log(-log_abs_phi(t2, x))
        - np.log(abs(t2)) * np.log(-log_abs_phi(t1, x))
    ) / np.log(abs(t1 / t2))
    gamma_hat = np.exp(log_gamma_hat)

    # 3. β̂ -------------------------------------------------------------------------
    u_t3 = u_hat(t3, x)
    u_t4 = u_hat(t4, x)

    beta_hat = (
        u_t4 / t4 - u_t3 / t3
    ) / (
        (abs(t4) ** (alpha_hat - 1) - abs(t3) ** (alpha_hat - 1))
        * (gamma_hat ** alpha_hat)
        * np.tan(np.pi * alpha_hat / 2)
    )

    # 4. δ̂ ------------------------------------------------------------------
    delta_hat = (
        abs(t4) ** (alpha_hat - 1) * u_t3 / t3
        - abs(t3) ** (alpha_hat - 1) * u_t4 / t4
    ) / (abs(t4) ** (alpha_hat - 1) - abs(t3) ** (alpha_hat - 1))

    return np.array((alpha_hat,beta_hat,log_gamma_hat,delta_hat))
#s4    
def s4(x):
    """
    Returns concatenated real and imaginary parts.
    """
    #positive grid
    t_pos = np.arange(0.5, 5.1, 0.5)
    # +-
    t_vals = np.concatenate([t_pos, -t_pos])
    #important to standartise! 
    new_x=(x-np.mean(x))/(np.quantile(x,0.75)-np.quantile(x,0.25))
    #compute φ̂(t) for each t in our grid
    phi = np.array([np.mean(np.exp(1j * t * new_x)) for t in t_vals])
    return np.hstack((phi.real, phi.imag))


def stat(statistic, x):
    """ 
    for mcmc abc we use S3,S4 statistic

    """
    if statistic == "s3":
        return s3(x)
    elif statistic == "s4":
        return s4(x)

    elif statistic=="s2":
        return summary_stat_2(x)
    else:
        raise ValueError(f"Unknown summary statistic '{statistic}'")
