import argparse
# python FEW_Info_Mat.py -dev 6
parser = argparse.ArgumentParser(description='MCMC few')
parser.add_argument('-dev','--dev', help='Cuda Device', required=True, type=int)
args = vars(parser.parse_args())

import os
print("process", os.getpid() )

os.system(f"CUDA_VISIBLE_DEVICES={args['dev']}")
os.environ["CUDA_VISIBLE_DEVICES"] = f"{args['dev']}"
os.system("echo $CUDA_VISIBLE_DEVICES")

from few.utils.utility import omp_get_num_threads, omp_set_num_threads
omp_set_num_threads(4)

import warnings
import time
import numpy as np
import matplotlib.pyplot as plt
# try to import cupy
try:
    import cupy as xp
    gpu_available = True

except (ImportError, ModuleNotFoundError) as e:
    import numpy as xp
    gpu_available = False

# Cython imports
from pyinterp_cpu import interpolate_arrays_wrap as interpolate_arrays_wrap_cpu
from pyinterp_cpu import get_waveform_wrap as get_waveform_wrap_cpu

# Python imports
from few.summation.interpolatedmodesum import InterpolatedModeSum
from few.utils.ylm import GetYlms
from few.utils.baseclasses import SummationBase, SchwarzschildEccentric
from few.utils.citations import *
from few.utils.utility import get_fundamental_frequencies
from few.utils.constants import *
from few.summation.interpolatedmodesum import CubicSplineInterpolant
from few.trajectory.inspiral import EMRIInspiral
from few.waveform import FastSchwarzschildEccentricFlux, SlowSchwarzschildEccentricFlux, GenerateEMRIWaveform
from few.amplitude.romannet import RomanAmplitude
from few.amplitude.interp2dcubicspline import Interp2DAmplitude
from few.utils.modeselector import ModeSelector
from few.infomat import InfoMatrixFastSchwarzschildEccentricFlux



# Attempt Cython imports of GPU functions
try:
    from pyinterp import interpolate_arrays_wrap, get_waveform_wrap

except (ImportError, ModuleNotFoundError) as e:
    pass

# for special functions
from scipy import special, signal
from scipy.interpolate import CubicSpline
import multiprocessing as mp

from lisatools.diagnostic import *



# keyword arguments for inspiral generator (RunSchwarzEccFluxInspiral)
inspiral_kwargs = {
    "DENSE_STEPPING": 0,  # we want a sparsely sampled trajectory
    "max_init_len": int(
        1e3
    ),  # all of the trajectories will be well under len = 1000
}

# keyword arguments for inspiral generator (RomanAmplitude)
amplitude_kwargs = {
    "max_init_len": int(
        1e3
    )  # all of the trajectories will be well under len = 1000
}

# keyword arguments for Ylm generator (GetYlms)
Ylm_kwargs = {
    "assume_positive_m": False  # if we assume positive m, it will generate negative m for all m>0
}

# keyword arguments for summation generator (InterpolatedModeSum)
sum_kwargs = {"pad_output":True}

gen_wave = FastSchwarzschildEccentricFlux(inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,)

fast = InfoMatrixFastSchwarzschildEccentricFlux(
    inspiral_kwargs=inspiral_kwargs,
    amplitude_kwargs=amplitude_kwargs,
    Ylm_kwargs=Ylm_kwargs,
    sum_kwargs=sum_kwargs,
    use_gpu=gpu_available,
    # normalize_amps=False
)

# parameters
T = 0.45#0.45  # years
dt = 15.0  # seconds
M = 1e6
mu = 3e1
p0 = 10.0
e0 = 0.3
theta = np.pi / 3  # polar viewing angle
phi = np.pi / 4  # azimuthal viewing angle
dist = 5.0  # distance
batch_size = int(1e4)
par = np.array([M, mu, p0, e0, theta, phi])

mode_sel = []
for ll in range(2,3):
    for mm in range(-ll, ll+1):
        for i in range(-10, 10):
            mode_sel.append((ll,mm,i))
wv_kw = dict(T=T, dt=dt, mode_selection=mode_sel, dist=dist)

from lisatools.diagnostic import *

inner_product_kwargs = dict(dt=dt, PSD="cornish_lisa_psd", use_gpu=gpu_available)

h = gen_wave(*par, **wv_kw)
window =1.0#signal.tukey(len(h),0.01)

print("snr", inner_product(h*window, h*window, **inner_product_kwargs)**(1/2))
print("h end", h[-10:])
###################################################
deriv_inds = [0, 1, 3, 4]
list_ind = [0, 1, 2, 3]
delta_deriv = [5.0, 1e-2, 1e-2, 1e-2]

dim = len(deriv_inds)
fast_wave = fast(M, mu, p0, e0, theta, phi, delta_deriv=delta_deriv, deriv_inds=deriv_inds, **wv_kw)

# cross corr
X = xp.array([[inner_product(fast_wave[i].real*window, fast_wave[j].real*window, **inner_product_kwargs) + \
    inner_product(fast_wave[i].imag*window, fast_wave[j].imag*window, **inner_product_kwargs) for i in range(dim)] for j in range(dim)])

X_my = X.copy()
print(X_my)
from scipy import linalg
inv_X = xp.linalg.inv(X_my)

print("zero check",[inner_product(h,fast_wave[i], **inner_product_kwargs) for i in range(4)] )
######################################################################################################
N_t = len(h)
freq_bin = np.fft.rfftfreq(N_t,dt)
PSD = get_sensitivity(freq_bin[1:],sens_fn="cornish_lisa_psd")
PSD = np.hstack((PSD[0],PSD))

variance_noise_f = PSD/(4.*dt*N_t)  # Calculate variance of noise 
sigma = np.sqrt(variance_noise_f)

n_f = np.random.normal(0, sigma)+ 1j * np.random.normal(0, sigma)
noise_t = np.fft.irfft(n_f)
if len(noise_t)!=N_t:
    noise_t = np.hstack((noise_t[0],noise_t))

noisy_data = h + xp.array(noise_t) 
################################
perc = 0.3

from eryn.prior import uniform_dist, log_uniform

# define priors, it really can only do uniform cube at the moment
priors_in = [uniform_dist(par[i]*(1.-perc), par[i]*(1.+perc)) for i in range(dim)]

def prior_prop(xx):
    return [priors_in[i].rvs() for i in range(dim)]


# test Fisher
class StochsticProposal():
    def __init__(self, data, waveform_kw, inner_product_kwargs, fisher_only=False):
        self.wv_kw = waveform_kw
        self.inn_kw = inner_product_kwargs
        self.data = data
        if fisher_only:
            self.fisher_only = 0.0
        else:
            self.fisher_only = 1.0
    
    def __call__(self, current_p, sigma=1.0, it=1.0):

        current_param = par.copy()
        current_param[list_ind] = current_p[list_ind]
        wv_kw_tmp = self.wv_kw.copy()
        wv_kw_tmp["T"] = it * self.wv_kw["T"]

        # make sure the current parameters cannot compute the waveform
        try:
            if current_p[2]<1.2*(6+2*current_p[3])or(current_p[1]/current_p[0]>1e-4):
                print("prior proposed")
                return prior_prop(current_p)
            gen_wave.sanity_check_init(*current_p)
            htmp = gen_wave(*current_param, **wv_kw_tmp)
        except:
            print("prior proposed")
            return prior_prop(current_p)
        
        # len
        Ntmp = len(htmp)

        # h_h = inner_product([htmp.real, htmp.imag],[htmp.real, htmp.imag], **self.inn_kw)
        # d_h = inner_product([self.data[:Ntmp].real, self.data[:Ntmp].imag],[htmp.real, htmp.imag], **self.inn_kw)
        # print("snr",d_h/h_h**.5)
        d_m_h =self.data[:Ntmp] - htmp# self.data[:Ntmp] * h_h/d_h - htmp # 
        
        d_m_h = [d_m_h.real, d_m_h.imag]
        fast_wave = fast(*current_param, delta_deriv=delta_deriv, deriv_inds=deriv_inds, **wv_kw_tmp)
        X = xp.array([[inner_product(fast_wave[i].real*window, fast_wave[j].real*window, **self.inn_kw) + inner_product(fast_wave[i].imag*window, fast_wave[j].imag*window, **self.inn_kw) for i in range(dim)] for j in range(dim)])
        invGamma = xp.linalg.inv(X)
        vec = xp.array([inner_product(d_m_h, [fast_wave[i].real, fast_wave[i].imag], **self.inn_kw) for i in range(4)])
        proposed_step = xp.dot(invGamma,vec)
        
        # add eigenvector proposal
        w,v =xp.linalg.eigh(X) # xp.linalg.eigh(X) #
        
        candidate = []
        snr_matched = []
        for i in range(1):
            pp = current_param.copy()
            ii = np.random.randint(0,4)
            v_tilde = xp.random.normal(0,sigma) *v[:,ii] * w[ii]**(-0.5)
            pp[list_ind] += self.fisher_only*proposed_step.get() + v_tilde.get()

            # local search
            # htmp = gen_wave(*pp, **wv_kw_tmp)
            # h_h = inner_product([htmp.real, htmp.imag],[htmp.real, htmp.imag], **self.inn_kw)
            # d_h = inner_product([self.data[:Ntmp].real, self.data[:Ntmp].imag],[htmp.real, htmp.imag], **self.inn_kw)
            
            # snr_matched.append(d_h/h_h**.5)
            # candidate.append(pp)
        # print(snr_matched)
        # ind = np.argmax(xp.array(snr_matched).get())

        return pp[list_ind]#candidate[ind][list_ind]

propose_param = StochsticProposal(noisy_data, wv_kw, inner_product_kwargs)
propose_fisher= StochsticProposal(noisy_data, wv_kw, inner_product_kwargs, fisher_only=True)
#############################################



d_d = inner_product([noisy_data.real, noisy_data.imag],[noisy_data.real, noisy_data.imag],**inner_product_kwargs)
def loglike(xx, it=1.0):

    wv_kw_tmp = wv_kw.copy()
    wv_kw_tmp["T"] = it * wv_kw["T"]
    tmpx = par.copy()
    tmpx[:4]= xx
    try:
        if (tmpx[2]<1.2*(6+2*tmpx[3]))or(tmpx[1]/tmpx[0]>1e-4):
            print("-inf ll")
            return xp.array(-1e20)
        gen_wave.sanity_check_init(*tmpx[:4])
        htmp = gen_wave(*tmpx, **wv_kw_tmp)
    except:
        return xp.array(-1e20)
    Ntmp = len(htmp)

    # maximize overall phase
    sig1 = [noisy_data[:Ntmp].real, noisy_data[:Ntmp].imag]
    sig2 = [htmp.real, htmp.imag]
    ft_sig1 = [
            xp.fft.rfft(sig)[1:] * dt for sig in sig1
        ]  # remove DC / dt factor helps to cast to proper dimensionality
    ft_sig2 = [xp.fft.rfft(sig)[1:] * dt for sig in sig2]  # remove DC
    f_arr = xp.linspace(1/(dt*Ntmp),1/(2*dt),num=int(Ntmp/2))
    angles = [np.exp(1j*xp.angle(xp.dot(xp.conj(ft_sig1[i]), ft_sig2[i]/get_sensitivity(f_arr,sens_fn="cornish_lisa_psd")))) for i in range(2)]
    # print("angle",angles)
    inn_fd = dict(f_arr=f_arr , PSD="cornish_lisa_psd", use_gpu=gpu_available)
    ft_sig1[0] = ft_sig1[0]*angles[0]
    ft_sig1[1] = ft_sig1[1]*angles[1]
    d_h = inner_product(ft_sig1,ft_sig2,**inn_fd)
    # print("dh", d_h)
    # breakpoint()
    h_h = inner_product([htmp.real, htmp.imag],[htmp.real, htmp.imag], **inner_product_kwargs)
    # d_h = inner_product([noisy_data[:Ntmp].real, noisy_data[:Ntmp].imag],[htmp.real, htmp.imag],**inner_product_kwargs)
    # print("dh", d_h)
    # d_m_h = noisy_data[:Ntmp] - htmp
    # d_m_h = [d_m_h.real, d_m_h.imag]
    # ll = inner_product(d_m_h, d_m_h, **inner_product_kwargs)
    ll = d_d-d_h**2.0 / h_h
    
    snr_tmp = np.abs(d_h) / h_h**0.5
    if snr_tmp<6.0:
        beta = (snr_tmp/6.0)**-3.0
    res  =-0.5*ll * beta
    print("ll",res, -0.5*ll)
    return res


def log_prob(x):

        prior_vals = np.zeros((x.shape[0]))

        for prior_i, x_i in zip(priors_in, x.T):
            # print("x=",x_i)
            temp = prior_i.logpdf(x_i)
            prior_vals[np.isinf(temp)] += -1e20

        inds_eval = np.atleast_1d(np.squeeze(np.where(np.isinf(prior_vals) != True)))

        loglike_vals = np.full(x.shape[0], -1e20)

        if len(inds_eval) == 0:
            return np.array([-loglike_vals, prior_vals]).T

        x_ll = x[inds_eval]

        temp = np.array([loglike(x_i).get() for x_i in x_ll ])

        loglike_vals[inds_eval] = temp
        return np.array([loglike_vals, prior_vals]).T


print("loglike_inj:",loglike(par[:4]))

import emcee
seed=42
np.random.seed(seed)

fac = 1e5
init = np.array([prior_prop( par[list_ind]) for i in range(16)])
print("init",init)
nwalkers, ndim = init.shape

filename = f"tutorial_half_perc{perc}_seed{seed}.h5"
backend = emcee.backends.HDFBackend(filename)
backend.reset(nwalkers, ndim)

sampler0 = emcee.EnsembleSampler(nwalkers, ndim, log_prob, vectorize=True,backend=backend, moves=[
        (emcee.moves.MyProposal(propose_param), 0.5),
        (emcee.moves.MyDE(), 0.5),
    ],
    )
sampler0.run_mcmc(init, 10000, progress=True,tune=False)

plt.figure(); plt.plot(sampler0.get_log_prob(), lw=0.5); plt.savefig(f"test_{fac:.2e}_half_perc{perc}_seed{seed}.pdf");