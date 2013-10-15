#!/usr/bin/env python
# title          : Euler equations HOSM solution for gravity waves
#description     : Solves Euler eqs for M-order non-linearity using
#                  West (1987) implementation of the HOSM
#author          : Martin Goycoolea
#notes           : Adapted from fortran90 code provided by Prof. Takuji Waseda
#python_version  : python 3.3.1 , numpy 1.7.1 , scipy 0.12.0
#==========================================================================
# critical things to do:
# rogue wave detector
# fourier transformations pyfftw implment!
# optimize using cython

import numpy as np
import sympy as sy
from scipy.fftpack import fftfreq, fft2, ifft2
from math import factorial
#import pyfftw


# Global Variables
pi = np.pi
grav = 9.80665

# Program Input Variables
xdomain = 1600
ydomain = 1600
xpoints = 400
ypoints = 400
M = 5
nl_damping = 2

spec_threshold = 20
spectrum_file = 0

dt = 0.08
save_freq = 1
nstep = 50

save_on_disk = False

np.random.seed(1) # comment if you want truly random


total_time = nstep*dt
number_saved_steps = nstep//save_freq

# save file name will contain this unique string
tag = ('_M'+str(M)+'_dim'+str(xdomain)+'_'+str(ydomain)+'_time'+str(total_time)+
       '_dt'+str(dt)+'_savedensity'+str(save_freq)+'_specthresh'+str(spec_threshold))


# FFTW ffts, these are much much faster!
#test_array = np.complex128(np.random.random((ypoints,xpoints)))
#ifft2 = pyfftw.builders.ifft2(test_array, threads=8)
#fft2 = pyfftw.builders.fft2(test_array, threads=8)



def integration_scheme(scheme):
    '''Experimental, variable integration scheme feature.'''
    if scheme == 'euler':
        time_vector = np.array([0])
        weight_vector = np.array([[1]])
    elif scheme == 'rk4':
        time_vector = np.array([0,0.5,0.5,1])
        weight_vector = np.array([1/6,1/3,1/3,1/6])
    else:
        print("Sorry, we don't have that integration scheme")

    return time_vector, weight_vector


def suggest_dt(xdomain, ydomain,
               xpoints, ypoints):
    '''
    Give a  maximum delta t based on the maximum speed
    of energy propagation in the wave using the CFL condition
    with Courant_num = 1. 
    '''
    min_kx = 2*pi/xdomain
    min_ky = 2*pi/ydomain
    vx_max = 0.5 * np.sqrt(grav/min_kx)
    vy_max = 0.5 * np.sqrt(grav/min_ky)
    xstep = xdomain / (xpoints-1)
    ystep = ydomain / (ypoints-1)
    dt = 1 / (vx_max/xstep + vy_max/ystep)

    return dt


def create_2d_grids(xdomain, ydomain,
                    xpoints, ypoints):

    x = np.linspace(0,xdomain,xpoints)
    y = np.linspace(0,ydomain,ypoints)
    delx = x[1]-x[0]
    dely = y[1]-y[0]
    xgrid, ygrid = np.meshgrid(x,y)
    
    kx = fftfreq(xpoints, xdomain/xpoints)*(2*pi)
    ky = fftfreq(ypoints, ydomain/ypoints)*(2*pi)
    dkx_dky = (2*pi)/xdomain * (2*pi)/ydomain
    kxgrid, kygrid = np.meshgrid(kx,ky)
    ktgrid = np.empty_like(kxgrid)
    ktgrid = np.sqrt(kxgrid**2 + kygrid**2)

    return xgrid, ygrid, kxgrid, kygrid, ktgrid, dkx_dky


def make_spectrum(spectrum_file, kxgrid, kygrid, ktgrid, spec_threshold):
    if spectrum_file == 0:
        spectrum = jonswap_kspectrum(kxgrid, kygrid, ktgrid)
    else:
        None #make load file here np.genfromtxt(

        spectrum *= (spectrum > spec_threshold) #filtering
    return spectrum

def jonswap_kspectrum(kxgrid, kygrid, ktgrid, wcut=10*pi,
                      tp=2*pi, gamma=3.3, alpha=0.016395, sprd=1):

    wk = np.sqrt(grav * ktgrid)
    wk[0,0] = 1
    fw = wk / (2*pi)
    thr = np.arctan2(kygrid,kxgrid)

    fp = 1/tp
    fcut = wcut*fp

    shigh = 0.9
    slow = 0.7
    scond = fw <= fp
    s = scond * (shigh - slow) + slow


    a11 = alpha * grav**2
    a12 = (1/(2*pi))**4
    a13 = 1/fw**5
    a2 = (-5/4)*(fp/fw)**4
    a31 = -1*(fw-fp)**2
    a32 = 2*(s**2)*(fp**2)
    a3 = np.exp(a31/a32)

    spcj = a11 * a12 * a13 * np.exp(a2) * gamma**a3 / (2*pi)
    spcj *= fw < fcut
    spcj *= fw != 0

    gnorm = 0
    for i in range(0,315):
        thdum=(-0.5 * pi + 0.000001) + (0.01*i)
        gnorm+=np.cos(thdum)**(2.0*sprd)

    gtheta = gnorm * np.cos(thr)**(2*sprd)
    gtheta *= np.cos(thr) > 0.000001

    corrk= grav**2/(2*wk**3)

    spectrum_k =  spcj * gtheta * corrk
    spectrum_k[0,0] = 0

    return spectrum_k


def make_initial_surface(spectrum, ktgrid, dkx_dky):
    '''Make an initial surface and potential from a spectrum
    by giving each mode a random phase'''

    random_phase = (2*pi)*np.random.random(np.shape(spectrum))
    amplitude_k = np.sqrt(2*spectrum*dkx_dky)
    wk = np.sqrt(ktgrid*grav)
    wk[0,0] = 1
    surface_k = amplitude_k * np.exp(1j * random_phase)
    potential_k = 1j * grav / wk * surface_k
    potential_k[0,0] = 0.0
    surface = np.real(ifft2(surface_k))
    potential = np.real(ifft2(potential_k))

    return surface, potential




def spectrum_statistics(spectrum, ktgrid, dkx_dky):
    '''
    Returns the main statistical parameters of a given spectrum in the
    form of a dictionary:
    am0: 0th moment
    am2: 2nd moment
    am4: 4th moment
    sig_h: significant height
    epsilon: steepness
    '''
    size = np.size(ktgrid)
    wk = np.sqrt(ktgrid*grav)
    am0 = np.sum(spectrum) 
    am2 = np.sum(spectrum*wk**2)
    am4 = np.sum(spectrum*wk**4)
    sig_h = 4*np.sqrt(am0*dkx_dky) / size
    steepness = (am0*am4 - am2**2) / (am0*am4)
    output = dict(am0=am0, am2=am2, am4=am4, sig_h=sig_h, steepness=steepness)
    return output

def surface_statistics(zeta):

     mean= np.sum(zeta)/np.size(zeta)
     sum_variance = np.sum((zeta-mean)**2)
     sum_skewness = np.sum((zeta-mean)**3)
     sum_kurtosis = np.sum((zeta-mean)**4)
     std_dev = np.sqrt(sum_variance/np.size(zeta))
     kurtosis = sum_kurtosis/(std_dev**4*np.size(zeta))
     skewness = sum_skewness/(std_dev**3*np.size(zeta))

     output = dict(mean=mean, std_dev=std_dev, kurtosis=kurtosis, skewness=skewness)
     return output

def maximum_height(zeta, old_max, time):
    output = {}
    current_max = np.max(zeta)
    if current_max > old_max:
        out['max'] = current_max
        out['time'] = time
        return output


def detect_rogue_waves(old_zeta, new_zeta, wmax, wmin, sig_h, storage, step):

    tracker = np.where((old_zeta < 0) * (new_zeta > 0), 0, 1)
    difference = np.where(tracker == 0, wmax - wmin > 2*sig_h, 0)
    indices = difference.nonzero()
     
    if np.size(indices) > 0:
        print('You found rogue wave(s) at time ' + str(step) + ' and coordinates ' + str(indices))
        storage[str(step)] = indices
    wmax = np.where(new_zeta > wmax, new_zeta, wmax)
    wmin = np.where(new_zeta < wmin, new_zeta, wmin)
    wmax *= tracker
    wmin *= tracker
    return storage, wmax, wmin



def monitor_conserved_quantities(phi, zeta, dzeta_dt, kxgrid, kygrid):
    
    dphi_dx, dphi_dy, gradphi2 = calc_gradients(phi, kxgrid, kygrid)

    mass = np.sum(zeta)
    mom_x = np.sum(zeta*dphi_dx)
    mom_y = np.sum(zeta*dphi_dy)
    kin = 0.5*np.sum(phi*dzeta_dt)
    poten = 0.5*np.sum(grav*zeta**2)

    output = dict(mass=mass, mom_x=mom_x, mom_y=mom_y, kin=kin, poten=poten)
    return output


def derive_euler_equation_functions(M):
    '''Output Python functions dphi/dt and dzeta/dt for M-order
    Euler eqs. these functions' input arguments are:
    w0, ... , w(M-1), gradphi^2, gradzeta^2, zeta, gradzeta*gradphi,
    time, nonlinear_damping_coefficient
    taken from west and the zakharov fromultation.'''

    # define all symbols
    eps = sy.Symbol('eps') # smallness paramater
    zeta = sy.Symbol('zeta')
    gradphi2 = sy.Symbol('gradphi2')
    gradzeta2 = sy.Symbol('gradzeta2')
    gradzeta_gradphi = sy.Symbol('gradzeta_gradphi')
    time =  sy.Symbol('t')
    damping_factor = sy.Symbol('ta')
    damping= 1-sy.exp(-(time/damping_factor)**2)

    # gives order to the list of w(M), w(M) has order M+1
    w_list = [None]*M
    eps_list = [None]*M
    for m in range(M):
        w_list[m] = sy.Symbol('w_' + str(m))
        eps_list[m] = eps**(m+1)

    # make these arrays to allow array multiplication
    # w is the sum of all w_m with right order assigned
    w_array = np.array(w_list)
    eps_array = np.array(eps_list)
    w_eps = sum(eps_array*w_array)

    # symbolic euler eqs with order assigned
    dzeta_dt = -(gradzeta_gradphi*(eps**2)) + w_eps*(1+(gradzeta2*eps**2))
    dphi_dt = (-0.5*(gradphi2*eps**2) - (grav*zeta*eps) +
                0.5*(w_eps**2)*(1 + (gradzeta2*eps**2)) )

    args = tuple(w_list + [gradphi2, gradzeta2, zeta, gradzeta_gradphi, time, damping_factor])
    expressions = [dphi_dt, dzeta_dt]
    functions = [None]*2
    final_eqs = [None]*2

    for eqn in expressions:
        coeffs = sy.Poly(eqn, eps).all_coeffs() #gives equation as a list with coefficient of eps
        coeffs.reverse()
        coeffs_array = np.array(coeffs) # allows array multiplication
        if M > 1:
            coeffs_array[2:M+1] *= damping #applying damping factor on non_linear terms
        eqn_truncated =  np.sum(coeffs_array[1:M+1]) # sums the needed M coefficients, discards rest
        s = expressions.index(eqn)
        final_eqs[s] = sy.collect(eqn_truncated, damping) #factorizes to reduce number of computations
        functions[s] = sy.lambdify(args, final_eqs[s], modules='numpy') # now it is a numpy python function

    return functions[0], functions[1], final_eqs[0], final_eqs[1]

def dealias(array_k, M):
    '''
    Filter high-frequencies of a given 2D-array that
    represents a variable in k-space as seen in West 1987.

    ndim: number of entries in one dimension
    M: order of non-linearity).
    '''

    ydim, xdim = np.shape(array_k)

    min_kx = int(xdim/(M+1))
    max_kx = xdim - min_kx

    min_ky = int(ydim/(M+1))
    max_ky = ydim - min_ky

    array_k[min_ky:max_ky,:] = 0
    array_k[:,min_kx:max_kx] = 0

    return array_k


def integration_and_analysis(zeta0, phi0, f_zeta_t, f_phi_t, M,
                             kxgrid, kygrid, ktgrid, dt, time, damping,
                             storage, wmax, wmin):
    '''
    Perform the 4th order Runge-Kutta integration scheme for
    surface and potential.

    zeta0: surface at time step n.
    phi0: potential at time step n.
    f_zeta_t, f_phi_t: functions with Euler eqs of M-order to be solved
    found with derive_euler_equation_functions(M).
    time: time in which the integration takes place
    damping: non-linear damping factor in Euler eqs.

    Return: surface and potential at time step n+1. 
    Other variables like the orders of phi can easily be returned!
    '''

    zeta = zeta0
    phi = phi0

    rk1_zeta, rk1_phi, phi_m = tderiv_surface_potential(zeta, phi, f_zeta_t, f_phi_t, M,
                                                 kxgrid, kygrid, ktgrid, time, damping,
                                                 return_phi_m = 1)
    zeta = zeta0 + rk1_zeta*dt/2
    phi = phi0 + rk1_phi*dt/2

    rk2_zeta, rk2_phi = tderiv_surface_potential(zeta, phi, f_zeta_t, f_phi_t, M,
                                                 kxgrid, kygrid, ktgrid, time + dt/2, damping)
    zeta = zeta0 + rk2_zeta*dt/2
    phi = phi0 + rk2_phi*dt/2

    rk3_zeta, rk3_phi = tderiv_surface_potential(zeta, phi, f_zeta_t, f_phi_t, M,
                                                 kxgrid, kygrid, ktgrid, time + dt/2, damping)
    zeta = zeta0 + rk3_zeta*dt
    phi = phi0 + rk3_phi*dt

    rk4_zeta, rk4_phi = tderiv_surface_potential(zeta, phi, f_zeta_t, f_phi_t, M,
                                                 kxgrid, kygrid, ktgrid, time + dt, damping)

    dzeta_dt = 1/6 * (rk1_zeta + rk4_zeta + 2*(rk2_zeta + rk3_zeta))
    dphi_dt = 1/6 * (rk1_phi + rk4_phi + 2*(rk2_phi + rk3_phi))

    
    kernel = monitor_conserved_quantities(phi0, zeta0, dzeta_dt, kxgrid, kygrid)
    print('Total Energy: ' + str(kernel['kin'] + kernel['poten']) + ' Total Mass:' +  str(kernel['mass']))

    zeta_next = np.real(ifft2(dealias(fft2(zeta0 + dt*dzeta_dt), M))) #We transform, dealias, then transform back as real. 
    phi_next = np.real(ifft2(dealias(fft2(phi0 + dt*dphi_dt), M)))

    storage, wmax, wmin = detect_rogue_waves(zeta0, zeta_next, wmax, wmin, sig_h, storage, time)

    return zeta_next, phi_next, phi_m, storage, wmax, wmin

def tderiv_surface_potential(zeta, phi, f_dzeta_dt, f_dphi_dt, M,
                            kxgrid, kygrid, ktgrid, time, damping,
                            return_phi_m = 0):
    '''
    Calculate the rate of change of surface and potential for given initial conditions.
    See rk4 for explanation of variables.
    '''

    phi_order, phi_deriv = phi_m_and_zderivs(phi, zeta, ktgrid, M)
    vel_order = calculate_vel_order(zeta, ktgrid, M, phi_deriv)
    dzeta_dx, dzeta_dy, gradzeta2 = calc_gradients(zeta, kxgrid, kygrid)
    dphi_dx, dphi_dy, gradphi2 = calc_gradients(phi, kxgrid, kygrid)
    gradzeta_gradphi = dzeta_dx*dphi_dx + dzeta_dy*dphi_dy

    args = [None]*M
    fixed_args = [gradphi2, gradzeta2, zeta, gradzeta_gradphi, time, damping]
    for m in range(M):
        args[m]= vel_order[m]
    args[M:] = fixed_args
    dzeta_out = f_dzeta_dt(*args)
    dphi_out = f_dphi_dt(*args)

    if return_phi_m:
        return dzeta_out, dphi_out, phi_order
    else:
        return dzeta_out, dphi_out

def phi_m_and_zderivs(phi_0, zeta, ktgrid, M):
    '''
    Compute M orders of the potential Phi along with the necessary
    derivatives to compute the orders.
    '''
    phi = 0
    phi_zderivs = {}
    get_phi_zderivs(phi_0, 0, M, ktgrid, phi_zderivs)

    if M > 1:
        y, x = np.shape(phi_0)
        phi = np.zeros((M, y, x))

    for m in range(1,M):
        for s in range(m):
            nderiv = s+1
            nphi = m-(s+1)
            phi[m] -= zeta**(nderiv) * phi_zderivs[str(nphi)+str(nderiv)] / factorial(nderiv)
        get_phi_zderivs(phi[m], m, M, ktgrid, phi_zderivs)
    return phi, phi_zderivs

def get_phi_zderivs(phi_i, index, M, ktgrid, output):
    '''
    Calculate all needed derivatives for an M-order integration
    using spectral method as seen in West 1987.

    phi_i: 2D array with the pontential of order index+1
    index: order of the input 2D array
    M: order of the system of Euler eqs
    output: dictionary in which to store the derivatives as ['index'+'deriv']
    e.g out['21'] would call the first derivative of the 3rd order phi (phi_2)

    Does not return any value, but modifies dictionary output.
    '''
    phi_k = fft2(np.complex128(phi_i))
    for deriv in range(1,M+1-index):
        phi_k *= ktgrid
        dphi_dz = ifft2(phi_k)
        output[str(index)+str(deriv)] = np.copy(np.real(dphi_dz))

def calculate_vel_order(zeta, ktgrid, M, phi_deriv):
    y, x = np.shape(zeta)
    vel = np.zeros((M, y, x))
    for m in range(M):
        for s in range(m+1):
            # phi_index=(m-s), deriv=(s+1)
            vel[m] += zeta**s * phi_deriv[str(m-s)+str(s+1)] / factorial(s)
    return vel


def calc_gradients(inputvar, kxgrid, kygrid):
    var_k = fft2(np.complex128(inputvar))
    gradx_k = 1j * kxgrid * var_k
    grady_k = 1j * kygrid * var_k
    dvar_dx = np.real(ifft2(gradx_k))
    dvar_dy = np.real(ifft2(grady_k))
    grad2 = dvar_dx**2 + dvar_dy**2
    return dvar_dx, dvar_dy, grad2 


# Main Program
f_dphi_dt, f_dzeta_dt, eq_dphi_dt, eq_dzeta_dt = derive_euler_equation_functions(M)

xgrid, ygrid, kxgrid, kygrid, ktgrid, dkx_dky = create_2d_grids(xdomain, ydomain,
                                                                xpoints, ypoints)

spectrum = make_spectrum(spectrum_file, kxgrid, kygrid, ktgrid, spec_threshold)
spectrum = dealias(spectrum, M)


initial_spectrum_stats = spectrum_statistics(spectrum, ktgrid, dkx_dky)

surface, potential = make_initial_surface(spectrum, ktgrid, dkx_dky)

zeta_output = np.empty((number_saved_steps, ypoints, xpoints))
phi_output = np.empty((number_saved_steps, ypoints, xpoints))
phi_m_output = np.empty((number_saved_steps, M, ypoints, xpoints))

rogue_waves = {}
wmax = np.zeros_like(surface)
wmin= np.zeros_like(surface)
sig_h = initial_spectrum_stats['sig_h']

for n in range(nstep):
    time = dt*n  

    if (n%save_freq) == 0:
        zeta_output[n//save_freq,:,:] = surface
        phi_output[n//save_freq,:,:] = potential
        
    (surface, potential, phi_m,
     rogue_waves, wmax, wmin) = integration_and_analysis(surface, potential, f_dzeta_dt, f_dphi_dt, M,
                                                         kxgrid, kygrid, ktgrid, dt, time, nl_damping, 
                                                         rogue_waves, wmax, wmin)


if save_on_disk:
    print('Saving results, hold on a second, almost there')
    np.save('zeta_'+tag, zeta_output)
    np.save('phi_'+tag, phi_output)
    

print('The run ended.')
