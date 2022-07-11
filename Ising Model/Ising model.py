"""Hamilton Trust Summer Project: Ising model.

@author: Ruaidhrí Campion
"""

import numpy as np
import matplotlib.pyplot as plt


def corr(spins):
    """Calculate the spin-spin correlation of a given state of the system."""
    eye = np.shape(spins)[0]
    jay = np.shape(spins)[1]
    total = 0.
    for a in range(eye):
        for b in range(jay):
            total += spins[a, b] * (spins[(a+1) % eye, b]
                                    + spins[a, (b+1) % jay])
    return total


def mag(spins):
    """Calculate the total magnetisation of a given state of the system."""
    total = 0.
    for A in spins:
        for B in A:
            total += B
    return total


def H(spins, g, B):
    """Calculate the energy of a given state of the system."""
    return -g*corr(spins) - B*mag(spins)


def average(array):
    """Calculate the mean of an array."""
    total = 0.
    for i in array:
        total += i
    return total / len(array)


def variance(array):
    """Calculate the variance of an array."""
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    return total / (len(array) - 1.)


def mean_error(array):
    """Calculate the mean and the corresponding error of an array."""
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    return mean, (total / ((len(array) - 1.) * (len(array)))) ** 0.5


def m_e_v_e(array):
    """Calculate the mean and variance and corresponding errors of an array."""
    mean = average(array)
    total = 0.
    for i in range(len(array)):
        total += (array[i] - mean) ** 2.
    variance = total / (len(array) - 1.)
    return (mean, (variance / len(array)) ** 0.5, variance,
            variance * (2. / (len(array) - 1.)) ** 0.5)


def Met_Ising(width, length, tau, g, B, T):
    """Metropolis-Hastings Ising model.

    Run a Metropolis-Hastings simulation of an Ising model of a given size with
    given interaction and magnetic terms for a given length of time.
    """
    spins = np.ones((tau, width, length))
    for i in range(width):
        for j in range(length):
            if np.random.uniform(0, 1) < 0.5:
                spins[0, i, j] = -1.
    c = corr(spins[0])
    M = np.empty(tau)
    m = mag(spins[0])
    M[0] = m
    U = np.empty(tau)
    E = -g*c - B*m
    U[0] = E
    for t in range(1, tau):
        spins[t] = spins[t-1]
        for i in range(width):
            for j in range(length):
                c_diff = -2.*spins[t, i, j] * (spins[t, (i+1) % width, j]
                                               + spins[t, i, (j+1) % length]
                                               + spins[t, (i-1) % width, j]
                                               + spins[t, i, (j-1) % length])
                m_diff = -2. * spins[t, i, j]
                E_diff = -g*c_diff - B*m_diff
                if E_diff <= 0. or np.random.uniform(0, 1) <= np.exp(-E_diff
                                                                     / T):
                    spins[t, i, j] *= -1.
                    c += c_diff
                    m += m_diff
                    E += E_diff
        M[t] = m
        U[t] = E
    return U / (width*length), M / (width*length), spins[0], spins[-1]


def HB_Ising(width, length, tau, g, B, T):
    """Heat bath Ising model.

    Run a heat bath simulation of an Ising model of a given size with given
    interaction and magnetic terms for a given length of time.
    """
    spins = np.ones((tau, width, length))
    for i in range(width):
        for j in range(length):
            if np.random.uniform(0, 1) < 0.5:
                spins[0, i, j] = -1.
    c = corr(spins[0])
    M = np.empty(tau)
    m = mag(spins[0])
    M[0] = m
    U = np.empty(tau)
    E = -g*c - B*m
    U[0] = E
    for t in range(1, tau):
        spins[t] = spins[t-1]
        for i in range(width):
            for j in range(length):
                c_diff = -2.*spins[t, i, j] * (spins[t, (i+1) % width, j]
                                               + spins[t, i, (j+1) % length]
                                               + spins[t, (i-1) % width, j]
                                               + spins[t, i, (j-1) % length])
                m_diff = -2. * spins[t, i, j]
                E_diff = -g*c_diff - B*m_diff
                p = np.exp(-E_diff/T)
                if np.random.uniform(0, 1) <= p / (p+1.):
                    spins[t, i, j] *= -1.
                    c += c_diff
                    m += m_diff
                    E += E_diff
        M[t] = m
        U[t] = E
    return U / (width*length), M / (width*length), spins[0], spins[-1]


def equib_Met_Ising(width, length, tau, g, B, T):
    """Equilibrium Metropolis-Hastings Ising model.

    Run a Metropolis-Hastings simulation of an Ising model of a given size with
    given interaction and magnetic terms for a given length of time, starting
    in an equilibrium state for the zero magnetic field model for temperatures
    less than the critical temperature.
    """
    spins = np.ones((tau, width, length))
    if np.random.uniform(0, 1) < 0.5:
        spins *= -1.
    if g < 0.:
        for i in range(width):
            for j in range(length):
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                    spins[0, i, j] *= -1.
    c = corr(spins[0])
    M = np.empty(tau)
    m = mag(spins[0])
    M[0] = m
    U = np.empty(tau)
    E = -g*c - B*m
    U[0] = E
    for t in range(1, tau):
        spins[t] = spins[t-1]
        for i in range(width):
            for j in range(length):
                c_diff = -2.*spins[t, i, j] * (spins[t, (i+1) % width, j]
                                               + spins[t, i, (j+1) % length]
                                               + spins[t, (i-1) % width, j]
                                               + spins[t, i, (j-1) % length])
                m_diff = -2. * spins[t, i, j]
                E_diff = -g*c_diff - B*m_diff
                if E_diff <= 0. or np.random.uniform(0, 1) <= np.exp(-E_diff
                                                                     / T):
                    spins[t, i, j] *= -1.
                    c += c_diff
                    m += m_diff
                    E += E_diff
        M[t] = m
        U[t] = E
    return U / (width*length), M / (width*length), spins[0], spins[-1]


def equib_HB_Ising(width, length, tau, g, B, T):
    """Equilibrium heat bath Ising model.

    Run a heat bath simulation of an Ising model of a given size with given
    interaction and magnetic terms for a given length of time, starting in an
    equilibrium state for the zero magnetic field model for temperatures less
    than the critical temperature.
    """
    spins = np.ones((tau, width, length))
    if np.random.uniform(0, 1) < 0.5:
        spins *= -1.
    if g < 0.:
        for i in range(width):
            for j in range(length):
                if (i % 2 == 0 and j % 2 == 0) or (i % 2 == 1 and j % 2 == 1):
                    spins[0, i, j] *= -1.
    c = corr(spins[0])
    M = np.empty(tau)
    m = mag(spins[0])
    M[0] = m
    U = np.empty(tau)
    E = -g*c - B*m
    U[0] = E
    for t in range(1, tau):
        spins[t] = spins[t-1]
        for i in range(width):
            for j in range(length):
                c_diff = -2.*spins[t, i, j] * (spins[t, (i+1) % width, j]
                                               + spins[t, i, (j+1) % length]
                                               + spins[t, (i-1) % width, j]
                                               + spins[t, i, (j-1) % length])
                m_diff = -2. * spins[t, i, j]
                E_diff = -g*c_diff - B*m_diff
                p = np.exp(-E_diff/T)
                if np.random.uniform(0, 1) <= p / (p+1.):
                    spins[t, i, j] *= -1.
                    c += c_diff
                    m += m_diff
                    E += E_diff
        M[t] = m
        U[t] = E
    return U / (width*length), M / (width*length), spins[0], spins[-1]


colours = ["r",
           "darkorange",
           "y",
           "g",
           "b",
           "purple",
           "k",
           "saddlebrown",
           "c",
           "violet",
           "dimgray"]

# for t in range(0,tau,50):
#     pic = spins[t]
#     %varexp --imshow pic

L = 32
g = 1.
B = 0.
T_crit = 2.*g / (np.log(1. + 2.**.5))
Ts = np.concatenate((np.linspace(.2, .8, 4),
                     np.linspace(1., 1.9, 10),
                     np.linspace(2., 2.25, 6),
                     [2.26, T_crit, 2.28],
                     np.linspace(2.3, 2.45, 4),
                     np.linspace(2.5, 2.9, 5),
                     np.linspace(3., 3.8, 5),
                     np.linspace(4., 6., 9)))
T_strings = ["0.2", "0.4", "0.6", "0.8", "1", "1.1", "1.2", "1.3", "1.4",
             "1.5", "1.6", "1.7", "1.8", "1.9", "2", "2.05", "2.1", "2.15",
             "2.2", "2.25", "2.26", "T_c", "2.28", "2.3", "2.35", "2.4",
             "2.45", "2.5", "2.6", "2.7", "2.8", "2.9", "3", "3.2", "3.4",
             "3.6", "3.8", "4", "4.25", "4.5", "4.75", "5", "5.25", "5.5",
             "5.75", "6"]
t1 = len(Ts)
Tcolours = np.empty((t1, 3))
for i in range(t1):
    Tcolours[i, 0] = 1. - i/(t1-1.)
    Tcolours[i, 1] = 0.
    Tcolours[i, 2] = i/(t1-1.)
width = L
length = L
K = 17
tau = 2 ** K
disc = 2000
factors = np.empty(K)
for i in range(K):
    factors[i] = 2 ** i

(Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror) = (np.empty(len(Ts)) for a in
                                                 range(6))
(U_mses, U_mses_error, M_mses, M_mses_error, V_mses) = (np.empty((len(Ts), K))
                                                        for a in range(5))
init_spins, final_spins = (np.empty((len(Ts), width, length)) for a in
                           range(2))
for T in range(len(Ts)):
    temp_Us, temp_Ms, init_spins[T], final_spins[T] = Met_Ising(
        width, length, tau+disc, g, B, Ts[T])
    temp_Us = temp_Us[disc:]
    temp_Ms = temp_Ms[disc:]

    Us[T], Us_nerror[T], Vs[T], Vs_nerror[T] = m_e_v_e(temp_Us)
    U_mses[T, 0] = Us_nerror[T] ** 2.
    U_mses_error[T, 0] = U_mses[T, 0] * (2. / (tau - 1.)) ** 0.5
    V_mses[T, 0] = Vs_nerror[T] ** 2.

    Ms[T], Ms_nerror[T] = mean_error(temp_Ms)
    M_mses[T, 0] = Ms_nerror[T] ** 2.
    M_mses_error[T, 0] = M_mses[T, 0] * (2. / (tau - 1.)) ** 0.5

    for k in range(1, K):
        bins = int(len(temp_Us) / 2)
        binned_Us, binned_Ms = (np.empty(bins) for a in range(2))
        for b in range(bins):
            binned_Us[b] = 0.5 * (temp_Us[2*b] + temp_Us[2*b + 1])
            binned_Ms[b] = 0.5 * (temp_Ms[2*b] + temp_Ms[2*b + 1])
        temp_Us = binned_Us
        temp_Ms = binned_Ms

        U_mses[T, k] = variance(temp_Us) / bins
        U_mses_error[T, k] = U_mses[T, k] * (2. / (bins - 1.)) ** 0.5
        V_mses[T, k] = U_mses_error[T, k] ** 2.

        M_mses[T, k] = variance(temp_Ms) / bins
        M_mses_error[T, k] = M_mses[T, k] * (2. / (bins - 1.)) ** 0.5

(Us1, Us_nerror1, Ms1, Ms_nerror1, Vs1, Vs_nerror1) = (
    Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror)
(U_mses1, U_mses_error1, M_mses1, M_mses_error1, V_mses1) = (
    U_mses, U_mses_error, M_mses, M_mses_error, V_mses)
init_spins1, final_spins1 = init_spins, final_spins


(Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror) = (np.empty(len(Ts)) for a in
                                                 range(6))
(U_mses, U_mses_error, M_mses, M_mses_error, V_mses) = (np.empty((len(Ts), K))
                                                        for a in range(5))
init_spins, final_spins = (np.empty((len(Ts), width, length)) for a in
                           range(2))
for T in range(len(Ts)):
    temp_Us, temp_Ms, init_spins[T], final_spins[T] = HB_Ising(
        width, length, tau+disc, g, B, Ts[T])
    temp_Us = temp_Us[disc:]
    temp_Ms = temp_Ms[disc:]

    Us[T], Us_nerror[T], Vs[T], Vs_nerror[T] = m_e_v_e(temp_Us)
    U_mses[T, 0] = Us_nerror[T] ** 2.
    U_mses_error[T, 0] = U_mses[T, 0] * (2. / (tau - 1.)) ** 0.5
    V_mses[T, 0] = Vs_nerror[T] ** 2.

    Ms[T], Ms_nerror[T] = mean_error(temp_Ms)
    M_mses[T, 0] = Ms_nerror[T] ** 2.
    M_mses_error[T, 0] = M_mses[T, 0] * (2. / (tau - 1.)) ** 0.5

    for k in range(1, K):
        bins = int(len(temp_Us) / 2)
        binned_Us, binned_Ms = (np.empty(bins) for a in range(2))
        for b in range(bins):
            binned_Us[b] = 0.5 * (temp_Us[2*b] + temp_Us[2*b + 1])
            binned_Ms[b] = 0.5 * (temp_Ms[2*b] + temp_Ms[2*b + 1])
        temp_Us = binned_Us
        temp_Ms = binned_Ms

        U_mses[T, k] = variance(temp_Us) / bins
        U_mses_error[T, k] = U_mses[T, k] * (2. / (bins - 1.)) ** 0.5
        V_mses[T, k] = U_mses_error[T, k] ** 2.

        M_mses[T, k] = variance(temp_Ms) / bins
        M_mses_error[T, k] = M_mses[T, k] * (2. / (bins - 1.)) ** 0.5

(Us2, Us_nerror2, Ms2, Ms_nerror2, Vs2, Vs_nerror2) = (
    Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror)
(U_mses2, U_mses_error2, M_mses2, M_mses_error2, V_mses2) = (
    U_mses, U_mses_error, M_mses, M_mses_error, V_mses)
init_spins2, final_spins2 = init_spins, final_spins

g = -1.

(Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror) = (np.empty(len(Ts)) for a in
                                                 range(6))
(U_mses, U_mses_error, M_mses, M_mses_error, V_mses) = (np.empty((len(Ts), K))
                                                        for a in range(5))
init_spins, final_spins = (np.empty((len(Ts), width, length)) for a in
                           range(2))
for T in range(len(Ts)):
    temp_Us, temp_Ms, init_spins[T], final_spins[T] = Met_Ising(
        width, length, tau+disc, g, B, Ts[T])
    temp_Us = temp_Us[disc:]
    temp_Ms = temp_Ms[disc:]

    Us[T], Us_nerror[T], Vs[T], Vs_nerror[T] = m_e_v_e(temp_Us)
    U_mses[T, 0] = Us_nerror[T] ** 2.
    U_mses_error[T, 0] = U_mses[T, 0] * (2. / (tau - 1.)) ** 0.5
    V_mses[T, 0] = Vs_nerror[T] ** 2.

    Ms[T], Ms_nerror[T] = mean_error(temp_Ms)
    M_mses[T, 0] = Ms_nerror[T] ** 2.
    M_mses_error[T, 0] = M_mses[T, 0] * (2. / (tau - 1.)) ** 0.5

    for k in range(1, K):
        bins = int(len(temp_Us) / 2)
        binned_Us, binned_Ms = (np.empty(bins) for a in range(2))
        for b in range(bins):
            binned_Us[b] = 0.5 * (temp_Us[2*b] + temp_Us[2*b + 1])
            binned_Ms[b] = 0.5 * (temp_Ms[2*b] + temp_Ms[2*b + 1])
        temp_Us = binned_Us
        temp_Ms = binned_Ms

        U_mses[T, k] = variance(temp_Us) / bins
        U_mses_error[T, k] = U_mses[T, k] * (2. / (bins - 1.)) ** 0.5
        V_mses[T, k] = U_mses_error[T, k] ** 2.

        M_mses[T, k] = variance(temp_Ms) / bins
        M_mses_error[T, k] = M_mses[T, k] * (2. / (bins - 1.)) ** 0.5

(Us3, Us_nerror3, Ms3, Ms_nerror3, Vs3, Vs_nerror3) = (
    Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror)
(U_mses3, U_mses_error3, M_mses3, M_mses_error3, V_mses3) = (
    U_mses, U_mses_error, M_mses, M_mses_error, V_mses)
init_spins3, final_spins3 = init_spins, final_spins


(Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror) = (np.empty(len(Ts)) for a in
                                                 range(6))
(U_mses, U_mses_error, M_mses, M_mses_error, V_mses) = (np.empty((len(Ts), K))
                                                        for a in range(5))
init_spins, final_spins = (np.empty((len(Ts), width, length)) for a in
                           range(2))
for T in range(len(Ts)):
    temp_Us, temp_Ms, init_spins[T], final_spins[T] = HB_Ising(
        width, length, tau+disc, g, B, Ts[T])
    temp_Us = temp_Us[disc:]
    temp_Ms = temp_Ms[disc:]

    Us[T], Us_nerror[T], Vs[T], Vs_nerror[T] = m_e_v_e(temp_Us)
    U_mses[T, 0] = Us_nerror[T] ** 2.
    U_mses_error[T, 0] = U_mses[T, 0] * (2. / (tau - 1.)) ** 0.5
    V_mses[T, 0] = Vs_nerror[T] ** 2.

    Ms[T], Ms_nerror[T] = mean_error(temp_Ms)
    M_mses[T, 0] = Ms_nerror[T] ** 2.
    M_mses_error[T, 0] = M_mses[T, 0] * (2. / (tau - 1.)) ** 0.5

    for k in range(1, K):
        bins = int(len(temp_Us) / 2)
        binned_Us, binned_Ms = (np.empty(bins) for a in range(2))
        for b in range(bins):
            binned_Us[b] = 0.5 * (temp_Us[2*b] + temp_Us[2*b + 1])
            binned_Ms[b] = 0.5 * (temp_Ms[2*b] + temp_Ms[2*b + 1])
        temp_Us = binned_Us
        temp_Ms = binned_Ms

        U_mses[T, k] = variance(temp_Us) / bins
        U_mses_error[T, k] = U_mses[T, k] * (2. / (bins - 1.)) ** 0.5
        V_mses[T, k] = U_mses_error[T, k] ** 2.

        M_mses[T, k] = variance(temp_Ms) / bins
        M_mses_error[T, k] = M_mses[T, k] * (2. / (bins - 1.)) ** 0.5

(Us4, Us_nerror4, Ms4, Ms_nerror4, Vs4, Vs_nerror4) = (
    Us, Us_nerror, Ms, Ms_nerror, Vs, Vs_nerror)
(U_mses4, U_mses_error4, M_mses4, M_mses_error4, V_mses4) = (
    U_mses, U_mses_error, M_mses, M_mses_error, V_mses)
init_spins4, final_spins4 = init_spins, final_spins


for T in range(len(Ts)):
    plt.imshow(init_spins1[T])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.title(r"Initial state (M-H, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Initial (M-H, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.imshow(final_spins1[T])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.title(r"Final state (M-H, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Final (M-H, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.errorbar(factors, U_mses1[T], yerr=U_mses_error1[T],
                 color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{U}$)")
    plt.title(r"Binned MSE($\bar{U}$) (M-H, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned U (M-H, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.errorbar(factors, M_mses1[T], yerr=M_mses_error1[T],
                 color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{M}$)")
    plt.title(r"Binned MSE($\bar{M}$) (M-H, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned M (M-H, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.plot(factors, V_mses1[T]/(Ts[T]*Ts[T]),
             color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{c_V}$)")
    plt.title(r"Binned MSE($\bar{c_V}$) (M-H, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned c_V (M-H, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()
for T in range(len(Ts)):
    plt.imshow(init_spins2[T])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.title(r"Initial state (HB, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Initial (HB, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.imshow(final_spins2[T])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.title(r"Final state (HB, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Final (HB, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.errorbar(factors, U_mses2[T], yerr=U_mses_error2[T],
                 color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{U}$)")
    plt.title(r"Binned MSE($\bar{U}$) (HB, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned U (HB, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.errorbar(factors, M_mses2[T], yerr=M_mses_error2[T],
                 color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{M}$)")
    plt.title(r"Binned MSE($\bar{M}$) (HB, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned M (HB, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.plot(factors, V_mses2[T]/(Ts[T]*Ts[T]),
             color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{c_V}$)")
    plt.title(r"Binned MSE($\bar{c_V}$) (HB, $g=1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned c_V (HB, g = 1, T = %s).pdf" % T_strings[T])
    plt.clf()
for T in range(len(Ts)):
    plt.imshow(init_spins3[T])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.title(r"Initial state (M-H, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Initial (M-H, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.imshow(final_spins3[T])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.title(r"Final state (M-H, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Final (M-H, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.errorbar(factors, U_mses3[T], yerr=U_mses_error3[T],
                 color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{U}$)")
    plt.title(r"Binned MSE($\bar{U}$) (M-H, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned U (M-H, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.errorbar(factors, M_mses3[T], yerr=M_mses_error3[T],
                 color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{M}$)")
    plt.title(r"Binned MSE($\bar{M}$) (M-H, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned M (M-H, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.plot(factors, V_mses3[T]/(Ts[T]*Ts[T]),
             color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{c_V}$)")
    plt.title(r"Binned MSE($\bar{c_V}$) (M-H, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned c_V (M-H, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()
for T in range(len(Ts)):
    plt.imshow(init_spins4[T])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.title(r"Initial state (HB, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Initial (HB, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.imshow(final_spins1[T])
    plt.tick_params(axis="both", which="both", bottom=False, top=False,
                    labelbottom=False, left=False, right=False,
                    labelleft=False)
    plt.title(r"Final state (HB, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Final (HB, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.errorbar(factors, U_mses4[T], yerr=U_mses_error4[T],
                 color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{U}$)")
    plt.title(r"Binned MSE($\bar{U}$) (HB, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned U (HB, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.errorbar(factors, M_mses4[T], yerr=M_mses_error4[T],
                 color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{M}$)")
    plt.title(r"Binned MSE($\bar{M}$) (HB, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned M (HB, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()

    plt.plot(factors, V_mses4[T]/(Ts[T]*Ts[T]),
             color=(Tcolours[T, 0], Tcolours[T, 1], Tcolours[T, 2]))
    plt.xscale("log", base=2)
    plt.xlim(1., factors[-1])
    plt.xlabel(r"$b$")
    plt.ylabel(r"Var($\bar{c_V}$)")
    plt.title(r"Binned MSE($\bar{c_V}$) (HB, $g=-1$, $T=%s$)" % T_strings[T])
    plt.savefig("Binned c_V (HB, g = -1, T = %s).pdf" % T_strings[T])
    plt.clf()


Us_indexes1 = np.array([0, 0, 11, 8, 4, 4, 6, 4, 4, 4, 4, 6, 7, 7, 8, 9, 10, 9,
                        9, 10, 10, 9, 9, 9, 8, 9, 10, 7, 8, 6, 5, 7, 8, 8, 7,
                        6, 6, 7, 5, 6, 7, 8, 8, 8, 8, 6])
Ms_indexes1 = np.array([0, 0, 11, 8, 4, 4, 6, 4, 4, 4, 4, 5, 6, 7, 8, 9, 9, 9,
                        10, 13, 13, 13, 13, 12, 11, 10, 9, 10, 8, 8, 8, 7, 9,
                        6, 6, 8, 6, 7, 5, 3, 6, 7, 6, 6, 8, 7])
Us_indexes2 = np.array([0, 0, 10, 10, 3, 4, 3, 6, 6, 6, 8, 6, 6, 7, 8, 7, 8, 8,
                        10, 10, 12, 10, 10, 9, 11, 9, 9, 11, 9, 8, 7, 6, 8, 10,
                        7, 7, 5, 4, 6, 4, 6, 4, 5, 4, 5, 6])
Ms_indexes2 = np.array([0, 0, 10, 10, 3, 4, 3, 6, 5, 5, 8, 6, 6, 6, 9, 7, 9, 8,
                        10, 10, 12, 10, 10, 9, 12, 11, 10, 11, 9, 8, 10, 8, 8,
                        7, 6, 8, 6, 7, 5, 6, 5, 7, 7, 5, 5, 6])
Us_indexes3 = np.array([0, 0, 2, 4, 3, 4, 6, 3, 4, 5, 5, 4, 6, 7, 5, 5, 7, 7,
                        10, 9, 8, 9, 9, 9, 10, 8, 8, 8, 6, 5, 5, 5, 5, 6, 6, 6,
                        5, 5, 7, 6, 6, 6, 7, 5, 5, 6])
Ms_indexes3 = np.array([0, 0, 2, 4, 3, 4, 3, 4, 5, 5, 4, 5, 5, 5, 5, 5, 6, 5,
                        6, 6, 6, 5, 5, 5, 5, 6, 6, 7, 6, 7, 8, 6, 6, 6, 7, 6,
                        7, 6, 7, 6, 7, 7, 7, 7, 7, 6])
Us_indexes4 = np.array([0, 0, 5, 3, 2, 4, 3, 4, 4, 7, 5, 5, 6, 6, 6, 10, 7, 9,
                        9, 11, 9, 11, 9, 10, 10, 10, 10, 8, 7, 7, 7, 7, 6, 5,
                        5, 5, 5, 6, 4, 4, 4, 4, 5, 4, 4, 4])
Ms_indexes4 = np.array([0, 0, 4, 6, 3, 5, 4, 4, 5, 4, 4, 5, 5, 5, 5, 5, 6, 5,
                        6, 4, 5, 4, 6, 5, 5, 4, 6, 6, 6, 6, 5, 5, 6, 5, 7, 5,
                        5, 4, 4, 5, 4, 5, 6, 6, 6, 6])
(Us_tau1, Us_terror1, Ms_tau1, Ms_terror1,
 Us_tau2, Us_terror2, Ms_tau2, Ms_terror2,
 Us_tau3, Us_terror3, Ms_tau3, Ms_terror3,
 Us_tau4, Us_terror4, Ms_tau4, Ms_terror4) = (np.empty(len(Ts)) for a in
                                              range(16))
for i in range(len(Ts)):
    a = U_mses1[i, Us_indexes1[i]]
    b = M_mses1[i, Ms_indexes1[i]]
    Us_terror1[i] = a ** 0.5
    Ms_terror1[i] = b ** 0.5
    Us_tau1[i] = a / U_mses1[i, 0]
    Ms_tau1[i] = b / M_mses1[i, 0]

    a = U_mses2[i, Us_indexes2[i]]
    b = M_mses2[i, Ms_indexes2[i]]
    Us_terror2[i] = a ** 0.5
    Ms_terror2[i] = b ** 0.5
    Us_tau2[i] = a / U_mses2[i, 0]
    Ms_tau2[i] = b / M_mses2[i, 0]

    a = U_mses3[i, Us_indexes3[i]]
    b = M_mses3[i, Ms_indexes3[i]]
    Us_terror3[i] = a ** 0.5
    Ms_terror3[i] = b ** 0.5
    Us_tau3[i] = a / U_mses3[i, 0]
    Ms_tau3[i] = b / M_mses3[i, 0]

    a = U_mses4[i, Us_indexes4[i]]
    b = M_mses4[i, Ms_indexes4[i]]
    Us_terror4[i] = a ** 0.5
    Ms_terror4[i] = b ** 0.5
    Us_tau4[i] = a / U_mses4[i, 0]
    Ms_tau4[i] = b / M_mses4[i, 0]


plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Us_tau1, color="r", label=r"M-H, $g=1$")
plt.plot(Ts, Us_tau2, color="g", label=r"HB, $g=1$")
plt.xlim(Ts[2], Ts[-1])
plt.xlabel(r"$T$")
plt.ylabel(r"$\tau$")
plt.title("Energy correlation times")
plt.legend()
plt.savefig("U times (g = 1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Us_tau3, color="b", label=r"M-H, $g=-1$")
plt.plot(Ts, Us_tau4, color="m", label=r"HB, $g=-1$")
plt.xlim(Ts[1], Ts[-1])
plt.xlabel(r"$T$")
plt.ylabel(r"$\tau$")
plt.title("Energy correlation times")
plt.legend()
plt.savefig("U times (g = -1).pdf")
plt.show()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Ms_tau1, color="r", label=r"M-H, $g=1$")
plt.plot(Ts, Ms_tau2, color="g", label=r"HB, $g=1$")
plt.xlim(Ts[2], Ts[-1])
plt.xlabel(r"$T$")
plt.ylabel(r"$\tau$")
plt.title("Magnetisation correlation times")
plt.legend()
plt.savefig("M times (g = 1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Ms_tau3, color="b", label=r"M-H, $g=-1$")
plt.plot(Ts, Ms_tau4, color="m", label=r"HB, $g=-1$")
plt.xlim(Ts[1], Ts[-1])
plt.xlabel(r"$T$")
plt.ylabel(r"$\tau$")
plt.title("Magnetisation correlation times")
plt.legend()
plt.savefig("M times (g = -1).pdf")
plt.show()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, Us1, yerr=Us_terror1, color="r")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")
plt.title(r"Average energy (M-H, $g=1$)")
plt.legend()
plt.savefig("U (M-H, g = 1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, Us2, yerr=Us_terror2, color="g")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")
plt.title(r"Average energy (HB, $g=1$)")
plt.legend()
plt.savefig("U (HB, g = 1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, Us3, yerr=Us_terror3, color="b")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")
plt.title(r"Average energy (M-H, $g=-1$)")
plt.legend()
plt.savefig("U (M-H, g = -1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, Us4, yerr=Us_terror4, color="m")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")
plt.title(r"Average energy (HB, $g=-1$)")
plt.legend()
plt.savefig("U (HB, g = -1).pdf")
plt.show()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, abs(Ms1), yerr=Ms_terror1, color="r")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$|M|$")
plt.title(r"Average magnetisation (M-H, $g=1$)")
plt.legend()
plt.savefig("M (M-H, g = 1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, abs(Ms2), yerr=Ms_terror2, color="g")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$|M|$")
plt.title(r"Average magnetisation (HB, $g=-1$)")
plt.legend()
plt.savefig("M (HB, g = 1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, abs(Ms3), yerr=Ms_terror3, color="b")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$|M|$")
plt.title(r"Average magnetisation (M-H, $g=1$-)")
plt.legend()
plt.savefig("M (M-H, g = -1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, abs(Ms4), yerr=Ms_terror4, color="m")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$|M|$")
plt.title(r"Average magnetisation (HB, $g=-1$)")
plt.legend()
plt.savefig("M (HB, g = -1).pdf")
plt.show()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Vs1/(Ts*Ts), color="r")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$c_V$")
plt.title(r"Average heat capacity (M-H, $g=1$)")
plt.legend()
plt.savefig("c_V (M-H, g = 1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Vs2/(Ts*Ts), color="g")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$c_V$")
plt.title(r"Average heat capacity (HB, $g=1$)")
plt.legend()
plt.savefig("c_V (HB, g = 1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Vs3/(Ts*Ts), color="b")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$c_V$")
plt.title(r"Average heat capacity (M-H, $g=-1$)")
plt.legend()
plt.savefig("c_V (M-H, g = -1).pdf")
plt.show()
plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Vs4/(Ts*Ts), color="m")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$c_V$")
plt.title(r"Average heat capacity (HB, $g=-1$)")
plt.legend()
plt.savefig("c_V (HB, g = -1).pdf")
plt.show()


plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, Us1, yerr=Us_terror1, color="r", label=r"M-H, $g=1$")
plt.errorbar(Ts, Us2, yerr=Us_terror2, color="g", label=r"HB, $g=1$")
plt.errorbar(Ts, Us3, yerr=Us_terror3, color="b", label=r"M-H, $g=-1$")
plt.errorbar(Ts, Us4, yerr=Us_terror4, color="m", label=r"HB"", $g=-1$")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$U$")
plt.title(r"Average energy")
plt.legend()
plt.savefig("U.pdf")
plt.show()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, abs(Ms1), yerr=Ms_terror1, color="r", label=r"M-H, $g=1$")
plt.errorbar(Ts, abs(Ms2), yerr=Ms_terror2, color="g", label=r"HB, $g=1$")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$|M|$")
plt.title(r"Average magnetisation")
plt.legend()
plt.savefig("M1.pdf")
plt.show()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.errorbar(Ts, abs(Ms3), yerr=Ms_terror3, color="b", label=r"M-H, $g=-1$")
plt.errorbar(Ts, abs(Ms4), yerr=Ms_terror4, color="m", label=r"HB, $g=-1$")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$|M|$")
plt.title(r"Average magnetisation absolute value")
plt.legend(loc="upper left")
plt.savefig("M2.pdf")
plt.show()

plt.axvline(x=T_crit, color="k", linestyle="dashed", label=r"$T_c$")
plt.plot(Ts, Vs1/(Ts*Ts), color="r", label=r"M-H, $g=1$")
plt.plot(Ts, Vs2/(Ts*Ts), color="g", label=r"HB, $g=1$")
plt.plot(Ts, Vs3/(Ts*Ts), color="b", label=r"M-H, $g=-1$")
plt.plot(Ts, Vs4/(Ts*Ts), color="m", label=r"HB, $g=-1$")
plt.xlim(min(Ts), max(Ts))
plt.xlabel(r"$T$")
plt.ylabel(r"$c_V$")
plt.title(r"Average heat capacity")
plt.legend()
plt.savefig("c_V.pdf")
plt.show()
