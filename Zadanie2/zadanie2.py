
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(20260108)
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11

# ===== Pomocnicze: gęstość normalna =====
def pdf_normal(x, mu=0.0, sigma=1.0):
    return (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

# ===== Generator spaceru =====
def simulate_final_positions(M=6000, N=200, p=0.11, x0=0):
    steps = np.where(np.random.rand(M, N) < p, 1, -1)
    final_pos = steps.sum(axis=1) + x0
    return final_pos

# ===== Trajektorie =====
def random_walk_paths(M=12, N=500, p=0.11, x0=0):
    steps = np.where(np.random.rand(M, N) < p, 1, -1)
    S = np.cumsum(steps, axis=1)
    S = np.hstack([np.full((M,1), x0), x0 + S])
    return S

# ===== 1) Średnia pozycja + 95% CI dla p=0.11 vs N =====
def plot_mean_ci_vs_N(p=0.11, Ns=(20,50,100,200,500,1000), M=6000, filename='mean_ci_p0_11.png'):
    means = []
    ci_low = []
    ci_high = []
    theo_means = []
    theo_ci_low = []
    theo_ci_high = []
    for N in Ns:
        final_pos = simulate_final_positions(M=M, N=N, p=p)
        m = final_pos.mean()
        s = final_pos.std(ddof=1)
        z = 1.96
        se = s/np.sqrt(M)
        means.append(m)
        ci_low.append(m - z*se)
        ci_high.append(m + z*se)
        mu_theo = N*(2*p-1)
        sigma_SN = 2*np.sqrt(N*p*(1-p))
        theo_means.append(mu_theo)
        theo_ci_low.append(mu_theo - z*(sigma_SN/np.sqrt(M)))
        theo_ci_high.append(mu_theo + z*(sigma_SN/np.sqrt(M)))
    Ns_arr = np.array(Ns)
    plt.figure(figsize=(9,6))
    plt.errorbar(Ns_arr, means, yerr=np.array(ci_high)-np.array(means), fmt='o', capsize=5,
                 color='#1f77b4', ecolor='#1f77b4', label='Empiryczna średnia ±95% CI')
    plt.plot(Ns_arr, theo_means, color='black', lw=2.2, label='Teoretyczna średnia')
    plt.fill_between(Ns_arr, theo_ci_low, theo_ci_high, color='gray', alpha=0.25, label='Teoretyczne 95% CI')
    plt.title(f'Średnia pozycja S_N z 95% CI dla p={p} (M={M})')
    plt.xlabel('Liczba kroków N')
    plt.ylabel('Średnia pozycja S_N')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    return {
        'Ns': Ns,
        'emp_means': means,
        'emp_ci': list(zip(ci_low, ci_high)),
        'theo_means': theo_means,
        'theo_ci': list(zip(theo_ci_low, theo_ci_high))
    }

# ===== 2) Histogramy pozycji końcowych dla różnych p (stałe N) =====
def plot_histograms_var_p(N=200, ps=(0.11,0.5,0.89), M=8000, filename='hist_var_p_N200.png'):
    plt.figure(figsize=(10,7))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    summaries = {}
    for i,p in enumerate(ps):
        final_pos = simulate_final_positions(M=M, N=N, p=p)
        mu = N*(2*p-1)
        sigma = 2*np.sqrt(N*p*(1-p))
        bins = np.arange(final_pos.min()-1, final_pos.max()+2)
        hist_vals, bin_edges = np.histogram(final_pos, bins=bins, density=True)
        centers = (bin_edges[:-1] + bin_edges[1:]) / 2
        plt.plot(centers, hist_vals, drawstyle='steps-mid', color=colors[i%len(colors)], label=f'p={p}')
        xs = np.linspace(centers.min(), centers.max(), 1000)
        plt.plot(xs, pdf_normal(xs, mu=mu, sigma=sigma), color=colors[i%len(colors)], ls='--')
        summaries[p] = {
            'emp_mean': float(final_pos.mean()),
            'emp_std': float(final_pos.std(ddof=1)),
            'theo_mean': float(mu),
            'theo_std': float(sigma)
        }
    plt.title(f'Histogramy (krokowe) S_N dla różnych p, N={N}, M={M} + przybliżenia normalne')
    plt.xlabel('S_N (pozycja końcowa)')
    plt.ylabel('Gęstość (kroki)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    return summaries

# ===== 3) Przykładowe trajektorie =====
def plot_sample_trajectories(p=0.11, N=500, M_show=12, filename='trajectories_p0_11_N500.png'):
    S = random_walk_paths(M=M_show, N=N, p=p, x0=0)
    t = np.arange(N+1)
    plt.figure(figsize=(9,6))
    for i in range(M_show):
        plt.plot(t, S[i], lw=1.3, alpha=0.9)
    mu_t = t*(2*p-1)
    plt.plot(t, mu_t, color='black', lw=2.2, label='E[S_t] = t(2p-1)')
    plt.title(f'Przykładowe trajektorie wędrówki (p={p}, N={N})')
    plt.xlabel('Krok t')
    plt.ylabel('Pozycja S_t')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

# ===== Uruchomienie symulacji zgodnie z zadaniem =====
summary = {}
summary['mean_ci_p0_11'] = plot_mean_ci_vs_N(p=0.11, Ns=(20,50,100,200,500,1000), M=6000, filename='mean_ci_p0_11.png')
summary['hist_var_p_N200'] = plot_histograms_var_p(N=200, ps=(0.11,0.5,0.89), M=8000, filename='hist_var_p_N200.png')
plot_sample_trajectories(p=0.11, N=500, M_show=12, filename='trajectories_p0_11_N500.png')

print('--- PODSUMOWANIA ---')
for k,v in summary.items():
    print(k, v if k=='hist_var_p_N200' else {kk: (vv if kk=='Ns' else '...') for kk,vv in v.items()})
