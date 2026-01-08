
import numpy as np
import matplotlib.pyplot as plt

np.random.seed(123)
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11

# ===== Pomocnicze: gęstość normalna =====
def pdf_normal(x, mu=0.0, sigma=1.0):
    return (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

# ===== Generator spaceru =====
def random_walk_paths(M=100, N=100, p=0.5, x0=0):
    # Zwraca macierz M x (N+1) z trajektoriami (z pozycją początkową)
    steps = np.where(np.random.rand(M, N) < p, 1, -1)
    S = np.cumsum(steps, axis=1)
    S = np.hstack([np.full((M,1), x0), x0 + S])
    return S

# ===== Histogram pozycji końcowych + overlay normalny =====
def plot_final_hist(M=6000, N=200, p=0.11, title='', filename='hist.png'):
    steps = np.where(np.random.rand(M, N) < p, 1, -1)
    final_pos = steps.sum(axis=1)  # S_N
    mu = N * (2*p - 1)
    sigma = 2 * np.sqrt(N * p * (1-p))

    plt.figure()
    bins = np.arange(final_pos.min()-1, final_pos.max()+2)
    plt.hist(final_pos, bins=bins, density=True, alpha=0.7, color='#1f77b4', edgecolor='white', label='Pozycje końcowe')
    xs = np.linspace(final_pos.min(), final_pos.max(), 1000)
    plt.plot(xs, pdf_normal(xs, mu=mu, sigma=sigma), color='#d62728', lw=2.2,
             label=f'Przybliżenie N({mu:.1f}, {sigma:.1f}²)')
    plt.title(title if title else f'Wędrówka: p={p}, N={N}, M={M}')
    plt.xlabel('S_N (pozycja końcowa)')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    return {
        'emp_mean': float(final_pos.mean()),
        'emp_std': float(final_pos.std(ddof=1)),
        'theo_mean': float(mu),
        'theo_std': float(sigma)
    }

# ===== Wykres trajektorii =====
def plot_trajectories(M_show=12, N=500, p=0.11, x0=0, filename='traj.png'):
    S = random_walk_paths(M=M_show, N=N, p=p, x0=x0)
    t = np.arange(N+1)
    plt.figure(figsize=(9, 6))
    for i in range(M_show):
        plt.plot(t, S[i], lw=1.2, alpha=0.9)
    mu_t = t * (2*p - 1)
    plt.plot(t, mu_t, color='black', lw=2.5, label='Ścieżka średniej E[S_t]')
    plt.title(f'Traje ktorie wędrówki: p={p}, N={N}, x0={x0}')
    plt.xlabel('Krok t')
    plt.ylabel('Pozycja S_t')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

# ===== Porównanie różnych p przy stałym N =====
def plot_compare_p(N=200, ps=(0.11, 0.5, 0.89), M=8000, filename='compare_p.png'):
    plt.figure(figsize=(9, 6))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#9467bd', '#8c564b']
    summaries = {}
    for i, p in enumerate(ps):
        steps = np.where(np.random.rand(M, N) < p, 1, -1)
        final_pos = steps.sum(axis=1)
        mu = N * (2*p - 1)
        sigma = 2 * np.sqrt(N * p * (1-p))
        # Gęstość przybliżona histogramem krokowym
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
    plt.title(f'Porównanie rozkładów S_N dla różnych p (N={N}, M={M})')
    plt.xlabel('S_N (pozycja końcowa)')
    plt.ylabel('Gęstość (kroki)')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()
    return summaries

# ===== Uruchomienie: p=0.11 kluczowy, plus porównania =====
summary = {}
# Trajektorie dla p=0.11
plot_trajectories(M_show=12, N=500, p=0.11, filename='rw_traj_p0_11_N500.png')

# Histogramy dla p=0.11 i różnych N
summary['p0_11_N50']   = plot_final_hist(M=6000, N=50,  p=0.11, filename='rw_hist_p0_11_N50.png')
summary['p0_11_N200']  = plot_final_hist(M=6000, N=200, p=0.11, filename='rw_hist_p0_11_N200.png')
summary['p0_11_N1000'] = plot_final_hist(M=6000, N=1000,p=0.11, filename='rw_hist_p0_11_N1000.png')

# Porównanie p przy N=200
summary['compare_p_N200'] = plot_compare_p(N=200, ps=(0.11, 0.5, 0.89), M=8000, filename='rw_compare_p_N200.png')

print('--- PODSUMOWANIE ---')
for k,v in summary.items():
    print(k, v)

