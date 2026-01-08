
import numpy as np
import matplotlib.pyplot as plt

# ===== USTAWIENIA OGÓLNE =====
np.random.seed(42)
plt.rcParams['figure.figsize'] = (8, 5)
plt.rcParams['axes.grid'] = True
plt.rcParams['font.size'] = 11

# ===== FUNKCJE GENERUJĄCE =====
def generate_uniform(n, a=0.0, b=1.0):
    return np.random.uniform(a, b, size=n)

# Rozkład wykładniczy Exp(λ): metoda odwrotnej dystrybuanty
# X = -ln(U)/λ, U ~ U(0,1)
def generate_exponential(n, lam=1.0):
    u = np.random.uniform(0, 1, size=n)
    return -np.log(u) / lam

# Rozkład Cauchy'ego: X = tan(pi*(U - 1/2)), U ~ U(0,1)
def generate_cauchy(n, x0=0.0, gamma=1.0):
    u = np.random.uniform(0, 1, size=n)
    return x0 + gamma * np.tan(np.pi * (u - 0.5))

# ===== GĘSTOŚCI DO NAKŁADANIA NA HISTOGRAM =====
def pdf_exponential(x, lam=1.0):
    return lam * np.exp(-lam * x) * (x >= 0)

def pdf_cauchy(x, x0=0.0, gamma=1.0):
    return (1/np.pi) * (gamma / (gamma**2 + (x - x0)**2))

# Gęstość normalna (do CLT)
def pdf_normal(x, mu=0.0, sigma=1.0):
    return (1 / (np.sqrt(2*np.pi) * sigma)) * np.exp(-0.5 * ((x - mu)/sigma)**2)

# ===== WIZUALIZACJE =====
def plot_hist_with_density(samples, pdf_func, pdf_kwargs, x_min, x_max, bins, title, filename):
    # Histogram z normalizacją do gęstości
    plt.figure()
    plt.hist(samples, bins=bins, density=True, alpha=0.65, color='#4472C4', edgecolor='white', label='Histogram (gęstość)')
    x = np.linspace(x_min, x_max, 1000)
    y = pdf_func(x, **pdf_kwargs)
    plt.plot(x, y, color='#D95319', lw=2.2, label='Funkcja gęstości')
    plt.title(title)
    plt.xlabel('Wartość zmiennej')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

# Rozkład średnich dla M niezależnych symulacji po N próbkach
def plot_means_with_normal_overlay(means, mu, sigma_over_sqrtN, title, filename):
    plt.figure()
    plt.hist(means, bins=60, density=True, alpha=0.65, color='#2CA02C', edgecolor='white', label='Histogram średnich')
    x = np.linspace(np.min(means), np.max(means), 1000)
    y = pdf_normal(x, mu=mu, sigma=sigma_over_sqrtN)
    plt.plot(x, y, color='#D95319', lw=2.2, label=f'N({mu:.3f}, {sigma_over_sqrtN:.3f}²) – przybliżenie CLT')
    plt.title(title)
    plt.xlabel('Średnia z N próbek')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.tight_layout()
    plt.savefig(filename, dpi=160)
    plt.close()

# ===== SYMULACJE CLT =====
def clt_for_exponential(M=4000, N=50, lam=1.0):
    # Pojedyncza duża próbka do histogramu z gęstością
    raw = generate_exponential(100000, lam=lam)
    plot_hist_with_density(
        raw,
        pdf_func=pdf_exponential,
        pdf_kwargs={'lam': lam},
        x_min=0.0, x_max=np.quantile(raw, 0.995),
        bins=80,
        title=f'Exp(λ={lam}) – histogram i gęstość',
        filename='exp_hist_density.png'
    )
    # M średnich z N próbek
    means = []
    for _ in range(M):
        x = generate_exponential(N, lam=lam)
        means.append(np.mean(x))
    means = np.array(means)
    mu, sigma = 1.0/lam, 1.0/lam
    plot_means_with_normal_overlay(
        means,
        mu=mu,
        sigma_over_sqrtN=sigma/np.sqrt(N),
        title=f'Exp(λ={lam}) – rozkład średnich (M={M}, N={N})',
        filename='exp_means_hist.png'
    )
    return {
        'emp_means_mean': float(means.mean()),
        'emp_means_std': float(means.std(ddof=1)),
        'theo_mean': mu,
        'theo_std': sigma/np.sqrt(N)
    }

def clt_for_cauchy(M=4000, N=50, x0=0.0, gamma=1.0):
    # Pojedyncza duża próbka do histogramu z gęstością – zakres przycięty do wizualizacji
    raw = generate_cauchy(150000, x0=x0, gamma=gamma)
    x_min, x_max = -25, 25
    plot_hist_with_density(
        raw[(raw >= x_min) & (raw <= x_max)],
        pdf_func=pdf_cauchy,
        pdf_kwargs={'x0': x0, 'gamma': gamma},
        x_min=x_min, x_max=x_max,
        bins=120,
        title=f'Cauchy(x0={x0}, γ={gamma}) – histogram (przycięte) i gęstość',
        filename='cauchy_hist_density.png'
    )
    # M średnich z N próbek: dla Cauchy, średnia nie stabilizuje się; wciąż Cauchy
    means = []
    for _ in range(M):
        x = generate_cauchy(N, x0=x0, gamma=gamma)
        means.append(np.mean(x))
    means = np.array(means)
    # Nakładamy gęstość Cauchy (ta sama skala) – własność stabilna
    plt.figure()
    plt.hist(means[(means >= x_min) & (means <= x_max)], bins=120, density=True,
             alpha=0.65, color='#8E44AD', edgecolor='white', label='Histogram średnich (przycięte)')
    xs = np.linspace(x_min, x_max, 1000)
    plt.plot(xs, pdf_cauchy(xs, x0=x0, gamma=gamma), color='#D95319', lw=2.2,
             label='Gęstość Cauchy – brak zbieżności do N')
    plt.title(f'Cauchy – rozkład średnich (M={M}, N={N})')
    plt.xlabel('Średnia z N próbek')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.tight_layout()
    plt.savefig('cauchy_means_hist.png', dpi=160)
    plt.close()

    # Zwracamy kilka statystyk opisowych (uwaga: średnia/odchylenie dla Cauchy bywa niestabilna)
    return {
        'emp_means_median': float(np.median(means)),
        'emp_means_iqr': float(np.subtract(*np.percentile(means, [75, 25])))
    }

# ===== LOSOWY SPACER (random walk) – wpływ p i N =====
def random_walk_final_positions(M=5000, N=50, p=0.5):
    # Krok: +1 z prawdopodobieństwem p, -1 z prawdopodobieństwem 1-p
    steps = np.where(np.random.rand(M, N) < p, 1, -1)
    final_pos = steps.sum(axis=1)
    mu = N * (2*p - 1)
    sigma = 2 * np.sqrt(N * p * (1-p))
    # Wykres histogramu pozycji końcowych + normalne przybliżenie
    plt.figure()
    bins = np.arange(final_pos.min()-1, final_pos.max()+2)  # całkowite przedziały
    plt.hist(final_pos, bins=bins, density=True, alpha=0.65, color='#17BECF', edgecolor='white', label='Pozycje końcowe')
    xs = np.linspace(final_pos.min(), final_pos.max(), 1000)
    plt.plot(xs, pdf_normal(xs, mu=mu, sigma=sigma), color='#D95319', lw=2.2,
             label=f'Przybliżenie N({mu:.1f}, {sigma:.1f}²)')
    plt.title(f'Random walk: p={p}, N={N}, M={M}')
    plt.xlabel('Pozycja końcowa S_N')
    plt.ylabel('Gęstość')
    plt.legend()
    plt.tight_layout()
    plt.savefig(f'rw_p{str(p).replace(".", "_")}_N{N}_hist.png', dpi=160)
    plt.close()
    return {
        'emp_mean': float(final_pos.mean()),
        'emp_std': float(final_pos.std(ddof=1)),
        'theo_mean': float(mu),
        'theo_std': float(sigma)
    }

# ===== URUCHOMIENIE DEMO =====
results = {}
results['exp'] = clt_for_exponential(M=4000, N=50, lam=1.0)
results['cauchy'] = clt_for_cauchy(M=4000, N=50, x0=0.0, gamma=1.0)
results['rw_p05_N50'] = random_walk_final_positions(M=6000, N=50, p=0.5)
results['rw_p07_N50'] = random_walk_final_positions(M=6000, N=50, p=0.7)
results['rw_p07_N200'] = random_walk_final_positions(M=6000, N=200, p=0.7)

print('--- PODSUMOWANIA ---')
for k,v in results.items():
    print(k, v)
