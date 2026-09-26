import numpy as np
import matplotlib.pyplot as plt
from scipy import integrate

HBAR = 1.0      # constante de Planck reduzida (unidades naturais)
MASS = 1.0      # massa da partícula (unidades naturais)
L = 1.0         # largura da caixa (unidades naturais)
N_STATES = 5    # número de autoestados a exibir


def psi_n(n, x, L=L):
    """Autofunção normalizada do poço infinito."""
    if n < 1 or not float(n).is_integer():
        raise ValueError("n deve ser um inteiro positivo (n = 1, 2, 3, ...)")
    return np.sqrt(2.0 / L) * np.sin(n * np.pi * x / L)


def energy_n(n, L=L, hbar=HBAR, m=MASS):
    """Energia do n-ésimo autoestado."""
    if n < 1:
        raise ValueError("n deve ser um inteiro positivo")
    return (n ** 2 * np.pi ** 2 * hbar ** 2) / (2.0 * m * L ** 2)


def verificar_normalizacao_ortogonalidade(n_max=N_STATES, L=L, tol=1e-6):
    """
    Confere, por integração numérica, que:
        <psi_n | psi_n> = 1        (normalização)
        <psi_n | psi_m> = 0, n!=m  (ortogonalidade)
    Lança um erro se a tolerância não for satisfeita, garantindo que
    o restante do script trabalha com autoestados fisicamente válidos.
    """
    for n in range(1, n_max + 1):
        norm, _ = integrate.quad(lambda x: psi_n(n, x) ** 2, 0, L)
        assert abs(norm - 1.0) < tol, f"Falha de normalização em n={n}: {norm}"
    for n in range(1, n_max + 1):
        for m in range(n + 1, n_max + 1):
            overlap, _ = integrate.quad(lambda x: psi_n(n, x) * psi_n(m, x), 0, L)
            assert abs(overlap) < tol, f"Falha de ortogonalidade ({n},{m}): {overlap}"
    return True


def plotar(n_max=N_STATES, L=L, out_path="particula_na_caixa.png"):
    x = np.linspace(0, L, 800)
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), sharey=False)

    energias = [energy_n(n) for n in range(1, n_max + 1)]
    escala = 0.6 * max(energias[1] - energias[0], 1e-9)  

    
    ax = axes[0]
    for n in range(1, n_max + 1):
        E = energy_n(n)
        ax.axhline(E, color="gray", lw=0.6, ls="--")
        ax.plot(x, E + escala * psi_n(n, x), label=f"n={n}")
        ax.text(L * 1.02, E, f"E_{n}={E:.2f}", va="center", fontsize=9)
    ax.set_xlim(0, L * 1.25)
    ax.set_xlabel("posição x / L")
    ax.set_ylabel("Energia (níveis) + ψₙ(x) deslocada")
    ax.set_title("Níveis de energia e autofunções ψₙ(x)")
    ax.legend(loc="upper left", fontsize=8)

    
    ax2 = axes[1]
    for n in range(1, n_max + 1):
        E = energy_n(n)
        ax2.axhline(E, color="gray", lw=0.6, ls="--")
        ax2.plot(x, E + escala * psi_n(n, x) ** 2, label=f"n={n}")
    ax2.set_xlim(0, L)
    ax2.set_xlabel("posição x / L")
    ax2.set_title("Densidades de probabilidade |ψₙ(x)|²")

    fig.suptitle("Partícula na Caixa (Poço de Potencial Infinito)", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ok = verificar_normalizacao_ortogonalidade()
    print(f"Verificação de normalização/ortogonalidade: {'OK' if ok else 'FALHOU'}")
    for n in range(1, N_STATES + 1):
        print(f"  n={n}: E_n = {energy_n(n):.4f} (unidades naturais)")
    caminho = plotar()
    print(f"Gráfico salvo em: {caminho}")