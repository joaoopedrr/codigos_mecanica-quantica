import numpy as np
import matplotlib.pyplot as plt
from scipy.special import eval_hermite, factorial
from scipy import integrate


HBAR = 1.0
MASS = 1.0
OMEGA = 1.0
N_STATES = 6


def psi_n(n, x, hbar=HBAR, m=MASS, omega=OMEGA):
    """Autofunção normalizada do oscilador harmônico quântico."""
    if n < 0 or not float(n).is_integer():
        raise ValueError("n deve ser um inteiro >= 0")
    alpha = m * omega / hbar
    norm = (alpha / np.pi) ** 0.25 / np.sqrt(2.0 ** n * factorial(n))
    xi = np.sqrt(alpha) * x
    return norm * eval_hermite(n, xi) * np.exp(-xi ** 2 / 2.0)


def energy_n(n, hbar=HBAR, omega=OMEGA):
    """Energia do n-ésimo autoestado (quantizada, com energia de ponto zero)."""
    if n < 0:
        raise ValueError("n deve ser um inteiro >= 0")
    return hbar * omega * (n + 0.5)


def potencial(x, m=MASS, omega=OMEGA):
    return 0.5 * m * omega ** 2 * x ** 2


def verificar_normalizacao(n_max=N_STATES, tol=1e-4):
    """Confirma <psi_n|psi_n> = 1 por integração numérica em [-inf, inf]."""
    for n in range(n_max + 1):
        norm, _ = integrate.quad(lambda x: psi_n(n, x) ** 2, -20, 20, limit=200)
        assert abs(norm - 1.0) < tol, f"Falha de normalização em n={n}: {norm}"
    return True


def plotar(n_max=N_STATES, out_path="oscilador_harmonico.png"):
    x = np.linspace(-6, 6, 1000)
    V = potencial(x)

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))

    escala = 0.8  

    ax = axes[0]
    ax.plot(x, V, color="black", lw=1.5, label="V(x) = ½mω²x²")
    for n in range(n_max + 1):
        E = energy_n(n)
        y = E + escala * psi_n(n, x)
        ax.plot(x, y, lw=1.2)
        ax.axhline(E, color="gray", lw=0.5, ls="--")
        ax.text(6.1, E, f"n={n}", va="center", fontsize=8)
    ax.set_ylim(0, energy_n(n_max) + 2)
    ax.set_xlim(-7, 7.5)
    ax.set_xlabel("posição x")
    ax.set_ylabel("Energia + ψₙ(x) deslocada")
    ax.set_title("Potencial V(x), níveis de energia e autofunções")
    ax.legend(loc="upper center", fontsize=8)

    ax2 = axes[1]
    ax2.plot(x, V, color="black", lw=1.5)
    for n in range(n_max + 1):
        E = energy_n(n)
        y = E + escala * psi_n(n, x) ** 2
        ax2.plot(x, y, lw=1.2)
        ax2.axhline(E, color="gray", lw=0.5, ls="--")
    ax2.set_ylim(0, energy_n(n_max) + 2)
    ax2.set_xlim(-7, 7)
    ax2.set_xlabel("posição x")
    ax2.set_title("Densidades de probabilidade |ψₙ(x)|²\n(note a penetração na região classicamente proibida)")

    fig.suptitle("Oscilador Harmônico Quântico", fontsize=14)
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    return out_path


if __name__ == "__main__":
    ok = verificar_normalizacao()
    print(f"Verificação de normalização: {'OK' if ok else 'FALHOU'}")
    for n in range(N_STATES + 1):
        print(f"  n={n}: E_n = {energy_n(n):.4f} (energia de ponto zero incluída)")
    caminho = plotar()
    print(f"Gráfico salvo em: {caminho}")