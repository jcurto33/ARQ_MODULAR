"""
scripts.iris_eda
─────────────────
Genera gráficas de análisis exploratorio (EDA) del dataset Iris
y las guarda como imágenes PNG en data/files/eda/iris/.

Uso:
    python scripts/iris_eda.py
"""

import sys
from pathlib import Path

_REPO_ROOT = str(Path(__file__).resolve().parent.parent)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import matplotlib
matplotlib.use("Agg")  # Backend no interactivo

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import json

from shared.config_loader import load_config

# ── Configuración ────────────────────────────────────────
_config = load_config()
_RAW_DIR = Path(_config["paths"]["raw_dir"])
_EDA_DIR = Path(_config["paths"]["data_dir"]) / "files" / "eda" / "iris"
_METRICS_DIR = Path(_config["paths"]["models_dir"]) / "metrics" / "iris_classifier"

FEATURE_COLS = ["sepal_length", "sepal_width", "petal_length", "petal_width"]
FEATURE_LABELS = {
    "sepal_length": "Sepal Length (cm)",
    "sepal_width": "Sepal Width (cm)",
    "petal_length": "Petal Length (cm)",
    "petal_width": "Petal Width (cm)",
}
SPECIES_PALETTE = {"setosa": "#2ecc71", "versicolor": "#3498db", "virginica": "#e74c3c"}


def _load_dataset() -> pd.DataFrame:
    """Carga el CSV de Iris."""
    csv_path = _RAW_DIR / "iris.csv"
    if not csv_path.exists():
        raise FileNotFoundError(
            f"No se encontró {csv_path}. Ejecuta primero: python scripts/iris_ingest.py"
        )
    return pd.read_csv(csv_path)


def _apply_style():
    """Configura el estilo global de las gráficas."""
    sns.set_theme(style="whitegrid", font_scale=1.2)
    plt.rcParams.update({
        "figure.facecolor": "#0e1117",
        "axes.facecolor": "#1a1d23",
        "axes.edgecolor": "#444",
        "axes.labelcolor": "#fafafa",
        "text.color": "#fafafa",
        "xtick.color": "#ccc",
        "ytick.color": "#ccc",
        "grid.color": "#333",
        "legend.facecolor": "#1a1d23",
        "legend.edgecolor": "#444",
        "figure.dpi": 150,
    })


def generate_distributions(df: pd.DataFrame) -> None:
    """Histogramas + KDE de cada feature coloreados por especie."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Distribución de Features por Especie", fontsize=18, fontweight="bold", y=0.98)

    for ax, col in zip(axes.flat, FEATURE_COLS):
        for species, color in SPECIES_PALETTE.items():
            subset = df[df["species"] == species][col]
            ax.hist(subset, bins=15, alpha=0.45, label=species, color=color, edgecolor="none")
            subset.plot.kde(ax=ax, color=color, linewidth=2)
        ax.set_xlabel(FEATURE_LABELS[col], fontsize=13)
        ax.set_ylabel("Densidad", fontsize=12)
        ax.legend(fontsize=11)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_EDA_DIR / "distributions.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ distributions.png")


def generate_boxplots(df: pd.DataFrame) -> None:
    """Box plots de cada feature por especie."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Box Plots por Especie", fontsize=18, fontweight="bold", y=0.98)

    for ax, col in zip(axes.flat, FEATURE_COLS):
        sns.boxplot(
            data=df, x="species", y=col, ax=ax,
            palette=SPECIES_PALETTE, linewidth=1.5,
            flierprops={"marker": "o", "markerfacecolor": "#e74c3c", "markersize": 5},
        )
        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_LABELS[col], fontsize=13)
        ax.set_xticklabels(ax.get_xticklabels(), fontsize=12)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_EDA_DIR / "boxplots.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ boxplots.png")


def generate_violin_plots(df: pd.DataFrame) -> None:
    """Violin plots de cada feature por especie."""
    _apply_style()
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Violin Plots por Especie", fontsize=18, fontweight="bold", y=0.98)

    for ax, col in zip(axes.flat, FEATURE_COLS):
        sns.violinplot(
            data=df, x="species", y=col, ax=ax,
            palette=SPECIES_PALETTE, inner="quart", linewidth=1,
        )
        ax.set_xlabel("")
        ax.set_ylabel(FEATURE_LABELS[col], fontsize=13)

    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(_EDA_DIR / "violin_plots.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ violin_plots.png")


def generate_correlation_heatmap(df: pd.DataFrame) -> None:
    """Mapa de calor de correlaciones entre features."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(8, 7))
    corr = df[FEATURE_COLS].corr()
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(
        corr, mask=mask, annot=True, fmt=".2f", cmap="coolwarm",
        ax=ax, vmin=-1, vmax=1, linewidths=0.5,
        annot_kws={"fontsize": 14, "fontweight": "bold"},
        cbar_kws={"shrink": 0.8},
    )
    ax.set_title("Correlación entre Features", fontsize=16, fontweight="bold", pad=15)
    fig.tight_layout()
    fig.savefig(_EDA_DIR / "correlation_heatmap.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ correlation_heatmap.png")


def generate_pairplot(df: pd.DataFrame) -> None:
    """Scatter matrix (pairplot) coloreado por especie."""
    _apply_style()
    g = sns.pairplot(
        df, vars=FEATURE_COLS, hue="species",
        palette=SPECIES_PALETTE, diag_kind="kde",
        plot_kws={"alpha": 0.6, "s": 40, "edgecolor": "none"},
        diag_kws={"linewidth": 2},
        height=2.5,
    )
    g.figure.suptitle("Scatter Matrix por Especie", fontsize=18, fontweight="bold", y=1.02)
    g.savefig(_EDA_DIR / "pairplot.png", bbox_inches="tight")
    plt.close(g.figure)
    print("  ✓ pairplot.png")


def generate_class_balance(df: pd.DataFrame) -> None:
    """Gráfica de barras con el balance de clases."""
    _apply_style()
    fig, ax = plt.subplots(figsize=(7, 5))
    counts = df["species"].value_counts()
    bars = ax.bar(counts.index, counts.values, color=list(SPECIES_PALETTE.values()),
                  edgecolor="none", width=0.6)
    for bar, val in zip(bars, counts.values):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 1,
                str(val), ha="center", va="bottom", fontsize=14, fontweight="bold", color="#fafafa")
    ax.set_ylabel("Número de muestras", fontsize=13)
    ax.set_title("Balance de Clases", fontsize=16, fontweight="bold")
    ax.set_ylim(0, counts.max() * 1.15)
    fig.tight_layout()
    fig.savefig(_EDA_DIR / "class_balance.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ class_balance.png")


def generate_feature_ranges(df: pd.DataFrame) -> None:
    """Tabla resumen de rangos min/max por feature y especie, guardada como imagen."""
    _apply_style()

    species_list = list(SPECIES_PALETTE.keys())
    summary_data = []
    for species in species_list:
        subset = df[df["species"] == species]
        for col in FEATURE_COLS:
            summary_data.append({
                "Especie": species.capitalize(),
                "Feature": FEATURE_LABELS[col],
                "Mín": f"{subset[col].min():.1f}",
                "Máx": f"{subset[col].max():.1f}",
                "Media": f"{subset[col].mean():.2f}",
                "Std": f"{subset[col].std():.2f}",
            })

    summary_df = pd.DataFrame(summary_data)

    # Guardar también como CSV para referencia
    summary_df.to_csv(_EDA_DIR / "feature_summary.csv", index=False)

    # Crear imagen tipo tabla
    fig, ax = plt.subplots(figsize=(12, 8))
    ax.axis("off")
    ax.set_title("Resumen Estadístico por Especie", fontsize=16, fontweight="bold", pad=20)

    table = ax.table(
        cellText=summary_df.values,
        colLabels=summary_df.columns,
        cellLoc="center",
        loc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(11)
    table.scale(1.0, 1.8)

    # Estilo de la tabla
    for key, cell in table.get_celld().items():
        cell.set_edgecolor("#555")
        if key[0] == 0:  # Header
            cell.set_facecolor("#3498db")
            cell.set_text_props(color="white", fontweight="bold")
        else:
            row_species = summary_df.iloc[key[0] - 1]["Especie"].lower()
            alpha = 0.15
            color = SPECIES_PALETTE.get(row_species, "#333")
            # Convert hex to RGBA with alpha
            r, g, b = int(color[1:3], 16)/255, int(color[3:5], 16)/255, int(color[5:7], 16)/255
            cell.set_facecolor((r, g, b, alpha))
            cell.set_text_props(color="#fafafa")

    fig.tight_layout()
    fig.savefig(_EDA_DIR / "feature_summary.png", bbox_inches="tight")
    plt.close(fig)
    print("  ✓ feature_summary.png / feature_summary.csv")


def main():
    """Genera todas las gráficas EDA."""
    print("═" * 50)
    print("  Generando gráficas EDA del dataset Iris")
    print("═" * 50)

    _EDA_DIR.mkdir(parents=True, exist_ok=True)

    df = _load_dataset()
    print(f"\n  Dataset cargado: {len(df)} muestras, {len(FEATURE_COLS)} features\n")

    generate_distributions(df)
    generate_boxplots(df)
    generate_violin_plots(df)
    generate_correlation_heatmap(df)
    generate_pairplot(df)
    generate_class_balance(df)
    generate_feature_ranges(df)

    print(f"\n  ✅ Todas las gráficas guardadas en: {_EDA_DIR.resolve()}")
    print("═" * 50)


if __name__ == "__main__":
    main()
