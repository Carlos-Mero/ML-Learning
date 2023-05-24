import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl


mpl.rc("font", family='Songti SC')


def show_corr(corr):
    corr.to_csv("correlation_matrix.csv", encoding="utf-8")
    print(corr)


def plot_corr(corr):
    fig, ax = plt.subplots(figsize=(8, 6))

    im = ax.imshow(corr, cmap="coolwarm")
    cbar = ax.figure.colorbar(im, ax=ax)

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))
    ax.set_xticklabels(corr.columns)
    ax.set_yticklabels(corr.columns)
    plt.setp(
        ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    ax.set_title("Correlation Matrix")

    plt.savefig('correlation_matrix.png', bbox_inches='tight', dpi=300)


if __name__ == "__main__":
    df = pd.read_csv("bili-refined.csv")
    corr = df.corr(numeric_only=True)
    show_corr(corr)
    plot_corr(corr)
