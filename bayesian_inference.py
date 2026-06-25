import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import numpy as np
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator
    import matplotlib.colors as mcolors
    import pandas as pd
    from mpltern.datasets import get_triangular_grid
    import seaborn as sns
    import statsmodels.api as sm
    import matplotlib as mpl
    from dataclasses import dataclass
    import pocomc as pc
    import scipy
    import mpltern
    from aquarel import load_theme
    from dataclasses import dataclass

    theme = load_theme("umbra_light")
    theme.apply()
    plt.rc('font', size=16)
    return MultipleLocator, mo, np, pc, pd, plt, scipy, sns


@app.cell
def _(plt, sns):
    # colors = sns.color_palette("Set2")
    colors = sns.color_palette(
        plt.rcParams['axes.prop_cycle'].by_key()['color'] + [
            sns.color_palette("Set2")[-1]
        ]
    )
    colors
    return (colors,)


@app.cell
def _(colors):
    wine_to_color = {
        # pure variety
        "Cab. Sauv.": colors[0],
        "Merlot": colors[1],
        "Cab. Franc": colors[2],

        # blends
        0: colors[3],
        1: colors[4],
        2: colors[5],
        3: colors[6]
    }
    return (wine_to_color,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # 🍷 reverse-engineering a wine blend

    ## data

    ### chemical fingerprints of the pure-variety wines
    """)
    return


@app.cell
def _(np, pd):
    _wines = ["Cab. Sauv.", "Merlot", "Cab. Franc"]

    _pv_fprints_1 = pd.DataFrame(
        {
            "alcohol [vol. %]": [12.96, 13.20, 14.35],
            "sugar [g/L]": [4.51,  3.12,  2.44],
            "acid [g/L]": [5.6,  6.08,  5.41]
        },
        index=_wines
    ) # day 1

    _pv_fprints_2 = pd.DataFrame(
        {
            "alcohol [vol. %]": [12.89, 13.125, 14.225],
            "sugar [g/L]": [4.58,  3.11,  2.47],
            "acid [g/L]": [5.71,  5.87,  5.33]
        },
        index=_wines
    ) # day 2

    pv_fprints = (_pv_fprints_1 + _pv_fprints_2) / 2

    # calculate std over the two days for the sigmas
    sigmas = {}
    trust_factor = 1.0 # inflate to account for bias
    for _feature in pv_fprints.columns:
        sigmas[_feature] = np.zeros(3)
        for (_w, _wine) in enumerate(_wines):
            sigmas[_feature][_w] = np.std(
                [
                    _pv_fprints_1.loc[_wine, _feature], 
                    _pv_fprints_2.loc[_wine, _feature]
                ],
                ddof=1
            ) * trust_factor

        sigmas[_feature] = np.mean(sigmas[_feature])

    # mean std
    sigma_bar = np.mean(
        [
            sigmas[constituent] 
            for constituent in pv_fprints.columns
        ]
    )
    return pv_fprints, sigmas


@app.cell
def _(pv_fprints):
    pv_fprints
    return


@app.cell
def _(sigmas):
    sigmas
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### chemical fprints of the blends
    """)
    return


@app.cell
def _(pd):
    blend_fprints = pd.DataFrame(
        {
            "alcohol [vol. %]": [13.37, 13.19, 13.18, 12.975],
            "sugar [g/L]": [3.1, 3.97, 3.4, 4.0],
            "acid [g/L]": [5.73, 5.56, 5.69, 5.63]
        }
    )
    blend_fprints
    return (blend_fprints,)


@app.cell
def _(blend_fprints, pv_fprints):
    # blend features should be within range of parent pure variety wines
    for _constituent in blend_fprints.columns:
        max_conc = pv_fprints[_constituent].max()
        min_conc = pv_fprints[_constituent].min()

        for _b in range(blend_fprints.shape[0]):
            conc_blend = blend_fprints.loc[_b, _constituent]
            assert conc_blend <= max_conc
            assert conc_blend >= min_conc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ::lucide:eye:: visualize the fprints
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    incl_blends_checkbox = mo.ui.checkbox(label="include blends?", value=True)
    incl_blends_checkbox
    return (incl_blends_checkbox,)


@app.cell
def _(blend_fprints, incl_blends_checkbox, plt, pv_fprints, wine_to_color):
    def _draw_lollipop(ax, x, y, color):
        ax.plot([x, x], [0, y], color=color, zorder=100)
        ax.scatter([x], [y], color=color, zorder=100, s=8**2)

    constituent_to_ymax = { # use later
        'alcohol [vol. %]': 15.0,
        'sugar [g/L]': 5.05, 
        'acid [g/L]': 6.1
    }

    def viz_features(pv_fprints, blend_fprints):
        wines = pv_fprints.index
        n_wines = len(wines)
        constituents = pv_fprints.columns
        if incl_blends_checkbox.value:
            n_blends = blend_fprints.shape[0]
        else:
            n_blends = 0

        figsize = [6.4, 4.8]
        if incl_blends_checkbox.value:
            figsize[0] *= 1.1

        fig, axs = plt.subplots(1, 3, figsize=figsize)

        for c, constituent in enumerate(constituents):
            axs[c].minorticks_off()
            axs[c].xaxis.grid(False)
            for w, wine in enumerate(wines):
                conc = pv_fprints.loc[wine, constituent]
                _draw_lollipop(axs[c], w, conc, wine_to_color[wine])

            xticks = list(range(n_wines))
            xticklabels = list(wines)
            xmax = n_wines - 0.5

            if incl_blends_checkbox.value:
                d = 1 # spacing
                for blend_id in range(n_blends):
                    conc = blend_fprints.loc[blend_id, constituent]
                    _draw_lollipop(
                        axs[c], 
                        n_wines + blend_id + d, conc, 
                        wine_to_color[blend_id]
                    )

                xticks = xticks + list(range(n_wines+d, n_wines+n_blends+d))
                xticklabels = xticklabels + [
                    f"blend {b+1}" for b in range(n_blends)
                ]

                # highlight blends to distinguish them
                axs[c].axvspan(
                    n_wines+d - 0.5, 
                    n_wines+n_blends+d, 
                    color='gray', alpha=0.1, lw=0
                )
                xmax += n_blends + d

            axs[c].set_xticks(xticks, xticklabels, rotation=90)
            axs[c].set_xlim(-0.5, xmax)
            axs[c].set_ylabel(constituent)
            axs[c].set_ylim(0.0, constituent_to_ymax[constituent])
        plt.tight_layout()

        _savename = "features"
        if incl_blends_checkbox.value:
            _savename += "_w_blends"
        else:
            _savename += "_wo_blends"

        plt.savefig(_savename + ".pdf", format="pdf")
        return fig

    viz_features(pv_fprints, blend_fprints)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ground-truth blend make-up
    """)
    return


@app.cell
def _(pd):
    blend_makeup = pd.DataFrame(
        {
            "Cab. Sauv.": [0.1, 0.7, 0.3, 0.6],
            "Merlot": [0.6, 0.1, 0.5, 0.4],
            "Cab. Franc": [0.3, 0.2, 0.2, 0.0]
        }
    )
    blend_makeup
    return


@app.cell
def _(mo):
    mo.md(r"""
    # prior
    """)
    return


@app.cell
def _(np, pc, scipy):
    alpha = np.array([1.0, 1.0, 1.0])
    dirichlet_distn = scipy.stats.dirichlet(alpha)
    prior = pc.Prior(dirichlet_distn)

    xs_prior = dirichlet_distn.rvs(250)

    xs_prior[0][0]
    return (xs_prior,)


@app.cell
def _(np):
    def softmax(z):
        z_full = np.concatenate([z, [0.0]])
        e = np.exp(z_full - np.max(z_full))
        return e / e.sum()

    softmax([-1, 3.0, 0.1])
    return


@app.cell
def _(np):
    def log_likelihood(x):
        return -np.sum(10.0 * (x[:, ::2] ** 2.0 - x[:, 1::2]) ** 2.0 + (x[:, ::2] - 1.0) ** 2.0, axis=1)
    return


@app.cell
def _(MultipleLocator, blend_fprints, plt, pv_fprints, sigmas, xs_prior):
    def draw_simplex(
        xs_prior, pv_fprints, blend_fprints, sigmas
    ):
        wines = pv_fprints.index
        n_blends = blend_fprints.shape[0]

        all_errors = [] # for error analysis

        ax = plt.subplot(projection="ternary")

        ax.set_tlabel(wines[0] + " [volume fraction]")
        ax.set_llabel(wines[1] + " [volume fraction]")
        ax.set_rlabel(wines[2] + " [volume fraction]")

        for _ax in [ax.taxis, ax.laxis, ax.raxis]:
            _ax.set_label_position("tick1")
            _ax.set_major_locator(MultipleLocator(0.1))

        # ax.grid(which='both')
        ax.grid(which='major', linestyle='-', alpha=0.3)
        ax.grid(which='minor', linestyle=':', alpha=0.3)

        for x in xs_prior:
            ax.scatter(
                [x[0]], [x[1]], [x[2]],
                marker="o", #color=wine_to_color[blend_id], alpha=0.1,
                s=25
            )

        ###
        #   true blend composition
        ###
        # for blend_id in ble
        # ax.scatter(
        #     [blend_makeup.loc[blend_id, wines[0]]],
        #     [blend_makeup.loc[blend_id, wines[1]]],
        #     [blend_makeup.loc[blend_id, wines[2]]],
        #     label="truth", marker="^", facecolor=wine_to_color[blend_id],
        #     s=markersize ** 2, color="black", clip_on=False, zorder=1000
        # )
        plt.show()

    draw_simplex(xs_prior, pv_fprints, blend_fprints, sigmas)
    return


if __name__ == "__main__":
    app.run()
