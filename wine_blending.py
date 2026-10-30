import marimo

__generated_with = "0.17.6"
app = marimo.App()


@app.cell
def _():
    import marimo as mo
    import matplotlib.pyplot as plt
    import copy
    from matplotlib.lines import Line2D
    import numpy as np
    from matplotlib.ticker import MultipleLocator, AutoMinorLocator, PercentFormatter, MaxNLocator
    import matplotlib.colors as mcolors
    from scipy.optimize import nnls
    import pandas as pd
    from itertools import combinations
    from scipy.stats import gaussian_kde
    from mpltern.datasets import get_triangular_grid
    import seaborn as sns
    import statsmodels.api as sm
    import matplotlib as mpl
    import scipy
    import mpltern
    from dataclasses import dataclass

    import scienceplots
    plt.style.use(["science", "notebook"])
    fs = 12
    plt.rcParams.update({
        'font.size': fs,
        'axes.labelsize': fs,
        'axes.titlesize': fs,
        'xtick.labelsize': fs,
        'ytick.labelsize': fs,
        'legend.fontsize': fs,
        'legend.title_fontsize': fs,
        'figure.titlesize': fs,
        'axes.spines.top': False,
        'axes.spines.right': False,
        'xtick.top': False,
        'ytick.right': False,
    })
    return (
        Line2D,
        MaxNLocator,
        MultipleLocator,
        PercentFormatter,
        combinations,
        copy,
        dataclass,
        get_triangular_grid,
        mo,
        mpl,
        nnls,
        np,
        pd,
        plt,
        scipy,
        sm,
        sns,
    )


@app.cell
def _(sns):
    colors = sns.color_palette("Set3")
    colors
    return (colors,)


@app.cell
def _(colors):
    list(colors[9])
    return


@app.cell
def _(colors):
    wine_to_color = {
        # parent (from KeyNote)
        "Cab. Sauv.": [134/255, 40/255, 35/255],# colors[9],
        "Merlot": [153/255, 45/255, 61/255],#colors[9],
        "Cab. Franc": [175/255, 75/255, 92/255],#colors[9],

        "Sauv. Blanc": [232/255, 220/255, 136/255],# "gold",
        "Bordeaux Blanc": [200/255, 175/255, 73/255],#"gold",

        # red blends
        0: colors[0],
        1: colors[2],
        2: colors[3],
        3: colors[4],

        # white blends
        4: colors[5],
        5: colors[6],
        6: colors[7],
        7: colors[8]
    }
    return (wine_to_color,)


@app.cell
def _(sns):
    wine_to_pie_color = {
        # reds: three shades of red/purple
        **dict(zip(["Cab. Sauv.", "Merlot", "Cab. Franc"],
                   sns.color_palette("Reds_r", 4)[:3])),
        # whites: two shades of yellow
        **dict(zip(["Sauv. Blanc", "Bordeaux Blanc"],
                   sns.color_palette("YlOrBr", 4)[1:3])),
    }
    return (wine_to_pie_color,)


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
    def build_parent_fprint_and_sigmas():
        wines = ["Cab. Sauv.", "Merlot", "Cab. Franc"]

        ###
        #  parent fingerprints
        ###
        parent_fprints_1 = pd.DataFrame(
            {
                "alcohol [vol. %]": [12.96, 13.20, 14.35],
                "sugar [g/L]": [4.51,  3.12,  2.44],
                "acid [g/L]": [5.6,  6.08,  5.41]
            },
            index=wines
        ) # day 1

        parent_fprints_2 = pd.DataFrame(
            {
                "alcohol [vol. %]": [12.89, 13.125, 14.225],
                "sugar [g/L]": [4.58,  3.11,  2.47],
                "acid [g/L]": [5.71,  5.87,  5.33]
            },
            index=wines
        ) # day 2

        parent_fprints = (parent_fprints_1 + parent_fprints_2) / 2
        constituents = parent_fprints.columns

        ###
        #  sigmas
        ###
        sigmas = {}
        trust_factor = 1.0 # inflate to account for bias
        print("trust factor: ", trust_factor)
        for constituent in constituents:
            sigmas[constituent] = np.zeros(3)
            for (w, wine) in enumerate(wines):
                sigmas[constituent][w] = np.std(
                    [
                        parent_fprints_1.loc[wine, constituent], 
                        parent_fprints_2.loc[wine, constituent]
                    ],
                    ddof=1
                ) * trust_factor

            sigmas[constituent] = np.mean(sigmas[constituent])

        return parent_fprints, list(constituents), sigmas

    parent_fprints, constituents, sigmas = build_parent_fprint_and_sigmas()
    return constituents, parent_fprints, sigmas


@app.cell
def _(constituents):
    constituents
    return


@app.cell
def _(parent_fprints):
    parent_fprints
    return


@app.cell
def _(sigmas):
    sigmas
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    #### 🤔 direction of chemical fingerprints

    this could explain anisotropy in UQ.
    """)
    return


@app.cell
def _(np):
    def fingerprint_angles_and_distances(
        parent_fprints, wine_a, wine_b, sigmas
    ):
        constituents = parent_fprints.columns

        x_a = np.array(
            [
                parent_fprints.loc[wine_a, constituent] / sigmas[constituent]
                for constituent in constituents
            ]
        )
        x_b = np.array(
            [
                parent_fprints.loc[wine_b, constituent] / sigmas[constituent]
                for constituent in constituents
            ]
        )
        cos_theta = np.dot(x_a, x_b) / (
            np.linalg.norm(x_a) * np.linalg.norm(x_b)
        )
        angle = np.acos(np.clip(cos_theta, -1.0, 1.0)) * 180 / np.pi
        distance = np.linalg.norm(x_a - x_b)  # in units of measurement std devs
        return angle, distance
    return (fingerprint_angles_and_distances,)


@app.cell
def _(
    combinations,
    fingerprint_angles_and_distances,
    parent_fprints,
    pd,
    sigmas,
):
    _pairs = list(combinations(parent_fprints.index, 2))
    pd.DataFrame(
        [
            fingerprint_angles_and_distances(parent_fprints, wine_a, wine_b, sigmas)
            for wine_a, wine_b in _pairs
        ],
        index=pd.MultiIndex.from_tuples(_pairs, names=["wine a", "wine b"]),
        columns=["angle [°]", "distance [σ]"],
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### chemical fprints of the blends
    """)
    return


@app.cell
def _(parent_fprints, pd):
    blend_data = pd.DataFrame(
        {
            # features
            "alcohol [vol. %]": [13.37, 13.19, 13.18, 12.975],
            "sugar [g/L]": [3.1, 3.97, 3.4, 4.0],
            "acid [g/L]": [5.73, 5.56, 5.69, 5.63],

            # true make-up
            "Cab. Sauv.": [0.1, 0.7, 0.3, 0.6],
            "Merlot": [0.6, 0.1, 0.5, 0.4],
            "Cab. Franc": [0.3, 0.2, 0.2, 0.0]
        }
    )

    def _abbrev(s):
        return "".join(word[0].upper() for word in s.split())

    def _makeup_string(row):
        parts = []
        for wine in parent_fprints.index:
            pct = round(row[wine] * 100)
            if pct > 0:
                parts.append(f"{pct}%{_abbrev(wine)}")
        return "  ".join(parts)

    blend_data["name"] = blend_data.apply(_makeup_string, axis=1)

    # re-order
    blend_data = blend_data.iloc[
        [1, 3, 2, 0], :].reset_index(drop=True)

    blend_data
    return (blend_data,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ✅ check that blend features are within range of parent pure variety wines
    """)
    return


@app.cell
def _(blend_data, constituents, parent_fprints):
    for _constituent in constituents:
        max_conc = parent_fprints[_constituent].max()
        min_conc = parent_fprints[_constituent].min()

        for _b in range(blend_data.shape[0]):
            conc_blend = blend_data.loc[_b, _constituent]
            assert conc_blend <= max_conc
            assert conc_blend >= min_conc
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ::lucide:eye:: visualize the fprints
    """)
    return


@app.cell
def _():
    constituent_to_xmax = {  # use later
        'alcohol [vol. %]': 15.0,
        'sugar [g/L]': 5.05,
        'acid [g/L]': 6.1
    }
    return (constituent_to_xmax,)


@app.cell
def _(
    blend_data,
    constituent_to_xmax,
    constituents,
    parent_fprints,
    plt,
    sigmas,
    sns,
):
    def viz_cute_fingerprint(
        fprints, wine, sigmas, constituents,
        constituent_to_xmax=constituent_to_xmax, 
        savename=None, incl_title=False
    ):
        fp = fprints.loc[wine, constituents]
        # colors = dict(zip(constituents, sns.color_palette("Pastel1")))
        color = sns.color_palette("Pastel1")[2]

        if wine in ["Cab. Sauv.", "Merlot", "Cab. Franc"] or "red" in savename:
            w = 0.55
        else:
            w = 0.65
        fig, axs = plt.subplots(
            1, len(fp), figsize=(w * len(fp), 2), layout="constrained"
        )
        for ax, constituent in zip(axs, constituents):
            ax.bar(
                0, fp[constituent], width=0.6, color=color,#colors[constituent], 
                  yerr=sigmas[constituent]
            )
            ax.set_ylim(0, constituent_to_xmax[constituent])
            ax.set_xticks([])
            clabel = constituent.replace(" ", "\n", 1)
            ax.set_title(clabel, fontsize=10)
            ax.spines[["top", "right"]].set_visible(False)
            ax.tick_params(labelsize=7)
        if incl_title:
            fig.suptitle(wine, fontsize=10)

        if savename is not None:
            fig.savefig(
                savename + "_fingerprint.pdf", format="pdf", bbox_inches="tight"
            )
        return fig

    # for schematic
    for _wine in parent_fprints.index:
        viz_cute_fingerprint(
            parent_fprints, _wine, sigmas, constituents, savename=_wine
        )

    viz_cute_fingerprint(
        blend_data, 0, sigmas, constituents, savename="red_blend_0"
    )
    return (viz_cute_fingerprint,)


@app.cell
def _(
    MaxNLocator,
    blend_data,
    constituent_to_xmax,
    constituents,
    parent_fprints,
    plt,
    sigmas,
    wine_to_color,
):
    def draw_lollipop(ax, pos, val, color, sigma=None):
        ax.plot([0, val], [pos, pos], color=color, zorder=100)
        ax.scatter([val], [pos], color=color, zorder=100, s=8**2, clip_on=False)
        if sigma is not None:
            ax.errorbar(
                val, pos, xerr=sigma, fmt="none", ecolor="black",
                elinewidth=3, capsize=3, zorder=102, clip_on=False,
            )

    def draw_bar(ax, pos, val, color, sigma=None):
        ax.barh(
            pos, val, height=0.6, color=color, zorder=100,
            xerr=sigma,
            error_kw={"ecolor": "black", "elinewidth": 1, "capsize": 0, "zorder": 102},
        )

    def viz_fingerprints(
        parent_fprints, blend_data, constituents, sigmas,
        constituent_to_xmax=constituent_to_xmax, savename=None
    ):
        wines = parent_fprints.index
        n_wines = len(wines)
        n_blends = blend_data.shape[0]

        panel_w = 1.4        # inches per constituent panel
        label_w = 1.8        # extra inches for the y tick labels + group labels

        panel_h = 2 + 0.3 * n_wines

        figsize = [1.1 * (label_w + panel_w * len(constituents)), panel_h]

        fig, axs = plt.subplots(
            1, len(constituents), sharey=True, figsize=figsize
        )

        for c, constituent in enumerate(constituents):
            axs[c].minorticks_off()
            axs[c].yaxis.grid(False)
            axs[c].xaxis.set_major_locator(MaxNLocator(nbins=4))
            for w, wine in enumerate(wines):
                conc = parent_fprints.loc[wine, constituent]
                draw_bar(
                    axs[c], w, conc, 
                    wine_to_color[wine], sigma=sigmas[constituent]
                )

            yticks = list(range(n_wines))
            yticklabels = list(wines)
            ymax = n_wines - 0.5

            d = 1  # spacing
            for blend_id in range(n_blends):
                if n_wines == 2:
                    color = wine_to_color[blend_id+4] # white
                else:
                    color = wine_to_color[blend_id]
                conc = blend_data.loc[blend_id, constituent]
                draw_bar(
                    axs[c],
                    n_wines + blend_id + d, conc,
                    color, sigma=sigmas[constituent]
                )

            yticks = yticks + list(range(n_wines + d, n_wines + n_blends + d))
            yticklabels = yticklabels + [
                blend_data.loc[b, "name"] for b in range(n_blends)
            ]

            ymax += n_blends + d

            axs[c].set_yticks(yticks, yticklabels)
            axs[c].set_ylim(ymax, -0.5)  # inverted so first wine is at the top
            axs[c].set_xlabel(constituent)
            axs[c].set_xlim(0.0, constituent_to_xmax[constituent])

        # label parent and blend
        # measure how wide the tick labels are so the group labels clear them
        fig.canvas.draw()
        renderer = fig.canvas.get_renderer()
        max_w_px = max(
            t.get_window_extent(renderer).width 
            for t in axs[0].get_yticklabels()
        )
        pad_pts = 8
        offset_pts = -(max_w_px * 72 / fig.dpi + pad_pts)

        groups = [("parents", (n_wines - 1) / 2)]          # centered on the wines

        d = 1  # same spacing as above
        groups.append(("blends", n_wines + d + (n_blends - 1) / 2))

        for label, ypos in groups:
            axs[0].annotate(
                label,
                xy=(0, ypos),
                xycoords=axs[0].get_yaxis_transform(), # x axes coords, y data
                xytext=(offset_pts, 0),
                textcoords="offset points",
                rotation=90, ha="right", va="center",
                fontweight="bold",
                annotation_clip=False,
            )

        plt.tight_layout()

        if savename is not None:
            plt.savefig(savename + ".pdf", format="pdf")
        return fig

    viz_fingerprints(
        parent_fprints, blend_data, constituents, sigmas,
        constituent_to_xmax=constituent_to_xmax,
        savename="red_fingerprints"
    )
    return (viz_fingerprints,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## setting up an un-mixing problem
    """)
    return


@app.cell
def _(dataclass, pd):
    @dataclass
    class WineBlendingProb:
        # name
        name : str

        # wine & constituent order
        wines : list
        constituents : list

        # parent fingerprints
        parent_fprints : pd.DataFrame

        # blend fingerprint
        blend_fprint : pd.Series

        # measurement noise
        sigmas : dict

        # true makeup
        blend_makeup : dict
    return (WineBlendingProb,)


@app.cell(hide_code=True)
def _(mo):
    blend_selector = mo.ui.dropdown(
        options=[0, 1, 2, 3], value=1, label="blend ID"
    )
    blend_selector
    return (blend_selector,)


@app.cell
def _(
    WineBlendingProb,
    blend_data,
    blend_selector,
    constituents,
    parent_fprints,
    sigmas,
):
    blending_probs = [
        WineBlendingProb(
            blend_data.loc[blend_id, "name"],
            list(parent_fprints.index),
            constituents,
            parent_fprints,
            blend_data.loc[blend_id, constituents],
            sigmas,
            blend_data.loc[blend_id, parent_fprints.index].to_dict()
        )
        for blend_id in range(4)
    ]

    blending_prob = blending_probs[blend_selector.value]
    return blending_prob, blending_probs


@app.cell
def _(blending_prob, plt, wine_to_pie_color):
    def viz_makeup_pie(bp, savename=None):
        wines = [wine for wine in bp.wines if bp.blend_makeup[wine] > 0]

        fig, ax = plt.subplots()
        ax.pie(
            [bp.blend_makeup[wine] for wine in wines], labels=wines,
            colors=[wine_to_pie_color[wine] for wine in wines],
            wedgeprops={"edgecolor": "white", "linewidth": 1.5},
        )
        ax.set_title(bp.name)
        if savename:
            fig.savefig(savename, format="pdf", bbox_inches="tight")
        return fig

    viz_makeup_pie(blending_prob)
    return


@app.cell
def _(np):
    def build_linear_system(bp):
        M = len(bp.constituents)
        N = len(bp.wines)

        A = np.transpose(bp.parent_fprints.loc[bp.wines, bp.constituents].values)
        b = bp.blend_fprint[bp.constituents].to_numpy(dtype=float)
        b = b.reshape((M, 1))

        sigma_bar = np.mean(
            [bp.sigmas[constituent] for constituent in bp.constituents]
        )
        W = np.diag(
            [
                sigma_bar / bp.sigmas[constituent] 
                for constituent in bp.constituents
            ]
        )

        assert A.shape == (M, N)
        assert W.shape == (M, M)
        assert b.shape == (M, 1)

        return W, A, b
    return (build_linear_system,)


@app.cell
def _(blending_prob, build_linear_system):
    W, A, b = build_linear_system(blending_prob)
    W, A, b
    return (b,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## constrained total least squares solution
    """)
    return


@app.cell
def _(np):
    def assert_feasible_vol_fracs(x):
        assert np.all(x <= 1.0)
        assert np.all(x >= 0.0)
        assert np.isclose(np.sum(x), 1.0)
    return (assert_feasible_vol_fracs,)


@app.cell
def _(blending_probs, np, ols_baselines):
    def print_stuff(blending_probs, err_type="MAE"):
        for b, blending_prob in enumerate(blending_probs):
            print("BLEND", blending_prob.name)
            res = ols_baselines(blending_prob, verbose=False)
            for method in ("SCTLS", "SCOLS", "COLS", "SOLS", "OLS"):
                err = np.round(res[f"{method} {err_type}"]*100, decimals=1)
                if method == "SCTLS":
                    # list prediction
                    x = np.round(100*res[f"x_SCTLS"], decimals=1)
                    print(f"\t\tpred:", x, " vol %")

                print(f"\t{method} MAE: {err} vol %")

        ress = [
            ols_baselines(blending_prob, verbose=False)
            for blending_prob in blending_probs
        ]
        errs = [res[f"SCTLS {err_type}"] for res in ress]
        err_vecs = [np.abs(res[f"SCTLS error vector"]) for res in ress]

        print(
            f"SCTLS avg {err_type} over blends: ", 
            np.round(np.mean(errs) * 100, decimals=1)
        )
        print(
            f"...by parent: ", 
            np.round(np.mean(err_vecs, axis=0) * 100, decimals=1)
        )
    print_stuff(blending_probs)
    return (print_stuff,)


@app.cell
def _(
    assert_feasible_vol_fracs,
    blending_prob,
    build_linear_system,
    constrained_tls_soln,
    nnls,
    np,
):
    def ols_soln(
        blending_prob, sum_to_one=True, scale=True, w=1e3,
        verbose=True
    ):
        if verbose:
            print("sum-to-one constraint? ", sum_to_one)
            print("scale? ", scale)
        W, A, b = build_linear_system(blending_prob)
        n = len(b)

        if scale:
            A = W @ A
            b = W @ b

        if sum_to_one:
            A = np.vstack([A, w * np.ones(A.shape[1])])
            b = np.append(b, w)
        else:
            b = b.reshape((n, ))

        if verbose:
            print("shape A: ", np.shape(A))
            print("shape b: ", np.shape(b))

        x, _ = nnls(A, b)
        if verbose:
            print("sum b4 normalize: ", np.sum(x))
            print("\n")
        x = x / np.sum(x)
        assert_feasible_vol_fracs(x)

        # residual (in scaled units if scale=True)
        r = b[:n] - A[:n] @ x

        return x, r

    def ols_baselines(bp, verbose=True):
        res = {}
        res["x_SCOLS"] = ols_soln(bp, sum_to_one=True, scale=True, verbose=verbose)[0]
        res["x_COLS"]  = ols_soln(bp, sum_to_one=True, scale=False, verbose=verbose)[0]
        res["x_SOLS"]  = ols_soln(bp, sum_to_one=False, scale=True, verbose=verbose)[0]
        res["x_OLS"]   = ols_soln(bp, sum_to_one=False, scale=False, verbose=verbose)[0]

        res["x_SCTLS"] = constrained_tls_soln(bp)[0]

        for method in ["SCOLS", "OLS", "COLS", "SOLS", "SCTLS"]:
            x = [bp.blend_makeup[wine] for wine in bp.wines]
            res[method + " error vector"] = x - res["x_" + method]
            res[method + " RMSE"] = np.sqrt(np.mean((x - res["x_" + method]) ** 2))
            res[method + " MAE"] = np.mean(np.abs(x - res["x_" + method]))
        return res

    ols_baselines(blending_prob)
    return ols_baselines, ols_soln


@app.cell
def _(
    Line2D,
    MultipleLocator,
    PercentFormatter,
    blending_probs,
    mpl,
    np,
    ols_baselines,
    plt,
    wine_to_color,
):
    def compare_with_ols_simplex(
        blending_probs,
        methods=("SCTLS", "SCOLS", "COLS", "SOLS", "OLS"),
        savename=None,
    ):
        wines = blending_probs[0].wines
        markers = dict(zip(methods, ["*", "o", "s", "D", "v"]))

        fig = plt.figure(figsize=(7, 6))
        ax = fig.add_subplot(projection="ternary")
        ax.set_tlabel(wines[0] + " [vol. %]")
        ax.set_llabel(wines[1] + " [vol. %]")
        ax.set_rlabel(wines[2] + " [vol. %]")
        for _ax in [ax.taxis, ax.laxis, ax.raxis]:
            _ax.set_label_position("tick1")
            _ax.set_major_locator(MultipleLocator(0.1))
            _ax.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.grid(which="major", linestyle="-", alpha=0.3)

        err_type = "MAE"
        errors = dict(zip(methods, np.zeros(len(methods))))
        error_labels = {}
        for blend_id, bp in enumerate(blending_probs):
            res = ols_baselines(bp, verbose=False)
            x_true = np.array([bp.blend_makeup[w] for w in wines])
            c = wine_to_color[blend_id]

            ax.scatter(
                *x_true, marker="^", facecolor=c, edgecolor="black",
                s=90, zorder=10, clip_on=False,
            ) # true
            for m in methods:
                x = res["x_" + m]
                errors[m] += res[m + " " + err_type]

                # line from truth to prediction = error
                ax.plot(
                    *np.column_stack([x_true, x]), 
                    color=c, lw=0.8, alpha=0.6, zorder=1
                )
                ax.scatter(
                    *x, marker=markers[m], facecolor="none", edgecolor=c,
                    linewidths=1.5, s=60, zorder=5, clip_on=False,
                )
        error_labels = {}
        for m in methods:
            errors[m] /= len(blending_probs)
            error_labels[m] = f"{m} ({err_type} {errors[m]:.1%})"

        blend_handles = [
            mpl.patches.Patch(color=wine_to_color[i], label=bp.name)
            for i, bp in enumerate(blending_probs)
        ]

        method_handles = [
            Line2D([0], [0], marker="^", ls="None", mfc="white", mec="black",
                   ms=9, label="truth")
            ] + [
            Line2D([0], [0], marker=markers[m], ls="None", 
                   mfc="white", mec="black",
                   ms=8, label=error_labels[m])
            for m in methods
        ]
        leg1 = ax.legend(handles=blend_handles, loc="upper left",
                         bbox_to_anchor=(0.9, 1.0), title="blend", fontsize=9)
        ax.legend(handles=method_handles, loc="upper left",
                  bbox_to_anchor=(1.02, 0.7), title="method", fontsize=9)
        ax.add_artist(leg1)

        if savename:
            plt.tight_layout()
            fig.savefig(savename, format="pdf", bbox_inches="tight")
        return fig

    compare_with_ols_simplex(blending_probs, savename="red_ols_comparison.pdf")
    return


@app.cell
def _(assert_feasible_vol_fracs, build_linear_system, copy, np, scipy):
    def constrained_tls_soln(
        blending_prob, 
        verbose=False, run_checks=False, handle_non_negativity=True
    ):
        W, A, b = build_linear_system(blending_prob)

        # build augmented matrix [A; b]
        Ab = np.hstack([A, b])

        # build scaled augmented matrix
        WAb = W @ Ab

        # case M+1 > N
        M = len(blending_prob.constituents)
        N = len(blending_prob.wines)

        if run_checks:
            # ☑️ [A b] full rank
            assert np.linalg.matrix_rank(WAb) == min(WAb.shape)

            # ☑️ C = [A b; 1] full rank
            C = np.vstack([WAb, np.ones(N+1)])
            assert C.shape == (M+1, N+1)
            assert np.linalg.matrix_rank(C) == min(C.shape)
            assert np.linalg.matrix_rank(C) == C.shape[1] # column rank

            # ☑️ C's null space is just the zero vector
            assert scipy.linalg.null_space(C).size == 0

            # ☑️ W [A; b]'s null space. not orthogonal to 1 like we want.
            WAb_null = scipy.linalg.null_space(WAb)
            if M == N:
                # 1D null space. 
                assert WAb_null.size == N+1
                # but not in direction we want
                assert not np.isclose(
                    np.dot(np.ravel(WAb_null), np.ones(N+1)), 0.0, atol=0.001
                )
                # null space here is just solution to Ax = b! so
                #  just saying, if we naively do A x = b, x is not gonna
                #  satisfy the sum to one constraint
                x_naive = np.linalg.solve(A, b)
                assert np.allclose(
                    x_naive, - WAb_null[:N] / WAb_null[N]
                )
            else:
                assert WAb_null.size == 0

        # enforce linear constraint 1^T x = 1
        O = np.ones(N+1).reshape((1, N+1)) # [1^T; 1]

        QR = np.linalg.qr(np.transpose(O), mode="complete")
        Q2 = QR.Q[:, 1:] # basis for null space of O
        if run_checks:
            # ☑️ size of Q, Q2, R
            assert QR.Q.shape == (N+1, N+1)
            assert Q2.shape == (N+1, N)
            assert QR.R.shape == (N+1, 1)

            # ☑️ first column of Q is just the one vector
            assert np.allclose(
                -1 * QR.Q[:, 0], # direction doesn't matter
                np.ones(N+1) / np.sqrt(N+1)
            )

            # ☑️ vectors in cols of Q2 = vectors in null space of O
            assert np.allclose(
                np.ravel(O @ Q2), np.zeros(N), atol=1e-5
            )

            # ☑️ orthogonal invariance of Frob norm
            assert np.isclose(
                np.linalg.norm(WAb, ord='fro'), 
                np.linalg.norm(WAb @ QR.Q, ord='fro')
            )
            assert np.isclose(
                np.linalg.norm(WAb, ord='fro') ** 2, 
                np.linalg.norm(
                    WAb @ QR.Q[:, 0] # cuz a vector
                ) ** 2 + np.linalg.norm(
                    WAb @ Q2, ord='fro'
                ) ** 2
            )

        # W [A b] Q2
        WAbQ2 = WAb @ Q2
        if run_checks:
            # ☑️ size of W [A b] Q2
            assert WAbQ2.shape == (M, N)

            # ☑️ W [A b] Q2 is full [column] rank
            assert np.linalg.matrix_rank(WAbQ2) == min(WAbQ2.shape)
            assert np.linalg.matrix_rank(WAbQ2) == N

            # ☑️ W [A b] Q2 does not admit a null space.
            assert scipy.linalg.null_space(WAbQ2).size == 0

        # SVD of W [A; b] Q2
        f = np.linalg.svd(WAbQ2, full_matrices=True)
        if run_checks:
            assert WAbQ2.shape == (M, N)
            assert f.Vh.shape == (N, N)
            assert f.U.shape == (M, M)
            assert f.S[-1] > 0.0

        # Nth singular vectors (N-1 in Python)
        sigma_n = f.S[N-1]
        u_n = f.U[:, N-1].reshape(M, 1)
        v_n = f.Vh[N-1, :].reshape(N, 1)

        # total least squares solution
        z = Q2 @ v_n
        x_tls = -z[:N] / z[-1]

        if run_checks:
            # ☑️ z size
            assert np.size(z) == N+1
            # ☑️ z orthogonal to 1
            assert np.isclose(np.dot(np.ravel(z), np.ones(N+1)), 0.0)
            # ☑️ vn in null space of perturbed matrix
            Delta = sigma_n * u_n @ np.transpose(v_n)
            assert np.allclose(
                (WAbQ2 - Delta) @ v_n,
                np.zeros(M)
            )

        # total perturbation
        delta_Ab = sigma_n * np.linalg.inv(W) @  u_n @ np.transpose(Q2 @ v_n)

        if run_checks:
            # ☑️ [dA db] perp to one i.e. [dA dB] Q1 = 0
            assert np.allclose(
                delta_Ab @ np.ones((N+1, 1)),
                np.zeros((M, 1))
            )

        if run_checks:
            # ☑️ sum to one constraint satisfied
            assert np.isclose(np.sum(x_tls), 1.0)

            # ☑️ does optimal total perturbation give null space that
            #    matches the TLS solution?
            z_alt = scipy.linalg.null_space(Ab - delta_Ab)
            z_alt = -z_alt / z_alt[-1] # normalize
            assert np.allclose(
                x_tls, z_alt[:N]
            )

            # ☑️ does optimal total perturbation give Ax = b
            #    a solution that satisfies constraint?
            if M == N:
                x_tls_w_perturb = np.linalg.solve(
                    A - delta_Ab[:, :N], b - delta_Ab[:, N].reshape(M, 1)
                )
                assert np.allclose(x_tls_w_perturb, x_tls, atol=0.000001)

        # handle non-negative volume fractions
        if handle_non_negativity and np.any(x_tls < 0.0):
            # eliminate the wine with the negative volume fraction
            id_wine_remove = np.argmin(x_tls)
            wine_to_remove = blending_prob.wines[id_wine_remove]
            if verbose:
                print(f"Eliminating wine: {wine_to_remove}")

            new_wines = [w for w in blending_prob.wines if w != wine_to_remove]

            # new blending problem
            new_blending_prob = copy.deepcopy(blending_prob)
            new_blending_prob.wines = new_wines

            # run on reduced TLS problem
            _x_tls, _delta_Ab = constrained_tls_soln(
                new_blending_prob, verbose=verbose, run_checks=run_checks
            )

            # handle re-sizing (we removed a wine)
            x_tls = np.zeros(N)
            delta_Ab = np.zeros((M, N+1))
            mask = np.array([w != wine_to_remove for w in blending_prob.wines])
            x_tls[mask] = _x_tls
            delta_Ab[:, np.append(mask, True)] = _delta_Ab

        if handle_non_negativity:
            assert_feasible_vol_fracs(x_tls)

        return np.ravel(x_tls), delta_Ab
    return (constrained_tls_soln,)


@app.cell
def _(blending_prob, constrained_tls_soln):
    x_tls, delta_Ab = constrained_tls_soln(
        blending_prob, verbose=True, run_checks=True
    )
    x_tls, delta_Ab
    return delta_Ab, x_tls


@app.cell
def _(delta_Ab, x_tls):
    # correction on col i of A = x_i * correction on b
    _j = 0 # parent mixture ID
    delta_Ab[:, _j] + delta_Ab[:, -1] * x_tls[_j]
    return


@app.cell
def _(delta_Ab):
    delta_Ab[:, -1]
    return


@app.cell
def _(delta_Ab, x_tls):
    delta_Ab[:, :3] + delta_Ab[:, -1].reshape((3, 1)) @ x_tls.reshape((1, 3))
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### test situation when $x_i<0$.
    """)
    return


@app.cell
def _(blending_prob, constrained_tls_soln, copy):
    bad_blending_prob = copy.deepcopy(blending_prob)
    bad_blending_prob.blend_fprint = bad_blending_prob.blend_fprint - 0.02

    constrained_tls_soln(
        bad_blending_prob, verbose=True, run_checks=True,
        handle_non_negativity=True
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### compare TLS with OLS approaches
    """)
    return


@app.cell
def _(blending_prob, ols_soln):
    ols_soln(blending_prob, sum_to_one=False, scale=False)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### check shapes with bogus problem
    (add another constituent "blah")
    """)
    return


@app.cell
def _(blending_prob, constrained_tls_soln, copy, np):
    bogus_blending_prob = copy.deepcopy(blending_prob)
    bogus_blending_prob.constituents = blending_prob.constituents + ["blah [g/L]"]
    bogus_blending_prob.parent_fprints["blah [g/L]"] = np.random.rand(3)
    bogus_blending_prob.blend_fprint["blah [g/L]"] = np.random.rand()
    bogus_blending_prob.sigmas["blah [g/L]"] = 1.2

    constrained_tls_soln(bogus_blending_prob, run_checks=True) # it runs checks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## uncertainty quantification
    """)
    return


@app.cell
def _(copy, np):
    def sample_perturbed_problem(bp):
        new_bp = copy.deepcopy(bp)

        for constituent in bp.constituents:
            sigma = bp.sigmas[constituent]

            # perturb wine fingerprint
            for wine in bp.wines:
                delta = sigma * np.random.normal()
                new_bp.parent_fprints.loc[wine, constituent] += delta

            # perturb blend fingerprint
            new_bp.blend_fprint.loc[constituent] += sigma * np.random.normal()

        return new_bp
    return (sample_perturbed_problem,)


@app.cell
def _(blending_prob, np, sample_perturbed_problem):
    perturbed_prob = sample_perturbed_problem(blending_prob)

    # check blend fingerprint changed
    delta_blend = perturbed_prob.blend_fprint - blending_prob.blend_fprint
    assert delta_blend.is_unique
    assert np.all(delta_blend.values != 0.0)

    # check pure variety fingerprints changed
    delta_pv = perturbed_prob.parent_fprints.values - blending_prob.parent_fprints.values
    assert np.unique(delta_pv).size == delta_pv.size
    assert np.all(delta_pv != 0.0)

    # see perturbations; run again to see they change.
    delta_pv, delta_blend
    return


@app.cell
def _(constrained_tls_soln, sample_perturbed_problem):
    def sample_perturbed_solutions(
        blending_prob,
        n_samples=100
    ):
        x_tlss = []
        for s in range(n_samples):
            perturbed_prob = sample_perturbed_problem(blending_prob)
            x_tls, _ = constrained_tls_soln(perturbed_prob)
            x_tlss.append(x_tls)

        return x_tlss
    return (sample_perturbed_solutions,)


@app.cell
def _(blending_prob, sample_perturbed_solutions):
    perturbed_x_tlss = sample_perturbed_solutions(
        blending_prob, n_samples=1000
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the mean over these peturbations should be the x_tls.
    """)
    return


@app.cell
def _(get_triangular_grid, np, sm):
    def do_trinary_kde(x_tlss, n=30):
        # create grid in ternary space
        tn0, tn1, tn2 = get_triangular_grid(n)

        # perform KDE
        kde = sm.nonparametric.KDEMultivariate(
            data=np.array(x_tlss).T, var_type='ccc'
        )

        # evaluate KDE at grid points
        positions = np.vstack([tn0, tn1, tn2])
        density = kde.pdf(positions)

        return tn0, tn1, tn2, density

    # _tn0, _tn1, _tn2, my_density = do_trinary_kde(perturbed_x_tlss)
    return (do_trinary_kde,)


@app.cell
def _(np):
    def get_level(rho, alpha, n=250):
        rho_max = np.max(rho)
        rho_sum = np.sum(rho)

        levels = np.linspace(0.0, rho_max, n)
        for i in range(1, n):
            # fraction of density above this value
            f_density = np.sum(rho[rho > levels[-i]]) / rho_sum
            if f_density > alpha:
                return levels[-i]

    # get_level(my_density, 0.95)
    return (get_level,)


@app.cell
def _(blending_probs, np, sample_perturbed_solutions, sm):
    def hdr_coverage(x_samples, x_true):
        """Smallest HDR mass containing x_true, using the same KDE as do_trinary_kde."""
        data = np.asarray(x_samples).T                          # (3, n)
        kde = sm.nonparametric.KDEMultivariate(data=data, var_type='ccc')
        dens_samples = kde.pdf(data)
        dens_true = kde.pdf(np.asarray(x_true, dtype=float).reshape(3, 1))
        return np.mean(dens_samples > dens_true)

    def hdr_coverage_all(blending_probs):
        for blending_prob in blending_probs:
            print(blending_prob.name)
            xs = sample_perturbed_solutions(blending_prob, n_samples=2500)
            x_true = [blending_prob.blend_makeup[w] for w in blending_prob.wines]
            print("hdr:", hdr_coverage(xs, x_true) * 100)

    hdr_coverage_all(blending_probs)
    return


@app.cell
def _(mo):
    checkbox_kde = mo.ui.checkbox(label="include KDE?")
    checkbox_kde
    return (checkbox_kde,)


@app.cell
def _(
    Line2D,
    MultipleLocator,
    PercentFormatter,
    blending_probs,
    checkbox_kde,
    constrained_tls_soln,
    do_trinary_kde,
    get_level,
    mpl,
    np,
    plt,
    sample_perturbed_solutions,
    wine_to_color,
):
    def draw_simplex(blending_probs):    
        wines = blending_probs[0].wines

        all_errors = [] # for error analysis

        fig = plt.figure(figsize=(5, 5))
        ax = fig.add_subplot(projection="ternary")

        ax.set_tlabel(wines[0] + " [vol. %]")
        ax.set_llabel(wines[1] + " [vol. %]")
        ax.set_rlabel(wines[2] + " [vol. %]")

        for _ax in [ax.taxis, ax.laxis, ax.raxis]:
            _ax.set_label_position("tick1")
            _ax.set_major_locator(MultipleLocator(0.1))
            _ax.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

        # ax.grid(which='both')
        ax.grid(which='major', linestyle='-', alpha=0.3)
        ax.grid(which='minor', linestyle=':', alpha=0.3)

        markersize = 8
        nb_samples = 2500
        # loop over blends
        for blend_id, blending_prob in enumerate(blending_probs):
            ###
            #   uncertainty quantification (predictions)
            ###
            x_tlss = sample_perturbed_solutions(
                blending_prob, 
                n_samples=nb_samples if checkbox_kde.value else 100
            )
            alpha = 10.0
            lower = np.round(
                100 * np.percentile(x_tlss, alpha, axis=0), decimals=1
            )
            upper = np.round(
                100 * np.percentile(x_tlss, 100 - alpha, axis=0), decimals=1
            )
            x_tls_mean = np.mean(x_tlss, axis=0)

            if checkbox_kde.value:
                tn0, tn1, tn2, density = do_trinary_kde(x_tlss, n=100)

                # N_levels = 4
                # cmap = mcolors.LinearSegmentedColormap.from_list(
                #     "", ["white", wine_to_color[blend_id]], N=N_levels+2,
                # )

                levels = [get_level(density, l) for l in 
                          [0.60, 0.70, 0.80, 0.90]
                ]
                for level in levels:
                    ax.tricontourf(
                        tn0, tn1, tn2, density,
                        levels=[level, density.max()],
                        colors=[wine_to_color[blend_id]],
                        alpha=0.2,
                    )
            else:
                for s in range(len(x_tlss)):
                    ax.scatter(
                        [x_tlss[s][0]], [x_tlss[s][1]], [x_tlss[s][2]],
                        marker="o", color=wine_to_color[blend_id], alpha=0.1,
                        s=markersize ** 2
                    )

            ###
            #   true blend composition
            ###
            ax.scatter(
                blending_prob.blend_makeup[wines[0]],
                blending_prob.blend_makeup[wines[1]],
                blending_prob.blend_makeup[wines[2]],
                label="truth", marker="^", facecolor=wine_to_color[blend_id],
                s=markersize ** 2, color="black", clip_on=False, zorder=1000
            )

            ###
            #   single predicted blend composition
            ###
            x_tls, delta_Ab = constrained_tls_soln(blending_prob)
            ax.scatter(
                *x_tls,
                label="TLS", marker="*", facecolor=wine_to_color[blend_id],
                s=markersize ** 2, color="black", clip_on=False, zorder=1000
            )

            ###
            #   print estimate and UQ
            ###
            print(f"blend {blend_id+1} TLS:")
            print(blending_probs[blend_id].name)
            for (w, wine) in enumerate(wines):
                print(f"\t{wine}:")
                print(f"\t\tpred: {np.round(100*x_tls[w], decimals=1):.1f}%")
                print(f"\t\t{100 - 2*alpha:.0f}% CI: [{lower[w]}, {upper[w]}]")

            # errors
            x_true = np.array(
                [
                    blending_prob.blend_makeup[wine] 
                    for wine in blending_prob.wines
                ]
            )
            errors = np.abs(x_tls - x_true)
            all_errors.append(errors)
            print(f"\tmean MAE: {np.round(100*np.mean(errors), decimals=1)}%")
            print(f"\tmean RMSE: {np.round(100*np.sqrt(np.mean(errors**2)), decimals=1)}%")

        print("mean error over all blends: ")
        mean_all_errors = np.round(np.mean(all_errors, axis=0)*100, decimals=1)
        for (w, wine) in enumerate(wines):
            print(f"\t{wine}: {mean_all_errors[w]}")
        print(
            "mean error over all pure-variety wines and blend: ",
            np.round(np.mean(all_errors) * 100, decimals=1)
        )

        if not checkbox_kde.value:
            ax.legend(
                handles=[
                    Line2D([0], [0], marker='^', color='black', label='true',
                           markersize=markersize, linestyle='None'),
                    Line2D([0], [0], marker='o', color='None', label='predicted',
                           markersize=markersize, linestyle='None', markerfacecolor='black'),
                ], 
                fontsize=12
            )
        else:
            blend_handles = [
                mpl.patches.Patch(
                    color=wine_to_color[blend_id], 
                    label=bp.name
                ) 
                for blend_id, bp in enumerate(blending_probs)
            ]

            symbol_handles = [
                Line2D(
                    [0], [0], marker='^', color='none', markeredgecolor='black',
                    markerfacecolor='none', markersize=markersize, label='actual'
                ),
                Line2D(
                    [0], [0], marker='*', color='none', markeredgecolor='black',
                    markerfacecolor='none', markersize=markersize, label='predicted'
                )
            ]

            leg1 = ax.legend(
                handles=blend_handles, loc='upper left',
                bbox_to_anchor=(0.7, 1.12), borderaxespad=0, title='blend',
                prop={'size': 8}, title_fontsize=10, fontsize=6
            )

            leg2 = ax.legend(
                handles=symbol_handles, loc='upper left',
                bbox_to_anchor=(0.875, 0.8),
                borderaxespad=0, prop={'size': 10}
            )

            ax.add_artist(leg1)  # re-add first legend

        plt.tight_layout()
        _savename = "red_unmix"
        if checkbox_kde.value:
            _savename += "_kde"
        else:
            _savename += "_scatter"

        plt.savefig(_savename + ".pdf", format="pdf")
        plt.show()

    draw_simplex(blending_probs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### MC convergence study
    """)
    return


@app.cell
def _(
    PercentFormatter,
    blending_probs,
    constrained_tls_soln,
    np,
    pd,
    plt,
    sample_perturbed_solutions,
):
    def mc_sample_size_convergence(
        blending_prob,
        n_samples_grid=(10, 25, 50, 100, 250, 500, 1000, 2500),
        n_trials=20,
        alpha=10,# same as draw_simplex: [alpha, 100-alpha] percentiles
        nested=True,
        seed=0,
        x_point=None,                    # optional point estimate, e.g. constrained_tls_soln(bp)[0]
        savename=None,
    ):
        """
        How many MC samples do we need for the UQ of one blend to settle down?

        For each trial, draw MC samples of the predicted make-up and, for each
        n in n_samples_grid, compute the [alpha, 50, 100-alpha] percentiles of
        each wine's predicted volume fraction. Across trials, report mean +/- std.

        nested=True draws one pool of max(n_samples_grid) samples per trial and
        uses its first n samples for each n (much cheaper; the curves within a
        trial are then correlated, but the across-trial std at each n is still
        the right Monte Carlo error). nested=False draws fresh samples for
        every (trial, n).

        Returns (fig, results), with results a tidy DataFrame of every estimate.
        """
        np.random.seed(seed)  # sample_perturbed_problem uses the global RNG
        wines = blending_prob.wines
        n_grid = np.sort(np.asarray(n_samples_grid, dtype=int))
        qs = {"lower": alpha, "median": 50.0, "upper": 100.0 - alpha}

        rows = []
        for trial in range(n_trials):
            if nested:
                pool = np.asarray(
                    sample_perturbed_solutions(
                        blending_prob, n_samples=int(n_grid[-1]))
                )
            for n in n_grid:
                xs = pool[:n] if nested else np.asarray(
                    sample_perturbed_solutions(blending_prob, n_samples=int(n))
                )
                for stat, q in qs.items():
                    vals = np.percentile(xs, q, axis=0)
                    for w, wine in enumerate(wines):
                        rows.append(dict(trial=trial, n_samples=n, wine=wine,
                                         stat=stat, value=vals[w]))
        results = pd.DataFrame(rows)
        summary = (results.groupby(["wine", "stat", "n_samples"])["value"]
                   .agg(["mean", "std"]).reset_index())

        # ---- plot -----------------------------------------------------------
        fig, axs = plt.subplots(
            2, len(wines), figsize=(3.3 * len(wines), 5.5),
            sharex=True, layout="constrained",
            gridspec_kw={"height_ratios": [2, 1]},
        )
        stat_style = {
            "lower":  dict(color="C0", label=f"{alpha:g}th pct."),
            "median": dict(color="C2", label="median"),
            "upper":  dict(color="C3", label=f"{100 - alpha:g}th pct."),
        }
        for w, wine in enumerate(wines):
            ax, ax_err = axs[0, w], axs[1, w]
            for stat, style in stat_style.items():
                s = summary[(summary.wine == wine) & (summary.stat == stat)]
                ax.plot(s.n_samples, s["mean"], marker="o", ms=3, **style)
                ax.fill_between(s.n_samples, s["mean"] - s["std"], s["mean"] + s["std"],
                                color=style["color"], alpha=0.25, lw=0)
                pos = s["std"] > 0  # e.g. a 10th pct. pinned at 0 by the non-negativity step
                ax_err.plot(s.n_samples[pos], s["std"][pos], marker="o", ms=3,
                            color=style["color"])

            if x_point is not None:
                ax.axhline(
                    x_point[w], color="k", ls=":", lw=1, label="point est."
                )

            # 1/sqrt(n) guide anchored at the first grid point
            s0 = summary[(summary.wine == wine) & (summary.n_samples == n_grid[0])]["std"].max()
            ax_err.plot(n_grid, s0 * np.sqrt(n_grid[0] / n_grid), color="gray",
                        ls="--", lw=1, label=r"$\propto 1/\sqrt{n}$")

            ax.set_title(wine)
            ax.set_xscale("log")
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=None))
            ax_err.set_yscale("log")
            ax_err.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=None))
            ax_err.set_xlabel("# MC samples")
        axs[0, 0].set_ylabel("predicted vol. fraction\n(mean ± std over trials)")
        axs[1, 0].set_ylabel("MC std of estimate")
        axs[0, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        axs[1, -1].legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.02, 1.0))
        fig.suptitle(f"blend {blending_prob.name}: MC convergence ({n_trials} trials)")

        if savename is not None:
            fig.savefig(savename + ".pdf", format="pdf", bbox_inches="tight")
        return fig, results

    mc_sample_size_convergence(
        blending_probs[0], 
        x_point=constrained_tls_soln(blending_probs[0])[0],
        savename="red_MC_study"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## analyze corrections

    consistently, we're increasing values in A and decreasing values in b.

    from a mass balance perspective, this means "we think there is less sugar/acid/alcohol in the blend" than what you're measuring, to make the equations consistent.

    this could reflect bias in the model. think about mixing. [here](https://commons.wikimedia.org/wiki/File:Excess_Volume_Mixture_of_Ethanol_and_Water.png) is an excess volume chart.
    """)
    return


@app.cell
def _(blending_probs, constrained_tls_soln, pd):
    def analyze_corrections(blending_probs):
        corrections = pd.DataFrame()
        for blend_id, blending_prob in enumerate(blending_probs):   
            x_tls, delta_Ab = constrained_tls_soln(blending_prob)

            for c, constituent in enumerate(blending_prob.constituents):
                for w, wine in enumerate(blending_prob.wines + ['blend']):
                    new_row = pd.DataFrame(
                        [
                            {
                                'wine': wine,
                                'blend': blend_id,
                                'constituent': constituent,
                                'Δ': delta_Ab[c, w]
                            }
                        ]
                    )

                    corrections = pd.concat(
                        [corrections, new_row], ignore_index=True
                    )
        return corrections

    corrections = analyze_corrections(blending_probs)
    corrections.head()
    return (corrections,)


@app.cell
def _(corrections, sigmas):
    _blend = corrections[corrections["wine"] == "blend"]
    (_blend["Δ"] / _blend["constituent"].map(sigmas)).abs().groupby(
        _blend["constituent"]
    ).mean()
    return


@app.cell
def _(corrections):
    (corrections.groupby("wine").get_group("blend")["Δ"] < 0.0).all()
    return


@app.cell
def _(corrections):
    (corrections.groupby("wine").get_group("Cab. Franc")["Δ"] > 0.0).all()
    return


@app.cell
def _(corrections):
    (corrections.groupby("wine").get_group("Cab. Sauv.")["Δ"] > 0.0).all()
    return


@app.cell
def _(corrections):
    (corrections.groupby("wine").get_group("Merlot")["Δ"] > 0.0).all()
    return


@app.cell
def _(blending_probs, constrained_tls_soln, np, pd, plt, sns):
    def viz_correction_heatmaps(
        blending_probs, savename=None, ncols=2
    ):
        wines = blending_probs[0].wines
        constituents = blending_probs[0].constituents
        row_order = wines + ["blend"]

        tables = []
        for prob in blending_probs:
            x, delta_Ab = constrained_tls_soln(prob)

            s = np.array([prob.sigmas[c] for c in constituents])

            # corrected = measured - Δ, so (measured - corrected) / σ = Δ / σ
            t = pd.DataFrame(
                (delta_Ab / s[:, None]).T,
                index=wines + ["blend"], columns=constituents
            )
            # wines removed by the non-negativity step have Δ = 0; show them as blank
            t.iloc[np.flatnonzero(x == 0), :] = np.nan
            t = t.reindex(index=row_order, columns=constituents)
            tables.append(t)

        lim = max(np.nanmax(np.abs(t.values)) for t in tables)
        n = len(tables)
        n_rows, n_cols = tables[0].shape
        nrows = int(np.ceil(n / ncols))

        fig, axs = plt.subplots(
            nrows, ncols, sharex=True, sharey=True, layout="constrained",
            figsize=(6, 6),
            squeeze=False,
        )
        axs_flat = axs.ravel()
        used_axs = axs_flat[:n]

        for b, (ax, t) in enumerate(zip(used_axs, tables)):
            sns.heatmap(
                t, ax=ax, cmap="bwr", center=0, vmin=-lim, vmax=lim,
                annot=True, fmt=".1f", cbar=False,
                linewidths=0.5, linecolor="white",
            )
            ax.set_facecolor("0.85")  # NaN (removed wine) cells show as gray
            ax.set_title("blend:\n" + blending_probs[b].name, fontsize=10)
            ax.tick_params(axis="x", rotation=45, length=0)
            ax.tick_params(axis="y", rotation=0, length=0)
            ax.minorticks_off()
            for lbl in ax.get_xticklabels():
                lbl.set_ha("right")

        # hide any unused panels (e.g. 3 blends in a 2x2 grid)
        for ax in axs_flat[n:]:
            ax.set_visible(False)

        # tick labels only on the bottom row / left column
        for ax in used_axs:
            ax.label_outer()

        fig.colorbar(
            used_axs[0].collections[0], ax=list(used_axs), shrink=0.8,
            label=r"TLS correction, $\Delta / \sigma$",
        )
        if savename is not None:
            plt.savefig(savename + "_corrections_heatmap.pdf", bbox_inches="tight")
        return fig

    viz_correction_heatmaps(blending_probs, savename="red")
    return (viz_correction_heatmaps,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## questions

    I didn't account for the excess volume due to mixing. this is a small error.

    mix two wines, volume always less right?

    should the paper be narrowly focused on reverse-engineering of wine blending or about inverse problems of blending in general, across chemical engineering (e.g. fuel blends), the food industry (e.g., milk, wine, fruit juice, etc. blends), and environmental forensics (e.g., pollutant streams from multiple factories blending in a body of water).
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### excess volume

    worst-case analysis based on [data](https://en.wikipedia.org/wiki/Ethanol_(data_page)).

    at 25 deg C., 12.4 vol %: 0.98043, 15 vol %: 0.97753

    another source [here](https://wissen.science-and-fun.de/chemistry/chemistry/density-tables/ethanol-water-mixtures/):
    """)
    return


@app.cell
def _():
    (0.98043 - 0.97753) / 0.97753 * 100 # a 0.2 % error...
    return


@app.cell
def _():
    (0.89040 - 0.88339) / 0.88339 * 100 # a less than 1 % error
    return


@app.cell
def _(delta_Ab):
    delta_Ab[:, -1]
    return


@app.cell
def _(b):
    b
    return


@app.cell
def _(parent_fprints):
    parent_fprints["alcohol [vol. %]"].max()
    return


@app.cell
def _(parent_fprints):
    parent_fprints["alcohol [vol. %]"].min()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### checking assumptions

    first, vapor pressure. second, excess volume using data from [here](https://chem.libretexts.org/Bookshelves/Physical_and_Theoretical_Chemistry_Textbook_Maps/Exercises%3A_Physical_and_Theoretical_Chemistry/Data-Driven_Exercises/Bartender's_Conundrum_-_Partial_Molar_Volume_in_Water-Ethanol_Mixtures).
    """)
    return


@app.cell
def _():
    def vapor_pressures():
        T = 298.0 # K
        # Ethanol
        # 292.77 to 366.63 K
        # https://webbook.nist.gov/cgi/cbook.cgi?ID=C64175&Mask=4&Type=ANTOINE&Plot=on
        A = 5.24677
        B = 1598.673
        C = -46.424

        print("vapor pressure ethanol at 298 K: ", 
              10 ** (A - B / (T + C)),
              " bar"
        )

        # water
        # https://webbook.nist.gov/cgi/cbook.cgi?ID=C7732185&Mask=4
        A = 5.40221
        B = 1838.675
        C = -31.737
        print("vapor pressure water at 298 K: ", 
              10 ** (A - B / (T + C)),
              " bar"
        )

    vapor_pressures()
    return


@app.cell
def _():
    def excess_volume(et_vol_per):
        # 22 deg C
        # mixtures
        rho_1 = 0.98061 # g/mL
        x_1 = 10.0 # wt %
        rho_2 = 0.96808 # g/mL
        x_2 = 20.0 # wt %

        # pure densities
        rho_et = 0.78808
        rho_wa = 0.99799

        # basis
        V = 100 # L
        V_et = et_vol_per / 100 * V
        w_et = V_et * rho_et
        V_wa = (1 - et_vol_per / 100) * V
        w_wa = V_wa * rho_wa

        rho_ideal = (V_et * rho_et + V_wa * rho_wa) / V
        print("ideal density [g/mL]: ", rho_ideal) 

        x = w_et / (w_et + w_wa) * 100
        m = (rho_2 - rho_1) / (x_2 - x_1)
        print("actual density [g/mL]:", rho_1 + m * (x - x_1))

    excess_volume(12.9)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # white wine un-blending

    ## true make-up
    """)
    return


@app.cell
def _(pd):
    wblend_makeup = pd.DataFrame(
        {
            "Sauv. Blanc": [0.15, 0.4, 0.5, 0.85],
        }
    )
    wblend_makeup.insert(0, "Bordeaux Blanc", 1.0 - wblend_makeup["Sauv. Blanc"])

    wblend_makeup["name"] = [
        f"{bb*100:.0f}%BB {sb*100:.0f}%SB"
        for bb, sb in zip(
            wblend_makeup["Bordeaux Blanc"], wblend_makeup["Sauv. Blanc"]
        )
    ]

    # reorder. sort by composition
    wblend_perumutation = [1, 3, 0, 2]

    wblend_makeup
    return wblend_makeup, wblend_perumutation


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## fingerprints
    """)
    return


@app.cell
def _():
    wconstituent_to_xmax = { # use later
        'alcohol [vol. %]': 13.0,
        'sugar [g/L]': 4.0, 
        'acid [g/L]': 8.0,
        "abs. [280 nm]": 8.0, 
        "abs. [320 nm]": 5.0
    }
    return (wconstituent_to_xmax,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### chemistry (alcohol, sugar, acid)
    """)
    return


@app.cell
def _(pd, wblend_perumutation):
    wblend_fprints_chem = pd.DataFrame(
        {
            "alcohol [vol. %]": [12.10, 12.11, 12.10, 12.09],
            "sugar [g/L]": [2.2, 1.02, 3.45, 1.86],
            "acid [g/L]": [6.95, 6.71, 7.78, 6.52]
        }
    )
    wblend_fprints_chem = wblend_fprints_chem.iloc[
        wblend_perumutation, :].reset_index(drop=True)
    wblend_fprints_chem
    return (wblend_fprints_chem,)


@app.cell
def _(pd):
    wparent_fprints_chem = pd.DataFrame(
           {
            "alcohol [vol. %]": [12.16, 12.08],
            "sugar [g/L]": [0.48, 3.95],
            "acid [g/L]": [6.12, 7.63]
        },
        index=["Bordeaux Blanc", "Sauv. Blanc"]
    )
    wparent_fprints_chem
    return (wparent_fprints_chem,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### UV-viz sec

    280 nm is the standard total-phenolics index. It measures the aromatic rings in essentially all phenolic compounds and is the usual single number for a wine’s phenolic load.

    320 nm is the white-wine-specific band. It mainly reflects hydroxycinnamic acids (caftaric, coutaric, fertaric acid and their derivatives), which are the dominant phenolics in white wines. Their levels depend strongly on grape variety and on how the juice was handled, so 320 nm is often the most varietally distinctive wavelength for whites.
    """)
    return


@app.cell
def _():
    spec_cols = ["abs. [280 nm]", "abs. [320 nm]"]
    return (spec_cols,)


@app.cell
def _(wsigmas):
    # B2
    outlier_deviance = (
        (8.952 - (7.283 + 7.272) / 2) / wsigmas["abs. [280 nm]"]
        +
        (5.776 - (4.133 + 4.147) / 2) / wsigmas["abs. [320 nm]"]
    ) / 2
    outlier_deviance
    return


@app.cell
def _(pd, spec_cols):
    def read_uv_viz_data(sheet_name="Table", rm_outlier=True):
        path = "data/whitewineblendtrial.xlsx"
        # Find the header row (first column == "#") to skip the instrument metadata
        raw = pd.read_excel(path, sheet_name=sheet_name, header=None)
        hdr = raw.index[raw.iloc[:, 0].astype(str).str.strip() == "#"][0]
        df = pd.read_excel(path, sheet_name=sheet_name, header=hdr, na_values=["N/A"])
        df = df.dropna(how="all").set_index("#")

        # Drop blank rows
        df = df[~df["Name"].str.startswith("Blank")]

        # Keep only Abs. columns, renamed to their wavelength (e.g. 280)
        abs_cols = [c for c in df.columns if c.startswith("Abs.")]
        lam_cols = [c.replace("Abs.", "λ") for c in abs_cols]
        wavelengths = df[lam_cols].iloc[0].astype(int).tolist()

        out = df.set_index("Name")[abs_cols]
        out.columns = [f"abs. [{w} nm]" for w in wavelengths]

        if rm_outlier:
            print("remove outlier B2 1")
            out = out[out.index != "B2 1"]

        out = out[spec_cols]

        return out

    raw_spec_data = read_uv_viz_data(rm_outlier=True)
    group = raw_spec_data.index.str.replace(r"\s+\d+$", "", regex=True)
    raw_spec_data
    return group, raw_spec_data


@app.cell
def _(group, raw_spec_data, spec_cols):
    spec_data = raw_spec_data.groupby(group).mean()
    spec_data = spec_data[spec_cols]
    spec_data = spec_data.rename(index={"BB": "Bordeaux Blanc", "SB": "Sauv. Blanc"})
    spec_data.index.name = "Name"
    spec_data
    return (spec_data,)


@app.cell
def _(spec_data):
    wparent_fprints_spec = spec_data.loc[["Bordeaux Blanc", "Sauv. Blanc"]]
    wparent_fprints_spec
    return (wparent_fprints_spec,)


@app.cell
def _(group, raw_spec_data):
    spec_data_sigma_data = raw_spec_data.groupby(group).std()

    sigma_spec = spec_data_sigma_data.mean().to_dict()
    sigma_spec
    return sigma_spec, spec_data_sigma_data


@app.cell
def _(spec_data_sigma_data):
    spec_data_sigma_data#.mean()
    return


@app.cell
def _(spec_data, wblend_perumutation):
    wblend_fprints_spec = spec_data.loc[["B1", "B2", "B3", "B4"]]
    wblend_fprints_spec = wblend_fprints_spec.iloc[wblend_perumutation, :].reset_index(drop=True)

    wblend_fprints_spec
    return (wblend_fprints_spec,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## un-mix

    ### fingerprints
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    ww_fp = mo.ui.dropdown(options=["chemistry", "UV-viz", "both", "sugar"], label="choose white wine FP", value="both")
    ww_fp
    return (ww_fp,)


@app.cell
def _(
    constituents,
    pd,
    sigma_spec,
    sigmas,
    wblend_fprints_chem,
    wblend_fprints_spec,
    wblend_makeup,
    wparent_fprints_chem,
    wparent_fprints_spec,
    ww_fp,
):
    if ww_fp.value == "chemistry":
        wparent_fprints = wparent_fprints_chem.copy()
        wblend_fprints = wblend_fprints_chem.copy()
        wconstituents = constituents.copy()
    elif ww_fp.value == "UV-viz":
        wparent_fprints = wparent_fprints_spec.copy()
        wblend_fprints = wblend_fprints_spec.copy()
        wconstituents = list(wparent_fprints_spec.columns)
    elif ww_fp.value == "both":
        wparent_fprints = pd.concat([wparent_fprints_chem, wparent_fprints_spec], axis=1)
        wblend_fprints = pd.concat([wblend_fprints_chem, wblend_fprints_spec], axis=1)
        wconstituents = constituents.copy() + list(wparent_fprints_spec.columns)

    wblend_fprints["name"] = wblend_makeup["name"].values # for viz_fingerprints

    wsigmas = sigmas | sigma_spec
    wsigmas
    return wblend_fprints, wconstituents, wparent_fprints, wsigmas


@app.cell
def _(
    viz_cute_fingerprint,
    wblend_fprints,
    wconstituent_to_xmax,
    wconstituents,
    wparent_fprints,
    wsigmas,
):
    for _wine in wparent_fprints.index:
        viz_cute_fingerprint(
            wparent_fprints, _wine, wsigmas, wconstituents,
            constituent_to_xmax=wconstituent_to_xmax, savename=_wine
        )

    viz_cute_fingerprint(
        wblend_fprints, 0, wsigmas, wconstituents,
        constituent_to_xmax=wconstituent_to_xmax, savename="white_blend_0"
    )
    return


@app.cell
def _(
    WineBlendingProb,
    wblend_fprints,
    wblend_makeup,
    wconstituents,
    wparent_fprints,
    wsigmas,
):
    wblending_probs = [
        WineBlendingProb(
            wblend_makeup.loc[blend_id, "name"],
            list(wparent_fprints.index),
            wconstituents,
            wparent_fprints,
            wblend_fprints.loc[blend_id, wconstituents],
            wsigmas,
            wblend_makeup.loc[blend_id, wparent_fprints.index].to_dict()
        )
        for blend_id in range(4)
    ]
    return (wblending_probs,)


@app.cell
def _(
    viz_fingerprints,
    wblend_fprints,
    wconstituent_to_xmax,
    wconstituents,
    wparent_fprints,
    wsigmas,
):
    viz_fingerprints(
        wparent_fprints, wblend_fprints, wconstituents, wsigmas,
        constituent_to_xmax=wconstituent_to_xmax, savename="white_fingerprints"
    )
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### unmix
    """)
    return


@app.cell
def _(
    Line2D,
    PercentFormatter,
    constrained_tls_soln,
    mpl,
    np,
    plt,
    sample_perturbed_solutions,
    sns,
    wblending_probs,
    wine_to_color,
):
    def viz_white_wine_unblending(wblending_probs):
        wwbp_solns = [sample_perturbed_solutions(wwbp) for wwbp in wblending_probs]

        wines = wblending_probs[0].wines

        fig, ax = plt.subplots(figsize=(5, 2.5))

        ax.set_xlim(0, 1)
        ax.set_xlabel(wines[0] + " [volume %]")

        top = ax.secondary_xaxis(
            "top", functions=(lambda v: 1 - v, lambda v: 1 - v)
        )
        top.set_xlabel(wines[1] + " [volume %]")

        ax.set_ylabel("density")
        ax.set_yticks([0])

        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        top.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

        ms = 10
        for b in range(4):
            color = wine_to_color[b+4]

            # distribution of predicted make-ups
            vs = [x[0] for x in wwbp_solns[b]]
            print(f"blend {b} {np.mean(vs)*100:.1f}+/-{np.std(vs)*100:.1f}%")

            # true comp
            plt.scatter(
                wblending_probs[b].blend_makeup[wines[0]], 0.0,
                color=color, marker='^',
                clip_on=False, s=ms**2
            )

            # predicted
            x_tls, delta_Ab = constrained_tls_soln(
                wblending_probs[b]
            )
            plt.scatter(
                x_tls[0], 1.0,
                color=color, marker='*',
                s=ms**2,
                clip_on=False, transform=ax.get_xaxis_transform()
            )

            sns.kdeplot(
                vs, ax=ax, clip=(0, 1), fill=True, alpha=0.4, 
                label=wblending_probs[b].name,
                color=color
            )

        blend_handles = [
            mpl.patches.Patch(
                color=wine_to_color[b+4], 
                label=wblending_probs[b].name
            ) 
            for b in range(4)
        ]

        symbol_handles = [
            Line2D(
                [0], [0], marker='^', color='none', markeredgecolor='black',
                markerfacecolor='none', markersize=ms, label='actual'
            ),
            Line2D(
                [0], [0], marker='*', color='none', markeredgecolor='black',
                markerfacecolor='none', markersize=ms, label='predicted'
            )
        ]

        leg1 = ax.legend(
            handles=blend_handles, loc='upper left',
            bbox_to_anchor=(1.01, 0.9), borderaxespad=0, title='blend',
            prop={'size': 10}, title_fontsize=12, fontsize=10
        )
        ax.add_artist(leg1)
        ax.tick_params(axis="y", pad=8)   # nudge the y "0" left
        # ax.tick_params(axis="x", pad=6)   # nudge the x labels down

        leg2 = ax.legend(
            handles=symbol_handles, loc='upper left',
            bbox_to_anchor=(1.01, 0.3),
            borderaxespad=0, prop={'size': 10}
        )

        # fig.tight_layout()
        plt.savefig(
            "white_unmix.pdf", format="pdf", 
            bbox_extra_artists=(leg1, leg2, top, top.xaxis.label),
            bbox_inches="tight"
        )
        plt.show()

    viz_white_wine_unblending(wblending_probs)
    return


@app.cell
def _(viz_correction_heatmaps, wblending_probs):
    # wcorrections = analyze_corrections(wpv_fprints, wblend_fprints, wsigmas)
    viz_correction_heatmaps(wblending_probs, savename="white")
    return


@app.cell
def _(print_stuff, wblending_probs):
    print_stuff(wblending_probs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### compare with OLS
    """)
    return


@app.cell
def _(PercentFormatter, np, ols_baselines, plt, wblending_probs):
    def compare_with_ols(wblending_probs):
        wines = wblending_probs[0].wines

        fig, ax = plt.subplots(figsize=(6, 4))

        ax.set_xlim(0, 1)
        ax.set_xlabel("true " + wines[0] + " [volume %]")

        top = ax.secondary_xaxis(
            "top", functions=(lambda v: 1 - v, lambda v: 1 - v)
        )
        top.set_xlabel("true " + wines[1] + " [volume %]")

        ax.set_ylim(0, 1)
        ax.set_ylabel("pred. " + wines[0] + " [volume %]")
        right = ax.secondary_yaxis(
            "right", functions=(lambda v: 1 - v, lambda v: 1 - v)
        )
        right.set_ylabel("pred. " + wines[1] + " [volume %]")

        ax.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        top.xaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))
        right.yaxis.set_major_formatter(PercentFormatter(xmax=1, decimals=0))

        ax.set_aspect('equal', 'box')

        ress = [ols_baselines(bp, verbose=False) for bp in wblending_probs]
        x_trues = np.array([wbp.blend_makeup[wines[0]] for wbp in wblending_probs])
        err_type = "MAE"
        for k, method in enumerate(["SCTLS", "OLS", "COLS", "SOLS", "SCOLS"]):
            x_preds = np.array([res["x_" + method][0] for res in ress])

            errs = np.array([res[method + " " + err_type] for res in ress])
            err = np.mean(errs)

            plt.scatter(
                x_trues, x_preds, label=f"{method} ({err_type}: {err:.1%})",
                facecolors="none", edgecolors=f"C{k}"
            )
        plt.plot([0, 1], [0, 1], color="lightgray", linestyle="--", zorder=0)
        plt.legend(fontsize=10)

        plt.show()

    compare_with_ols(wblending_probs)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## ethanol noise
    """)
    return


@app.cell
def _(np, pd):
    etoh_raw_data = np.array([
        [14.63, 14.63, 14.61],  # 12% target
        [15.79, 15.75, 15.72],
        [16.81, 16.77, 16.75],
        [17.87, 17.87, 17.85],  # 15% target
    ])

    etoh_data = pd.DataFrame({
        "experiment": np.repeat(range(4), etoh_raw_data.shape[1]),
        "EtOH [vol. %]": etoh_raw_data.ravel(),
    })
    etoh_data["mean EtOH [vol. %]"] = etoh_data.groupby("experiment")["EtOH [vol. %]"].transform("mean")
    etoh_data["std EtOH [vol. %]"] = etoh_data.groupby("experiment")["EtOH [vol. %]"].transform("std")
    etoh_data
    return (etoh_data,)


@app.cell
def _(etoh_data, plt, sns):
    def viz_etoh_data(etoh_data):
        fig, ax = plt.subplots(figsize=(5, 4))
        sns.swarmplot(
            data=etoh_data, x="mean EtOH [vol. %]", y="EtOH [vol. %]",
            native_scale=True, size=7, ax=ax,
        )
        ax.set_xticks(etoh_data["mean EtOH [vol. %]"].unique())
        ax.set_xticklabels([f"{m:.2f}" for m in etoh_data["mean EtOH [vol. %]"].unique()])
        plt.show()

    viz_etoh_data(etoh_data)
    return


@app.cell
def _(etoh_data, plt):
    def viz_etoh_std(etoh_data):
        means = etoh_data["mean EtOH [vol. %]"]
        stds = etoh_data["std EtOH [vol. %]"]

        fig, ax = plt.subplots(figsize=(5, 4))
        ax.plot(means, stds, "o-")
        ax.set_xlabel("mean measured EtOH [vol. %]")
        ax.set_ylabel("std measured EtOH [vol. %]")
        ax.set_ylim(ymin=0)
        plt.show()

    viz_etoh_std(etoh_data)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
