import marimo

__generated_with = "0.18.4"
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
    import scipy
    import mpltern
    from aquarel import load_theme

    theme = load_theme("umbra_light")
    theme.apply()
    plt.rc('font', size=16)
    return (
        Line2D,
        MultipleLocator,
        get_triangular_grid,
        mcolors,
        mo,
        mpl,
        np,
        pd,
        plt,
        scipy,
        sm,
        sns,
    )


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

    ### features of the pure wine varieties
    """)
    return


@app.cell
def _(np, pd):
    wines = ["Cab. Sauv.", "Merlot", "Cab. Franc"]
    n_wines = len(wines)

    _pure_features_1 = pd.DataFrame(
        {
            "alcohol [vol. %]": [12.96, 13.20, 14.35],
            "sugar [g/L]": [4.51,  3.12,  2.44],
            "acid [g/L]": [5.6,  6.08,  5.41]
        },
        index=wines
    ) # day 1

    _pure_features_2 = pd.DataFrame(
        {
            "alcohol [vol. %]": [12.89, 13.125, 14.225],
            "sugar [g/L]": [4.58,  3.11,  2.47],
            "acid [g/L]": [5.71,  5.87,  5.33]
        },
        index=wines
    ) # day 2

    pure_features = (_pure_features_1 + _pure_features_2) / 2

    features = pure_features.columns

    # calculate std over the two days for the sigmas
    sigmas = {}
    trust_factor = 1.1 # inflate to account for bias
    for _feature in features:
        sigmas[_feature] = np.zeros(3)
        for (_w, _wine) in enumerate(wines):
            sigmas[_feature][_w] = np.std(
                [
                    _pure_features_1.loc[_wine, _feature], 
                    _pure_features_2.loc[_wine, _feature]
                ]
            ) * trust_factor

        sigmas[_feature] = np.mean(sigmas[_feature])

    pure_features
    return features, n_wines, pure_features, sigmas, wines


@app.cell
def _(sigmas):
    sigmas
    return


@app.cell
def _(features, np, sigmas):
    # mean std
    sigma_bar = np.mean([sigmas[feature] for feature in features])
    sigma_bar
    return (sigma_bar,)


@app.cell
def _(features, np, sigma_bar, sigmas):
    # weight matrix
    W = np.diag([sigma_bar / sigmas[feature] for feature in features])
    W
    return (W,)


@app.cell
def _(W):
    W
    return


@app.cell(hide_code=True)
def _(mo):
    checkbox = mo.ui.checkbox(label="include blends?")
    checkbox
    return (checkbox,)


@app.cell
def _(checkbox):
    include_blends = checkbox.value
    return (include_blends,)


@app.cell
def _(
    blend_features,
    features,
    include_blends,
    n_blends,
    n_wines,
    plt,
    pure_features,
    wine_to_color,
    wines,
):
    def _draw_lollipop(ax, x, y, color):
        ax.plot([x, x], [0, y], color=color, zorder=100)
        ax.scatter([x], [y], color=color, zorder=100, s=8**2)

    def viz_features(pure_features, blend_features):
        figsize = [6.4, 4.8]
        if include_blends:
            figsize[0] *= 1.1

        fig, axs = plt.subplots(1, 3, figsize=figsize)
        # loop over features
        for f, feature in enumerate(features):
            # loop over wines
            for w, wine in enumerate(wines):
                feature_val = pure_features.loc[wine, feature]
                _draw_lollipop(axs[f], w, feature_val, wine_to_color[wine])

            if include_blends:
                d = 1 # spacing
                for blend_id in range(n_blends):
                    feature_val = blend_features.loc[blend_id, feature]
                    _draw_lollipop(
                        axs[f], 
                        n_wines + blend_id + d, feature_val, 
                        wine_to_color[blend_id]
                    )

                axs[f].set_xticks(
                    list(range(n_wines)) + list(range(n_wines+d, n_wines+n_blends+d)),
                    wines + [f"blend {b+1}" for b in range(n_blends)],
                    rotation=90
                )
                axs[f].set_xlim(-0.5, n_wines + n_blends + 0.5)

                axs[f].axvspan(
                    n_wines+d - 0.5, n_wines+n_blends+d, color='gray', alpha=0.1, lw=0
                )
            else:  
                axs[f].set_xticks(range(n_wines), wines, rotation=90)
                axs[f].set_xlim(-0.5, n_wines - 0.5)
            axs[f].set_ylabel(feature)
            axs[f].set_ylim(ymin=0.0)
        plt.tight_layout()

        _savename = "features"
        if include_blends:
            _savename += "_w_blends"
        else:
            _savename += "_wo_blends"

        plt.savefig(_savename + ".pdf", format="pdf")
        return fig

    viz_features(pure_features, blend_features)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### features of the blends
    """)
    return


@app.cell
def _(pd):
    blend_features = pd.DataFrame(
        {
            "alcohol [vol. %]": [13.37, 13.19, 13.18, 12.975],
            "sugar [g/L]": [3.1, 3.97, 3.4, 4.0],
            "acid [g/L]": [5.73, 5.56, 5.69, 5.63]
        }
    )

    n_blends = blend_features.shape[0]

    blend_features
    return blend_features, n_blends


@app.cell
def _(blend_features, features, n_blends, pure_features):
    # blend features should be within range of parent pure variety wines
    for _feature in features:
        for _blend_id in range(n_blends):
            assert blend_features.loc[_blend_id, _feature] > pure_features[_feature].min()
            assert blend_features.loc[_blend_id, _feature] < pure_features[_feature].max()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### ground-truth blend composition
    """)
    return


@app.cell
def _(pd):
    blend_composition = pd.DataFrame(
        {
            "Cab. Sauv.": [0.1, 0.7, 0.3, 0.6],
            "Merlot": [0.6, 0.1, 0.5, 0.4],
            "Cab. Franc": [0.3, 0.2, 0.2, 0.0]
        }
    )
    blend_composition
    return (blend_composition,)


@app.cell
def _(plt, wine_to_color, wines):
    def viz_makeup_pie(blend_composition, blend_id, savename=None):
        blend_text = f"blend {blend_id + 1}"

        fig, ax = plt.subplots()
        ax.pie(
            blend_composition.loc[blend_id, wines], labels=wines, 
            colors=[wine_to_color[wine] for wine in wines]
        )
        plt.title(blend_text)
        if savename:
            plt.savefig(blend_text + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()
    return (viz_makeup_pie,)


@app.cell
def _(blend_composition, viz_makeup_pie):
    viz_makeup_pie(blend_composition, 3)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## setting up the inverse problem

    *unknowns:*

    * $c$: vol. fraction Cab. Sauv.
    * $m$: vol. fraction Merlot
    * $1-c-m$: vol. fraction Cab. Franc

    *ingredient:* $f$

    $f_{cs} cs + f_m m + f_{cf} cf = f_{b}$

    $f_{cs} cs + f_m m + f_{cf} (1-cs-m) = f_{b}$

    $(f_{cs}-f_{cf}) cs + (f_m-f_{cf}) m = f_{b}-f_{cf}$
    """)
    return


@app.cell(hide_code=True)
def _(mo):
    dropdown = mo.ui.dropdown(
        options=[0, 1, 2, 3], value=1, label="blend ID"
    )
    dropdown
    return (dropdown,)


@app.cell
def _(dropdown):
    blend_id = dropdown.value
    return (blend_id,)


@app.cell
def _(features, np, wines):
    def build_linear_problem(
        pure_features, blend_features, blend_id
    ):
        A = np.transpose(pure_features.loc[wines, features].values)

        b = blend_features.loc[blend_id, features].values.reshape((3, 1))

        return A, b
    return (build_linear_problem,)


@app.cell
def _(blend_features, blend_id, build_linear_problem, pure_features):
    A, b = build_linear_problem(pure_features, blend_features, blend_id)
    A, b
    return A, b


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ☑️ if we solve naively, sum-to-one constraint is violated.
    """)
    return


@app.cell
def _(A, b, np):
    x_naive = np.linalg.solve(A, b)
    assert not np.isclose(np.sum(x_naive), 1.0)
    print("sum of vol fracs: ", np.sum(x_naive))
    return (x_naive,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### direction of chemical fingerprints

    this could explain anisotropy in UQ.
    """)
    return


@app.cell
def _(pure_features):
    pure_features
    return


@app.cell
def _(W, features, np):
    def fingerprint_angles(pure_features, wine_a, wine_b):
        x_i = W @ pure_features.loc[wine_a, features].values
        x_j = W @ pure_features.loc[wine_b, features].values
        cos_theta = np.dot(x_i, x_j) / (np.linalg.norm(x_i) * np.linalg.norm(x_j))
        return np.acos(cos_theta) * 180 / np.pi
    return (fingerprint_angles,)


@app.cell
def _(fingerprint_angles, pure_features):
    fingerprint_angles(pure_features, "Cab. Sauv.", "Merlot")
    return


@app.cell
def _(fingerprint_angles, pure_features):
    fingerprint_angles(pure_features, "Cab. Sauv.", "Cab. Franc")
    return


@app.cell
def _(fingerprint_angles, pure_features):
    fingerprint_angles(pure_features, "Cab. Franc", "Merlot")
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## constrained total least squares solution
    """)
    return


@app.cell
def _(W, build_linear_problem, np):
    def constrained_tls_soln(
        pure_features, blend_features, blend_id, verbose=False, scale_by_W=True
    ):
        A, b = build_linear_problem(pure_features, blend_features, blend_id)

        # build augmented matrix [A; b]
        Ab = np.hstack([A, b])

        # scale for variances
        if scale_by_W:
            Ab = W @ Ab

        if verbose:
            print("rank of [A b]: ", np.linalg.matrix_rank(Ab))
            print("size of [A b]: ", Ab.shape, "\n")

            print("rank of [A; b; 1]: ", np.linalg.matrix_rank(
                np.vstack([Ab, np.ones(4)])), "\n")

        # enforce linear constraint Psi x = d
        Psid = np.ones(4).reshape((1, 4)) # [Psi; d]

        QR = np.linalg.qr(np.transpose(Psid), mode="complete")
        Q2 = QR.Q[:, 1:] # basis for null space; Psid @ Q2 == 0
        np.testing.assert_allclose(
            (Psid @ Q2).reshape(3, 1), 
            np.zeros(3).reshape(3, 1), atol=1e-5
        )

        # new augmented matrix
        C = Ab @ Q2 @ np.transpose(Q2)
        if verbose:
            print("rank of [A b] Q2 Q2': ", np.linalg.matrix_rank(C))
            print("size of [A b] Q2 Q2': ", C.shape)
        assert C.shape == (3, 4)

        # SVD of [A; b] Q2 Q2'
        f = np.linalg.svd(C, full_matrices=True)
        if verbose:
            print("SVs of [A b] Q2 Q2': ", f.S, "\n")

        # n + 1 - p = 3 + 1 - 1 = 3rd singular vector.
        z = -f.Vh[2, :] / f.Vh[2, 3] # row 2 Vh = col 2 of V
        assert f.S[2] > 0.0

        x_tls = z[:3]
        assert np.isclose(sum(x_tls), 1.0)

        # corrections
        delta_Ab = f.S[2] * f.U[:, 2].reshape(3, 1) @ f.Vh[2, :].reshape(1, 4)
        if verbose:
            print("shape of [A b] Q2 Q2' Q2: ", (C @ Q2).shape)
            print("SVs of [A b] Q2 Q2' Q2: ", np.linalg.svd(
                C @ Q2, full_matrices=True).S, "\n")
            print("SVs of ([A b] Q2 Q2'- Delta) Q2: ", np.linalg.svd(
                (C - delta_Ab) @ Q2, full_matrices=True).S)
            print("rank of [[A; b] - Delta; 1]: ", np.linalg.matrix_rank(
                np.vstack([Ab - delta_Ab, np.ones(4)])), "\n")

        # unscale
        if scale_by_W:
            delta_Ab = np.linalg.inv(W) @ delta_Ab

        # make physical
        if np.any(x_tls < 0.0):
            x_tls[x_tls < 0.0] = 0.0
            x_tls = x_tls / np.sum(x_tls)

        return x_tls, delta_Ab
    return (constrained_tls_soln,)


@app.cell
def _(blend_features, blend_id, constrained_tls_soln, pure_features):
    constrained_tls_soln(
        pure_features, blend_features, blend_id, verbose=True, scale_by_W=False
    )
    return


@app.cell
def _(blend_features, blend_id, constrained_tls_soln, pure_features):
    x_tls, delta_Ab = constrained_tls_soln(
        pure_features, blend_features, blend_id, verbose=True
    )
    x_tls, delta_Ab
    return delta_Ab, x_tls


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ☑️  null space of W [A ; b] is related to the naive solution to Ax = b which doesn't satisfy the constraint.
    """)
    return


@app.cell
def _(A, W, b, np, scipy, x_naive):
    _z = scipy.linalg.null_space(W @ np.hstack([A, b]))
    _x_null = -(_z / _z[-1])[:3]
    assert np.allclose(_x_null, x_naive)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ☑️ now, we've modified [A; b] with the total perturbation. so we're adjusting the null space to allow for the constraint to be satisfied. this gives the CTLS solution.
    """)
    return


@app.cell
def _(A, b, delta_Ab, np, scipy, x_tls):
    _z = scipy.linalg.null_space(np.hstack([A, b]) - delta_Ab)
    _x_null = -(_z / _z[-1])[:3]
    assert np.allclose(_x_null, x_tls.reshape(3, 1))
    _x_null
    return


@app.cell
def _(x_tls):
    x_tls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ☑️ directly check perturbed problem.
    """)
    return


@app.cell
def _(A, b, delta_Ab, np, x_tls):
    x_tls_w_perturb = np.linalg.solve(
        A - delta_Ab[:, :3], b - delta_Ab[:, 3].reshape(3, 1)
    ) 
    assert np.allclose(x_tls_w_perturb, x_tls.reshape(3, 1))
    x_tls_w_perturb
    return


@app.cell
def _(x_tls):
    x_tls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ☑️ the matrix [A b ; 1] is full rank.
    """)
    return


@app.cell
def _(A, b, np):
    Ab_ones = np.vstack([np.hstack([A, b]), np.ones(4)])
    assert np.linalg.matrix_rank(Ab_ones) == 4
    print("singular values of [Ab; 1]: ", np.linalg.svd(Ab_ones).S)
    Ab_ones
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ☑️ difference with naive solution
    """)
    return


@app.cell
def _(np, x_naive, x_tls):
    x_tls - np.ravel(x_naive)
    return


@app.cell
def _(np, x_naive, x_tls):
    x_tls - np.ravel(x_naive) / np.sum(x_naive)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ☑️ $z=Q_2y=v_n$ has solution.

    $v_{n+1}$ is the one outside col space.
    """)
    return


@app.cell
def _(A, W, b, np):
    _Q2 = np.linalg.qr(np.ones(4).reshape((4, 1)), mode="complete").Q[:, 1:]
    _V = np.transpose(
        np.linalg.svd(W @ np.hstack([A, b]) @ _Q2 @ np.transpose(_Q2)).Vh
    )
    np.transpose(_Q2) @ _Q2
    _V[:, 3]
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## uncertainty quantification
    """)
    return


@app.cell
def _(W):
    W
    return


@app.cell
def _(pure_features):
    pure_features
    return


@app.cell
def _(new_blend_features):
    new_blend_features
    return


@app.cell
def _(features, n_blends, n_wines, np):
    def perturb_data(
        pure_features, blend_features, sigmas
    ):
        new_pure_features = pure_features.copy()
        new_blend_features = blend_features.copy()

        for feature in features:
            new_pure_features.loc[:, feature] += np.random.normal(
            scale=sigmas[feature], size=(n_wines, )
        )
            new_blend_features.loc[:, feature] += np.random.normal(
            scale=sigmas[feature], size=(n_blends, )
        )

        return new_pure_features, new_blend_features
    return (perturb_data,)


@app.cell
def _(blend_features, np, perturb_data, pure_features, sigmas):
    new_pure_features, new_blend_features = perturb_data(
        pure_features, blend_features, sigmas
    )

    _differences_pure = np.ravel((pure_features - new_pure_features).values)
    assert len(_differences_pure) == len(set(_differences_pure))
    _differences_blend = np.ravel((blend_features - new_blend_features).values)
    assert len(_differences_blend) == len(set(_differences_blend))

    new_pure_features, new_blend_features
    return (new_blend_features,)


@app.cell
def _(constrained_tls_soln, perturb_data):
    def sample_perturbed_solutions(
        pure_features, blend_features, sigmas, blend_id, n_samples=100
    ):
        x_tlss = []

        for s in range(n_samples):
            new_pure_features, new_blend_features = perturb_data(
                pure_features, blend_features, sigmas
            )

            x_tls, _ = constrained_tls_soln(
                new_pure_features, new_blend_features, blend_id
            )
            x_tlss.append(x_tls)

        return x_tlss
    return (sample_perturbed_solutions,)


@app.cell
def _(
    blend_features,
    blend_id,
    pure_features,
    sample_perturbed_solutions,
    sigmas,
):
    perturbed_x_tlss = sample_perturbed_solutions(
        pure_features, blend_features, sigmas, blend_id
    )
    perturbed_x_tlss
    return (perturbed_x_tlss,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    the mean over these peturbations should be the x_tls.
    """)
    return


@app.cell
def _(np, perturbed_x_tlss, x_tls):
    assert np.allclose(
        np.mean(perturbed_x_tlss, axis=0),
        x_tls, 
        atol=0.01
    )
    return


@app.cell
def _(get_triangular_grid, np, perturbed_x_tlss, sm):
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

    _tn0, _tn1, _tn2, my_density = do_trinary_kde(perturbed_x_tlss)
    # _density
    return do_trinary_kde, my_density


@app.cell
def _(my_density, np):
    def get_level(rho, alpha, n=250):
        rho_max = np.max(rho)
        rho_sum = np.sum(rho)

        levels = np.linspace(0.0, 1.0, n) * rho_max
        for i in range(1, n):
            # fraction of density above this value
            f_density = np.sum(rho[rho > levels[-i]]) / rho_sum
            if f_density > alpha:
                return levels[-i]

    get_level(my_density, 0.98)
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
    blend_composition,
    blend_features,
    checkbox_kde,
    constrained_tls_soln,
    do_trinary_kde,
    mcolors,
    mpl,
    n_blends,
    np,
    perturbed_x_tlss,
    plt,
    pure_features,
    sample_perturbed_solutions,
    sigmas,
    wine_to_color,
    wines,
):
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

    markersize = 6

    dx_text = [[0.05, 0.0, 0.05], [0.0, 0.1, 0.0]]

    # loop over blends
    for _blend_id in range(n_blends):
        ###
        #   uncertainty quantification (predictions)
        ###
        x_tlss = sample_perturbed_solutions(
            pure_features, blend_features, sigmas, _blend_id, 
            n_samples=1000 if checkbox_kde.value else 100
        )
        x_UQ = np.std(perturbed_x_tlss, axis=0)

        if checkbox_kde.value:
            tn0, tn1, tn2, density = do_trinary_kde(x_tlss, n=100)

            # alphas = [0.9, 0.8, 0.7, 0.6]
            N_levels = 4
            cmap = mcolors.LinearSegmentedColormap.from_list(
                "", ["white", wine_to_color[_blend_id]], N=N_levels+2,
            )
            # cmap.set_under("white", alpha=0.0)

            ax.tricontour(
                tn0, tn1, tn2, density, 
                levels=N_levels,# [get_level(density, alpha) for alpha in alphas], 
                cmap=cmap,
                # extend='min'
            )

        else:
            for s in range(len(x_tlss)):
                ax.scatter(
                    [x_tlss[s][0]], [x_tlss[s][1]], [x_tlss[s][2]],
                    marker="^", color=wine_to_color[_blend_id], alpha=0.1,
                    s=markersize ** 2
                )

        ###
        #   true blend composition
        ###
        ax.scatter(
            [blend_composition.loc[_blend_id, wines[0]]],
            [blend_composition.loc[_blend_id, wines[1]]],
            [blend_composition.loc[_blend_id, wines[2]]],
            label="truth", marker="^", facecolor=wine_to_color[_blend_id],
            s=markersize ** 2, color="black", clip_on=False, zorder=1000
        )

        ###
        #   single predicted blend composition
        ###
        _x_tls, _delta_Ab = constrained_tls_soln(
            pure_features, blend_features, _blend_id
        )
        print(f"blend {_blend_id} TLS:")
        for (_w, _wine) in enumerate(wines):
            print(f"\t\t{_wine}: {_x_tls[_w]:.2f} vol % +/- {x_UQ[_w]:.2f} vol. %")
        # errors
        errors = np.abs(_x_tls - blend_composition.loc[_blend_id, wines].values)
        all_errors.append(errors)
        print("\terrors = ", errors)
        print("\tmean error: ", np.mean(errors))


        # ax.scatter(
        #     [_x_tls[0]], [_x_tls[1]], [_x_tls[2]],
        #     label="prediction", marker="^", color=wine_to_color[_blend_id],
        #     s=markersize ** 2
        # )

    print("mean error over all blends: ", np.mean(all_errors, axis=0))
    print("mean error over all pure-variety wines and blend: ", np.mean(all_errors))

    if not checkbox_kde.value:
        ax.legend(
            handles=[
                Line2D([0], [0], marker='o', color='black', label='true',
                       markersize=markersize, linestyle='None'),
                Line2D([0], [0], marker='^', color='None', label='predicted',
                       markersize=markersize, linestyle='None', markerfacecolor='black'),
            ], 
            fontsize=12
        )
    else:
        patches = [
            mpl.patches.Patch(
                color=wine_to_color[_blend_id], 
                label=f"{_blend_id+1}"
            ) for _blend_id in range(n_blends)
        ]
        plt.legend(
            handles=patches, title="blend", loc='upper right',
            prop={'size': 10}, title_fontsize='small',
            bbox_to_anchor=(1.05, 1.05)
        )

    plt.tight_layout()
    _savename = "composition_triangle"
    if checkbox_kde.value:
        _savename += "_kde"
    else:
        _savename += "_scatter"

    plt.savefig(_savename + ".pdf", format="pdf")
    ax
    return (x_tlss,)


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
def _(np, x_tlss):
    np.mean(x_tlss, axis=0)
    return


@app.cell
def _(x_tls):
    x_tls
    return


@app.cell
def _(pure_features):
    pure_features
    return


@app.cell
def _(
    blend_features,
    constrained_tls_soln,
    features,
    n_blends,
    np,
    pd,
    pure_features,
):
    def get_corrections(pure_features, blend_features):
        corrections = pd.DataFrame(index=range(n_blends))
        for feature in features:
            corrections[feature] = np.zeros(n_blends)

        for blend_id in range(n_blends):
            x_tls, delta_Ab = constrained_tls_soln(
                pure_features, blend_features, blend_id
            )
            print(delta_Ab)
            # for f, feature in enumerate(features):
            #     corrections.loc[blend_id, feature] = delta_Ab[f]

        corrections

    get_corrections(pure_features, blend_features)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## all blends at once
    """)
    return


@app.cell
def _(build_linear_problem, n_blends, np):
    def constrained_tls_soln_all(pure_features, blend_features):
        A = build_linear_problem(pure_features, blend_features, 0)[0]
        bs = [
            build_linear_problem(pure_features, blend_features, blend_id)[1] 
            for blend_id in range(n_blends)
        ]
        B = np.hstack(bs)

        # build augmented matrix [A; B]
        AB = np.hstack([A, B])
        print("shape AB: ", AB.shape)

        # TODO

        return AB
    return (constrained_tls_soln_all,)


@app.cell
def _(blend_features, constrained_tls_soln_all, pure_features):
    constrained_tls_soln_all(pure_features, blend_features)
    return


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
    ## excess volume

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
def _(pure_features):
    pure_features["alcohol [vol. %]"].max()
    return


@app.cell
def _(pure_features):
    pure_features["alcohol [vol. %]"].min()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## checking assumptions

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
        print("ideal density [g/L]: ", rho_ideal) 

        x = w_et / (w_et + w_wa)
        m = (rho_2 - rho_1) / (x_2 - x_1)
        print("actual density [g/L]:", rho_1 + m * (x - x_1))

    excess_volume(12.9)
    return


@app.cell
def _():
    def excess_volume2():
        # https://en.wikipedia.org/wiki/Ethanol_(data_page)#Properties_of_aqueous_ethanol_solutions
        # x : vol %, 15 deg C
        rho_1 = 0.98043 # relative to water
        x_1 = 12.4
        rho_2 = 0.97753
        x_2 = 15.0

        rho = lambda x: rho_1 + (rho_2 - rho_1) / (x_2 - x_1) * (x - x_1)

        # pure
        rho_wa = 0.99708
        rho_et = 0.78506

        # scenario
        V = 50.0 # L basis
        # mix together these two solutions
        v1 = 12.9 / 100 # Et vol frac
        v2 = 14.3 / 100
        m_et = V * (v1 + v2) * rho_et
        m_wa = V * (2 - v1 - v2) * rho_wa
        m = m_et + m_wa # total mass
        V_new = m / rho((v1+v2)/2 * 100)
        print("calc V : ", V_new)
        print("V reduction: ", (1 - V_new / (2 * V)) * 100)

    excess_volume2()
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
