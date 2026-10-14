import marimo

__generated_with = "0.25.1"
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
    # import pyvista as pv
    import seaborn as sns
    import statsmodels.api as sm
    import matplotlib as mpl
    from dataclasses import dataclass
    import scipy
    import mpltern
    from aquarel import load_theme
    from dataclasses import dataclass

    theme = load_theme("umbra_light")
    theme.apply()
    plt.rc('font', size=16)
    return (
        Line2D,
        MultipleLocator,
        dataclass,
        get_triangular_grid,
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

        "Sauv. Blanc": "gold",
        "Bordeaux Blanc": "darkgoldenrod",

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
    #### 🤔 direction of chemical fingerprints

    this could explain anisotropy in UQ.
    """)
    return


@app.cell
def _(np):
    def fingerprint_angles(
        pv_fprints, wine_a, wine_b, sigmas
    ):
        constituents = pv_fprints.columns

        x_a = np.array(
            [
                pv_fprints.loc[wine_a, constituent] / sigmas[constituent]
                for constituent in constituents
            ]
        )
        x_b = np.array(
            [
                pv_fprints.loc[wine_b, constituent] / sigmas[constituent]
                for constituent in constituents
            ]
        )
        cos_theta = np.dot(x_a, x_b) / (
            np.linalg.norm(x_a) * np.linalg.norm(x_b)
        )
        return np.acos(cos_theta) * 180 / np.pi

    return (fingerprint_angles,)


@app.cell
def _(fingerprint_angles, pv_fprints, sigmas):
    fingerprint_angles(
        pv_fprints, "Cab. Sauv.", "Merlot", sigmas
    )
    return


@app.cell
def _(fingerprint_angles, pv_fprints, sigmas):
    fingerprint_angles(
        pv_fprints, "Cab. Sauv.", "Cab. Franc", sigmas
    )
    return


@app.cell
def _(fingerprint_angles, pv_fprints, sigmas):
    fingerprint_angles(
        pv_fprints, "Merlot", "Cab. Franc", sigmas
    )
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
        ax.scatter([x], [y], color=color, zorder=100, s=8**2, clip_on=False)

    constituent_to_ymax = { # use later
        'alcohol [vol. %]': 15.0,
        'sugar [g/L]': 5.05, 
        'acid [g/L]': 6.1
    }

    def viz_features(pv_fprints, blend_fprints, constituent_to_ymax):
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

    viz_features(pv_fprints, blend_fprints, constituent_to_ymax)
    return constituent_to_ymax, viz_features


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
    return (blend_makeup,)


@app.cell
def _(plt, wine_to_color):
    def viz_makeup_pie(blend_makeup, blend_id, savename=None):
        wines = blend_makeup.columns
        blend_text = f"blend {blend_id + 1}"

        fig, ax = plt.subplots()
        ax.pie(
            blend_makeup.loc[blend_id, wines], labels=wines, 
            colors=[wine_to_color[wine] for wine in wines]
        )
        plt.title(blend_text)
        if savename:
            plt.savefig(blend_text + ".pdf", format="pdf", bbox_inches="tight")
        plt.show()

    return (viz_makeup_pie,)


@app.cell
def _(blend_makeup, viz_makeup_pie):
    viz_makeup_pie(blend_makeup, 3)
    return


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
        pv_fprints : pd.DataFrame
        blend_fprint : pd.Series
        sigmas : dict

        def __post_init__(self):
            # order of constiuents same
            c1 = self.pv_fprints.columns
            c2 = self.blend_fprint.index
            assert all(c1 == c2)

    return (WineBlendingProb,)


@app.cell(hide_code=True)
def _(mo):
    blend_selector = mo.ui.dropdown(
        options=[0, 1, 2, 3], value=1, label="blend ID"
    )
    blend_selector
    return (blend_selector,)


@app.cell
def _(WineBlendingProb, blend_fprints, blend_selector, pv_fprints, sigmas):
    blending_prob = WineBlendingProb(
        pv_fprints,
        blend_fprints.loc[blend_selector.value],
        sigmas
    )
    blending_prob
    return (blending_prob,)


@app.cell
def _(np):
    def build_linear_system(blending_prob):
        wines = blending_prob.pv_fprints.index
        constituents = blending_prob.pv_fprints.columns

        M = len(constituents)
        N = len(wines)

        A = np.transpose(blending_prob.pv_fprints.values)
        b = blending_prob.blend_fprint.values.reshape((M, 1))

        sigma_bar = np.mean(
            [blending_prob.sigmas[constituent] for constituent in constituents]
        )
        W = np.diag(
            [
                sigma_bar / blending_prob.sigmas[constituent] 
                for constituent in constituents
            ]
        )

        assert A.shape == (M, N)
        assert W.shape == (M, M)

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
def _(
    WineBlendingProb,
    assert_feasible_vol_fracs,
    build_linear_system,
    np,
    scipy,
):
    def constrained_tls_soln(
        blending_prob, 
        verbose=False, run_checks=False, use_naive_method=False,
        handle_non_negativity=True
    ):
        # W pulled from outside.
        W, A, b = build_linear_system(blending_prob)

        # build augmented matrix [A; b]
        Ab = np.hstack([A, b])

        # build scaled augmented matrix
        WAb = W @ Ab

        # case M+1 > N
        M = blending_prob.pv_fprints.shape[1] # constituents
        N = blending_prob.pv_fprints.shape[0] # parent mixtures

        if use_naive_method:
            if M == N:
                x = np.linalg.solve(A, b)
                if run_checks:
                    print("sum xi naive: ", np.sum(x))
                    assert not np.isclose(np.sum(x), 1.0, atol=0.005)
                x = x / np.sum(x)

                delta_Ab = np.zeros((M, N+1))
                return x, delta_Ab

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
            wines = blending_prob.pv_fprints.index

            # eliminate the wine with the negative volume fraction
            id_wine_remove = np.argmin(x_tls)
            wine_to_remove = wines[id_wine_remove]
            if verbose:
                print(f"Eliminating wine: {wine_to_remove}")

            new_wines = wines.drop(wine_to_remove)

            # new blending problem
            new_blending_prob = WineBlendingProb(
                blending_prob.pv_fprints.loc[new_wines, :], 
                blending_prob.blend_fprint, 
                blending_prob.sigmas
            )

            # run on reduced TLS problem
            _x_tls, _delta_Ab = constrained_tls_soln(
                new_blending_prob, verbose=verbose, run_checks=run_checks
            )

            # handle re-sizing (we removed a wine)
            x_tls = np.zeros(N)
            delta_Ab = np.zeros((M, N+1))
            mask = wines != wine_to_remove
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### test situation when $x_i<0$.
    """)
    return


@app.cell
def _(
    WineBlendingProb,
    blend_fprints,
    constrained_tls_soln,
    pv_fprints,
    sigmas,
):
    bad_blending_prob = WineBlendingProb(
        pv_fprints,
        blend_fprints.loc[3] - 0.02,
        sigmas
    )
    _x_tls, _delta_Ab = constrained_tls_soln(
        bad_blending_prob, verbose=True, run_checks=True,
        handle_non_negativity=True
    )
    _x_tls
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### 🙄 compare TLS with naive approaches
    """)
    return


@app.cell
def _(blending_prob, constrained_tls_soln):
    x_naive, _ = constrained_tls_soln(
        blending_prob, 
        run_checks=True, use_naive_method=True
    )
    return (x_naive,)


@app.cell
def _(np, x_naive, x_tls):
    x_tls - np.ravel(x_naive)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ### check shapes with bogus problem
    (add another constituent "blah")
    """)
    return


@app.cell
def _(
    WineBlendingProb,
    blend_fprints,
    constrained_tls_soln,
    np,
    pv_fprints,
    sigmas,
):
    bogus_pv_fprints = pv_fprints.copy()
    bogus_pv_fprints["blah [g/L]"] = np.random.rand(3)

    bogus_blend_fprint = blend_fprints.loc[0].copy()
    bogus_blend_fprint["blah [g/L]"] = np.random.rand()

    bogus_sigmas = sigmas.copy()
    bogus_sigmas["blah [g/L]"] = 1.2

    bogus_blending_prob = WineBlendingProb(
        bogus_pv_fprints, bogus_blend_fprint, bogus_sigmas
    )

    constrained_tls_soln(bogus_blending_prob, run_checks=True) # it runs checks
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## uncertainty quantification
    """)
    return


@app.cell
def _(WineBlendingProb, np):
    def sample_perturbed_problem(blending_prob):
        # copy these b/c we'll change them
        pv_fprints = blending_prob.pv_fprints.copy()
        blend_fprint         = blending_prob.blend_fprint.copy()

        wines = pv_fprints.index
        constituents = pv_fprints.columns

        for constituent in constituents:
            sigma = blending_prob.sigmas[constituent]

            # perturb wine fingerprint
            for wine in wines:
                delta = sigma * np.random.normal()
                pv_fprints.loc[wine, constituent] += delta

            # perturb blend fingerprint
            delta = sigma * np.random.normal()
            blend_fprint.loc[constituent] += delta

        return WineBlendingProb(
            pv_fprints,
            blend_fprint,
            blending_prob.sigmas
        )

    return (sample_perturbed_problem,)


@app.cell
def _(blending_prob, np, sample_perturbed_problem):
    perturbed_prob = sample_perturbed_problem(blending_prob)

    # check blend fingerprint changed
    delta_blend = perturbed_prob.blend_fprint - blending_prob.blend_fprint
    assert delta_blend.is_unique
    assert np.all(delta_blend.values != 0.0)

    # check pure variety fingerprints changed
    delta_pv = perturbed_prob.pv_fprints.values - blending_prob.pv_fprints.values
    assert np.unique(delta_pv).size == delta_pv.size
    assert np.all(delta_pv != 0.0)

    # see perturbations; run again to see they change.
    delta_pv, delta_blend
    return


@app.cell
def _(constrained_tls_soln, sample_perturbed_problem):
    def sample_perturbed_solutions(
        blending_prob,
        n_samples=100, use_naive_method=False
    ):
        x_tlss = []

        for s in range(n_samples):
            perturbed_prob = sample_perturbed_problem(blending_prob)

            x_tls, _ = constrained_tls_soln(
                perturbed_prob,
                use_naive_method=use_naive_method
            )

            x_tlss.append(x_tls)

        return x_tlss

    return (sample_perturbed_solutions,)


@app.cell
def _(blending_prob, sample_perturbed_solutions):
    perturbed_x_tlss = sample_perturbed_solutions(
        blending_prob, n_samples=1000
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
def _():
    # fails for any blend where we see x<0 and need to resolve.
    # assert np.allclose(
    #     np.mean(perturbed_x_tlss, axis=0),
    #     x_tls, 
    #     atol=0.01
    # )
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

        levels = np.linspace(0.0, rho_max, n)
        for i in range(1, n):
            # fraction of density above this value
            f_density = np.sum(rho[rho > levels[-i]]) / rho_sum
            if f_density > alpha:
                return levels[-i]

    get_level(my_density, 0.95)
    return (get_level,)


@app.cell
def _(mo):
    checkbox_kde = mo.ui.checkbox(label="include KDE?")
    checkbox_kde
    return (checkbox_kde,)


@app.cell
def _(
    Line2D,
    MultipleLocator,
    WineBlendingProb,
    blend_fprints,
    blend_makeup,
    checkbox_kde,
    constrained_tls_soln,
    do_trinary_kde,
    get_level,
    mpl,
    np,
    plt,
    pv_fprints,
    sample_perturbed_solutions,
    sigmas,
    wine_to_color,
):
    def draw_simplex(
        pv_fprints, blend_fprints, sigmas
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

        markersize = 8
        nb_samples = 2500
        # loop over blends
        for blend_id in range(n_blends):
            # set up blending problem
            blending_prob = WineBlendingProb(
                pv_fprints,
                blend_fprints.loc[blend_id],
                sigmas
            )

            ###
            #   uncertainty quantification (predictions)
            ###
            x_tlss = sample_perturbed_solutions(
                blending_prob, 
                n_samples=nb_samples if checkbox_kde.value else 100,
                use_naive_method=False
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

                level = get_level(density, 0.90)

                ax.tricontourf(
                    tn0, tn1, tn2, density,
                    levels=[level, density.max()],
                    colors=[wine_to_color[blend_id]],
                    alpha=0.4,
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
                [blend_makeup.loc[blend_id, wines[0]]],
                [blend_makeup.loc[blend_id, wines[1]]],
                [blend_makeup.loc[blend_id, wines[2]]],
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
            for (w, wine) in enumerate(wines):
                print(f"\t{wine}:")
                print(f"\t\tpred: {np.round(100*x_tls[w], decimals=1):.1f}%")
                print(f"\t\t{alpha}% CI: ")
                print(f"\t\t[{lower[w]}, {upper[w]}]")

            # errors
            errors = np.abs(x_tls - blend_makeup.loc[blend_id, wines].values)
            all_errors.append(errors)
            print(f"\tmean error: {np.round(100*np.mean(errors), decimals=1)}%")

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
                    Line2D([0], [0], marker='o', color='black', label='true',
                           markersize=markersize, linestyle='None'),
                    Line2D([0], [0], marker='^', color='None', label='predicted',
                           markersize=markersize, linestyle='None', markerfacecolor='black'),
                ], 
                fontsize=12
            )
        else:
            blend_handles = [
                mpl.patches.Patch(
                    color=wine_to_color[_blend_id], label=f"{_blend_id+1}"
                ) 
                for _blend_id in range(n_blends)
            ]

            symbol_handles = [
                Line2D(
                    [0], [0], marker='^', color='none', markeredgecolor='black',
                    markerfacecolor='none', markersize=8, label='actual'
                ),
                Line2D(
                    [0], [0], marker='*', color='none', markeredgecolor='black',
                    markerfacecolor='none', markersize=10, label='predicted'
                )
            ]

            leg1 = ax.legend(
                handles=blend_handles, loc='upper left',
                bbox_to_anchor=(0.95, 1.1), borderaxespad=0, title='blend',
                prop={'size': 10}, title_fontsize=12
            )

            leg2 = ax.legend(
                handles=symbol_handles, loc='upper left',
                bbox_to_anchor=(1.01, 1 - (len(blend_handles) + 1) * 0.06),
                borderaxespad=0, prop={'size': 10}
            )

            ax.add_artist(leg1)  # re-add first legend
            # plt.legend(
            #     handles=patches, title="blend", loc='upper right',
            #     prop={'size': 10}, title_fontsize='small',
            #     bbox_to_anchor=(1.05, 1.05)
            # )

        plt.tight_layout()
        _savename = "composition_triangle"
        if checkbox_kde.value:
            _savename += "_kde"
        else:
            _savename += "_scatter"

        plt.savefig(_savename + ".pdf", format="pdf")
        plt.show()

    draw_simplex(pv_fprints, blend_fprints, sigmas)
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
def _(
    WineBlendingProb,
    blend_fprints,
    constrained_tls_soln,
    pd,
    pv_fprints,
    sigmas,
):
    def analyze_corrections(pv_fprints, blend_fprints, sigmas):
        corrections = pd.DataFrame()
        for blend_id in range(blend_fprints.shape[0]):   
            blending_prob = WineBlendingProb(
                pv_fprints,
                blend_fprints.loc[blend_id],
                sigmas
            )

            x_tls, delta_Ab = constrained_tls_soln(blending_prob)

            for c, constituent in enumerate(pv_fprints.columns):
                for w, wine in enumerate(list(pv_fprints.index) + ['blend']):
                    new_row = pd.DataFrame(
                        [
                            {
                                'wine': wine,
                                'blend': blend_id + 1,
                                'constituent': constituent,
                                'Δ': delta_Ab[c, w]
                            }
                        ]
                    )

                    corrections = pd.concat(
                        [corrections, new_row], ignore_index=True
                    )
        return corrections

    corrections = analyze_corrections(pv_fprints, blend_fprints, sigmas)
    corrections.head()
    return (corrections,)


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
def _(plt, sns, wine_to_color):
    def viz_corrections(corrections, sigmas):
        groups = corrections.groupby("constituent")

        constituents = corrections["constituent"].unique()

        fig, axes = plt.subplots(
            1, len(groups), figsize=(3.4 * len(groups), 5)
        )

        palette = {i+1: wine_to_color[i] for i in range(4)}
        markers = ['o', 's', '^', 'D']
        for i, (ax, constituent) in enumerate(zip(axes, constituents)):
            group = groups.get_group(constituent)
            sns.stripplot(
                group, x="wine", y="Δ", hue="blend", ax=ax, dodge=True,
                palette=palette, s=10, marker="X"
            )
            if i != 1:
                ax.get_legend().remove()
            else:
                ax.legend(
                    title="un-blending problem", fontsize=14, title_fontsize=14,
                    ncol=2
                )
            ax.minorticks_off()
            ax.set_ylabel("Δ " + constituent)
            ax.tick_params(axis='x', labelrotation=90)
            ax.axhline(sigmas[constituent], color="gray", linestyle="--")
            ax.axhline(-sigmas[constituent], color="gray", linestyle="--")

            # shade every other wine-group band to visually separate them
            for j in [1, 3]:
                ax.axvspan(
                    j - 0.5, j + 0.5, color="gray", alpha=0.12, zorder=0
                )

        plt.tight_layout()
        plt.savefig("corrections.pdf", format="pdf")
        plt.show()

    return (viz_corrections,)


@app.cell
def _(corrections, sigmas, viz_corrections):
    viz_corrections(corrections, sigmas)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## viz geometry
    """)
    return


@app.cell
def _(pv):
    def draw_vector(pl, x, y, z, color):
        arrow = pv.Arrow(
            start=[0,0,0], direction=[x, y, z], scale="auto",
                tip_length=0.05,      # fraction of total length
                tip_radius=0.01,      # much slimmer
                shaft_radius=0.005,   # thin shaft
                tip_resolution=50
        )
        pl.add_mesh(arrow, color=color)

    return (draw_vector,)


@app.cell
def _(
    build_linear_system,
    constituent_to_ymax,
    constrained_tls_soln,
    draw_vector,
    np,
    pv,
    wine_to_color,
):
    def viz_setup(blending_prob, incl_blend=False):
        # get corrected system
        W, A, b = build_linear_system(blending_prob)
        x_tls, delta_Ab = constrained_tls_soln(blending_prob)
        A_hat = A - delta_Ab[:, :-1]
        b_hat = b - delta_Ab[:, -1].reshape((3, 1))

        pl = pv.Plotter()

        for wine in blending_prob.pv_fprints.index:
            fprint = blending_prob.pv_fprints.loc[wine, :]
            draw_vector(
                pl, fprint[0], fprint[1], fprint[2], wine_to_color[wine]
            )

        if incl_blend:
            blend_id = blending_prob.blend_fprint.name
            draw_vector(
                pl, b[0][0], b[1][0], b[2][0], wine_to_color[blend_id]
            )

        constituents = blending_prob.pv_fprints.columns

        pl.show_bounds(
            xtitle=constituents[0],
            ytitle=constituents[1],
            ztitle=constituents[2],
            n_xlabels=6,
            n_ylabels=5,
            n_zlabels=5,
            font_size=20,
            bold=False,
            location="origin",
            axes_ranges=(
                0, np.round(constituent_to_ymax[constituents[0]]), 
                0, np.round(constituent_to_ymax[constituents[1]]), 
                0, np.round(constituent_to_ymax[constituents[2]])
            )
        )

        pl.show()

    return (viz_setup,)


@app.cell
def _(mo):
    viz_setup_checkbox = mo.ui.checkbox(label="viz 3D setup?", value=False)
    viz_setup_checkbox
    return (viz_setup_checkbox,)


@app.cell
def _(blending_prob, viz_setup, viz_setup_checkbox):
    if viz_setup_checkbox.value:
        viz_setup(blending_prob)
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
def _(pv_fprints):
    pv_fprints["alcohol [vol. %]"].max()
    return


@app.cell
def _(pv_fprints):
    pv_fprints["alcohol [vol. %]"].min()
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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # white wine un-blending
    """)
    return


@app.cell
def _():
    wwb_true_makeup = [
        {"Sauv. Blanc": 0.5}, 
        {"Sauv. Blanc": 0.15}, 
        {"Sauv. Blanc": 0.85}, 
        {"Sauv. Blanc": 0.4}
    ]

    for _entry in wwb_true_makeup:
        _entry["Bordeaux Blanc"] = 1.0 - _entry["Sauv. Blanc"]
    wwb_true_makeup
    return (wwb_true_makeup,)


@app.cell
def _(pd):
    wblend_fprints = pd.DataFrame(
        {
            "alcohol [vol. %]": [12.10, 12.11, 12.10, 12.09],
            "sugar [g/L]": [2.2, 1.02, 3.45, 1.86],
            "acid [g/L]": [6.95, 6.71, 7.78, 6.52]
        }
    )
    wblend_fprints
    return (wblend_fprints,)


@app.cell
def _():
    wconstituent_to_ymax = { # use later
        'alcohol [vol. %]': 13.0,
        'sugar [g/L]': 4.0, 
        'acid [g/L]': 8.0
    }
    return (wconstituent_to_ymax,)


@app.cell
def _(pd):
    wpv_fprints = pd.DataFrame(
           {
            "alcohol [vol. %]": [12.16, 12.08],
            "sugar [g/L]": [0.48, 3.95],
            "acid [g/L]": [6.12, 7.63]
        },
        index=["Bordeaux Blanc", "Sauv. Blanc"]
    )
    wpv_fprints
    return (wpv_fprints,)


@app.cell
def _(viz_features, wblend_fprints, wconstituent_to_ymax, wpv_fprints):
    viz_features(wpv_fprints, wblend_fprints, wconstituent_to_ymax)
    return


@app.cell
def _(WineBlendingProb, sigmas, wblend_fprints, wpv_fprints):
    wwbps = [
        WineBlendingProb(wpv_fprints, wblend_fprints.loc[b, :], sigmas)
        for b in range(4)
    ]
    return (wwbps,)


@app.cell
def _(sample_perturbed_solutions, wwbps):
    wwbp_solns = [sample_perturbed_solutions(wwbp) for wwbp in wwbps]
    return (wwbp_solns,)


@app.cell
def _(plt, sns, wine_to_color, wwb_true_makeup, wwbp_solns, wwbps):
    def viz_white_wine_unblending(wwbps, wwbp_solns):
        wines = wwbps[0].pv_fprints.index
    
        fig, ax = plt.subplots()

        ax.set_xlim(0, 1)
        ax.set_xlabel(wines[0] + " [volume fraction]")

        top = ax.secondary_xaxis(
            "top", functions=(lambda v: 1 - v, lambda v: 1 - v)
        )
        top.set_xlabel(wines[1] + " [volume fraction]")

        ax.set_ylabel("density")
        ax.set_yticks([0])

        for b in range(4):
            # distribution of predicted make-ups
            vs = [x[0] for x in wwbp_solns[b]]

            plt.axvline(
                wwb_true_makeup[b][wines[0]], color=wine_to_color[b], linestyle="--"
            )
            sns.kdeplot(
                vs, ax=ax, clip=(0, 1), fill=True, alpha=0.4, label=f"{b}",
                color=wine_to_color[b]
            )
        
        ax.legend(
            title="blend", loc="center left", bbox_to_anchor=(1.02, 0.5), frameon=False
        )
    
        fig.tight_layout()
        plt.show()

    viz_white_wine_unblending(wwbps, wwbp_solns)
    return (viz_white_wine_unblending,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## with UV/viz spectra
    """)
    return


@app.cell
def _(pd):
    def read_uv_viz_data(sheet_name="Table"):
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
        return out
    
    raw_spec_data = read_uv_viz_data()
    group = raw_spec_data.index.str.replace(r"\s+\d+$", "", regex=True)
    raw_spec_data
    return group, raw_spec_data


@app.cell
def _(group, raw_spec_data):
    spec_data = raw_spec_data.groupby(group).mean()
    spec_data = spec_data[["abs. [280 nm]", "abs. [320 nm]"]]
    spec_data = spec_data.rename(index={"BB": "Bordeaux Blanc", "SB": "Sauv. Blanc"})
    spec_data.index.name = "Name"
    spec_data
    return (spec_data,)


@app.cell
def _(spec_data):
    wpv_fprints_spec = spec_data.loc[["Bordeaux Blanc", "Sauv. Blanc"]]
    wpv_fprints_spec
    return (wpv_fprints_spec,)


@app.cell
def _(blend_fprints):
    blend_fprints
    return


@app.cell
def _(spec_data):
    wblend_fprints_spec = spec_data.loc[["B1", "B2", "B3", "B4"]]
    wblend_fprints_spec
    return (wblend_fprints_spec,)


@app.cell
def _(group, raw_spec_data):
    spec_data_sigma_data = raw_spec_data.groupby(group).std()

    # sigma_spec = spec_data_sigma_data.std().to_dict() # sensitive to outlier
    dev = raw_spec_data - raw_spec_data.groupby(group).transform("median")
    sigma_spec = (1.4826 * dev.abs().median()).to_dict()

    spec_data_sigma_data
    return (sigma_spec,)


@app.cell
def _(sigma_spec):
    sigma_spec
    return


@app.cell
def _(wpv_fprints_spec):
    wpv_fprints_spec
    return


@app.cell
def _(sigma_spec):
    sigma_spec
    return


@app.cell
def _(WineBlendingProb, sigma_spec, wblend_fprints_spec, wpv_fprints_spec):
    wblending_prob_spec = WineBlendingProb(
        wpv_fprints_spec, wblend_fprints_spec.iloc[0], sigma_spec
    )
    return (wblending_prob_spec,)


@app.cell
def _(sample_perturbed_solutions, wblending_prob_spec):
    sample_perturbed_solutions(wblending_prob_spec)
    return


@app.cell
def _(constrained_tls_soln, wblending_prob_spec):
    wx_tls_spec, wdelta_Ab_spec = constrained_tls_soln(
        wblending_prob_spec, verbose=True, run_checks=True
    )
    wx_tls_spec
    return


@app.cell
def _(WineBlendingProb, sigma_spec, wblend_fprints_spec, wpv_fprints_spec):
    wwbps_spec = [
        WineBlendingProb(wpv_fprints_spec, wblend_fprints_spec.iloc[b], sigma_spec)
        for b in range(4)
    ]
    return (wwbps_spec,)


@app.cell
def _(sample_perturbed_solutions, wwbps_spec):
    wwbp_solns_spec = [sample_perturbed_solutions(wwbp) for wwbp in wwbps_spec]
    return (wwbp_solns_spec,)


@app.cell
def _(viz_white_wine_unblending, wwbp_solns_spec, wwbps_spec):
    viz_white_wine_unblending(wwbps_spec, wwbp_solns_spec)
    return


if __name__ == "__main__":
    app.run()
