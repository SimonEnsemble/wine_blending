import marimo

__generated_with = "0.20.4"
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
    from scipy.special import expit as inv_logit, logit
    from dataclasses import dataclass
    from scipy.stats import dirichlet
    import pocomc as pc
    import scipy
    import mpltern
    from scipy.stats import gaussian_kde
    from aquarel import load_theme
    from dataclasses import dataclass

    theme = load_theme("umbra_light")
    theme.apply()
    plt.rc('font', size=16)
    return (
        MultipleLocator,
        dirichlet,
        gaussian_kde,
        inv_logit,
        logit,
        mo,
        np,
        pc,
        pd,
        plt,
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
    return (blend_makeup,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # prior

    ## stick-breaking
    first illustrate stick breaking. see [here](https://mc-stan.org/docs/2_18/reference-manual/simplex-transform-section.html).
    """)
    return


@app.cell
def _(dirichlet, inv_logit, logit, np):
    class DirichletStickbreakPrior:
        """
        Dirichlet prior on x
        via stick-breaking. see:
        https://mc-stan.org/docs/2_18/reference-manual/simplex-transform-section.html
        """
        def __init__(self, alpha):
            # the alpha expresses our prior over x
            self.alpha = np.asarray(alpha)
            # number of components
            self.K = len(alpha)
            # dimension of inference space over y.
            self.dim = self.K - 1
            # y is boundless! that's the point of this.
            self.bounds = np.array([[-np.inf, np.inf]] * self.dim)

        def _stickbreak_forward(self, y):
            """
            inference variable is y, to represent x
            y: (n, K-1) -> x: (n, K), z: (n, K-1) 
            [z returned for reuse in logpdf]
            """
            y = np.atleast_2d(y)
        
            n = y.shape[0] # number of points
            assert y.shape[1] == self.K - 1
        
            # pre-allocate x and z that we want
            x = np.zeros((n, self.K))
            z = np.zeros((n, self.K - 1))
        
            # pre-allocate remaining length of stick
            remaining = np.ones(n)

            # break dat stick!
            for k in range(self.K - 1):
                c_k = np.log(1.0 / (self.K - (k + 1)))
                z[:, k] = inv_logit(y[:, k] + c_k)
                x[:, k] = remaining * z[:, k]
                remaining = remaining - x[:, k]
            x[:, self.K - 1] = remaining
        
            return x, z

        def _stickbreak_inverse(self, x):
            """
            x: (n, K) on simplex -> y: (n, K-1) unconstrained
            """
            x = np.atleast_2d(x)
        
            n = x.shape[0] # number of points

            # pre-allocate y that we want
            y = np.zeros((n, self.K - 1))

            # pre-allocate remaining length of stick
            remaining = np.ones(n)

            # inverse stick break!
            for k in range(self.K - 1):
                z_k = x[:, k] / remaining
                z_k = np.clip(z_k, 1e-12, 1 - 1e-12)  # numerical safety
                c_k = np.log(1.0 / (self.K - (k + 1)))
                y[:, k] = logit(z_k) - c_k
                remaining = remaining - x[:, k]
            return y

        def logpdf(self, y):
            y = np.atleast_2d(y)
            x, z = self._stickbreak_forward(y)
            logp_x = dirichlet(self.alpha).logpdf(x.T)
            remaining = np.ones(y.shape[0])
            log_jac = np.zeros(y.shape[0])
            for k in range(self.K - 1):
                log_jac += np.log(z[:, k]) + np.log1p(-z[:, k]) + np.log(remaining)
                remaining = remaining - x[:, k]
            return logp_x + log_jac

        def rvs(self, size=1):
            x = np.random.dirichlet(self.alpha, size=size)
            y = self._stickbreak_inverse(x)
            return y if size > 1 else y[0]

    return (DirichletStickbreakPrior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## tests
    """)
    return


@app.cell
def _(DirichletStickbreakPrior):
    prior_toy = DirichletStickbreakPrior([0.5, 2.0, 3.4])
    return (prior_toy,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    for intuition, draw the transformation.
    """)
    return


@app.cell
def _(DirichletStickbreakPrior, np, plt):
    prior_2D = DirichletStickbreakPrior([1, 1])

    plt.figure()
    _n = 100
    _ys = np.linspace(-4.0, 4.0, _n)
    _zs = np.zeros(_n)
    for _i in range(_n):
        _x, _z = prior_2D._stickbreak_forward([_ys[_i]])
        _zs[_i] = _z[0]
    plt.plot(_ys, _zs)
    plt.ylabel("proportion of stick broken first, $z_1$")
    plt.xlabel("new variable, $y_1$")
    plt.title("K=2")
    plt.show()
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    first, check we recover the $y$ with an inverse stick-breaking transform.
    """)
    return


@app.cell
def _(np, prior_toy):
    y_toy = np.array([-3.0, 4.0])
    x_toy, z_toy = prior_toy._stickbreak_forward(y_toy)
    y_reconstructed = prior_toy._stickbreak_inverse(x_toy)

    assert np.allclose(y_reconstructed, y_toy)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    second, check $y=0$ gives the uniform $x$.
    """)
    return


@app.cell
def _(np, prior_toy):
    y_uni = np.zeros(prior_toy.K-1)
    x_uni, z_uni = prior_toy._stickbreak_forward(y_uni)
    assert np.size(x_uni) == prior_toy.K
    assert np.allclose(x_uni, 1/prior_toy.K)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    third, check logpdf normalizes via a MC simulation
    """)
    return


@app.cell
def _(DirichletStickbreakPrior):
    mc_prior = DirichletStickbreakPrior([0.5, 2.0])
    return (mc_prior,)


@app.cell
def _(mc_prior, np):
    # uniform box wide enough that p_Y has decayed to ~0 at the edges
    _lo, _hi = -15.0, 15.0
    box_volume = (_hi - _lo)
    n_samples = 10_000
    y_mc = np.random.uniform(_lo, _hi, size=(n_samples, 1))

    logp = mc_prior.logpdf(y_mc)
    integral_est = np.mean(np.exp(logp)) * box_volume
    assert np.isclose(integral_est, 1.0, atol=0.03)
    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    fourth, check for consistency between the RVs and the PDF.
    """)
    return


@app.cell
def _(np, plt, prior_toy):
    y_draws = prior_toy.rvs(size=200_000) # samples via Dirichlet + inverse transform

    lo, hi = -10.0, 10.0
    # 2D histogram of the samples (normalized to be a density)
    H, xedges, yedges = np.histogram2d(
        y_draws[:, 0], y_draws[:, 1], bins=80, 
        range=[[lo, hi], [lo, hi]], density=True
    )

    # evaluate logpdf on the same grid of bin centers
    xc = 0.5 * (xedges[:-1] + xedges[1:])
    yc = 0.5 * (yedges[:-1] + yedges[1:])
    Xc, Yc = np.meshgrid(xc, yc, indexing="ij")
    grid_pts = np.stack([Xc.ravel(), Yc.ravel()], axis=1)
    logp_grid = prior_toy.logpdf(grid_pts).reshape(Xc.shape)
    p_grid = np.exp(logp_grid)

    fig, axes = plt.subplots(1, 2, figsize=(10, 4.5))
    axes[0].imshow(H.T, origin="lower", extent=[lo, hi, lo, hi], aspect="auto")
    axes[0].set_title("Empirical density (rvs)")
    axes[1].imshow(p_grid.T, origin="lower", extent=[lo, hi, lo, hi], aspect="auto")
    axes[1].set_title("logpdf, exponentiated")
    plt.tight_layout()
    plt.show()

    return


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ## set the real prior
    """)
    return


@app.cell
def _(DirichletStickbreakPrior, np):
    alpha = np.array([1.0, 1.0, 1.0])
    prior = DirichletStickbreakPrior(alpha)
    return (prior,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    sample from it
    """)
    return


@app.cell
def _(prior):
    ys_prior = prior.rvs(500)
    xs_prior = prior._stickbreak_forward(ys_prior)[0]
    return (xs_prior,)


@app.cell
def _(blend_fprints, draw_simplex, pv_fprints, xs_prior):
    draw_simplex(xs_prior, pv_fprints, blend_fprints)
    return


@app.cell
def _(MultipleLocator, gaussian_kde, np, plt):
    def draw_simplex(
        xs, pv_fprints, blend_fprints, b_true=None, 
        do_kde=False, do_scatter=True
    ):
        wines = pv_fprints.index
        n_blends = blend_fprints.shape[0]

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

        if do_scatter:
            ax.scatter(
                xs[:, 0], xs[:, 1], xs[:, 2], 
                marker="o", #color=wine_to_color[blend_id], alpha=0.1,
                s=25
            )

        # KDE
        if do_kde:
            grid_n = 500
            kde = gaussian_kde(xs[:, :2].T)
            t_vals = np.linspace(0, 1, grid_n)
            l_vals = np.linspace(0, 1, grid_n)
            T, L = np.meshgrid(t_vals, l_vals)
            R = 1 - T - L
            mask = R >= 0
            T_flat, L_flat, R_flat = T[mask], L[mask], R[mask]
    
            density = kde(np.vstack([T_flat, L_flat]))

            density_at_samples = kde(xs[:, :2].T)

            for alpha, hdr_mass in zip([0.5, 0.7, 0.9], [0.9, 0.8, 0.7]):
                hdr_threshold = np.quantile(density_at_samples, 1 - hdr_mass)
    
                ax.tricontourf(
                    T_flat, L_flat, R_flat, density,
                    cmap="Greens", levels=[hdr_threshold, density.max()],
                    alpha=alpha
                )

        if b_true is not None:
            ax.scatter(
                [b_true[0]], [b_true[1]], [b_true[2]], 
                marker="+", #color=wine_to_color[blend_id], alpha=0.1,
                s=35, color="white"
            )

        plt.show()

    return (draw_simplex,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # likelihood
    """)
    return


@app.cell
def _():
    blend_id = 1
    return (blend_id,)


@app.cell
def _(pv_fprints):
    A = pv_fprints.values.T
    A
    return (A,)


@app.cell
def _(blend_fprints, blend_id):
    b = blend_fprints.iloc[blend_id, :].values
    b
    return (b,)


@app.cell
def _(blend_id, blend_makeup):
    b_true = blend_makeup.iloc[blend_id, :].values
    b_true
    return (b_true,)


@app.cell
def _(blend_fprints):
    blend_fprints.columns
    return


@app.cell
def _(blend_fprints, np, sigmas):
    sigs = np.array([sigmas[c] for c in blend_fprints.columns])
    sigs
    return (sigs,)


@app.cell
def _(A, b, np, prior, sigs):
    def log_likelihood(y):
        y = np.atleast_2d(y)                      # (n_walkers, K-1)

        # stick-breaking transform to get x
        x, _ = prior._stickbreak_forward(y)       # (n_walkers, K)

        # predicted blend fingerprint
        b_pred = x @ A.T                          # (n_walkers, m)

        # residual
        resid = b_pred - b                       # (n_walkers, m)

        # log likelihood
        loglike = -0.5 * np.sum((resid / sigs) **2, axis=-1)
    
        return loglike

    return (log_likelihood,)


@app.cell
def _(log_likelihood, np):
    log_likelihood(np.array([-3.0, 5.0]))
    return


@app.cell
def _(log_likelihood, pc, prior):
    sampler = pc.Sampler(
        prior=prior,
        likelihood=log_likelihood,
        vectorize=True
    )
    sampler.run()
    return (sampler,)


@app.cell
def _(prior, sampler):
    y_samples, _, _, _ = sampler.posterior()
    x_samples, _ = prior._stickbreak_forward(y_samples)
    return (x_samples,)


@app.cell
def _(np, x_samples):
    assert np.allclose(np.sum(x_samples, axis=1), 1.0)
    return


@app.cell
def _(b_true, blend_fprints, draw_simplex, pv_fprints, x_samples):
    draw_simplex(
        x_samples, pv_fprints, blend_fprints, 
        b_true=b_true, do_scatter=False, do_kde=True
    )
    return


if __name__ == "__main__":
    app.run()
