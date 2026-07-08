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
    from aquarel import load_theme
    from dataclasses import dataclass

    theme = load_theme("umbra_light")
    theme.apply()
    plt.rc('font', size=16)
    return MultipleLocator, dirichlet, inv_logit, logit, mo, np, pd, plt, sns


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # prior

    ## stick-breaking
    first illustrate stick breaking. see [here](https://mc-stan.org/docs/2_18/reference-manual/simplex-transform-section.html).


    $x$ is what we want to construct under the constraint $\sum x_i = 1$ and $x_i>0$ for all $i$.

    we have a prior $f(x)$.

    we instead use $y$ that can vary freely, no constraint. we write $z(y)$ and $x(z)$. need Jacobian to get $f(y)$ we impose.
    """)
    return


@app.cell
def _(inv_logit, np):
    def z_of_y(y):
        """
        returns z, proportion of stick broken off in step 1 <= k < K.
        y: the inference variable that is in (-inf, inf)
        """
        K = np.size(y) + 1
        return inv_logit(y + np.log(1.0 / (K - np.arange(1, K))))

    return (z_of_y,)


@app.cell
def _(np):
    def x_of_z(z):
        """
        returns x, the blend make-up
        z: proportion of stick broken off in each step
        """
        K = np.size(z) + 1
        x = np.zeros(K)
        for i in range(K-1):
            x[i] = z[i] * (1.0 - np.sum(x[:i]))
        x[-1] = 1.0 - sum(x[:-1])
        return x

    return (x_of_z,)


@app.cell
def _(x_of_z, z_of_y):
    def xz_of_y(y):
        """
        stick-breaking transform.
        return intermediate variable z too since useful for computations.
        """
        z = z_of_y(y)
        x = x_of_z(z)
        return x, z

    return (xz_of_y,)


@app.cell
def _(np):
    def z_of_x(x):
        K = np.size(x)
        z = np.zeros(K-1)
        for i in range(K-1):
            z[i] = x[i] / (1.0 - np.sum(x[:i]))
        return z

    return (z_of_x,)


@app.cell
def _(logit, np):
    def y_of_z(z):
        K = np.size(z) + 1
        return logit(z) - np.log(1.0 / (K - np.arange(1, K)))

    return (y_of_z,)


@app.cell
def _(y_of_z, z_of_x):
    def y_of_x(x):
        """
        inverse stick-breaking transform.
        """
        z = z_of_x(x)
        y = y_of_z(z)
        return y

    return (y_of_x,)


@app.cell
def _(np):
    toy_y = np.array([-3.0, 4.0])
    toy_y
    return (toy_y,)


@app.cell
def _(toy_y, z_of_y):
    toy_z = z_of_y(toy_y)
    toy_z
    return (toy_z,)


@app.cell
def _(np):
    x = np.ones(3)
    x[:3]
    return


@app.cell
def _(toy_x):
    toy_x[:-1]
    return


@app.cell
def _(toy_z, x_of_z):
    toy_x = x_of_z(toy_z)
    toy_x
    return (toy_x,)


@app.cell
def _(np, toy_x, toy_y, y_of_x):
    # test inverse transform
    assert np.allclose(y_of_x(toy_x), toy_y)
    return


@app.cell
def _(np, toy_x, toy_z):
    assert toy_x[0] == toy_z[0]
    assert np.isclose(1.0, np.sum(toy_x))
    return


@app.cell
def _(np, xz_of_y):
    uniform_x, uniform_z = xz_of_y(np.zeros(4))
    assert np.size(uniform_x) == 5
    assert np.allclose(uniform_x, 1/5)
    return


@app.cell
def _(np, z_of_y):
    z_test = z_of_y(np.array([0, 0]))
    assert z_test[0] == 1 / 3
    assert z_test[1] == 1 / 2 # see stan docs
    z_test
    return


@app.cell
def _(np, plt, x_of_z, z_of_y):
    plt.figure()
    _ys = np.linspace(-4.0, 4.0, 100)
    _zs = [z_of_y([y0])[0] for y0 in _ys]
    _xs = [x_of_z([z0]) for z0 in _zs]
    plt.plot(_ys, _zs)
    plt.ylabel("proportion of stick broken first, $z_1$")
    plt.xlabel("new variable, $y_1$")
    plt.title("K=2")
    plt.show()
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


@app.cell
def _(DirichletStickbreakPrior, np, xz_of_y):
    alpha = np.array([1.0, 1.0, 1.0])
    prior = DirichletStickbreakPrior(alpha)
    prior.logpdf(np.array([4.0, 5.0]))
    ys_prior = prior.rvs(250)
    xs_prior = [xz_of_y(yi)[0] for yi in ys_prior]
    return (xs_prior,)


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
