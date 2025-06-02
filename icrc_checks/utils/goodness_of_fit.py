import os
import glob
import yaml
import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl
# from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
from .plot_utils import savefig
from .say_paper_llhs import LEff

import scipy
from scipy.special import loggamma
from scipy.stats import chi2

from NNMFit import AnalysisConfig, NNMFitter
from NNMFit.utilities.readout_graphs import HistogramGraph
from NNMFit.utilities import load_pickle
# from utils.GOF import , computeLEff,  calculate_indiv_sat_llh_values,calculate_indiv_sat_llh_values_DC
# from utils.GOF import calculate_sat_llh_values as calculate_sat_llh_values_ashwathi
from functools import cache


def calculate_chi2(d, mu, ssq, include_sigma_mc=True):
    """
    Calculate chi2 for binned quantities
    \chi^2 = \sum_i \frac{(d_i - \mu_i)^2}{\mu_i + \sigma_{MC}^2}

    Parameters
    ----------
    d : np.array
        data counts
    mu : np.array
        model prediction in each bin
    ssq : np.array
        mc statistical uncertainty per bin
    include_sigma_mc : bool, optional
        include mc statistical uncertainty in calculation, by default True

    Returns
    -------
    float
        chi^2 value
    """
    nom = (d - mu)**2
    denom = mu
    if include_sigma_mc:
        denom += ssq
    return np.sum(nom / denom)


def binwise_saturated_llh(data, mu, ssq, type='naive_say'):
    """
    logarithm of L_eff(\hat{\mu}, \sigma^2 | d) for one bin

    Parameters
    ----------
    data : np.array
        data count in each bin
    mu : np.array
        MC expecation in each bin
    ssq : np.array
        MC fluctutation in each bin
    """

    if type in ['naive_say', "say_simple"]:
        return float(LEff(data, data, ssq))

    elif type == 'poisson':
        return float(LEff(data, data, 0.))

    elif type == 'say':
        mu_hat = say_llh_find_best_mu(data, ssq)
        return float(LEff(data, mu_hat, ssq))

@cache
def say_llh_find_best_mu(data, ssq):

    # for one bin
    # find \hat{\mu}, maximizing LEff for given k and \sigma^2
    def say_minfunc(a):
        return -LEff(data, a[0], ssq)

    res = scipy.optimize.minimize(say_minfunc, x0=[data / 2.])

    # print(f"Using k={data} and ssq={ssq} found \hat(\mu)={res.x}")
    return float(res.x)


def binwise_saturated_poisson_llh(data):
    """
    Calculate saturated Poisson LLH, corresponding Poisson(d|d) per bin

    Parameters
    ----------
    data : np.array
        data counts per bin
    """

    return data * (-1. + np.log(data + 1E-300)) - loggamma(data + 1.0).real


def get_fit_results_restrict_astro(results, range_norm=(), range_gamma=()):
    """
    Helper function
    Returns one pseudoexp result (parameter dict) that falls into a range
    of fitted gamma_astro and astro_norm
    """

    mask_norm = np.where(
        np.logical_and(
            range_norm[0] < np.array(results['astro_norm']),
            np.array(results['astro_norm']) < range_norm[1]
        )
    )
    mask_gamma = np.where(
        np.logical_and(
            range_gamma[0] < np.array(results['gamma_astro']),
            np.array(results['gamma_astro']) < range_gamma[1]
        )
    )

    sect = np.intersect1d(mask_gamma, mask_norm)

    # pick first pseudoexp result fulfilling criterion
    # order is random, anyways
    idx = sect[0]

    res = {}
    for p in results['fit_params']:
        res[p] = float(results[p][idx])

    return res


def make_fit_config_for_gof_cscd(output_dir, input_params, llh='poisson'):
    """
    Config for fitting pseudodata generated from same MC that is also
    used for fittig (cscd only fit): Gradient_on_baseline
    """

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # setup config hdl
    config_hdl = AnalysisConfig.from_configs(
        main_config_file="main_datasets_rnaab.cfg",
        analysis_config_file=
        "/home/rnaab/analysis/configs_Globalfit/asimov_Cscd_Poisson.yaml",
        override_config_files=[
            "override/Cscd_FullSample_baseline.cfg",
            "override/SnowStorm_v2_Gradient_Histogram_external.cfg",
            "override/SnowStorm_v2_use_Gradient_Histogram.cfg"
        ],
        override_dict=None,
        config_dir="/home/rnaab/analysis/configs_Globalfit/"
    )

    # convert to dict
    config_dict = config_hdl.to_dict()

    # set analysis type to pseudoexp
    # not using "custom dataset": generate asimov set first, then do pseudoexp from this
    config_dict["analysis"]["analysis_type"] = "pseudoexp"

    # pick likelihood to use in fitting
    if llh == 'poisson':
        config_dict["analysis"]["llh"] = "PoissonLLH"
        config_dict["analysis"]["pseudoexp_type"] = "poisson"
    elif llh == 'say':
        config_dict["analysis"]["llh"] = "SAYLLH"
        config_dict["analysis"]["pseudoexp_type"] = "say"

    # set input_params to test specific point in parameter space
    config_dict["analysis"]["input_params"] = input_params

    outfile = os.path.join(output_dir, "Fit_Configuration.yaml")

    if not os.path.isfile(outfile):
        # store in FitConfiguration file in output_dir
        with open(outfile, "w") as f:
            yaml.dump(config_dict, f)
            print(f'Results written to {outfile}')
    else:
        print(f"File {outfile} already exists, do not overwrite")


def make_fit_config_for_gof_tracks(output_dir, input_params, llh='poisson'):
    """
    Config for fitting pseudodata generated from same MC that is also
    used for fittig (tracks only fit): Gradient_on_baseline
    """

    if not os.path.isdir(output_dir):
        os.mkdir(output_dir)

    # setup config hdl
    config_hdl = AnalysisConfig.from_configs(
        main_config_file="main.cfg",  # read from eganster's datasets
        analysis_config_file=
        "/home/rnaab/analysis/configs_Globalfit/asimov_Tracks_Poisson.yaml",
        override_config_files=[
            "override/Tracks_SparseBinning.cfg",
            "override/no_muon_template.cfg",
            "override/Tracks_FullSample_baseline.cfg",
            "override/SnowStorm_v2_Gradient_Histogram_external.cfg",
            "override/SnowStorm_v2_use_Gradient_Histogram.cfg"
        ],
        override_dict=None,
        config_dir="/home/rnaab/analysis/configs_Globalfit/"
    )

    # convert to dict
    config_dict = config_hdl.to_dict()

    # set analysis type to pseudoexp
    # not using "custom dataset": generate asimov set first, then do pseudoexp from this
    config_dict["analysis"]["analysis_type"] = "pseudoexp"

    # pick likelihood to use in fitting
    if llh == 'poisson':
        config_dict["analysis"]["llh"] = "PoissonLLH"
        config_dict["analysis"]["pseudoexp_type"] = "poisson"
    elif llh == 'say':
        config_dict["analysis"]["llh"] = "SAYLLH"
        config_dict["analysis"]["pseudoexp_type"] = "say"

    # set input_params to test specific point in parameter space
    config_dict["analysis"]["input_params"] = input_params

    outfile = os.path.join(output_dir, "Fit_Configuration.yaml")

    if not os.path.isfile(outfile):
        # store in FitConfiguration file in output_dir
        with open(outfile, "w") as f:
            yaml.dump(config_dict, f)
            print(f'Results written to {outfile}')
    else:
        print(f"File {outfile} already exists, do not overwrite")


def pseudoexp_result_dict(results, index):
    """
    Convenience function
    """

    res = {}
    for p in results['fit_params']:
        res[p] = results[p][index]

    return res


def get_sat_llh_values(
    res_dir,
    llh_values,
    datahists,
    bestfit_param_values,
    llh_type,
    say_sat_llh_type,
    graph_name="Precalculated_Graph.pickle",
    calculate_marginalTS = False,
    marginalTS_variable = None,
    det_config=None
):
    """
    Helper function to calculate saturated likelihood value

    Parameters
    ----------
    res_dir : str
        directory containing precalculated graph for fit
    llh_values : list
        llh values of each fit
    datahists : list
        data in each fit
    bestfit_param_values : list
        dict(param_name->value) for best fit
    llh_type : str
        type of likelihood that is used
    say_sat_llh_type : str
        type of saturated likelihood to use for the say likelihood
    graph_name : str
        file name of the precalculated graph, by default "Precalculated_Graph.pickle"

    Returns
    -------
    list
        -2 * (L_{best fit} - L{saturated})

    Raises
    ------
    NotImplementedError
        for unkown say_sat_llh_type
    """

    if llh_type == 'say' and say_sat_llh_type != 'poisson':
        # initialize histogram graph to access \mu and \ssq

        # this assumes that there is a precalculated graph in the same directory
        precalc_file = os.path.join(res_dir, graph_name)
        graph = HistogramGraph.from_precalculated_file(precalc_file)
    else:
        graph = None
    print('----------------')
    print(det_config)
    ts_vals = calculate_sat_llh_values(
        graph, llh_values, datahists, bestfit_param_values, llh_type,
        say_sat_llh_type,calculate_marginalTS = calculate_marginalTS,
    marginalTS_variable = marginalTS_variable,det_config=det_config
    )

    return ts_vals


def calculate_marginalTS_double(variable,dataset):
    
    if variable == 'reco_length':
        k = dataset.reshape(23,20,13).sum(axis=(0,2))
    elif variable == 'reco_energy':
        k = dataset.reshape(23,20,13).sum(axis=(1,2))
    elif variable == 'eratio':
        k = dataset.reshape(23,20,13).sum(axis=(1,0))
    else:
        raise NotImplementedError(
                    f'variable of type {variable} not implemented with this detector configuration!'
                )
    return k   

def calculate_marginalTS_nondouble(variable,dataset):
    
    
    if variable == 'reco_energy':
        k = dataset.reshape(23,10).sum(axis=(1))
    elif variable == 'reco_zenith':
        k = dataset.reshape(23,10).sum(axis=(0))
    else:
        raise NotImplementedError(
                    f'variable of type {variable} not implemented with this detector configuration!'
                )
    return k   
    
def calculate_sat_llh_values(
    graph,
    llh_values,
    datahists,
    bestfit_param_values,
    llh_type,
    say_sat_llh_type,
    det_config=None,
    calculate_marginalTS = False,
    marginalTS_variable = None,
    
):
    """
    Calculate saturated llh values for several fits
    Can be used for different cases
    (saturated Poisson & various saturated SAY likelihoods)

    Parameters
    ----------
    graph : NNMFit.HistogramGraph
        histogram graph of the analysis
    llh_values : list
        -llh value of each fit
    datahists : list
        dict(det_conf -> data_counts) of each fit
    bestfit_param_values : dict
        best-fit parameter values
    llh_type : str
        'poisson' or 'say'
    say_sat_llh_type : 'str'
        'poisson', 'naive_say', 'say_simple' or 'say'
    det_config = 'str'
        if not None, TS per detector config of combined fit shall be calculated 
    calculate_marginalTS = 'bool'
        if True, TS per detector config per hist dimension shall be calculated 
        if True det_config should not be None
    marginalTS_variable = 'str'
        specify for which analysis variable TS should be claculated 
        
    Returns
    -------
    list
        ts value of each fit

    Raises
    ------
    NotImplementedError
        for unknown `say_sat_llh_type`
    """

    ts_vals = []
       
    for i, llh in enumerate(llh_values):
        if llh_type == 'poisson':
            # PoissonLLH in NNMFit has saturated term substracted by default
            ts_vals.append(2 * float(llh))
        elif llh_type == 'say':
            sat_llh = 0

            if say_sat_llh_type == 'poisson':
                # substract saturated term if SAYLLH was used
                # falling back to log(Poisson(n,n)), i.e. choosing ssq=0. for now
                # TODO: is this the right choice?

                for det_conf, dataset in datahists[i].items():
                        
                        if det_config is not None:
                            print(f'per detector config GOF requested,for {det_config}')
                            if det_conf==det_config:
                                if calculate_marginalTS:
                                    print(f'1D marginal GOF requested, for {det_config} and for variable                                                                  {marginalTS_variable}')
                                    
                                    if det_conf=='IC86_pass2_SnowStorm_v2_Bfr_DoubleCascades':
                                            print(f'Calculating GOF for detcteor config {det_conf} for variable {marginalTS_variable}')
                                            k = calculate_marginalTS_double(variable = marginalTS_variable,
                                                                                      dataset=dataset)
                                            sat_llh += np.sum(
                                                binwise_saturated_poisson_llh(np.array(k)))
                                    elif det_conf=='IC86_pass2_SnowStorm_v2_Bfr_Cascades':
                                            print(f'Calculating GOF for detcteor config {det_conf} for variable {marginalTS_variable}')
                                            k = calculate_marginalTS_nondouble(variable = marginalTS_variable,
                                                                                      dataset=dataset)
                                            sat_llh += np.sum(
                                                binwise_saturated_poisson_llh(np.array(k)))
                                    elif det_conf=='IC86_pass2_SnowStorm_v2_Bfr_Tracks':
                                            print(f'Calculating GOF for detcteor config {det_conf} for variable {marginalTS_variable}')
                                            k = calculate_marginalTS_nondouble(variable = marginalTS_variable,
                                                                                      dataset=dataset)
                                            sat_llh += np.sum(
                                                binwise_saturated_poisson_llh(np.array(k)))
                                    
                                else:
                                    print(f'Calculating GOF for detcteor config {det_conf}')
                                    sat_llh += np.sum(
                                            binwise_saturated_poisson_llh(np.array(dataset)))

                            else:
                                continue
                        else:
                            print('Calculating overall GOF')
                            sat_llh += np.sum(
                                    binwise_saturated_poisson_llh(np.array(dataset)))

            elif say_sat_llh_type == 'naive_say' or say_sat_llh_type == 'say':
                # evaluate graph with best fit parameters
                for det_conf, dataset in datahists[i].items():
                    
                    if det_config is not None:
                            print(f'per detector config GOF requested,for {det_config}')
                            if det_conf==det_config:
                                print(f'Calculating GOF for detcteor config {det_conf}')
                                # pseudoexp bestfit
                                res = graph.get_evaled_histogram(
                                    det_config=det_conf,
                                    input_variables=bestfit_param_values[i],
                                    reshape=False
                                )

                                for j, mu in enumerate(res['mu']):
                                    sat_llh += binwise_saturated_llh(
                                        dataset[j], mu, res['ssq'][j], say_sat_llh_type
                                   )
                            else:
                                continue
                    else:
                        res = graph.get_evaled_histogram(
                                    det_config=det_conf,
                                    input_variables=bestfit_param_values[i],
                                    reshape=False
                                )

                        for j, mu in enumerate(res['mu']):
                                    sat_llh += binwise_saturated_llh(
                                        dataset[j], mu, res['ssq'][j], say_sat_llh_type
                                   )
                        
            else:
                raise NotImplementedError(
                    f'type of saturated likelihood {say_sat_llh_type} not implemented'
                )

            # note that result's llh is -Log(likelihood value)!
            ts_vals.append(2 * float(llh + sat_llh))

    return ts_vals


def plot_sat_llh_distribution(
    results,
    llh_type='poisson',
    say_sat_llh_type='poisson',
    override_injected={},
    graph_name="Precalculated_Graph.pickle",
    save=None,
    plot_dir=None,
    fit_chi2=True,
    eval_gof_ndf=None,
    det_config=None
):
    """
    Plot saturated likelihood TS distribution from pseudoexperiments
    Fit a chi^2 distribution if specified
    Evaluate GoF-p_values assuming a chi^2 with certain no. of dof.

    results is a dict that is the output of pseudoexp_old.read_pseudoexp
    """

    injected = dict(results['injected'])
    injected.update(override_injected)

    # make figure
    fig, (ax1) = plt.subplots(1, 1, figsize=(6.0, 4.0))

    # prepare input for ts value calculation
    res_dir = os.path.dirname(results['files'][0])
    llh_values = results['llh']
    datahists = results['datahists']
    bestfit_param_values = []
    for i in range(len(datahists)):
        bestfit_param_values.append(pseudoexp_result_dict(results, i))

    ts_vals = get_sat_llh_values(
        res_dir, llh_values, datahists, bestfit_param_values, llh_type,
        say_sat_llh_type, graph_name,det_config=det_config
    )

    if fit_chi2:
        # fit chi^2 distribution, fixing location and scale
        fit_res = chi2.fit(ts_vals, floc=0., fscale=1.)

        rv = chi2(fit_res[0])

        x = np.linspace(min(ts_vals), max(ts_vals))
        plt.plot(
            x, rv.pdf(x), 'k-', lw=2, label=f'chi^2 (k={fit_res[0]:.2f}) pdf'
        )

    # plot distribution
    hist, bins, _ = plt.hist(
        ts_vals,
        density=True,
        label=f"{len(ts_vals)} pseudoexperiments",
        color='b',
        alpha=0.5
    )

    norm = (len(ts_vals) * (bins[1] - bins[0]))

    bin_centers = bins[:-1] + 0.5 * (bins[1:] - bins[:-1])
    # error on normalized histogram count h^i: d_h^i = sqrt(h^i) / sqrt(norm)
    plt.errorbar(
        bin_centers,
        hist,
        yerr=np.sqrt(hist) / np.sqrt(norm),
        color='b',
        linestyle='None'
    )

    plt.xlabel(r'$-2 \mathrm{log}(\Lambda_\mathrm{sat})$')
    plt.ylabel('pdf')

    if eval_gof_ndf is None:
        # assuming these are not "asimov"-pseudoexperiments
        plt.legend(
            title=rf"Inject $\gamma_{{astro}}$={injected['gamma_astro']:.2f}, "
            + rf"$\Phi_{{astro}}$={injected['astro_norm']:.2f}"
        )

    if (save is not None) and (plot_dir is not None):
        savefig(fig, plot_dir, save)

    plt.show()

    if eval_gof_ndf is not None:
        print(
            f"Using {eval_gof_ndf} degrees of freedom for evaluating the GoF per pseudoexp"
        )
        rv = chi2(eval_gof_ndf)

        p_vals = []
        for ts in ts_vals:
            # print(rv.sf(ts))
            p_vals.append(rv.sf(ts))

        # make figure
        fig, (ax1) = plt.subplots(1, 1, figsize=(6.0, 4.0))

        plt.hist(
            np.log10(p_vals),
            density=True,
            label=f'Evaluated chi^2(k={eval_gof_ndf})'
        )
        plt.xlabel(r'GoF log10($p_\mathrm{value}$)')
        plt.ylabel('pdf')

        plt.legend()

        if (save is not None) and (plot_dir is not None):
            savefig(fig, plot_dir, save + 'eval_gof_pvalue')

        plt.show()


def calculate_TS_datafit(
    fit_res_file, llh_type='poisson', say_sat_llh_type='poisson',
    alt_config_file=None,det_config=None,
    calculate_marginalTS = False,
    marginalTS_variable = None
):
    """
    Get test statistic (L_{best fit} - L{saturated}) for a single fit

    Returns
    -------
    float
        TS value
    """

    fit_res = load_pickle(fit_res_file)

    if alt_config_file is None:
        # read fit config from scan dir
        fit_config_file = os.path.join(
            os.path.dirname(fit_res_file), 'Fit_Configuration.yaml'
        )
        if not os.path.isfile(fit_config_file):
            raise NotImplementedError(
                "Fit_Configuration.yaml must be in same directory as fit result"
            )
    else:
        print(f"Loading alternative config file (and corresponding graph) from {os.path.dirname(alt_config_file)}")
        fit_config_file = alt_config_file

    with open(fit_config_file) as hdl:
        fit_config = yaml.safe_load(hdl)

    # load precalculated graph (from same dir as config file)
    precalc_file = os.path.join(
        os.path.dirname(fit_config_file), 'Precalculated_Graph.pickle'
    )
    if os.path.isfile(precalc_file):
        precalculated = precalc_file
    else:
        precalculated = None

    config_hdl = AnalysisConfig.from_dict(fit_config)
    config_dict = config_hdl.to_dict()

    nnmfitter = NNMFitter(config_hdl, precalculated=precalculated)

    data = nnmfitter.get_data_hists(as_array=True)

    bestfit_values = dict(fit_res['fit-result'][1])
    # take into account fixed paratmeters
    bestfit_values.update(fit_res['fixed-parameters'])

    if alt_config_file is None:
        llh = fit_res['fit-result'][0]
    else:
        # need to evaluate llh for different set of parameters
        print("Setting up minimizer")
        minimizer = nnmfitter.setup_minimizer(
            fixed_pars=bestfit_values, profile=False, randomize_param_seeds=True
        )
        print("Evaluate")
        res, _ = minimizer.evaluate()
        # FIXME: write a better interface which is minimizer independent
        llh = res[0]
    print("Likelihood value:", llh)
    print(det_config)
    ts_val = get_sat_llh_values(
        res_dir=os.path.dirname(fit_config_file),
        llh_values=[llh],
        datahists=[data],
        bestfit_param_values=[bestfit_values],
        llh_type=llh_type,
        say_sat_llh_type=say_sat_llh_type,
        graph_name='Precalculated_Graph.pickle',
        det_config=det_config,
    calculate_marginalTS = calculate_marginalTS,
    marginalTS_variable = marginalTS_variable,
    )[0]

    return ts_val
