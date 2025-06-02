import os
import pandas as pd
import yaml
from copy import deepcopy
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import ConnectionPatch
from NNMFit import AnalysisConfig
from NNMFit.utilities.readout_graphs import HistogramGraph
# also need data histograms from NNMFitter
from NNMFit.core.nnm_fitter import NNMFitter
from NNMFit.utilities import override_dict_entries, load_pickle

from . import matplotlib_setup
from .plot_utils import plot_energy_and_zenith_data_MC,plot_3D_analysisvariables,plot_2D_DC_analysisvariables
from .plot_utils import initialize_figure, plot_data_hist_errorbar, plot_data_ratio_errorbar, plot_hist_errorbar, plot_ratio_errorbar, restrict_ylim_data,plot_2dHist
from .plot_utils import savefig
from .goodness_of_fit import calculate_chi2

import matplotlib.font_manager as font_manager
font_axis_label = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 22,
        }
font_title = {'family': 'serif',
        'color':  'black',
        'weight': 'bold',
        'size': 20,
        }
font_legend = font_manager.FontProperties(family='serif',
                                    weight='normal',
                                    style='normal', size=10)

def plot_2D_data_mc_LvsE(
    mc,
    data,
    bins_energy,
    bins_length,
    plot_settings=dict(),
    plot_title=None,
    plot_dir=None,
    save=None
):
    
    font_axis_label = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 16,
        }
    font_title = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 20,
            }
    font_legend = font_manager.FontProperties(family='serif',
                                        weight='normal',
                                        style='normal', size=4)


    # mc: dict(fit->res)
    mu_3d = {}
    ssq_3d = {}
    data_3d = data
    for fit, res in mc.items():
        mu_3d[fit] = res["mu"]
        ssq_3d[fit] = res["ssq"]
        
        
    
    
    

    
        
    
    
    plt.rcParams.update({'font.family':'serif'})
    
    fig= plt.figure(figsize=(10,6))
    ax = plt.subplot(111)
    for fit in mc.keys():
        
        h = ax.pcolormesh(bins_length, bins_energy,mu_3d[fit],norm = mpl.colors.LogNorm())
    lsoftcut = 20  # 15m according to ~68% background exclusion (where single-like are split from double)
    #lsoftcut = 132
    lhardcut = 10**1.301  # 20m according to ~90% background exclusion (where single-like are split from double)
    lsplitcut = 10**1.602 # 40m according to ~99% background exclusion (where single-like are split from track)
    
    xmin, xmax = 10, 1000
    ymin, ymax = 1000,10**7.2
    le2lowerboundmin = (10, 60000)
    le2lowerboundmax = (1000, 243426.44152138782)
    #le2upperboundmin = (10, 721327.5147613535)
#     le2lowerboundmax = (1000, 100000)
    le2upperboundmin = (10, 720000)
    le2upperboundmax = (1000, 1e7)
    sig_lines = []
    bkg_lines= []
    sig_lines.append([(le2lowerboundmin[0], le2lowerboundmax[0]), (le2lowerboundmin[1]-1000+0.2, le2lowerboundmax[1]-1000+0.2)])
    sig_lines.append([(le2upperboundmin[0], le2upperboundmax[0]), (le2upperboundmin[1]-1000+0.2, le2upperboundmax[1]-1000+0.2)])
    
    
  
    
    bkg_lines.append([(lsoftcut, lsoftcut), (ymin, ymax)])
    
    
    for line in sig_lines:
        ax.plot(line[0], line[1], ls='--',linewidth=1.5, color='white')
    
    for line in bkg_lines:
        ax.plot(line[0], line[1], ls='--',linewidth=1.5, color='midnightblue')
        
    ax.annotate("",
            xy=(110,3000000), xycoords='data',
            xytext=(110,120000), textcoords='data',
            arrowprops=dict(arrowstyle="<->",
                            connectionstyle="arc3", color='w', lw=2),
            )
#     con = ConnectionPatch((10**1.8, le2lowerboundmax[1]+0.2-1000-1.2), (10**1.8, le2upperboundmax[1]+0.2-1000-1.2), 
#                               "data", "data", arrowstyle="<->", shrinkA=5, shrinkB=5, mutation_scale=20, 
#                               color="w", lw=1, fc="w")
#     ax.add_artist(con)
    ax.text(0.56, 0.54, "68%\nSignal Region", transform=ax.transAxes,color='white',fontsize=14)
#     ax.text(0.008, 0.92, "68%", transform=ax.transAxes, color='midnightblue',fontsize=14)
    ax.text(0.02, 0.87, "68%", transform=ax.transAxes, color='midnightblue',fontsize=14)
    ax.text(0.02, 0.75, "Background\nRegion", transform=ax.transAxes, color='midnightblue',fontsize=14)
#     ax.text(0.32, 0.92, "99%", transform=ax.transAxes,color='midnightblue',fontsize=14)
    xcenters = (bins_length[:-1] + bins_length[1:]) / 2
    ycenters = (bins_energy[:-1] + bins_energy[1:]) / 2
#     for i_x in range(len(bins_energy)-1):
#         for i_y  in range(len(bins_length)-1):
#             if int(data_3d[i_x][i_y]) !=0:
    data_l = [11.5,17.3,10.9,96.2,14]
    data_e = [111158.7,97190.4,89116,76867.20,91166.10]
    ax.scatter(data_l,data_e,marker='1',s=160,facecolor='k',edgecolor='k',label='Data events')
                
#                 ax.text(xcenters[i_y],
#                         ycenters[i_x],
#                         int(data_3d[i_x, i_y]),
#                         color='k',
#                         ha='center',
#                         va='center',
#                         fontweight='normal',
#                         fontsize = 12,
#                         )
            # setup figure
    L = [0.05,0.5,5,50,500,5000]
    Length = np.array(L)
    E = [1e3,1e4,1e5,1e6,1e7,1e8]
    Energy = np.array(E)
#     ax.plot(Length,Energy,'w:',label= r'$E_\tau = \frac{1PeV}{50m} L$')
    plt.legend(fontsize=14,loc='upper right')
    ax.set_xlabel("Length [m]",fontdict=font_axis_label)
    ax.set_ylabel("Energy [GeV]",fontdict=font_axis_label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    clb = plt.colorbar(h)
    clb.set_label('Expected Number of events\nin 4268.7 days',rotation=270,labelpad=35,fontdict=font_axis_label)
    clb.ax.tick_params(labelsize=18)
    
    ax.set_xlim(min(bins_length),max(bins_length))
    ax.set_ylim(5e4,1.5e7)

    for item in (ax.get_xticklabels() + ax.get_yticklabels()):
        item.set_fontsize(15)
        item.set_family('serif')
    
    ax.tick_params(axis='both',which='major',width=2,length=4,direction='in')
    ax.tick_params(axis='both',which='minor',width=1,length=2,direction='in')
    
    
    
    
    print(save,plot_dir)
    
    if (save is not None) and (plot_dir is not None):
        savefig(fig, plot_dir, save + 'LvsE_2D_withData')
    plt.show()




def plot_3D_data_mc_LvsE(
    mc,
    data,
    bins_energy,
    bins_length,
    bins_eratio,
    plot_settings=dict(),
    plot_title=None,
    plot_dir=None,
    save=None
):
    
    font_axis_label = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 14,
        }
    font_title = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 20,
            }
    font_legend = font_manager.FontProperties(family='serif',
                                        weight='normal',
                                        style='normal', size=4)


    # mc: dict(fit->res)
    mu_3d = {}
    ssq_3d = {}
    data_3d = data
    for fit, res in mc.items():
        mu_3d[fit] = res["mu"]
        ssq_3d[fit] = res["ssq"]
        neratio_bins = mu_3d[fit].shape[2]  # same for all hists
        
    
    
    

    # energy spectrum for different zenith slices
    for eratio in range(neratio_bins):
        
        #successful_dict = main_dict_Poisson.mask_unsuccessful()
        
    
        
        hist_data = data_3d[:,:,eratio]
        hist_mc, error_mc = {}, {}
        for fit in mc.keys():
            # pick relevant zenith slice
            
            
            hist_mc[fit] = mu_3d[fit][:,:,eratio]
            
            error_mc[fit] = ssq_3d[fit][:,:,eratio]
            

        fig= plt.figure(figsize=(10,6))
        ax = plt.subplot(111)
        for fit in mc.keys():
            
            h = ax.pcolormesh(bins_length, bins_energy,hist_mc[fit],norm = mpl.colors.LogNorm())

        if not np.all(hist_data==0):
            xcenters = (bins_length[:-1] + bins_length[1:]) / 2
            ycenters = (bins_energy[:-1] + bins_energy[1:]) / 2
            for i_x in range(len(bins_energy)-1):
                for i_y  in range(len(bins_length)-1):
                    if int(hist_data[i_x][i_y]) !=0:
                        ax.text(xcenters[i_y],
                                ycenters[i_x],
                                int(hist_data[i_x, i_y]),
                                color='k',
                                ha='center',
                                va='center',
                                fontweight='normal',
                                fontsize = 12,
                               )
            # setup figure
        L = [0.05,0.5,5,50,500,5000]
        Length = np.array(L)
        E = [1e3,1e4,1e5,1e6,1e7,1e8]
        Energy = np.array(E)
        ax.plot(Length,Energy,'w:',label= r'$E_\tau = \frac{1PeV}{50m} L$')
        plt.legend(fontsize=14,loc='upper right',)
        ax.set_xlabel("Length[m]",fontdict=font_axis_label)
        ax.set_ylabel("Energy[GeV]",fontdict=font_axis_label)
        ax.set_xscale("log")
        ax.set_yscale("log")
        clb = plt.colorbar(h)
        clb.set_label('Probability Density',rotation=270,labelpad=20,fontdict=font_axis_label)
        clb.ax.tick_params(labelsize=18)
        
        ax.set_xlim(min(bins_length),max(bins_length))
        ax.set_ylim(5e4,1.5e7)
    
        for item in (ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(15)
            item.set_family('serif')
        
        ax.tick_params(axis='both',which='major',width=2,length=4,direction='in')
        ax.tick_params(axis='both',which='minor',width=1,length=2,direction='in')
        
        # range depending on zenith slice
        

        # save and show plot
        
        ax.set_title(
            f"Eratio in [{bins_eratio[eratio+1]:.2f}, {(bins_eratio[eratio]):.2f}]",
        fontdict=font_title)
        
        
        if (save is not None) and (plot_dir is not None):
            savefig(fig, plot_dir, save + '_LvsE_DataEvents')
        plt.show()



def plot_1D_data_MC_zenith_bands(
    data,
    mc,
    bins_energy,
    bins_zenith,
    det_conf='IC86_pass2_SnowStorm_v2_Bfr_Cascades',
    plot_data_mc=True,
    ratio_base_name=None,
    plot_settings=dict(),
    components=None,
    plot_title=None,
    plot_dir=None,
    save=None
):
    
    
    font_axis_label = {'family': 'serif',
        'color':  'black',
        'weight': 'normal',
        'size': 12,
        }
    font_title = {'family': 'serif',
            'color':  'black',
            'weight': 'bold',
            'size': 15,
            }
    font_legend = font_manager.FontProperties(family='serif',
                                        weight='normal',
                                        style='normal', size=8)
    # mc: dict(fit->res)
    mu_2d = {}
    ssq_2d = {}
    for fit, res in mc.items():
        mu_2d[fit] = res["mu"]
        ssq_2d[fit] = res["ssq"]
        nzen_bins = mu_2d[fit].shape[1]  # same for all hists
    data_2d = data
    data_label = 'data'

    # energy spectrum for different zenith slices
    for zen in range(nzen_bins):

        fig, (ax1, ax2) = initialize_figure(figsize=mpl.rcParams['figure.figsize'])

        if plot_title is not None:
            fig.suptitle(plot_title)

        hist_mc, error_mc = {}, {}
        for fit in mc.keys():
            # pick relevant zenith slice
            hist_mc[fit] = mu_2d[fit][:, zen]
            error_mc[fit] = np.sqrt(ssq_2d[fit][:, zen])

        if plot_data_mc:
            hist_data = data_2d[:, zen]
            error_data = np.sqrt(data_2d[:, zen])

        if ratio_base_name is None:
            if not plot_data_mc:
                raise NotImplementedError("Need to provide name for fit to base the ratio on if no data is plotted!")
            else:
                hist_ratio_base = hist_data
        else:
            hist_ratio_base = hist_mc[ratio_base_name]
        # yerr_ratio_base = yerr_base

        if plot_data_mc:
            plot_data_hist_errorbar(
                ax1,
                hist_data,
                bins_energy,
                error_data,
                label=data_label,
                color='k',
                marker='o'
            )
            plot_data_ratio_errorbar(
                ax2,
                hist_data,
                bins_energy,
                error_data,
                hist_ratio_base,
                yerr_baseline=None,
                color='k',
                marker='o'
            )
        for fit in mc.keys():

            kwargs_add = plot_settings[fit
                                      ] if fit in plot_settings.keys() else dict()

            
            plot_hist_errorbar(
                ax1,
                hist_mc[fit],
                bins_energy,
                error_mc[fit],
                label=fit,
                **kwargs_add
            )
            plot_ratio_errorbar(
                ax2,
                hist_mc[fit],
                bins_energy,
                error_mc[fit],
                hist_ratio_base,
                yerr_baseline=None,
                **kwargs_add
            )
        # optionally plot provided components
        if components is not None:
            for comp, d in components['settings'].items():
                hist = components['hists'][comp]["mu"][:, zen]
                yerror = np.sqrt(components['hists'][comp]["ssq"][:, zen])
                plot_hist_errorbar(ax1, hist, bins_energy, yerror, **d['plot_settings'])

        # setup figure
        plt.xlabel("Reco energy [GeV]",fontdict=font_axis_label)
        ax1.set_ylabel(r"$N_{\mathrm{Events}}$",fontdict=font_axis_label)
        ax1.set_xscale("log")
        ax1.set_yscale("log")
        ax1.set_xlim(bins_energy[0], bins_energy[-1])

        restrict_ylim_data(ax1)

        # range depending on zenith slice
        if np.cos(bins_zenith[zen + 1]) > 0.5:
            ax2.set_ylim(0., 2.)  # straight downgoing, case of very upgoing events, low stats
        elif np.cos(bins_zenith[zen+1]) <= 0.:
            # ax2.set_ylim(0.5, 1.5)  # southern sky
            ax2.set_ylim(0, 2)
        else:
            # ax2.set_ylim(0.25, 1.75)
            ax2.set_ylim(0, 2)
        for item in (ax1.get_xticklabels() + ax1.get_yticklabels()):
            item.set_fontsize(8)
            item.set_family('serif')
        for item in (ax2.get_xticklabels() + ax2.get_yticklabels()):
            item.set_fontsize(8)
            item.set_family('serif')
        ax1.tick_params(axis='both',which='major',width=2,length=8,direction='in')
        ax1.tick_params(axis='both',which='minor',width=1,length=4,direction='in')
        ax2.tick_params(axis='both',which='major',width=2,length=8,direction='in')
        ax2.tick_params(axis='both',which='minor',width=1,length=4,direction='in')
            # save and show plot
        ax1.legend()
        fig.suptitle(
            f"Cos(zen) in [{np.cos(bins_zenith[zen+1]):.2f}, {np.cos(bins_zenith[zen]):.2f}]"
        ,fontdict=font_title)
        
        fig.subplots_adjust(hspace=0.1)
        if (save is not None) and (plot_dir is not None):
            savefig(fig, plot_dir, save + f'_energy_zenith_slice_{zen}')
        plt.show()


def get_components_plot_sets(override_plot_components_dict=None):

    components_plot_dict = {
        
        "astro_allflavor":
            {
                'parameters': ['gamma_astro', 'astro_norm','astro_nue_ratio','astro_nutau_ratio','inel_scale'],
                'plot_settings': {
                    'label': 'Astro - All Flavor',
                    'color': 'C3'
                }
            },
        
       
        
        "conv":
            {
                'parameters':
                    [
                        'conv_norm', 'delta_gamma', 'CR_grad', 'barr_h',
                        'barr_w', 'barr_z', 'barr_y','inel_scale'
                    ],
                'plot_settings': {
                    'label': 'Conventional Atm.',
                    'color': 'C4'
                }
            },
        "prompt":
            {
                'parameters':
                    ['prompt_norm', 'delta_gamma', 'CR_grad','inel_scale'],
                'plot_settings':
                    {
                        'label': 'Prompt Atm.',
                        'color': 'C2',
                        # 'ls': ':'
                    }
            },
        "muon":
            {
                'parameters': ['muongun_norm'],
                'plot_settings': {
                    'label': 'Muongun',
                    'color': 'C1'
                },
                'skip_det_confs':
                    [
                        'IC86_pass2_SnowStorm_v2_Bfr_Cascades',
                        'IC86_pass2_SnowStorm_v2_Bfr_DoubleCascades'
                    ],
            },
        
    }
    if override_plot_components_dict is not None:
        print('overriding...')
        print(override_plot_components_dict)
        override_dict_entries(
            components_plot_dict, override_plot_components_dict
        )
    
    params_all_zero = {
    # atmospheric flux parameters
    'conv_norm': 0.0,
    'prompt_norm': 0.0,
    'delta_gamma': 0.0,
    'CR_grad': 0.0,
    'barr_h': 0.0,
    'barr_w': 0.0,
    'barr_z': 0.0,
    'barr_y': 0.0,

    
    # astro flux: default (generic) SPL
    'astro_norm': 0.0,
    'gamma_astro': 2.87,
    'astro_nue_ratio': 1.0,
    'astro_nutau_ratio': 1.0,
    'inel_scale':0.0,
    'muongun_norm':0.0,
    
# # #     #snow storm parameters
    'dom_eff': 1.0,
    'ice_abs': 1.0,
    'ice_scat': 1.0,
    'ice_aniso' : 1.0,
    'ice_holep0' : -0.27,
    'ice_holep1' : -0.042,
    
}
    
    
    
    return components_plot_dict, params_all_zero


def get_evaled_components(
    det_conf, graph,comp_graph, fit_params, override_plot_components_dict,params_all_zero_override
):
    # create dict(
    #   'settings' -> plot settings per comp,
    #   'hists' -> mu, ssq per comp for corresponding evaluated parameters
    # )
    components = {}

    components_plot_dict, params_all_zero = get_components_plot_sets(
        override_plot_components_dict
    )
    
    components['settings'] = components_plot_dict
    components['hists'] = {}
    comps_to_drop = []
    # params_all_zero
    for name, d in components_plot_dict.items():
        if 'skip_det_confs' in d.keys() and det_conf in d['skip_det_confs']:
            # do not add this component (because it is not applied to this det conf)
            
            print(f"Skipping component {name} for plotting det conf {det_conf}")
            comps_to_drop.append(name)
            continue

        input_vars = dict(
            params_all_zero,
            **{p: fit_params[p]
               for p in d['parameters'] if p in fit_params}
        )
        
        # get hist
#         print('generating histogram for {0} component with following values'.format(name))
#         print(input_vars)
        if comp_graph[name] is not None:
            print('generating histogram for {0} component with following values'.format(name))
            print(input_vars)
            graph = HistogramGraph.from_configdict(comp_graph[name])
            input_vars = dict(
                params_all_zero_override,
                **{p: fit_params[p]
                   for p in d['parameters'] if p in fit_params}
            )
            res_comp = graph.get_evaled_histogram(
                det_config=det_conf,
                input_variables=input_vars,
                debug=False,
                reshape=True
            )
        else:
            print('generating histogram for {0} component with following values'.format(name))
            print(input_vars)
            res_comp = graph.get_evaled_histogram(
                det_config=det_conf,
                input_variables=input_vars,
                debug=False,
                reshape=True
            )
        components['hists'][name] = {
            "mu": res_comp['mu'],
            "ssq": res_comp['ssq']
        }
    # drop unused components from settings
    for drop in comps_to_drop:
        components['settings'].pop(drop)
    
    
    
    return components


def plot_1D_data_MC_comparison(
    graph,
    fit_params,
    all_data,
    ylim_energy_ratio=(0, 2),
    ylim_zenith_ratio=(0, 2),
    ylim_length_ratio=(0, 2),
    ylim_eratio_ratio=(0, 2),
    plot_components=False,
    component_graphdict=None,
    perbin_plots = False,
    override_plot_components_dict=None,
    params_all_zero=None,
    det_conf='IC86_pass2_SnowStorm_v2_cscd_cascade',
    plot_title=None,
    plot_dir=None,
    save=None
):
    
    

        
    # actual plots
    if det_conf == 'IC86_pass2_SnowStorm_v2_Bfr_DoubleCascades':
        # get binning
        temp = graph.get_binning(det_conf)
        
        if len(temp)>2:
            bins_energy = temp["reco_energy"]
            bins_length = temp["reco_length"]
            bins_eratio = temp['eratio']

            shape = (len(bins_energy) - 1, len(bins_length) - 1,len(bins_eratio)-1)

            # get det_conf
            if det_conf is None:
                det_conf = graph.get_detconfig()

            # get data and reshape
            data = all_data[det_conf].reshape(shape)

            # pseudoexp bestfit
            res = graph.get_evaled_histogram(
                det_config=det_conf, input_variables=fit_params, reshape=True
            )

            # MC components
            components = None
            component_graph = None
            if plot_components:
                #component_graph = HistogramGraph.from_configdict(component_graphdict)
                components = get_evaled_components(
                    det_conf,  graph,component_graphdict, fit_params, override_plot_components_dict,params_all_zero
                )
            
            plot_3D_analysisvariables(
                res["mu"],
                res["ssq"],
                data,
                bins_energy,
                bins_length,
                bins_eratio,
                ylim_energy_ratio=ylim_energy_ratio,
                ylim_length_ratio=ylim_length_ratio,
                ylim_eratio_ratio=ylim_eratio_ratio,
                
                
                components=components,
                
                data_label='data',
                plot_title=plot_title,
                plot_dir=plot_dir,
                save=save,
            )

            if perbin_plots:
                plot_3D_data_mc_LvsE(
                        mc={"MC sum": res},
                        data=data,
                        bins_energy=bins_energy,
                        bins_length=bins_length,
                        bins_eratio=bins_eratio,
                        plot_settings=dict(),
                        plot_title=None,
                        plot_dir=plot_dir,
                        save=save
                    )
        else:
            bins_energy = temp["reco_energy"]
            bins_length = temp["reco_length"]
            

            shape = (len(bins_energy) - 1, len(bins_length) - 1)

            # get det_conf
            if det_conf is None:
                det_conf = graph.get_detconfig()

            # get data and reshape
            data = all_data[det_conf].reshape(shape)

            # pseudoexp bestfit
            res = graph.get_evaled_histogram(
                det_config=det_conf, input_variables=fit_params, reshape=True
            )

            # MC components
            components = None
            component_graph = None
            if plot_components:
                #component_graph = HistogramGraph.from_configdict(component_graphdict)
                components = get_evaled_components(
                    det_conf,  graph,component_graphdict, fit_params, override_plot_components_dict,params_all_zero
                )
                
#             plot_2D_DC_analysisvariables(
#                 res["mu"],
#                 res["ssq"],
#                 data,
#                 bins_energy,
#                 bins_length,
                
#                 ylim_energy_ratio=ylim_energy_ratio,
#                 ylim_length_ratio=ylim_length_ratio,
                
#                 components=components,
                
#                 data_label='data',
#                 plot_title=plot_title,
#                 plot_dir=plot_dir,
#                 save=save,
#             )
            if perbin_plots:
                plot_2D_data_mc_LvsE(
                        mc={"MC sum": res},
                        data=data,
                        bins_energy=bins_energy,
                        bins_length=bins_length,
                        plot_settings=dict(),
                        plot_title=None,
                        plot_dir=plot_dir,
                        save=save
                    )
        
        
    else:
        temp = graph.get_binning(det_conf)
    
        bins_energy = temp["reco_energy"]
        bins_zenith = temp["reco_zenith"]
        

        shape = (len(bins_energy) - 1, len(bins_zenith) - 1)

        # get det_conf
        if det_conf is None:
            det_conf = graph.get_detconfig()

        # get data and reshape
        data = all_data[det_conf].reshape(shape)

        # pseudoexp bestfit
        res = graph.get_evaled_histogram(
            det_config=det_conf, input_variables=fit_params, reshape=True
        )

        # MC components
        components = None
        component_graph = None
        if plot_components:
            
            #component_graph = HistogramGraph.from_configdict(component_graphdict)
            components = get_evaled_components(
                det_conf,  graph,component_graphdict, fit_params, override_plot_components_dict,params_all_zero
            )
            
        
        plot_energy_and_zenith_data_MC(
            res["mu"],
            res["ssq"],
            data,
            bins_energy,
            bins_zenith,
            ylim_energy_ratio=ylim_energy_ratio,
            ylim_zenith_ratio=ylim_zenith_ratio,
            components=components,
            
            data_label='data',
            plot_title=plot_title,
            plot_dir=plot_dir,
            save=save,
        )

    #1D distribution for different zenith bands for cascade signal sample
#     if perbin_plots:
#         if det_conf != 'IC86_pass2_SnowStorm_v2_Bfr_DoubleCascades':
#             print(f"Differnt zenith bands for {det_conf}")
#             plot_1D_data_MC_zenith_bands(
#                 data,
#                 {"MC sum": res},
#                 bins_energy,
#                 bins_zenith,
#                 det_conf=det_conf,
#                 ratio_base_name="MC sum",
#                 components=components,
#                 plot_title=plot_title,
#                 plot_dir=plot_dir,
#                 save=save,
#             )


def plot_data_mc_single_fit(
    fit_res_file,
    plot_dir=None,
    plot_name=None,
    from_precalc=False,
    ylim_energy_ratio=(0, 2),
    ylim_zenith_ratio=(0, 2),
    ylim_length_ratio=(0, 2),
    ylim_eratio_ratio=(0, 2),
    params_all_zero = None,
    component_graphdict = None,
    plot_components=False,
    
    print_results=False,
    perbin_plots = False,
    override_plot_components_dict=None,
    plot_title=None,
    override_params={},
):
    
    fit_res = pd.read_pickle(fit_res_file)

    if 'settings' in fit_res.keys():
        fit_config = fit_res['settings']
        
    else:
        # read fit config from scan dir
        fit_config_file = os.path.join(
            os.path.dirname(fit_res_file), 'Fit_Configuration.yaml'
        )
        
        with open(fit_config_file) as hdl:
            fit_config = yaml.safe_load(hdl)
            

    det_confs = fit_config['analysis']['detector_configs']

    if print_results:
        print(fit_res['fit-result'])
        for p, val in fit_res['fit-result'][1].items():
            print(p, np.round(val, 3))
        print(f"LLH {np.round(fit_res['fit-result'][0], 3)}")

    config_hdl = AnalysisConfig.from_dict(fit_config)
    config_dict = config_hdl.to_dict()
    

    if from_precalc:
        precalc_file = os.path.join(
                os.path.dirname(fit_res_file), 'Precalculated_Graph.pickle'
            )
        print(f"Using precalculated graph from {precalc_file}")
        hist_graph = HistogramGraph.from_precalculated_file(precalc_file)

        print("Setting up NNMFitter to access data hist...")
        nnmfitter = NNMFitter(config_hdl=config_hdl, precalculated=precalc_file)
        data = nnmfitter.get_data_hists(as_array=True)
        print("done!")
    else:
        hist_graph = HistogramGraph.from_configdict(config_dict)

        nnmfitter = NNMFitter(config_hdl)
        data = nnmfitter.get_data_hists(as_array=True)

    param_values = {}

    param_values['best_fit'] = dict(fit_res['fit-result'][1])
    # take into account fixed paratmeters
    param_values['best_fit'].update(fit_res['fixed-parameters'])

    #print("Param values, including fixed params")
    #print(param_values['best_fit'])

    for p, v in override_params.items():
        if p in param_values['best_fit'].keys():
            print(f"Overwriting param {p} to value {v}")
            param_values['best_fit'][p] = v

    for det_conf in det_confs:
        if det_conf != "IC86_pass2_SnowStorm_v2_Bfr_DoubleCascades":
            continue
        
        if plot_name is not None:
            plot_name_det_conf = plot_name + f"_{det_conf}"
        else:
            plot_name_det_conf = None
        # plot 1d projections for all detector configs
        plot_1D_data_MC_comparison(
            hist_graph,
            param_values['best_fit'],
            data,
            ylim_energy_ratio=ylim_energy_ratio,
            ylim_zenith_ratio=ylim_zenith_ratio,
            ylim_length_ratio=ylim_length_ratio,
            ylim_eratio_ratio=ylim_eratio_ratio,
            det_conf=det_conf,
            plot_components=plot_components,
            component_graphdict=component_graphdict,
            perbin_plots=perbin_plots,
            params_all_zero=params_all_zero,
            override_plot_components_dict=override_plot_components_dict,
            plot_title=plot_title,
            plot_dir=plot_dir,
            save=plot_name_det_conf
        )


def get_hists_from_graphs(
    graphs, det_confs, hist_patch_settings, get_data=True
):

    all_data = {}
    all_mc = {}
    binning = {}

    for det_conf in det_confs:

        all_data[det_conf] = {}
        all_mc[det_conf] = {}

        # get binning
        if hist_patch_settings is None:
            temp_fit = str(
                list(graphs.keys())[0]
            )  # just for getting the binning (fixed per detector config)
        else:
            temp_fit = hist_patch_settings[det_conf]['full_range_fit_name']
        temp = graphs[temp_fit]['graph'].get_binning(det_conf)
        bins_energy = temp["reco_energy"]
        bins_zenith = temp["reco_zenith"]
        shape = (len(bins_energy) - 1, len(bins_zenith) - 1)
        binning[det_conf] = (bins_energy, bins_zenith)

        if get_data:
            # get data (2d array)
            # initialise Fitter to load data explicitly
            fitter = NNMFitter(
                graphs[temp_fit]['graph'].config_hdl,
                precalculated=graphs[temp_fit]['graph_file']
            )
            data = fitter.get_data_hists(as_array=True)
            all_data[det_conf] = data[det_conf].reshape(
                shape
            )  # this should be the same for all

        for fit in graphs.keys():

            if "hist_file" in graphs[fit].keys():
                all_mc[det_conf][fit] = pd.read_pickle(graphs[fit]["hist_file"])
            else:
                # get mc (dict with mu and ssq)
                all_mc[det_conf][fit] = graphs[fit][
                    'graph'].get_evaled_histogram(
                        det_config=det_conf,
                        input_variables=graphs[fit]['fit_pars'],
                        reshape=True
                    )
                if hist_patch_settings is not None:
                    # doesn't modify all_mc[det_conf][fit] if not specified explicitly in settings
                    all_mc = do_hist_patching(
                        all_mc, det_conf, fit, hist_patch_settings
                    )

            if get_data:
                # also print chi^2 statistic for all fits
                chi2_val = calculate_chi2(
                    np.copy(all_data[det_conf]),
                    np.copy(all_mc[det_conf][fit]['mu']),
                    np.copy(all_mc[det_conf][fit]['ssq'])
                )
                print(
                    f"Fit {fit} obtaining a chi2 {chi2_val} for detector config {det_conf}"
                )

    return all_data, all_mc, binning


def compare_several_fits(
    graphs,
    det_confs,
    plot_data_mc=True,  # either set this to true so that data will be used as baseline in ratio, or:
    ratio_base_name=None,  # or set the name of the fit (and corresponding evaled graph) to be used as baseline in ratio
    hist_patch_settings=None,  # option to compare fits that use different binning (so far, only the case of dropped bins is implemented)
    ylim_energy_ratio=(0.5, 1.5),
    ylim_zenith_ratio=(0.95, 1.05),
    ratio_logscale=False,
    plot_title=None,
    plot_dir=None,
    save=None,
):
    # graphs: dict(fit_name{
    #                       'graph': histogram_graph,
    #                       'fit_pars': bestfit_point,
    #                       'plot_settings': dict()})
    # hist_patch_settings: dict(det_conf ->
    #   dict(
    #       'fits_to_patch': dict(fit_name-> settings_dict),
    #       'full_range_fit_name': str)

    all_data, all_mc, binning = get_hists_from_graphs(
        graphs=graphs,
        det_confs=det_confs,
        hist_patch_settings=hist_patch_settings,
        get_data=plot_data_mc
    )

    plot_settings = {fit: graphs[fit]['plot_settings'] for fit in graphs.keys()}

    for det_conf in det_confs:
        several_fits_data_mc(
            det_conf,
            bins_energy=binning[det_conf][0],
            bins_zenith=binning[det_conf][1],
            mc=all_mc[det_conf],
            data=all_data[det_conf],
            plot_data_mc=plot_data_mc,
            ratio_base_name=ratio_base_name,
            plot_settings=plot_settings,
            ylim_energy_ratio=ylim_energy_ratio,
            ylim_zenith_ratio=ylim_zenith_ratio,
            ratio_logscale=ratio_logscale,
            plot_title=plot_title,
            save=save,
            plot_dir=plot_dir
        )
    if det_conf == 'IC86_pass2_SnowStorm_v2_cscd_cascade':
        plot_1D_data_MC_zenith_bands(
            data=all_data[det_conf],
            mc=all_mc[det_conf],
            plot_data_mc=plot_data_mc,
            ratio_base_name=ratio_base_name,
            bins_energy=binning[det_conf][0],
            bins_zenith=binning[det_conf][1],
            plot_settings=plot_settings,
            plot_title=plot_title,
            save=save,
            plot_dir=plot_dir
        )


def compare_fits_tracks_zenith(
    graphs,
    det_confs,
    energy_slices=[20, 25, 30, 35],
    plot_data_mc=True,  # either set this to true so that data will be used as baseline in ratio, or:
    ratio_base_name=None,  # or set the name of the fit (and corresponding evaled graph) to be used as baseline in ratio
    hist_patch_settings=None,  # option to compare fits that use different binning (so far, only the case of dropped bins is implemented)
    ylim_energy_ratio=(0.5, 1.5),
    ylim_zenith_ratio=(0.95, 1.05),
    ratio_logscale=False,
    plot_title=None,
    plot_dir=None,
    save=None,
):
    # graphs: dict(fit_name{
    #                       'graph': histogram_graph,
    #                       'fit_pars': bestfit_point,
    #                       'plot_settings': dict()})
    # hist_patch_settings: dict(det_conf ->
    #   dict(
    #       'fits_to_patch': dict(fit_name-> settings_dict),
    #       'full_range_fit_name': str)

    assert det_confs==['IC86_pass2_SnowStorm_v2_tracks'], \
        "detailed zenith comparison only implemented for tracks so far"

    all_data, all_mc, binning = get_hists_from_graphs(
        graphs=graphs,
        det_confs=det_confs,
        hist_patch_settings=hist_patch_settings,
        get_data=plot_data_mc
    )

    plot_settings = {fit: graphs[fit]['plot_settings'] for fit in graphs.keys()}

    # slice down data, mc and binning according to energy ranges
    for n_s, i_s in enumerate(energy_slices[:-1]):
        data = deepcopy(all_data)
        mc = deepcopy(all_mc)
        energy_slice_idx_width = energy_slices[n_s + 1] - i_s
        i_e = i_s + energy_slice_idx_width
        for det_conf in ['IC86_pass2_SnowStorm_v2_tracks']:
            data[det_conf] = data[det_conf][i_s:i_e, :]
            for fit in graphs.keys():
                mc[det_conf][fit]['mu'] = mc[det_conf][fit]['mu'][i_s:i_e, :]
                mc[det_conf][fit]['ssq'] = mc[det_conf][fit]['ssq'][i_s:i_e, :]
            energy_binning_temp = binning[det_conf][0][i_s:i_e + 1]
            zenith_binning_temp = binning[det_conf][1]

            log_binedges_low = f"{np.log10(energy_binning_temp[0]):.2f}"
            log_binedges_up = f"{np.log10(energy_binning_temp[-1]):.2f}"
            title_temp = r"$\mathrm{log}_{10}(E_\mathrm{reco}) \in$" + \
                f"[{log_binedges_low}, {log_binedges_up}]"

            if plot_title is not None:
                title_temp = plot_title + title_temp

            save_slice = None
            if save is not None:
                save_slice = f"{save}_{log_binedges_low}_{log_binedges_up}"
            several_fits_data_mc(
                det_conf,
                bins_energy=energy_binning_temp,
                bins_zenith=zenith_binning_temp,
                mc=mc[det_conf],
                data=data[det_conf],
                plot_data_mc=plot_data_mc,
                ratio_base_name=ratio_base_name,
                plot_settings=plot_settings,
                ylim_energy_ratio=ylim_energy_ratio,
                ylim_zenith_ratio=ylim_zenith_ratio,
                ratio_logscale=ratio_logscale,
                plot_title=title_temp,
                save=save_slice,
                plot_dir=plot_dir
            )

def do_hist_patching(all_mc, det_conf, fit_name, hist_patch_settings):
    """
    _summary_

    Parameters
    ----------
    all_mc : dict
        mu, ssq per det conf for several fits
    det_conf : str
        detector config to patch
    fit_name : str
        name of the fit/corresponding mc to patch
    hist_patch_settings : dict
        settings for patching
    """
    def patch_hist(hist, settings):

        assert hist.shape == settings['default_shape']
        hist_new = np.zeros(settings['patched_shape'])
        # so far only implemented patch for case where bins are dropped
        n_bin_to_patch = settings['n_bin_to_patch']
        assert settings['patched_shape'][
            0] == n_bin_to_patch + settings['default_shape'][0]
        hist_new[n_bin_to_patch:, ] = hist

        return hist_new

    if fit_name in hist_patch_settings[det_conf]['fits_to_patch'].keys():
        settings = hist_patch_settings[det_conf]['fits_to_patch'][fit_name]
        mu_default = np.copy(all_mc[det_conf][fit_name]['mu'])
        ssq_default = np.copy(all_mc[det_conf][fit_name]['ssq'])
        all_mc[det_conf][fit_name]['mu'] = patch_hist(mu_default, settings)
        all_mc[det_conf][fit_name]['ssq'] = patch_hist(ssq_default, settings)
        print(
            f"CAREFUL when looking at zenith distributions for {fit_name}!"
            "this is histogram is padded with zeros to match energy distributions"
            "but hence missing events when projected onto zenith"
        )

    return all_mc


def get_plot_additionals(fit, plot_settings):
    """
    Helper function to several_fits_data_mc

    Parameters
    ----------
    fit : str
        fit to plot
    plot_settings : dict
        settings

    Returns
    -------
    tuple(kwargs_add, label, skip_mc_unc)
        _description_
    """
    kwargs_add = deepcopy(plot_settings[fit]
                         ) if fit in plot_settings.keys() else dict()
    if 'skip_label' in kwargs_add.keys():
        label = None
        kwargs_add.pop('skip_label')
    else:
        label = f"{fit}"
    if 'skip_mc_unc' in kwargs_add.keys():
        skip_mc_unc = True
        kwargs_add.pop('skip_mc_unc')
    else:
        skip_mc_unc = False

    return kwargs_add, label, skip_mc_unc


def several_fits_data_mc(
    det_conf,
    bins_energy,
    bins_zenith,
    mc,
    data,
    plot_data_mc=True,
    ratio_base_name=None,
    plot_settings=dict(),
    ylim_energy_ratio=(0.5, 1.5),
    ylim_zenith_ratio=(0.95, 1.05),
    ratio_logscale=False,
    plot_title=None,
    save=None,
    plot_dir=None
):

    # energy spectrum

    fig, (ax1, ax2) = initialize_figure(figsize=None)

    if plot_title is not None:
        fig.suptitle(plot_title)

    hist_mc = {}
    error_mc = {}

    for fit in mc.keys():
        hist_mc[fit] = mc[fit]['mu'].sum(axis=1)
        error_mc[fit] = np.sqrt(mc[fit]['ssq'].sum(axis=1))

    # make the distinction between data/MC and MC/MC comparisons
    if plot_data_mc and (ratio_base_name is None):
        hist_baseline = data.sum(axis=1)
        error_baseline = np.sqrt(data.sum(axis=1))
        baseline_func_hist_errorbar = plot_data_hist_errorbar
        baseline_func_ratio_errorbar = plot_data_ratio_errorbar
        kwargs_add = {'color': 'k', 'marker': 'o'}
        label = "Data"
    # yerr_ratio_base = yerr_base
    else:
        hist_baseline = hist_mc[ratio_base_name]
        error_baseline = error_mc[ratio_base_name]
        baseline_func_hist_errorbar = plot_hist_errorbar
        baseline_func_ratio_errorbar = plot_ratio_errorbar
        label = f"{ratio_base_name}"
        kwargs_add = plot_settings[
            ratio_base_name] if ratio_base_name in plot_settings.keys() else dict(
            )
        if plot_data_mc:
            plot_data_hist_errorbar(
                ax1,
                data.sum(axis=1),
                bins_energy,
                np.sqrt(data.sum(axis=1)),
                label='Data', color='k',  marker='o'
            )
            plot_data_ratio_errorbar(
                ax2,
                data.sum(axis=1),
                bins_energy,
                np.sqrt(data.sum(axis=1)),
                hist_baseline,
                yerr_baseline=None,
                color='k',  marker='o'
            )

    hist_ratio_base = hist_baseline

    baseline_func_hist_errorbar(
        ax1,
        hist_baseline,
        bins_energy,
        error_baseline,
        label=label,
        **kwargs_add
    )
    baseline_func_ratio_errorbar(
        ax2,
        hist_baseline,
        bins_energy,
        error_baseline,
        hist_ratio_base,
        yerr_baseline=None,
        **kwargs_add
    )

    for fit in mc.keys():
        # plot all other MC lines

        if fit == ratio_base_name:
            continue

        # handle potential skip label, skip mc unc
        kwargs_add, label, skip_mc_unc = get_plot_additionals(fit, plot_settings)
        if skip_mc_unc:
            error_mc_to_plot = np.zeros_like(hist_mc[fit])
        else:
            error_mc_to_plot = error_mc[fit]

        plot_hist_errorbar(
            ax1,
            hist_mc[fit],
            bins_energy,
            error_mc_to_plot,
            label=label,
            **kwargs_add
        )
        plot_ratio_errorbar(
            ax2,
            hist_mc[fit],
            bins_energy,
            error_mc_to_plot,
            hist_ratio_base,
            yerr_baseline=None,
            **kwargs_add,
        )

    # setup figure
    plt.xlabel("reco energy [GeV]")
    ax1.set_ylabel(r"$N_{\mathrm{Events}}$")
    ax1.set_xscale("log")
    ax1.set_yscale("log")
    ax1.set_xlim(bins_energy[0], bins_energy[-1])
    if ratio_logscale:
        ax2.set_yscale('log')
    ax2.set_ylim(ylim_energy_ratio[0], ylim_energy_ratio[1])

    # save and show plot
    ax1.legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    if (save is not None) and (plot_dir is not None):
        savefig(fig, plot_dir, save + '_energy')
    plt.show()

    # zenith spectrum

    fig, (ax1, ax2) = initialize_figure(figsize=None)

    if plot_title is not None:
        fig.suptitle(plot_title)

    hist_mc = {}
    error_mc = {}
    for fit in mc.keys():
        hist_mc[fit] = mc[fit]['mu'].sum(axis=0)
        error_mc[fit] = np.sqrt(mc[fit]['ssq'].sum(axis=0))

    # make the distinction between data/MC and MC/MC comparisons
    if plot_data_mc and (ratio_base_name is None):
        hist_baseline = data.sum(axis=0)
        error_baseline = np.sqrt(data.sum(axis=0))
        kwargs_add = {'color': 'k', 'marker': 'o'}
        label = "Data"
    else:
        hist_baseline = hist_mc[ratio_base_name]
        error_baseline = error_mc[ratio_base_name]
        label = f"{ratio_base_name}"
        kwargs_add = plot_settings[
            ratio_base_name] if ratio_base_name in plot_settings.keys() else dict(
            )
        if plot_data_mc:
            plot_data_hist_errorbar(
                ax1,
                data.sum(axis=0),
                np.cos(bins_zenith),
                np.sqrt(data.sum(axis=0)),
                label='Data', color='k',  marker='o'
            )
            plot_data_ratio_errorbar(
                ax2,
                data.sum(axis=0),
                np.cos(bins_zenith),
                np.sqrt(data.sum(axis=0)),
                hist_baseline,
                yerr_baseline=None,
                color='k',  marker='o'
            )

    hist_ratio_base = hist_baseline

    baseline_func_hist_errorbar(
        ax1,
        hist_baseline,
        np.cos(bins_zenith),
        error_baseline,
        label=label,
        **kwargs_add,
    )
    baseline_func_ratio_errorbar(
        ax2,
        hist_baseline,
        np.cos(bins_zenith),
        error_baseline,
        hist_ratio_base,
        yerr_baseline=None,
        **kwargs_add
    )
    for fit in mc.keys():
        # plot all other MC predictions

        if fit == ratio_base_name:
            continue

        # handle potential skip label, skip mc unc
        kwargs_add, label, skip_mc_unc = get_plot_additionals(fit, plot_settings)
        if skip_mc_unc:
            error_mc_to_plot = np.zeros_like(hist_mc[fit])
        else:
            error_mc_to_plot = error_mc[fit]

        plot_hist_errorbar(
            ax1,
            hist_mc[fit],
            np.cos(bins_zenith),
            error_mc_to_plot,
            label=label,
            **kwargs_add
        )
        plot_ratio_errorbar(
            ax2,
            hist_mc[fit],
            np.cos(bins_zenith),
            error_mc_to_plot,
            hist_ratio_base,
            yerr_baseline=None,
            **kwargs_add
        )

    # setup figure
    plt.xlabel("cos(zenith)")
    ax1.set_ylabel(r"$N_{\mathrm{Events}}$")
    ax1.set_yscale("log")
    ax1.set_xlim(np.cos(bins_zenith)[-1], np.cos(bins_zenith)[0])
    if ratio_logscale:
        ax2.set_yscale('log')
    ax2.set_ylim(ylim_zenith_ratio[0], ylim_zenith_ratio[1])

    # save and show plot
    ax1.legend()
    fig.tight_layout()
    fig.subplots_adjust(hspace=0.0)
    if (save is not None) and (plot_dir is not None):
        savefig(fig, plot_dir, save + '_zenith')
    plt.show()
