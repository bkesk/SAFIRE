# This file is distributed under the Apache License, Version 2.0 License.
# See LICENSE file in top directory for details.
#
# Copyright (c) 2021-2025 The Simons Foundation, Inc.
#
# You may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Circle
from matplotlib.patheffects import withStroke
from matplotlib.ticker import AutoMinorLocator, MultipleLocator
import matplotlib.cm as cm
import numpy as np

from afqmctools.systems.lattice import Lattice,LatticeSite

def plot_site(ax,site:LatticeSite):

    ax.add_artist(
        Circle(
            xy = site.coord,
            radius = 0.1,
        )
    )

def plot_lattice(
        lattice:Lattice,
        show_direct_nearest=True,
        show_image_nearest=True,
        nth_neighbor=1,
        save=False,
        show_plot=True,
        show_labels = True,
        density = None,
        density_label = "Density",
        show_lattice_vecs = True,
        **kwargs
    ):
    """
    Plot a lattice with nearest neighbors.

    Parameters
    ----------
    lattice : afqmctools.systems.lattice.Lattice
        Lattice object to plot.
    show_direct_nearest : bool, optional
        Show direct nearest neighbors, by default True. Direct neighbors are connected by solid lines.
    show_image_nearest : bool, optional
        Show image nearest neighbors, by default True. Image neighbors are connected by dashed lines.
    save : bool, optional
        Save the plot to a file, by default False. If False, the plot is displayed.
    show_labels : bool, optional
        Show site labels, by default True.
    density : np.array, optional
        Array of site densities with shape (nsites,nsites), by default None. If provided, the site color is scaled by the density.
    norm_type : str, optional
        Type of normalization to use for the density, by default 'linear'. Options are 'linear' and 'log'.
    
    """
    num_sublattice = lattice.num_sublattice

    _cmap = kwargs.get("cmap",'seismic')

    Lx = lattice.L[0]
    Ly = lattice.L[1]

    # For plot limits
    # TODO: these are not correct for general {ai}, and/or for a basis!!
    Dx = Lx*lattice.a1[0] + Ly*lattice.a2[0]
    Dy = Lx*lattice.a1[1] + Ly*lattice.a2[1]

    if Dx == 0: Dx = max(1,Dy)
    if Dy == 0: Dy = max(1,Dx)

    fig_scale = 2
    fig,ax = plt.subplots(1,1,figsize=(Dx*fig_scale,Dy*fig_scale))
    

    ax.xaxis.set_major_locator(MultipleLocator(1.000))
    ax.xaxis.set_minor_locator(AutoMinorLocator(Dx))
    ax.yaxis.set_major_locator(MultipleLocator(1.000))
    ax.yaxis.set_minor_locator(AutoMinorLocator(Dy))
    ax.xaxis.set_minor_formatter("{x:.2f}")

    ax.minorticks_off()

    #ax.set_xlim(-0.5, Dx - 0.5) # doesn't work with basis, ignore for now
    #ax.set_ylim(-0.5, Dy - 0.5)
    ax.set_aspect(1.)

    ax.tick_params(which='major', width=1.0, length=10, labelsize=14)
    
    ax.grid(linestyle="--", linewidth=0.5, color='.25', zorder=-10)

    lattice_type = getattr(lattice,"_type","")

    if kwargs.get("title",None) is not None:
        title_string = kwargs.get("title")
    else:
        title_string = f"{Lx}x{Ly} {lattice_type} Lattice"

    ax.set_title(title_string, fontsize=20, verticalalignment='bottom')
    ax.set_xlabel("x", fontsize=14)
    ax.set_ylabel("y", fontsize=14)

    if density is not None:
        density = np.array(density)

        if density.shape == (Lx, Ly) and num_sublattice == 1:
            density = density[:, :, np.newaxis]

        vmin = kwargs.get('vmin', np.min(density))
        vmax = kwargs.get('vmax', np.max(density))
        # avoid machine precision issues when vmin == vmax
        if vmin == vmax:
            vmin -= 1e-10
            vmax += 1e-10

        norm_type = kwargs.pop('norm_type', 'linear')
        
        if norm_type == 'linear':
            norm = plt.Normalize(vmin=vmin, vmax=vmax)
            colormap = cm.get_cmap(_cmap)(norm(density))  # Choose a colormap
        elif norm_type == 'log':
            norm = LogNorm(vmin=vmin, vmax=vmax)
            # for single sublattice case
            normed_density = norm(density[:, :, 0])[:,:,np.newaxis]
            colormap = cm.get_cmap(_cmap)( normed_density)  # Choose a colormap

    def annotate(x, y, text, coord):
    
        royal_blue = [0, 20/256, 82/256]
        black = [0.0,0.0,0.0]

        if density is not None:
            facecolor = colormap[tuple(coord)]  # Map density to a color
        else:
            facecolor = 'lightgray'  # Default color if density is not provided

        # Circle marker
        c = Circle(
            (x, y),
            radius=0.10,
            clip_on=False,
            zorder=10,
            linewidth=1.0,
            edgecolor=black,
            facecolor=facecolor,
            path_effects=[withStroke(linewidth=7, foreground='white')]
        )
        ax.add_artist(c)

        if not show_labels: return

        for path_effects in [[withStroke(linewidth=7, foreground='white')], []]:
            color = 'white' if path_effects else royal_blue
            ax.text(x, y-0.2, text, zorder=100,
                    ha='center', va='top', weight='bold', color=color,
                    style='italic', #fontfamily='Courier New',
                    path_effects=path_effects)

            color = 'white' if path_effects else 'black'
            ax.text(x, y-0.35, f"{coord[:2]}", zorder=100,
                    ha='center', va='top', weight='normal', color=color,
                    fontfamily='monospace', fontsize='medium',
                    path_effects=path_effects)


    for site in lattice.sites:
        annotate(
            x=site.position[0],
            y=site.position[1],
            text=site.index,
            coord=site.coord
        )

    def add_pair_arrows(pairs,offset=None,**kwargs):
        
        for pair in pairs:
            coord1 = lattice[pair.i].position
            if pair._abs_r is None:
                coord2 = lattice[pair.j].position
            else:
                coord2 = pair._abs_r

            delta_coord = [ c2 - c1 for c1,c2 in zip(coord1,coord2)]

            if offset is not None:
                coord1 = [ c + o for c,o in zip(coord1,offset)]

            # using zorder to avoid arrows being covered by sites
            ax.arrow(*coord1, *delta_coord, zorder=20, **kwargs)

    if show_direct_nearest:
        add_pair_arrows(
            pairs=lattice.get_nth_direct_neighbors(nth_neighbor),
            color='red'
        )
        
    if show_image_nearest:
        add_pair_arrows(
            pairs=lattice.get_nth_image_neighbors(nth_neighbor),
            color='red',
            linestyle=':'
        )

    if show_lattice_vecs:

        lattice_vec_settings = dict(
            color='black',
            width=0.04,
            head_width=0.1,
            shape='full'
        )
       
        if lattice.L[0] > 1 and lattice.L[1] > 1: # hide lattice vectors for 1-D lattices
            offset = 0.75
            offset = offset*1.1
            ax.arrow(-offset, -offset, *lattice.a1, **lattice_vec_settings)
            ax.text(-offset + lattice.a1[0] / 2, -offset + lattice.a1[1] / 2, r'$\vec{a}_1$', color='black', fontsize=16, ha='center', va='top')

            ax.arrow(-offset, -offset, *lattice.a2, **lattice_vec_settings)
            ax.text(-offset + lattice.a2[0] / 2, -offset + lattice.a2[1] / 2, r'$\vec{a}_2$', color='black', fontsize=16, ha='right', va='center')

    if density is not None:
        plt.colorbar(cm.ScalarMappable(norm=norm,cmap=cm.get_cmap(_cmap)),ax=ax).set_label(density_label,fontsize=16)

    fig.patch.set(linewidth=4, edgecolor='0.5')

    if save:
        if isinstance(save, str):
            plt.savefig(save)
        else:
            plt.savefig("lattice.png")
    
    if show_plot:
        plt.show()

    return fig,ax
