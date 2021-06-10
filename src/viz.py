import numpy as np
from matplotlib import gridspec
from matplotlib import pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.patches import Ellipse
from matplotlib.ticker import MultipleLocator
from sklearn import metrics

import pred


def visualize_simple(
    dataset,
    reconstruction,
    crds1,
    crds2,
    metadata1,
    metadata2,
    iframe=0,
    markersize=24,
    cmap1=None,
    cmap2=None,
    # show_3dpsd=True, show_1dpsd=True,
    figname="",
):
    """ """
    font = {"family": "monospace", "weight": "bold", "size": 12}
    plt.rc("font", **font)
    ### SETUP
    width = 6
    height = 6
    n_row = 2
    n_col = 2
    data_image = dataset[iframe, 0, ...]
    reconstructed = reconstruction[iframe, 0, ...]
    ### START FIGURE
    fig = plt.figure(figsize=(width, height), dpi=180)
    #
    plt.subplot(n_row, n_col, 1)
    plt.title("Angle-colored\n Latent Space")
    plt.scatter(
        crds1[:, 0], crds1[:, 1], c=metadata1, cmap=cmap1, marker=".", linewidth=1
    )
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,  # ticks along the top edge are off
        labelleft=False,
        labelbottom=False,
    )  # labels along the bottom edge ar
    plt.plot(
        crds1[iframe, 0],
        crds1[iframe, 1],
        color="black",
        marker="o",
        markersize=markersize,
    )
    # plt.grid()
    # cbar = plt.colorbar(orientation='horizontal')
    # cbar.set_ticks([-90,90])
    # cbar.set_label('angle (degrees)')
    plt.subplot(n_row, n_col, 2)
    plt.title("Defocus-colored\n Latent Space")
    plt.scatter(
        crds2[:, 0], crds2[:, 1], c=metadata2, cmap=cmap2, marker=".", linewidth=1
    )
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,  # ticks along the top edge are off
        labelleft=False,
        labelbottom=False,
    )  # labels along the bottom edge ar
    plt.plot(
        crds2[iframe, 0],
        crds2[iframe, 1],
        color="black",
        marker="o",
        markersize=markersize,
    )
    # plt.grid()
    # cbar = plt.colorbar(orientation='horizontal')
    # cbar.set_ticks([1.0,2.0])
    # cbar.set_label('defocus (um)')
    #
    plt.subplot(n_row, n_col, 3)
    plt.title("Particle")
    plt.imshow(data_image, cmap="Greys_r")
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,  # ticks along the top edge are off
        labelleft=False,
        labelbottom=False,
    )  # labels along the bottom edge ar
    plt.subplot(n_row, n_col, 4)
    plt.title("Reconstruction")
    plt.imshow(reconstructed, cmap="Greys_r")
    plt.tick_params(
        axis="both",  # changes apply to the x-axis
        which="both",  # both major and minor ticks are affected
        bottom=False,  # ticks along the bottom edge are off
        top=False,
        left=False,  # ticks along the top edge are off
        labelleft=False,
        labelbottom=False,
    )  # labels along the bottom edge ar
    plt.show()
    plt.tight_layout()
    if figname:
        fig.savefig(figname)


def visualize(
    dataset,
    reconstruction,
    crds1,
    crds2,
    metadata1,
    metadata2,
    iframe=0,
    markersize=24,
    cmap1=None,
    cmap2=None,
    show_3dpsd=True,
    show_1dpsd=True,
    figname="",
):
    """ """
    ### SETUP
    width = 12
    n_row = 2
    if show_3dpsd:
        n_row += 1
    if show_1dpsd:
        n_row += 1
    height = n_row * width / 2
    data_image = dataset[iframe, 0, ...]
    reconstructed = reconstruction[iframe, 0, ...]

    ### START FIGURE
    fig = plt.figure(figsize=(width, height), dpi=180)
    #
    plt.subplot(n_row, 2, 1)
    plt.title("latent space")
    plt.scatter(
        crds1[:, 0], crds1[:, 1], c=metadata1, cmap=cmap1, marker=".", linewidth=1
    )
    plt.plot(
        crds1[iframe, 0],
        crds1[iframe, 1],
        color="black",
        marker="o",
        markersize=markersize,
    )
    plt.grid()
    plt.colorbar()
    plt.subplot(n_row, 2, 2)
    plt.title("latent space")
    plt.scatter(
        crds2[:, 0], crds2[:, 1], c=metadata2, cmap=cmap2, marker=".", linewidth=1
    )
    plt.plot(
        crds2[iframe, 0],
        crds2[iframe, 1],
        color="black",
        marker="o",
        markersize=markersize,
    )
    plt.grid()
    plt.colorbar()
    #
    plt.subplot(n_row, 2, 3)
    plt.title("image")
    plt.imshow(data_image, cmap="Greys_r")
    plt.subplot(n_row, 2, 4)
    plt.title("reconstruction")
    plt.imshow(reconstructed, cmap="Greys_r")
    #
    i_subplot = 4
    if show_1dpsd:
        data_image_1dpsd = radial_profile(psd(data_image))
        plt.subplot(n_row, 2, i_subplot + 1)
        plt.title("1D PSD(image)")
        plt.plot(radial_profile(psd(data_image)), color="black", linewidth=2)
        plt.xlim(0, data_image.shape[0] / 2)
        plt.ylim(np.min(data_image_1dpsd), np.max(data_image_1dpsd))
        plt.subplot(n_row, 2, i_subplot + 2)
        plt.title("1D PSD(reconstruction)")
        plt.plot(radial_profile(psd(reconstructed)), color="black", linewidth=2)
        plt.plot(
            radial_profile(psd(data_image)),
            color="black",
            linewidth=1,
            linestyle="dashed",
        )
        plt.xlim(0, data_image.shape[0] / 2)
        plt.ylim(np.min(data_image_1dpsd), np.max(data_image_1dpsd))
        i_subplot = 6
    #
    if show_3dpsd:
        plt.subplot(3, 3, i_subplot + 1)
        plt.title("PSD(image)")
        plt.imshow(psd(data_image), cmap="Greys_r")
        plt.subplot(3, 3, i_subplot + 2)
        plt.title("PSD(image-reconstruction)")
        plt.imshow(psd(data_image - reconstructed), cmap="Greys_r")
        plt.subplot(3, 3, i_subplot + 3)
        plt.title("PSD(reconstruction)")
        plt.imshow(psd(reconstructed), cmap="Greys_r")
        i_subplot = 6
    #
    # plt.tight_layout()
    plt.show()
    if figname:
        fig.savefig(figname)


def show_latentspace(
    mus,
    metadata,
    do_pca=False,
    label0="",
    label1="",
    figsize=10,
    dpi=180,
    fontsize=18,
    figname="",
):
    """show_latentspace"""
    proj = mus
    if do_pca:
        U, L, Vt = np.linalg.svd(mus, full_matrices=False)
        proj = U
    #
    fig = plt.figure(figsize=(figsize, figsize), dpi=dpi)
    plt.rcParams.update({"font.size": fontsize})
    #
    plt.subplot(2, 2, 3)
    if do_pca:
        plt.title("eigenvalues")
        plt.bar(np.arange(L.shape[0]), L, color="black")
    else:
        plt.title("dataset")
        if label1:
            plt.ylabel(label1)
        plt.scatter(
            np.arange(metadata.shape[0]),
            metadata[:, 1],
            c=metadata[:, 0],
            cmap="rainbow",
        )
        plt.tick_params(labelbottom=False)
        plt.xticks([])
        plt.yticks([-90, 0, 90])
    #
    plt.subplot(2, 2, 1)
    plt.ylabel("projection #0")
    plt.xlabel("projection #1")
    plt.grid()
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.scatter(proj[:, 1], proj[:, 0], c=metadata[:, 1], cmap="twilight")
    #
    plt.subplot(2, 2, 2)
    plt.ylabel("projection #0")
    plt.xlabel("projection #2")
    plt.grid()
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.scatter(proj[:, 2], proj[:, 0], c=metadata[:, 1], cmap="twilight")
    cbar = plt.colorbar(
        ticks=[np.min(metadata[:, 1]), np.mean(metadata[:, 1]), np.max(metadata[:, 1])]
    )
    cbar.set_label(label1, rotation=270)
    #
    plt.subplot(2, 2, 4)
    plt.ylabel("projection #1")
    plt.xlabel("projection #2")
    plt.grid()
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.scatter(proj[:, 2], proj[:, 1], c=metadata[:, 0], cmap="rainbow")
    cbar = plt.colorbar(
        ticks=[np.min(metadata[:, 0]), np.mean(metadata[:, 0]), np.max(metadata[:, 0])]
    )
    cbar.set_label(label0, rotation=270)
    plt.tight_layout()
    plt.show()
    if figname:
        fig.savefig(figname)


def show_latentspace_v2(
    mus,
    metadata,
    do_pca=False,
    # label0='',
    label1="",
    figsize=10,
    dpi=180,
    fontsize=18,
    figname="",
):
    """show_latentspace_v2"""
    proj = mus
    if do_pca:
        U, L, Vt = np.linalg.svd(mus, full_matrices=False)
        proj = U
    #
    fig = plt.figure(figsize=(figsize, np.int(figsize / 2)), dpi=dpi)
    plt.rcParams.update({"font.size": fontsize})
    #
    plt.subplot(1, 2, 1)
    # plt.ylabel('projection #1')
    # plt.xlabel('projection #0')
    plt.grid(which="major")
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.scatter(proj[:, 0], proj[:, 1], c=metadata[:, 1], cmap="twilight")
    cbar = plt.colorbar(ticks=[-180, 180])
    cbar.set_label(label1, rotation=270)
    #
    plt.subplot(1, 2, 2)
    # plt.ylabel('projection #0')
    # plt.xlabel('projection #2')
    plt.grid()
    plt.tick_params(labelbottom=False, labelleft=False)
    plt.scatter(proj[:, 0], proj[:, 1], c=metadata[:, 0], cmap="rainbow")
    cbar = plt.colorbar(ticks=[0, 3])
    cbar.set_label(label1, rotation=270)
    plt.tight_layout()
    plt.show()
    if figname:
        fig.savefig(figname)


def show_reconstruction(recon, dataset=None, iframe=0):
    if dataset is None:
        nplots = 1
    else:
        nplots = 3
    fig = plt.figure(figsize=(12, 36))
    plt.subplot(1, nplots, 1)
    plt.title("reconstruction")
    plt.imshow(recon[iframe, 0, ...], cmap="Greys_r")
    if dataset is not None:
        plt.subplot(1, 3, 2)
        plt.title("image")
        plt.imshow(dataset[iframe, 0, ...], cmap="Greys_r")
        plt.subplot(1, 3, 3)
        plt.title("PSD(image-reconstruction)")
        psddiff = psd(dataset[iframe, 0, ...] - recon[iframe, 0, ...])
        plt.imshow(psddiff, cmap="Greys_r")
        plt.tight_layout()


def psd(data):
    """ """
    data_psd = np.fft.fft2(data)
    data_psd = np.fft.fftshift(data_psd)
    data_psd = np.log(np.absolute(data_psd) ** 2)
    return data_psd


def radial_profile(image):
    """
    copied from https://gist.github.com/ViggieSmalls/3bc5ec52774cf6e672f49723f0aa4a47
    """
    y, x = image.shape
    cx = x // 2
    cy = y // 2
    Y, X = np.indices(image.shape)
    r = np.sqrt((X - cx) ** 2 + (Y - cy) ** 2)
    ind = np.argsort(r.flat)  # indices for sorted radii (need to use with im)
    sr = r.flat[ind]  # the sorted radii
    sim = image.flat[ind]  # image values sorted by radii
    ri = sr.astype(np.int16)  # integer part of radii (bin size = 1)
    # The particularly tricky part, must average values within each radii bin
    # Start by looking for where radii change values
    deltar = ri[1:] - ri[:-1]  # assume all radii represented (more work if not)
    rind = np.where(deltar)[0]  # location of changed radius
    nr = rind[1:] - rind[:-1]  # number of pixels in radius bin
    csim = np.cumsum(sim, dtype=np.float64)  # cumulative sum for increasing radius
    # total in one bin is simply difference between cumulative sum for adjacent bins
    tbin = csim[rind[1:]] - csim[rind[:-1]]  # sum for image values in radius bins
    radialprofile = tbin / nr  # compute average for each bin
    return radialprofile


def cart2pol(x, y):
    """
    copied from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    """
    rho = np.sqrt(x ** 2 + y ** 2)
    phi = np.arctan2(y, x)
    return (rho, phi)


def pol2cart(rho, phi):
    """
    copied from https://stackoverflow.com/questions/20924085/python-conversion-between-coordinates
    """
    x = rho * np.cos(phi)
    y = rho * np.sin(phi)
    return (x, y)


def plot_pred2d(angle_pred, defocus_pred, angle_true, defocus_true, figname=""):
    """ """
    fig = plt.figure(figsize=(10, 5), dpi=180)
    plt.rcParams.update({"font.size": 12})
    plt.subplot(1, 2, 1)
    plt.title("in-plane rotation")
    plt.xlabel("polar angle in image")
    plt.ylabel("polar angle in (2D) latent space")
    plt.hexbin(angle_true, angle_pred, gridsize=25, mincnt=1)
    # plt.scatter(angle_true, angle_pred, color='black', linewidths=0.1)
    plt.grid()
    plt.subplot(1, 2, 2)
    plt.title("defocus")
    plt.xlabel("defocus in image")
    plt.ylabel("radius in (2D) latent space")
    plt.hexbin(defocus_true, defocus_pred, gridsize=25, mincnt=1)
    # plt.scatter(defocus_true, defocus_pred, c=metadata)
    # cbar = plt.colorbar(ticks=[-0.2,0.9])
    # cbar.set_label('3rd coordinate in latent space', rotation=270)
    plt.tight_layout()
    plt.show()
    if figname:
        fig.savefig(fname=figname)


#
############
# < BIPLOTS


def biplot_histncontour(x, y, bins=150, levels=[1, 3, 5]):
    """ """
    fig = plt.figure(figsize=(18, 9))
    plt.subplot(121)
    counts, xbins, ybins, _ = plt.hist2d(
        x, y, bins=bins, norm=LogNorm(), cmap=plt.cm.inferno
    )
    plt.colorbar()
    plt.subplot(122)
    plt.contourf(
        np.log10(counts.transpose()),
        extent=[xbins[0], xbins[-1], ybins[0], ybins[-1]],
        cmap=plt.cm.Greys,
        levels=levels,
    )
    plt.grid()
    plt.colorbar()


def biplots(
    prj,
    prj2=None,
    n=1,
    plottype="hexbin",
    plottype2="hexbin",
    nbins=10,
    figsize=-1,
    c=None,
    c2=None,
    cmap=None,
    c2map=None,
    show_histo=False,
    figname="",
    scatter_size=None,
    minortick=0.1,
    majortick=0.2,
    colorbar_size=200,
):
    """biplots : plot populations in component space
    Description
    -----------
    For n components, shows biplots between all pairs of prj in upper triangle.
    If prj2 is provided, shows biplots between all pairs of prj2 in lower one.
    The possibility to color based on input assignment is offered.
    """
    cmap, c2map, plottype, plottype2 = biplot_c(c, c2, cmap, c2map, plottype, plottype2)
    figsize, nrow, ncol = biplot_size(figsize, n)
    nbins_coarse = int(nbins / 1)
    labels = get_labels(n)
    gs = gridspec.GridSpec(nrow, ncol, hspace=0, wspace=0)
    minorticklocator = MultipleLocator(minortick)
    majorticklocator = MultipleLocator(majortick)
    #
    minimum = np.min(prj)  # [:,0])
    maximum = np.max(prj)  # [:,0])
    valmax = np.maximum(np.abs(minimum), np.abs(maximum))
    x_range = [-1.1 * valmax, 1.1 * valmax]
    y_range = x_range
    if prj2 is not None:
        minimum2 = np.min(prj2)  # [:,0])
        maximum2 = np.max(prj2)  # [:,0])
        valmax2 = np.maximum(np.abs(minimum2), np.abs(maximum2))
        x_range2 = [-1.1 * valmax2, 1.1 * valmax2]
        y_range2 = x_range2
    #
    fig = plt.figure(figsize=(figsize, figsize), dpi=160, facecolor="w", edgecolor="k")
    for i in np.arange(nrow):
        for j in np.arange(ncol):
            if i < j:
                ax = biplot_axis(
                    n,
                    i,
                    j,
                    gs,
                    fig,
                    labels,
                    Ax=prj[:, j],
                    Ay=prj[:, i],
                    c=c,
                    cmap=cmap,
                    nbins=nbins,
                    majortick=majortick,
                    minortick=minortick,
                    linewidth=2.5,
                    plottype=plottype,
                    scatter_size=scatter_size,
                    x_range=x_range,
                    y_range=y_range,
                )
            elif i == j:
                if show_histo:
                    cm = plt.cm.get_cmap(cmap)
                    ax = fig.add_subplot(gs[i, j])
                    plt.grid()
                    plt.tick_params(
                        axis="both",
                        which="both",
                        bottom=False,
                        top=False,
                        left=False,
                        right=False,
                        labelbottom=False,
                        labeltop=False,
                        labelleft=False,
                        labelright=False,
                    )
                    Ax = prj[:, i]
                    plt.hist(Ax, bins=nbins_coarse)
                    if prj2 is not None:
                        Ay = prj2[:, i]
                        plt.hist(Ay, bins=nbins_coarse, rwidth=0.4)
                else:
                    if i == 0 and c is not None:
                        ax = fig.add_subplot(gs[i, j])
                        # ax.set_xlabel(labels[j],fontsize='xx-large')
                        # ax.set_ylabel(labels[i],fontsize='xx-large')
                        ax.xaxis.set_label_position("top")
                        for corner in ("top", "bottom", "right", "left"):
                            ax.spines[corner].set_linewidth(0)
                        ax.tick_params(
                            axis="both",
                            which="both",
                            bottom=False,
                            top=False,
                            left=False,
                            right=False,
                            labelbottom=False,
                            labeltop=False,
                            labelleft=False,
                            labelright=False,
                            labelsize="large",
                        )
                        colorbar = c  # range(1,np.max(c)+1,1)
                        ax.scatter(
                            colorbar, colorbar, c=colorbar, cmap=cmap, s=colorbar_size
                        )  # vmin=1, vmax=np.max(c), cmap=cmap)
                        # for x in [np.min(c),np.mean(c),np.max(c)]:
                        # ax.annotate('{0:.1f}'.format(x),(x+np.std(x),x))
                    if i == (n - 1) and c2 is not None:
                        ax = fig.add_subplot(gs[i, j])
                        # ax.set_xlabel(labels[j],fontsize='xx-large')
                        # ax.set_ylabel(labels[i],fontsize='xx-large')
                        ax.xaxis.set_label_position("top")
                        for corner in ("top", "bottom", "right", "left"):
                            ax.spines[corner].set_linewidth(0)
                        ax.tick_params(
                            axis="both",
                            which="both",
                            bottom=False,
                            top=False,
                            left=False,
                            right=False,
                            labelbottom=False,
                            labeltop=False,
                            labelleft=False,
                            labelright=False,
                            labelsize="large",
                        )
                        colorbar = c2  # range(1,np.max(c)+1,1)
                        ax.scatter(
                            colorbar, colorbar, c=colorbar, cmap=c2map, s=colorbar_size
                        )  # vmin=1, vmax=np.max(c), cmap=cmap)
                        # for x in [np.min(c2),np.mean(c2),np.max(c2)]:
                        # ax.annotate('{0:.1f}'.format(x),(x+np.std(x),x))
            else:
                if prj2 is not None:
                    ax = biplot_axis(
                        n,
                        i,
                        j,
                        gs,
                        fig,
                        labels,
                        Ax=prj2[:, j],
                        Ay=prj2[:, i],
                        c=c2,
                        cmap=c2map,
                        nbins=nbins,
                        majortick=majortick,
                        minortick=minortick,
                        linewidth=2.5,
                        plottype=plottype2,
                        scatter_size=scatter_size,
                        x_range=x_range2,
                        y_range=y_range2,
                    )
    plt.tight_layout()
    plt.show()
    if figname:
        fig.savefig(figname)


def biplot_c(c, c2, cmap, c2map, plottype, plottype2):
    """biplot_c"""
    if c is not None:
        plottype = "scatter"
    if c2 is not None:
        plottype2 = "scatter"
    #
    if plottype == "scatter":
        if cmap is None:
            cmap = "rainbow"
    else:
        if cmap is None:
            cmap = "plasma"
    #
    if plottype2 == "scatter":
        if c2map is None:
            c2map = "rainbow"
    else:
        if c2map is None:
            c2map = "plasma"
    #
    return cmap, c2map, plottype, plottype2


def biplot_size(figsize, n):
    """biplot_size"""
    if figsize < 0:
        if n == 1:
            figsize = 1
        else:
            figsize = 4
    figsize = figsize * 6
    nrow = n
    ncol = n
    return figsize, nrow, ncol


def biplot_axis(
    n,
    i,
    j,
    gs,
    fig,
    labels,
    Ax=np.zeros(1),
    Ay=np.zeros(1),
    c=None,
    cmap=None,
    nbins=1,
    majortick=0.2,
    minortick=0.1,
    linewidth=2.5,
    plottype="scatter",
    scatter_size=None,
    x_range=None,
    y_range=None,
):
    """biplot_axis"""
    minorticklocator = MultipleLocator(minortick)
    majorticklocator = MultipleLocator(majortick)
    ax = fig.add_subplot(gs[i, j])
    ax.xaxis.set_minor_locator(minorticklocator)
    ax.xaxis.set_major_locator(majorticklocator)
    ax.yaxis.set_minor_locator(minorticklocator)
    ax.yaxis.set_major_locator(majorticklocator)
    #
    if x_range is not None:
        ax.set_xlim(x_range[0], x_range[1])
    if y_range is not None:
        ax.set_ylim(y_range[0], y_range[1])
    #
    for corner in ("top", "bottom", "right", "left"):
        ax.spines[corner].set_linewidth(linewidth)
    ax.minorticks_on()
    ax.grid(which="major", axis="both", linestyle="-", linewidth=0.5)
    ax.grid(which="minor", axis="both", linestyle="--", linewidth=0.1)
    ax.axvline(0, linestyle=":", linewidth=2, color="k")
    ax.axhline(0, linestyle=":", linewidth=2, color="k")
    if i == 0 and j != n - 1:
        ax.set_xlabel(labels[j], fontsize="xx-large")
        ax.xaxis.set_label_position("top")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelsize="large",
        )
    elif i == 0 and j == n - 1:
        ax.set_xlabel(labels[j], fontsize="xx-large")
        ax.xaxis.set_label_position("top")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelsize="large",
        )
    elif i != 0 and j == n - 1:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelsize="large",
        )
    elif i < j and i != 0 and j != n - 1:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelsize="large",
        )
    if j == 0 and i != n - 1:
        ax.set_ylabel(labels[i], fontsize="xx-large")
        if i == 1:
            ax.xaxis.set_label_position("top")
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
                # bottom=False,top=True,left=False,right=False,
                # labelbottom=False,labeltop=True,labelleft=False,labelright=False,
                labelsize="large",
            )
        else:
            ax.tick_params(
                axis="both",
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False,
                labeltop=False,
                labelleft=False,
                labelright=False,
                labelsize="large",
            )
    elif j == 0 and i == n - 1:
        ax.set_ylabel(labels[i], fontsize="xx-large")
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelsize="large",
        )
    elif j != 0 and i == n - 1:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelsize="large",
        )
    elif j < i and j != 0 and i != n - 1:
        ax.tick_params(
            axis="both",
            which="both",
            bottom=False,
            top=False,
            left=False,
            right=False,
            labelbottom=False,
            labeltop=False,
            labelleft=False,
            labelright=False,
            labelsize="large",
        )
    if plottype == "scatter":
        if scatter_size is None:
            ax.scatter(Ax, Ay, c=c, cmap=cmap)
        else:
            ax.scatter(Ax, Ay, c=c, cmap=cmap, s=scatter_size)
    else:
        ax.hexbin(Ax, Ay, gridsize=nbins, cmap=cmap, mincnt=1)
    return ax


def get_labels(n):
    labels = []
    for i in np.arange(0, n, 1):
        labels.append("# " + str(i + 1))
    return labels


#############
# ROC curve #
#############


def plot_roc_curve(
    mus,
    Zscore,
    Zscore_set,
    methods=["robust_covar", "isolation_forest", "local_outlier_detection"],
    labels=["Robust Covariance Method", "Isolation Forest", "Local Outlier Detection"],
    xlabel="False Positive Rate",
    ylabel="True Positive Rate",
    figname="",
):
    """ """
    fig = plt.figure(figsize=(6, 6), dpi=180)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    for method in methods:
        measure, offset, assignment = pred.outlier_measure(mus, method=method)
        fpr, tpr, thresholds = metrics.roc_curve(
            np.where(Zscore < Zscore_set, 0, 1), measure
        )
        plt.plot(fpr, tpr, lw=3, linestyle="-")
        print("AUC({}) = {} / offset={}".format(method, metrics.auc(fpr, tpr), offset))
    plt.plot([0, 1], [0, 1], color="grey", lw=1, linestyle=":")
    plt.legend(labels, loc=0, fontsize="xx-large")
    plt.grid()
    if figname:
        fig.savefig(figname)
