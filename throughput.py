import lsst.afw.detection as afwDetect
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mplcolors
import gc

def remove_figure(fig):
    """
    Remove a figure to reduce memory footprint.

    Parameters
    ----------
    fig: matplotlib.figure.Figure
        Figure to be removed.

    Returns
    -------
    None
    """
    # get the axes and clear their images
    for ax in fig.get_axes():
        for im in ax.get_images():
            im.remove()
    fig.clf()       # clear the figure
    plt.close(fig)  # close the figure
    gc.collect()    # call the garbage collector

def footprint_signal_spans(im, footprint):
    spans = footprint.getSpans()
    total = 0
    for span in spans:
        total += sum(im.array[span.getY()][span.getX0():span.getX1()+1])
    return total

def get_spots_counts(exp_assembled, threshold_adu=100,minarea=30000, makePlot=False):
    
    det = exp_assembled.getDetector()
    im = exp_assembled.getImage()

    imarr = im.getArray()
    threshold = afwDetect.Threshold(threshold_adu)
    
    fpset = afwDetect.FootprintSet(im, threshold)
    culled_fpset = [fp for fp in
                        fpset.getFootprints() if fp.getArea() > minarea  ]
    signals = np.array([footprint_signal_spans(im, fp) for fp in
                        culled_fpset])
    indx = np.where(signals > threshold_adu*minarea)

    if makePlot:
        fp = culled_fpset[0]
        spans = fp.getSpans()

        new_imarr = np.zeros_like(imarr)
        for span in spans:
            new_imarr[span.getY()][span.getX0():span.getX1()+1] = 1

        color1 = mplcolors.colorConverter.to_rgba('white')
        color2 = mplcolors.colorConverter.to_rgba('blue')
        cmap_footprint = mplcolors.LinearSegmentedColormap.from_list('my_map', [color1, color2],256)
        cmap_footprint._init()
        alphas = np.linspace(0, 0.8, cmap_footprint.N+3)
        cmap_footprint._lut[:,-1] = alphas

        fig = plt.figure()
        plt.imshow(imarr,clim=np.percentile(imarr.flat, (1,99)))
        plt.colorbar()
        plt.imshow(new_imarr,cmap=cmap_footprint, clim=(0,1))

        plt.title(f'{exp_assembled.getMetadata()["OBSID"]} - {det.getName()}')
        plt.show()
        remove_figure(fig)
        print(signals[indx])
    return signals[indx]
    
