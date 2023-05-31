import lsst.afw.detection as afwDetect
import lsst.afw.geom
import lsst.ip.isr as isr
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

def isr_ccob_exposure(exp, dark_exp=None):
    assemblyTask = isr.AssembleCcdTask()
    overscanTask = isr.IsrTask()

    overscanTask.config.doOverscan = True
    overscanTask.config.overscan.fitType = 'MEDIAN_PER_ROW'
    overscanTask.config.doAssembleIsrExposures =True
    overscanTask.config.doAssembleCcd = True
    overscanTask.config.assembleCcd.doTrim = True
    overscanTask.config.doApplyGains = True
    overscanTask.config.doBias = False
    overscanTask.config.doLinearize = False
    overscanTask.config.doDark = (dark_exp is not None)
    overscanTask.config.doFlat = False
    overscanTask.config.doDefect = False

    return overscanTask.run(exp, dark=dark_exp).exposure

def isr_dark_exposure(dark_exp_raw):
    assemblyTask = isr.AssembleCcdTask()
    overscanTask = isr.IsrTask()

    overscanTask.config.doOverscan = True
    overscanTask.config.doAssembleIsrExposures =True
    overscanTask.config.doAssembleCcd = True
    overscanTask.config.assembleCcd.doTrim = True
    overscanTask.config.doApplyGains = True
    overscanTask.config.doBias = False
    overscanTask.config.doLinearize = False
    overscanTask.config.doDark = False
    overscanTask.config.doFlat = False
    overscanTask.config.doDefect = False

    return overscanTask.run(dark_exp_raw).exposure

def footprint_signal_spans(im, footprint):
    spans = footprint.getSpans()
    total = 0
    for span in spans:
        total += sum(im.array[span.getY()][max(0,span.getX0()):span.getX1()+1])
    return total

def footprint_center_of_mass(im, footprint):
    spans = footprint.getSpans()
    total_moment = np.array([0.,0.])
    partial_sums_x = []
    partial_sums_y = []
    total = 0
    for span in spans:
        span_y = span.getY()
        if span_y >= im.array.shape[0]: break
        if span_y < 0 : continue
        x_range = np.arange(max(0,span.getX0()),min(im.array.shape[1], span.getX1()+1))
        partial_sums_x.append(np.sum(x_range * im.array[span_y,x_range]))
        partial_sums_y.append(np.sum(im.array[span_y,x_range]) * span_y)
        total += np.sum(im.array[span_y, x_range])
    center_x = np.sum(partial_sums_x/total)
    center_y = np.sum(partial_sums_y/total)
    center_of_mass = np.array((center_x, center_y))
    return center_of_mass, total

def get_spots_counts(exp_assembled, threshold_adu=10, minarea=30000, maxarea=4000000, make_plot=False, force_circle=False):
    
    im = exp_assembled.getImage()

    imarr = im.getArray()
    threshold = afwDetect.Threshold(threshold_adu)
    
    fpset = afwDetect.FootprintSet(im, threshold)
    culled_fpset = [fp for fp in
                        fpset.getFootprints() if (fp.getArea() > minarea and fp.getArea() < maxarea ) ]

    
    if force_circle:
        new_culled_fpset = []
        for fp in culled_fpset:
            this_center_of_mass, this_signal = footprint_center_of_mass(im, fp)
            new_culled_fpset.append(afwDetect.Footprint(lsst.afw.geom.SpanSet.fromShape(int(np.floor(np.sqrt(fp.getArea()/np.pi))), 
                                                        offset=(int(this_center_of_mass[0]), int(this_center_of_mass[1])))))
        culled_fpset = new_culled_fpset
        
    signals = []
    centers = []
    for fp in culled_fpset:
        center_of_mass, total = footprint_center_of_mass(im, fp)
        signals.append(total)
        centers.append(center_of_mass)
    signals = np.asarray(signals)
    centers = np.asarray(centers)
        
    indx = np.where(signals > threshold_adu*minarea)[0]
    
    results = [(signals[i], centers[i], culled_fpset[i]) for i in indx]

    if make_plot:
        
        fig = plt.figure()
    
        plt.imshow(imarr,clim=np.percentile(imarr.flat, (1,99)))
        plt.colorbar()
        
        det = exp_assembled.getDetector()
        for fp in culled_fpset:
            spans = fp.getSpans()

            new_imarr = np.zeros_like(imarr)
            for span in spans:
                new_imarr[span.getY()][max(0,span.getX0()):span.getX1()+1] = 1

            color1 = mplcolors.colorConverter.to_rgba('white')
            color2 = mplcolors.colorConverter.to_rgba('blue')
            cmap_footprint = mplcolors.LinearSegmentedColormap.from_list('my_map', [color1, color2],256)
            cmap_footprint._init()
            alphas = np.linspace(0, 0.8, cmap_footprint.N+3)
            cmap_footprint._lut[:,-1] = alphas

            plt.imshow(new_imarr,cmap=cmap_footprint, clim=(0,1))

        plt.title(f'{exp_assembled.getMetadata()["OBSID"]} - {det.getName()}')
        plt.show()
        remove_figure(fig)
        print(signals[indx])
    return results
    
def get_spots_counts_from_raw(exp_raw, dark, threshold_adu=100,minarea=30000, makePlot=False):
    exp_assembled = isr_ccob_exposure(exp_raw, dark)
    return get_spots_counts(exp_assembled, threshold_adu=threshold_adu,minarea=minarea, makePlot=makePlot)