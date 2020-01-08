import argparse
import ternary
import numpy as np
import math
import os

import matplotlib
matplotlib.rcParams['figure.dpi'] = 200
matplotlib.rcParams['figure.figsize'] = (4, 4)
matplotlib.rcParams['figure.facecolor'] = '1.0'
import matplotlib.pyplot as plt

import os

parser = argparse.ArgumentParser()

# where to find the weight files
parser.add_argument('--results_dir', type=str)
# what to plot
parser.add_argument('--combined', type=bool, default=True)
parser.add_argument('--separate_folds', type=bool, default=False)
parser.add_argument('--per_feature', type=bool, default=False)
parser.add_argument('--features', type=int, default=1)
    
args = parser.parse_args()

def weights_to_points(weights):
    return weights / np.sum(weights, axis=2)[:,:, np.newaxis]

def plot_run(load_file, save_folder,
             combined, separate_folds,
             features, per_feature):
    """
    Produce ternary plots from weights data.
    - expects weights as ndarrays saved as .npy with dimensions [fold, epoch, weight]
    - expects aggregators*3 many weights (3 per aggregator) unless per_feature is True when
      instead expects aggregators*features*3 (currently assumes fixed features per layer but may
      update this)
    together these allow us to parse the weights per (gr)aggregator
    """
    weights = np.load(load_file)
    folds = weights.shape[0]
    
    # figure out how many aggregators there are
    if per_feature == False:
        aggregators = int(weights.shape[-1]/3)
    else:
        aggregators = int(weights.shape[-1]/(3*features))
    
    # get the name of the run from the load_file (dropping .npy)
    run_identifier = os.path.basename(load_file)[:-4]
    # make directories for saved images
    save_folder += run_identifier
    if not os.path.exists(save_folder):
        os.mkdir(save_folder)
    
    if not per_feature:
        if separate_folds:
            if not os.path.exists(save_folder+'/by_fold/'):
                os.mkdir(save_folder+'/by_fold/')
            for agg in range(aggregators):
                # slice by aggregator then convert to ternary points
                points_agg = weights_to_points(weights[...,agg:agg+3])
                for fold in range(folds):
                    # ------ plotting part 
                    # take the points
                    points = points_agg[fold]
                    ## Sample trajectory plot
                    figure, tax = ternary.figure(scale=1)
                    figure.set_size_inches(5, 5)

                    tax.boundary()
                    tax.gridlines(multiple=1./2, color="black")

                    title_string = run_identifier + ' – layer ' + str(agg) + ', fold ' + str(fold)

                    tax.right_corner_label("Max", fontsize=8)
                    tax.top_corner_label("Sum", fontsize=8,  offset=0.16)
                    tax.left_corner_label("Mean", fontsize=8)

                    tax.set_title(title_string, fontsize=8, pad=20)

                    tax.new_colored_trajectory(points, linewidth=1.0, label="Curve")
                    tax.arrow(points, arrows=0, start=False, end=True, lw=1., head_width=0.005)
                    tax.ticks(axis='lbr', multiple=1./2, linewidth=1, tick_formats="%.1f",
                              offset=0.03)

                    tax.get_axes().axis('off')
                    tax.clear_matplotlib_ticks()
                    # ------ end of plotting part

                    # save
                    save_string = 'agg_' + str(agg) + '_fold_' + str(fold) +'.png'
                    tax.savefig(save_folder+'/by_fold/'+save_string)
                    plt.close()
        if combined:
            if not os.path.exists(save_folder+'/combined/'):
                os.mkdir(save_folder+'/combined/')
            for agg in range(aggregators):
                # slice by aggregator then convert to ternary points
                points_agg = weights_to_points(weights[...,agg:agg+3])

                # ------ plotting part 
                ## Sample trajectory plot
                figure, tax = ternary.figure(scale=1)
                figure.set_size_inches(5, 5)

                tax.boundary()
                tax.gridlines(multiple=1./2, color="black")

                title_string = run_identifier + ' – layer ' + str(agg)

                tax.right_corner_label("Max", fontsize=8)
                tax.top_corner_label("Sum", fontsize=8,  offset=0.16)
                tax.left_corner_label("Mean", fontsize=8)

                tax.set_title(title_string, fontsize=8, pad=20)

                for fold in range(folds):
                    points = points_agg[fold]
                    tax.new_colored_trajectory(points, linewidth=1.0, label="Curve")
                    tax.arrow(points, arrows=0, start=False, end=True, lw=1., head_width=0.005)

                #tax.ticks(axis='lbr', multiple=1./2, linewidth=1, tick_formats="%.1f", offset=0.03)

                tax.get_axes().axis('off')
                tax.clear_matplotlib_ticks()
                # ------ end of plotting part

                # save
                save_string = 'agg_' + str(agg) + '.png'
                tax.savefig(save_folder+'/combined/'+save_string)
                plt.close()
                
                
# get a list of the weight files in the directory (assuming there aren't some rogue
# .npy files in there)
onlyfiles = [f for f in os.listdir(args.results_dir)
             if os.path.isfile(os.path.join(args.results_dir, f)) and f.endswith('.npy')]

if not os.path.exists(args.results_dir+'plots/'):
    os.mkdir(args.results_dir+'plots/')

for run in onlyfiles:
    plot_run(args.results_dir+run, args.results_dir+'plots/',
             combined=args.combined, separate_folds=args.separate_folds,
             features=args.features, per_feature=args.per_feature)
