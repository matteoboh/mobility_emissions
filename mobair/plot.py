import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors, cm
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
import matplotlib.dates as dt
from .speed import *
from .emissions import *
from .utils import *


###########################################################################################################
############################################ PLOTTING #####################################################
###########################################################################################################

def get_edge_colors_by_attribute(road_network, attribute_name, log_norm=True, linthresh=0.01, cmap='autumn_r', na_color='#999999'):
	"""Get colors based on edge attribute values.

	Parameters
	----------
	road_network : networkx.MultiDiGraph

	attribute_name : string
		name of a numerical edge attribute.

	log_norm : bool
		set the normalizer to use in cm.ScalarMappable.
		if True, uses colors.LogNorm, else uses colors.Normalize.

	linthresh : float
		range around zero that is linearly normalized.
		See the documentation of matplotlib.colors.SymLogNorm for details.

	cmap : string
		name of a matplotlib colormap.

	na_color : string
		what color to assign edges with missing attr values.

	Returns
	-------
	color_series : pandas.Series
		series labels are edge IDs (u, v, k) and values are colors.

	sm : cm.ScalarMappable object
		this is useful for creating a colorbar for the plot.

	References
	----------
	slightly inspired by get_edge_colors_by_attr in the osmnx.plot module.
	"""
	vals = pd.Series(nx.get_edge_attributes(road_network, attribute_name))

	min_val = vals.dropna().min()
	max_val = vals.dropna().max()

	if log_norm:
		norm = colors.SymLogNorm(vmin=min_val, vmax=max_val, linthresh=linthresh, linscale=0, base=10)
	else:
		norm = colors.Normalize(vmin=min_val, vmax=max_val)

	sm = cm.ScalarMappable(norm=norm, cmap=cmap)
	color_series = vals.map(sm.to_rgba)
	color_series.loc[pd.isnull(vals)] = na_color

	return color_series, sm

### DEPRECATED ###
def get_edge_colors_from_list(list_of_emissions_per_edge, num_bins=3, cmap='autumn_r', start=0, stop=1, na_color='none'):
	"""
	Get a list of edge colors by binning some continuous-variable attribute into
	quantiles.

	Parameters
	----------
	list_of_emissions_per_edge : list
		list in the form [u, v, cont_var], where u, v are OSM ids specifying an edge,
		and cont_var is the value of the continuous variable on that edge
	num_bins : int
		how many quantiles
	cmap : string
		name of a colormap
	start : float
		where to start in the colorspace
	stop : float
		where to end in the colorspace
	na_color : string
		what color to assign nodes with null attribute values

	Returns
	-------
	list

	References
	----------
	took from get_edge_colors_by_attr in the osmnx.plot module, and modified.
	"""
	if num_bins is None:
		num_bins = len(list_of_emissions_per_edge)
	bin_labels = range(num_bins)
	attr_values = pd.Series([co2 for u, v, co2 in list_of_emissions_per_edge])
	cats = pd.qcut(x=attr_values, q=num_bins, labels=bin_labels)
	colors = ox.plot.get_colors(num_bins, cmap, start, stop, return_hex=True)
	edge_colors = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in cats]
	return edge_colors


def plot_road_network_with_attribute(road_network, attribute_name, region_name, tdf_with_emissions=None,
									 normalization_factor=None,
									 fig_size=(20, 20), show_hist=True, n_bins=30, log_normalise=False,
									 quantile_cut=0.01, # deprecated
									 color_map='autumn_r', bounding_box=None,
									 save_fig=False):
	"""Plot roads' attribute

	Plotting the roads by attribute (e.g. road length or grade) using the module osmnx.plot_graph.
	Colors indicate intensity of the attribute on each road.

	Parameters
	----------
	road_network : networkx MultiGraph

	attribute_name : string
		the name of the attribute to plot. Must be one of the edges' attributes in the graph.

	region_name : string
		the name of the region. This is only used to save the figure.

	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.
		This is ignored if attribute_name not in {'CO_2', 'NO_x', 'PM', 'VOC'}.

	normalization_factor : string
		the type of normalization wanted. It can be one of {None, 'tot_emissions', 'road_length'}.

	fig_size : tuple
		size of the figure as (width, height).

	show_hist : bool
		whether to draw or not an histogram with the distribution of attribute per road.

	n_bins : int
		the number of bins for the histrogram.

	log_normalise : bool
		if True, matplotlib.colors.SymLogNorm is used for log-normalisation of the data before mapping each value to a color,
		and shows the histogram on log scale.
		Otherwise, matplotlib.colors.Normalize is used, and the histogram is shown on linear scale.
		The logarithmic scale is suggested when the attribute to plot is one of the pollutant.

	quantile_cut : float
		used when log_normalise=True: the histogram is cut at this quantile to address issues with the log-normalisation
		when the minimum value of the attribute is exactly 0.
		Default value is 0.01. (DEPRECATED)

	color_map : str
		name of the colormap to use.
		Default is 'autumn_r'.

	bounding_box : list
		the bounding box as north, south, east, west, if one wants to plot the emissions only in a certain bbox of the network.
		Default is None.
		Note that if show_hist=True, the histogram refers to the entire road network, and not only to the area inside the bbox.

	save_fig : bool
		whether or not to save the figure.
		Default is False.

	Returns
	-------
	fig, ax
	"""

	list_pollutants = ['CO_2', 'NO_x', 'PM', 'VOC']
	if attribute_name in list_pollutants:
		# if the attribute to plot is a pollutant, then it should be first added as an edges' attribute:
		map__road__cum_em, attribute_label = map_road_to_cumulate_emissions(tdf_with_emissions, road_network,
																			attribute_name, normalization_factor)
		road_network = add_edge_emissions(map__road__cum_em, road_network, attribute_name)
	else:
		attribute_label = attribute_name.replace("_", " ").capitalize()

	# size of labels and ticks for hist and colorbar
	hist__label_size = fig_size[0]
	cbar__label_size = fig_size[0] + 5
	ticklabel_size = cbar__label_size - 2

	series_attribute = pd.Series(nx.get_edge_attributes(road_network, attribute_name))
	first_nonzero = [x for x in series_attribute.dropna().sort_values() if x != 0][0]

	# colors and ScalarMappable
	color_series, sm = get_edge_colors_by_attribute(road_network, attribute_name, log_norm=log_normalise,
													linthresh=first_nonzero, #series_attribute.dropna().quantile(quantile_cut),
													cmap=color_map, na_color='#999999')

	# map
	fig, ax = ox.plot_graph(road_network,
							bbox=bounding_box,
							figsize=fig_size,
							edge_color=color_series,
							edge_linewidth=1.8,
							node_size=0,
							bgcolor='w',
							show=False, close=False)

	if show_hist:
		# colorbar
		axin1 = inset_axes(ax,
						   width="5%",  # width = 5% of parent_bbox width
						   height="50%",  # height : 50%
						   loc='lower left',
						   bbox_to_anchor=(1.06, 0.1, 0.8, 1),
						   bbox_transform=ax.transAxes,
						   borderpad=0)
		cbar = fig.colorbar(sm, cax=axin1, shrink=0.5, extend='max', pad=0.03)
		cbar.set_label(attribute_label, size=cbar__label_size,
					   labelpad=15)  # labelpad is for spacing between colorbar and its label
		cbar.ax.tick_params(labelsize=ticklabel_size)

		# histogram
		axin2 = inset_axes(ax,
						   width="10%",  # width = 10% of parent_bbox width
						   height="50%",  # height : 50%
						   loc='lower left',
						   bbox_to_anchor=(1.06, 0.7, 2, 0.5),
						   bbox_transform=ax.transAxes,
						   borderpad=0)
		if log_normalise:
			min_val = series_attribute.dropna().min()
			if min_val == 0.0:
				min_val = first_nonzero #series_attribute.dropna().quantile(quantile_cut)
			max_val = series_attribute.dropna().max()
			n, bins, patches = axin2.hist(series_attribute.dropna(),
										  bins=np.logspace(np.log10(min_val), np.log10(max_val), n_bins))
			plt.xscale('symlog', linthresh = first_nonzero)
			plt.yscale('log')
		else:
			n, bins, patches = axin2.hist(series_attribute.dropna(), bins=n_bins, log=True)
		plt.xlabel(attribute_label, size=hist__label_size)
		plt.ylabel('# roads', labelpad=-1, size=hist__label_size)
		plt.tick_params(labelsize=ticklabel_size)

		# coloring the bars:
		for c_bin, thispatch in zip(bins, patches):
			color = sm.to_rgba(c_bin)
			thispatch.set_facecolor(color)
	else:
		cbar = fig.colorbar(sm, ax=ax, shrink=0.5, extend='max', pad=0.03)
		cbar.set_label(attribute_label, size=cbar__label_size,
					   labelpad=15)  # labelpad is for spacing between colorbar and its label
		cbar.ax.tick_params(labelsize=ticklabel_size)

	# (eventually) saving the figure
	if save_fig:
		filename = str('plot_road_%s__%s_normalised__%s.png' % (
		attribute_name, str(normalization_factor).lower(), region_name.lower().replace(" ", "_")))
		fig.savefig(filename, format='png', bbox_inches='tight'
					# facecolor='white'  # use if want the cbar to be on white (and not transparent) background
					)
		plt.close(fig)
	else:
		fig.show()

	return fig, ax


def plot_road_network_with_attribute__OLD(road_network, attribute_name, region_name, tdf_with_emissions=None,
									 normalization_factor=None,
									 fig_size=(20, 20), n_bins=4, equal_size=False,
									 color_map='autumn_r', bounding_box=None,
									 save_fig=False):
	"""Plot roads' attribute

	Plotting the roads by attribute (e.g. road length or grade) using the module osmnx.plot_graph.
	Colors indicate intensity of the attribute on each road.

	Parameters
	----------
	road_network : networkx MultiGraph

	attribute_name : string
		the name of the attribute to plot. Must be one of the edges' attributes in the graph.

	region_name : string
		the name of the region. This is only used to save the figure.

	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.
		This is ignored if attribute_name not in {'CO_2', 'NO_x', 'PM', 'VOC'}.

	normalization_factor : string
		the type of normalization wanted. It can be one of {None, 'tot_emissions', 'road_length'}.

	fig_size : tuple
		size of the figure as (width, height).

	n_bins : int
		This is used by osmnx.plot.get_edge_colors_by_attr to get colors based on edge attribute values.
		If None, linearly map a color to each value. Otherwise, assign values to this many bins then assign a color to each bin.

	equal_size : bool
		ignored if num_bins is None.
		If True, bin into equal-sized quantiles (requires unique bin edges). if False, bin into equal-spaced bins.

	color_map : str
		name of the colormap to use.
		Default is 'autumn_r'.

	bounding_box : list
		the bounding box as north, south, east, west, if one wants to plot the emissions only in a certain bbox of the network.
		Default is None.

	save_fig : bool
		whether or not to save the figure.
		Default is False.

	Returns
	-------
	fig, ax
	"""

	list_pollutants = ['CO_2', 'NO_x', 'PM', 'VOC']
	if attribute_name in list_pollutants:
		# if the attribute to plot is a pollutant, then it should be first added as an edges' attribute:
		map__road__cum_em, colorbar_label = map_road_to_cumulate_emissions(
			tdf_with_emissions, road_network, attribute_name, normalization_factor)
		road_network = add_edge_emissions(map__road__cum_em, road_network, attribute_name)
	else:
		colorbar_label = attribute_name.replace("_", " ")

	edge_cols = ox.plot.get_edge_colors_by_attr(road_network, attribute_name, cmap=color_map, num_bins=n_bins,
												na_color='#999999', equal_size=equal_size)

	dict_road_to_attribute = nx.get_edge_attributes(road_network, attribute_name)

	min_val = np.nanmin([x for x in dict_road_to_attribute.values() if x is not None])
	max_val = np.nanmax([x for x in dict_road_to_attribute.values() if x is not None])
	sm = cm.ScalarMappable(cmap=color_map, norm=colors.Normalize(vmin=min_val,
																 vmax=max_val))

	fig, ax = ox.plot_graph(road_network,
							bbox=bounding_box,
							figsize=fig_size,
							edge_color=edge_cols,
							edge_linewidth=1.5,
							bgcolor='w',
							node_size=0,
							show=False, close=False)

	cbar = fig.colorbar(sm, ax=ax, shrink=0.5, extend='max', pad=0.03)
	cbar.set_label(colorbar_label, size=20,
				   labelpad=15)  # labelpad is for spacing between colorbar and its label
	cbar.ax.tick_params(labelsize=18)

	if save_fig:
		filename = str('plot_road_%s__%s_normalised__%s.png' % (attribute_name, str(normalization_factor).lower(), region_name.lower().replace(" ", "_")))
		fig.savefig(filename, format='png', bbox_inches='tight',
					#facecolor='white'  # use if want the cbar to be on white (and not transparent) background
					)
		plt.close(fig)
	else:
		fig.show()

	return fig, ax


def plot_corr_matrix(corr_matrix, list_ordered_feature_names, region_name, corr_coef='spearman', save_fig=False):
	"""Plots a correlation matrix with matplotlib.pyplot.imshow.

	Parameters
	----------
	corr_matrix : numpy.ndarray
		the correlation matrix which one wants to plot.

	list_ordered_feature_names : list
		the list of the features for which the correlation has been computed.
		They should be ordered as they appear in the correlation matrix.

	region_name : str
		the name of the region to which the features belong.

	corr_coef : str
		the name of the correlation coefficient for the label (e.g. 'spearman').

	save_fig : bool
		whether or not to save the figure.
		Default is False.

	Returns
	-------
	fig, ax
	"""

	fig, ax = plt.subplots()
	im = ax.imshow(corr_matrix)

	im.set_clim(-1, 1)

	ax.set_xticks(np.arange(len(list_ordered_feature_names)))
	ax.set_yticks(np.arange(len(list_ordered_feature_names)))
	ax.set_xticklabels(list_ordered_feature_names)
	ax.set_yticklabels(list_ordered_feature_names)

	plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
			 rotation_mode="anchor")

	# Loop over data dimensions and create text annotations.
	for i in range(len(list_ordered_feature_names)):
		for j in range(len(list_ordered_feature_names)):
			text = ax.text(j, i, corr_matrix[i, j].round(decimals=2),
						   ha="center", va="center", color="w", weight='bold')

	cbar = fig.colorbar(im, ax=ax, format='% .2f')
	cbar.set_label(r'%s correlation coefficient ($\rho$)' %corr_coef.capitalize(), size=13,
				   labelpad=13)  # labelpad is for spacing between colorbar and its label

	if save_fig:
		filename = str('plot_corr_matrix__%s.png' %region_name.lower().replace(" ", "_"))
		plt.savefig(filename, format='png', bbox_inches='tight', facecolor='white')
		plt.close(fig)
	else:
		fig.show()

	return fig, ax


def plot_speed_over_time_for_one_id(tdf, uid):
	"""Plot speed over time.

	For a given individual ('uid') in a TrajDataFrame, plots its speed over time.

	Parameters
	----------
	tdf : TrajDataFrame
		the trajectories of the individuals.

	uid : int
		the uid of the individual for which we want the plot.

	Returns
	-------

	Warnings
	--------
	if the time interval of the trajectory is >24 hours, probably the x labels will be unreadable;
	if speed has not been previously computed, a function is firstly called to compute it.
	"""

	set_of_ids = set(tdf['uid'])
	if str(uid) not in set_of_ids:
		print('ID ' + uid + ' not found.')
		return

	if 'speed' not in tdf.columns:
		tdf_with_speed = compute_speed_from_tdf(tdf)
	else:
		tdf_with_speed = tdf.copy()

	tdf_one_id = tdf_with_speed.loc[tdf_with_speed['uid'] == str(uid)]

	seconds = dt.SecondLocator()
	minutes = dt.MinuteLocator()
	hours = dt.HourLocator()
	date_fmt = dt.DateFormatter('%H:%M')

	fig, ax = plt.subplots()
	ax.plot_date(tdf_one_id['datetime'], tdf_one_id['speed'], fmt='-')

	ax.xaxis.set_major_locator(hours)
	ax.xaxis.set_major_formatter(date_fmt)

	ax.set_ylabel('speed')

	fig.autofmt_xdate()  # rotate x labels to make them readable
	plt.title('Speed over time for one vehicle')
	plt.show()

	return
