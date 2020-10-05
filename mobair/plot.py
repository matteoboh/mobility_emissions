import pandas as pd
import numpy as np
import osmnx as ox
import networkx as nx
import matplotlib.pyplot as plt
from matplotlib import colors, cm
import matplotlib.dates as dt
from .speed import *
from .emissions import *


###########################################################################################################
############################################ PLOTTING #####################################################
###########################################################################################################

def create_list_road_to_cumulate_emissions(tdf_with_emissions, road_network, name_of_pollutant, normalization_factor=None):
	## TODO description
	# outputs [[u,v,cumulate_emissions],[u,v,cumulate_emissions],...]

	if normalization_factor not in list([None, 'tot_emissions', 'road_length']):
		print('normalization_factor must be one of [None, tot_emissions, road_length]')
		return

	dict_road_to_emissions = map_road_to_emissions(tdf_with_emissions, road_network, name_of_pollutant)
	array_all_emissions = np.array(tdf_with_emissions[name_of_pollutant])
	list_road_to_cumulate_emissions = []
	label = ''

	if normalization_factor == None:
		label = r'$%s$ (g)' % name_of_pollutant
		list_road_to_cumulate_emissions = [[road[0], road[1], np.sum(em)] for road,em in dict_road_to_emissions.items()]
	else:
		if normalization_factor == 'road_length':
			label = r'$%s$ (grams per meter of road)' % name_of_pollutant
			dict_road_to_attribute = nx.get_edge_attributes(road_network, 'length')
			dict_road_to_cum_em_norm = {road: sum(dict_road_to_emissions[road]) / dict_road_to_attribute[road + (0,)]
										for road in dict_road_to_emissions.keys()}
			list_road_to_cumulate_emissions = [[road[0], road[1], em] for road, em in dict_road_to_cum_em_norm.items()]
		if normalization_factor == 'tot_emissions':
			label = '% ' + r'$%s$' % name_of_pollutant
			sum_all_emissions = np.sum(array_all_emissions)
			list_road_to_cumulate_emissions = [[road[0], road[1], np.sum(em) / sum_all_emissions * 100] for road, em in
											   dict_road_to_emissions.items()]

	return list_road_to_cumulate_emissions, label


def create_list_road_to_cumulate_emissions__OLD(tdf_with_emissions, road_network, name_of_pollutant, normalization_factor=None):
	## TODO description
	# outputs [[u,v,cumulate_emissions],[u,v,cumulate_emissions],...]

	if normalization_factor not in list([None, 'tot_emissions', 'road_length']):
		print('normalization_factor must be one of [None, tot_emissions, road_length]')
		return

	dict_road_to_emissions = map_road_to_emissions(tdf_with_emissions, road_network, name_of_pollutant)
	array_all_emissions = np.array(tdf_with_emissions[name_of_pollutant])
	list_road_to_cumulate_emissions = []
	for road, emission in dict_road_to_emissions.items():
		if normalization_factor == None:
			list_road_to_cumulate_emissions.extend([list(road) + [sum(emission)]])
			label = r'$%s$ (g)' % name_of_pollutant
		else:
			if normalization_factor == 'road_length':
				road_length = road_network.get_edge_data(road[0], road[1], key=0, default=80).get('length', 80)  # default set to 80 meters (~mean road length) --> can be improved TODO
				normalized_emissions = sum(emission) / road_length #* 100  # quantity of emissions per meter on that road
				#label = r'$%s$ (grams per 100 meters of road)' % name_of_pollutant
				label = r'$%s$ (grams per meter of road)' % name_of_pollutant
			if normalization_factor == 'tot_emissions':
				normalized_emissions = sum(emission) / np.sum(array_all_emissions) * 100  # emission percentage of the gross total
				label = '% ' + r'$%s$' % name_of_pollutant
			list_road_to_cumulate_emissions.extend([list(road) + [normalized_emissions]])

	return list_road_to_cumulate_emissions, label


def create_list_cumulate_emissions_per_vehicle(tdf_with_emissions, name_of_pollutant):
	## TODO description
	dict_road_to_vehicle = map_vehicle_to_emissions(tdf_with_emissions, name_of_pollutant)
	list_cumulate_emissions = [np.sum(em) for em in dict_road_to_vehicle.values()]
	label = r'$%s$ (g)' % name_of_pollutant

	return list_cumulate_emissions, label


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


def plot_road_network_with_emissions(tdf_with_emissions, road_network, region_name, normalization_factor = None,
									 name_of_pollutant='CO_2', fig_size = (20,20), color_map='autumn_r', bounding_box=None, save_fig=False):
	"""Plot emissions

	Plotting emissions of one of four pollutants using the module osmnx.plot_graph_routes.
	Colors indicate intensity of cumulate emissions on each road.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	road_network : networkx MultiDiGraph

	region_name : str
		the name of the region of the road network.

	normalization_factor : str
		the type of normalization wanted. It can be None, 'tot_emissions' or 'road_length'.

	name_of_pollutant : string
		the name of the pollutant to plot. Must be one of ['CO_2', 'NO_x', 'PM', 'VOC'].
		Default is 'CO_2'.

	fig_size : tuple
		size of the figure as (width, height).

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

	if name_of_pollutant not in tdf_with_emissions.columns:
		print('Emissions have not been previously computed: use compute_emissions first.')
		return
	if 'road_link' not in tdf_with_emissions.columns:
		print('Points of TrajDataFrame have not been previously map-matched: use find_nearest_edges_in_network first.')
		return
	if normalization_factor not in list([None, 'tot_emissions', 'road_length']):
		print('normalization_factor must be one of [None, tot_emissions, road_length]')
		return

	print('-- runtimes --')
	import time
	start_time = time.time()
	list_road_to_cumulate_emissions, colorbar_label = create_list_road_to_cumulate_emissions(tdf_with_emissions, road_network, name_of_pollutant, normalization_factor)
	list_all_cumulate_emissions = [em for u,v,em in list_road_to_cumulate_emissions]
	list_roads = [[u,v] for u,v,em in list_road_to_cumulate_emissions]
	print("create lists: %s seconds" % (time.time() - start_time))

	start_time = time.time()
	edge_cols = get_edge_colors_from_list(list_road_to_cumulate_emissions, cmap=color_map, num_bins=3)
	print("get_edge_colors_from_list: %s seconds" % (time.time() - start_time))
	start_time = time.time()
	sm = cm.ScalarMappable(cmap=color_map, norm=colors.Normalize(vmin=min(list_all_cumulate_emissions),
																 vmax=max(list_all_cumulate_emissions)))
	print("ScalarMappable: %s seconds" % (time.time() - start_time))

	start_time = time.time()
	fig, ax = ox.plot_graph_routes(road_network,
								   list_roads,
								   bbox=bounding_box,
								   figsize=fig_size,
								   route_colors = edge_cols,
								   route_linewidth=3,
								   bgcolor = 'white',
								   node_alpha = 0,
								   orig_dest_size = 0,
								   show=False, close=False)
	print("plot_graph_routes: %s seconds" % (time.time() - start_time))

	cbar = fig.colorbar(sm, ax=ax, shrink=0.5, extend='max', pad=0.03)
	cbar.set_label(colorbar_label, size=22, labelpad=15)  # labelpad is for spacing between colorbar and its label
	cbar.ax.tick_params(labelsize=18)

	if save_fig:
		start_time = time.time()
		filename = str('plot_emissions_%s__%s_normalized__%s.png' %(name_of_pollutant, normalization_factor, region_name.lower().replace(" ", "_")))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close(fig)
		print("savefig: %s seconds" % (time.time() - start_time))
	else:
		fig.show()

	return fig, ax


def add_edge_emissions(list_road_to_cumulate_emissions, road_network, name_of_pollutant='CO_2'):
	"""Add the value of emissions as a new attribute to the edges of the road network.

	Parameters
	----------
	list_road_to_cumulate_emissions : list
		list of type [[u,v,cumulate_emissions],[u,v,cumulate_emissions],...].

	road_network : networkx MultiGraph

	name_of_pollutant : string
		the name of the pollutant to plot. Must be in {'CO_2', 'NO_x', 'PM', 'VOC'}.
		Default is 'CO_2'.

	Returns
	-------
	road network with the new attribute on its edges.
	Note that for the edges with no value of emissions the attribute is set to None.

	"""
	dict_road_to_cumulate_emissions = {(u, v): em for [u, v, em] in list_road_to_cumulate_emissions}

	for u, v, data in road_network.edges(keys=False, data=True):
		try:
			c_emissions = dict_road_to_cumulate_emissions[(u, v)]
			data[name_of_pollutant] = c_emissions
		except KeyError:
			data[name_of_pollutant] = None

	return road_network


def plot_road_network_with_attribute(road_network, attribute_name, region_name, tdf_with_emissions=None,
									 normalization_factor=None,
									 fig_size=(20, 20), n_bins=4, color_map='autumn_r', bounding_box=None,
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
		list_road_to_cumulate_emissions, colorbar_label = create_list_road_to_cumulate_emissions(
			tdf_with_emissions, road_network, attribute_name, normalization_factor)
		road_network = add_edge_emissions(list_road_to_cumulate_emissions, road_network, attribute_name)
	else:
		colorbar_label = attribute_name.replace("_", " ")

	edge_cols = ox.plot.get_edge_colors_by_attr(road_network, attribute_name, cmap=color_map, num_bins=n_bins,
												na_color='#999999', equal_size=False)

	dict_road_to_attribute = nx.get_edge_attributes(road_network, attribute_name)

	min_val = np.nanmin([x for x in dict_road_to_attribute.values() if x is not None])
	max_val = np.nanmax([x for x in dict_road_to_attribute.values() if x is not None])
	sm = cm.ScalarMappable(cmap=color_map, norm=colors.Normalize(vmin=min_val,
																 vmax=max_val))

	fig, ax = ox.plot_graph(road_network,
							bbox=bounding_box,
							figsize=fig_size,
							edge_color=edge_cols,
							edge_linewidth=2,
							bgcolor='w',
							node_size=0,
							show=False, close=False)

	cbar = fig.colorbar(sm, ax=ax, shrink=0.5, extend='max', pad=0.03)
	cbar.set_label(colorbar_label, size=22,
				   labelpad=15)  # labelpad is for spacing between colorbar and its label
	cbar.ax.tick_params(labelsize=18)

	if save_fig:
		filename = str('plot_road_%s__%s.png' % (attribute_name, region_name.lower().replace(" ", "_")))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close(fig)
	else:
		fig.show()

	return fig, ax

def plot_road_network_with_attribute__OLD(road_network, attribute_name, region_name,
									 fig_size=(20, 20), color_map='coolwarm', bounding_box=None, save_fig=False):
	"""Plot roads' attribute

	Plotting the roads by attribute (e.g. road length or grade) using the module osmnx.plot_graph.
	Colors indicate intensity of the attribute on each road.

	Parameters
	----------
	road_network : networkx MultiGraph

	attribute_name : string
		the name of the attribute to plot. Must be one of the edges' attributes in the graph.

	fig_size : tuple
		size of the figure as (width, height).

	color_map : str
		name of the colormap to use.
		Default is 'coolwarm'.

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

	edge_cols = ox.plot.get_edge_colors_by_attr(road_network, attribute_name, cmap=color_map, num_bins=4,
												equal_size=True)

	dict_road_to_attribute = nx.get_edge_attributes(road_network, attribute_name)

	sm = cm.ScalarMappable(cmap=color_map, norm=colors.Normalize(vmin=min(dict_road_to_attribute.values()),
																 vmax=max(dict_road_to_attribute.values())))

	fig, ax = ox.plot_graph(road_network,
							bbox=bounding_box,
							figsize=fig_size,
							edge_color=edge_cols,
							edge_linewidth=2,
							bgcolor='w',
							node_size=0,
							show=False, close=False)

	cbar = fig.colorbar(sm, ax=ax, shrink=0.5, extend='max', pad=0.03)
	cbar.set_label(attribute_name.replace("_", " "), size=22,
				   labelpad=15)  # labelpad is for spacing between colorbar and its label
	cbar.ax.tick_params(labelsize=18)

	if save_fig:
		filename = str('plot_road_%s__%s.png' % (attribute_name, region_name.lower().replace(" ", "_")))
		plt.savefig(filename, format='png', bbox_inches='tight')
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