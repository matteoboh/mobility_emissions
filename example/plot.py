import osmnx as ox
import pandas as pd
from matplotlib import cm, colors
import matplotlib.pyplot as plt
import matplotlib.dates as dt
from .speed_and_acceleration import *


def plot_road_network_with_emissions(tdf_with_emissions, road_network, name_of_pollutant='CO_2',
									 color_map='autumn_r', bounding_box=None, save_fig=False):
	"""Plot emissions

	Plotting emissions of one of four pollutants using the module osmnx.plot_graph_routes.
	Colors indicate intensity of cumulate emissions on each road.

	Parameters
	----------
	tdf_with_emissions : TrajDataFrame
		TrajDataFrame with 4 columns ['CO_2', 'NO_x', 'PM', 'VOC'] collecting the instantaneous emissions for each point.

	road_network : networkx MultiDiGraph

	name_of_pollutant : string
		the name of the pollutant to plot. Must be one of ['CO_2', 'NO_x', 'PM', 'VOC'].
		Default is 'CO_2'.

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

	emissions = list(tdf_with_emissions[name_of_pollutant])

	dict_road_to_emissions = map_road_to_emissions(tdf_with_emissions, road_network, name_of_pollutant)

	# extracting the list of roads and creating a list of colors to color them:
	list_roads = []
	list_road_to_emissions_cumulates = []  # this is used to create the list of colors

	for road, emission in dict_road_to_emissions.items():
		list_roads.append(list(road))
		list_road_to_emissions_cumulates.extend([list(road) + [sum(emission)]])

	edge_cols = get_edge_colors_from_list(list_road_to_emissions_cumulates,
										  cmap=color_map, num_bins=3)

	sm = cm.ScalarMappable(cmap=color_map, norm=colors.Normalize(vmin=min(emissions), vmax=max(emissions)))

	fig, ax = ox.plot_graph_routes(road_network,
								   list_roads,
								   bbox=bounding_box,
								   fig_height=20,
								   route_color=edge_cols,
								   route_linewidth=3,
								   orig_dest_node_alpha=0,
								   show=False, close=False)

	cbar = fig.colorbar(sm, ax=ax, shrink=0.5, extend='max')
	cbar.set_label('%s (g)' % name_of_pollutant, size=25,
				   labelpad=15)  # labelpad is for spacing between colorbar and its label
	cbar.ax.tick_params(labelsize=20)

	if save_fig:
		filename = str('plot_emissions_%s.png' % name_of_pollutant)
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close(fig)
	else:
		fig.show()

	return fig, ax


def get_edge_colors_from_list(list_of_emissions_per_edge, num_bins=3, cmap='summer', start=0, stop=1, na_color='none'):
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
	colors = ox.get_colors(num_bins, cmap, start, stop)
	edge_colors = [colors[int(cat)] if pd.notnull(cat) else na_color for cat in cats]
	return edge_colors


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