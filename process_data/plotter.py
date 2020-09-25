import skmob
import pandas as pd
import numpy as np
import osmnx as ox
import time
from .all_methods import *
from PROCESS_DATA.stats_utils import *

###### PARAMETERS
PATH_TO_INPUT_FILE = './output_files/'
NAME_OF_INPUT_FILE = 'uk_5125users_2017-01-01_2017-02-02__greater_london_2800users__filtered_2570users_120sec__WITH_EMISSIONS.csv'
PATH_TO_ROAD_NETWORKS = './input_files/road_nets/'
region = 'Greater London'
subregion = None #'Quartiere XXXII Europa' #'Municipio Roma I' #'Rione XV Esquilino'
bbox_subregion = None #[41.904815, 41.886851, 12.495463, 12.471533]
list_of_pollutants = ['CO_2', 'NO_x', 'PM', 'VOC']
pollutant_to_map = 'CO_2' # one of: CO_2, NO_x, PM, VOC
normalization_factor = None  # one of: None, 'tot_emissions', 'road_length'
######

###### set to True/False based on what you want to obtain or not:
plot_distributions = False
fit_distributions = False   # in some cases, this can take time
fit_comparison_with_ = 'truncated_power_law'  # one of: 'exponential', 'lognormal', 'truncated_power_law'
map_emissions = True
######



### Loading tdf
print('Loading tdf...')
#df = pd.read_csv(PATH_TO_INPUT_FILE+NAME_OF_INPUT_FILE, header=0)
#tdf = skmob.TrajDataFrame(df, latitude='lat', longitude='lng', datetime='datetime', user_id='uid', trajectory_id='tid')
tdf = skmob.read(PATH_TO_INPUT_FILE + NAME_OF_INPUT_FILE)

### Loading road network
print('Loading road network...')
region_name = region.lower().replace(" ", "_")
try:
	### Load from (previously saved) graphml file
	if subregion == None:
		graphml_filename = '%s_network.graphml' % (region_name)
	else:
		subregion_name = subregion.lower().replace(" ", "_")
		graphml_filename = '%s_%s_network.graphml' % (region_name, subregion_name)
	road_network = ox.save_load.load_graphml(graphml_filename, folder=PATH_TO_ROAD_NETWORKS)
except:
	### ...or directly from osmnx:
	if subregion == None:
		road_network = ox.graph_from_place(region, network_type = 'drive_service')
	else:
		road_network = ox.graph_from_place(subregion + ', ' + region, network_type = 'drive_service')

### Saving road network (if not previously saved)
#ox.save_graphml(road_network, filename=graphml_filename, folder=PATH_TO_ROAD_NETWORKS)

### Normalizing emissions s.t. we can compare different regions
#print('Normalizing emissions...')
#tdf = normalize_emissions(tdf, percentage=True)


###########################################################################################################
################################### PLOT and FIT DISTRIBUTIONS ############################################
###########################################################################################################

if plot_distributions:
	for c_pollutant in list_of_pollutants:

		#print('--- runtimes ---')
		array_all_emissions = np.array(tdf[c_pollutant])
		#start_time = time.time()
		list_road_to_cumulate_emissions, x_label_road = create_list_road_to_cumulate_emissions(tdf, road_network, c_pollutant, normalization_factor)
		list_cumulate_emissions_per_road = [em for u, v, em in list_road_to_cumulate_emissions]
		#print("emissions per road: %s seconds" % (time.time() - start_time))
		#start_time = time.time()
		list_cumulate_emissions_per_vehicle, x_label_vehicle = create_list_cumulate_emissions_per_vehicle(tdf, c_pollutant)
		#print("emissions per vehicle: %s seconds" % (time.time() - start_time))

		print('Plotting %s...' %c_pollutant)
		#
		def plot_loglog(values_to_plot, x_label, y_label):
			fig = plt.figure()
			x, y = zip(*lbpdf(2.0, values_to_plot))
			plt.plot(x, y, marker='o')
			plt.grid(alpha=0.2)
			plt.xlabel(x_label, fontsize=10)
			plt.ylabel(y_label, fontsize=10)
			plt.loglog()
			return fig
		#

		###########################################
		#### plot overall distribution of emissions:
		x_label = r'$%s$ (g)' % c_pollutant
		y_label = 'Frequency'
		fig = plot_loglog(list(array_all_emissions), x_label, y_label)
		plt.title('Distribution of emissions of %s' % c_pollutant)
		filename = str('loglog_emissions_%s__%s.png' % (c_pollutant, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()

		#################################################
		#### plot distribution of emissions per road link:
		y_label = '# roads'
		fig = plot_loglog(list_cumulate_emissions_per_road, x_label_road, y_label)
		plt.title('Distribution of emissions of %s per road link' % c_pollutant)
		filename = str('loglog_emissions_per_road_link_%s__%s_normalised__%s.png' % (c_pollutant, normalization_factor, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()

		###############################################
		#### plot distribution of emissions per vehicle:
		y_label = '# vehicles'
		fig = plot_loglog(list_cumulate_emissions_per_vehicle, x_label_vehicle, y_label)
		plt.title('Distribution of emissions of %s per vehicle' % c_pollutant)
		filename = str('loglog_emissions_per_vehicle_%s__%s.png' % (c_pollutant, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()


####################
if fit_distributions:
	for c_pollutant in list_of_pollutants:

		list_road_to_cumulate_emissions, x_label_road = create_list_road_to_cumulate_emissions(tdf, road_network, c_pollutant, normalization_factor)
		list_cumulate_emissions_per_road = [em for u, v, em in list_road_to_cumulate_emissions]
		# print("emissions per road: %s seconds" % (time.time() - start_time))
		# start_time = time.time()
		list_cumulate_emissions_per_vehicle, x_label_vehicle = create_list_cumulate_emissions_per_vehicle(tdf, c_pollutant)
		# print("emissions per vehicle: %s seconds" % (time.time() - start_time))

		print('Fitting %s...' % c_pollutant)
		import powerlaw
		name__fit_comparison_with_ = fit_comparison_with_.replace("_", " ")

		print('')
		print('Fitting the distribution of emissions of %s per road link...' % c_pollutant)
		fit = powerlaw.Fit(list_cumulate_emissions_per_road)
		print('---------------')
		print('Fit parameters:')
		print('alpha ', fit.power_law.alpha)
		print('x_min ', fit.power_law.xmin)
		print('---------------')

		print('Comparing with %s:' %name__fit_comparison_with_)
		R, p = fit.distribution_compare('power_law', fit_comparison_with_, normalized_ratio=True)
		print('log-likelihood ratio ', R)  # positive if the data is more likely in the first distribution, and negative if the data is more likely in the second distribution.
		print('p-val ', p)
		if p <= 0.05:
			if R > 0:
				print('=> A power law better fits the data.')
			if R < 0:
				print('=> A %s better fits the data.' %name__fit_comparison_with_)
		else:
			print('=> Neither distribution is a significantly stronger fit (p > 0.05).')
		print('---------------')

		# powerlaw.plot_pdf(list_emissions_per_road, color='b')
		fig = fit.plot_ccdf(color='navy', linewidth=2, label='Data')
		fit.power_law.plot_ccdf(color='b', linestyle='--', ax=fig,
								label=r'Power law fit, $\alpha$=%.2f' % fit.power_law.alpha)

		#### NOTE: comment/uncomment the following lines w.r.t. the comparison wanted:
		fit.truncated_power_law.plot_ccdf(color='lightblue', linestyle='--', ax=fig, label=r'Truncated pw law fit, $\alpha$=%.2f, $\lambda$=%.2f' % (fit.truncated_power_law.alpha, fit.truncated_power_law.parameter2))  # pars: alpha, lambda
		# fit.exponential.plot_ccdf(color='r', linestyle='--', ax=fig, label=r'Exponential fit, $\lambda$=%.2f' %fit.exponential.parameter1)
		# fit.lognormal.plot_ccdf(color='r', linestyle='--', ax=fig, label=r'Lognormal fit, $\mu$=%.2f, $\sigma$=%.2f' %(fit.lognormal.mu, fit.lognormal.sigma))

		plt.legend(loc="lower left", frameon=False)
		plt.xlabel(x_label_road)
		plt.ylabel(r'$P(X \geq x)$')
		plt.title('Fit emissions of %s per road link' % c_pollutant)
		filename = str('fit_emissions_per_road_%s__%s__%s__%s.png' % (c_pollutant, fit_comparison_with_, normalization_factor, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()

		###
		print('')
		print('Fitting the distribution of emissions of %s per vehicle...' % c_pollutant)
		fit = powerlaw.Fit(list_cumulate_emissions_per_vehicle)
		print('---------------')
		print('Fit parameters:')
		print('alpha ', fit.power_law.alpha)
		print('x_min ', fit.power_law.xmin)
		print('---------------')

		print('Comparing with %s:' %name__fit_comparison_with_)
		R, p = fit.distribution_compare('power_law', fit_comparison_with_, normalized_ratio=True)
		print('log-likelihood ratio ', R)  # positive if the data is more likely in the first distribution, and negative if the data is more likely in the second distribution.
		print('p-val ', p)
		if p <= 0.05:
			if R > 0:
				print('=> A power law better fits the data.')
			if R < 0:
				print('=> A %s better fits the data.' %name__fit_comparison_with_)
		else:
			print('=> Neither distribution is a significantly stronger fit (p > 0.05).')
		print('---------------')

		# powerlaw.plot_pdf(list_emissions_per_road, color='b')
		fig = fit.plot_ccdf(color='navy', linewidth=2, label='Data')
		fit.power_law.plot_ccdf(color='b', linestyle='--', ax=fig,
								label=r'Power law fit, $\alpha$=%.2f' % fit.power_law.alpha)

		#### NOTE: comment/uncomment the following lines w.r.t. the comparison wanted:
		fit.truncated_power_law.plot_ccdf(color='lightblue', linestyle='--', ax=fig, label=r'Truncated pw law fit, $\alpha$=%.2f, $\lambda$=%.2f' % (fit.truncated_power_law.alpha, fit.truncated_power_law.parameter2))  # pars: alpha, lambda
		# fit.exponential.plot_ccdf(color='r', linestyle='--', ax=fig, label=r'Exponential fit, $\lambda$=%.2f' %fit.exponential.parameter1)
		# fit.lognormal.plot_ccdf(color='r', linestyle='--', ax=fig, label=r'Lognormal fit, $\mu$=%.2f, $\sigma$=%.2f' %(fit.lognormal.mu, fit.lognormal.sigma))

		plt.legend(loc="lower left", frameon=False)
		plt.xlabel(x_label_vehicle)
		plt.ylabel(r'$P(X \geq x)$')
		plt.title('Fit emissions of %s per vehicle' % c_pollutant)
		filename = str('fit_emissions_per_vehicle_%s__%s__%s.png' % (c_pollutant, fit_comparison_with_, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()


###########################################################################################################
################################## MAP EMISSIONS OVER ROAD NETWORK #######################################
###########################################################################################################

if map_emissions:
	print('Mapping %s...' %pollutant_to_map)
	fig, ax = plot_road_network_with_emissions(tdf, road_network,
											   normalization_factor = normalization_factor,
											   name_of_pollutant=pollutant_to_map,
											   color_map='autumn_r',
											   bounding_box=bbox_subregion,
											   save_fig=True)