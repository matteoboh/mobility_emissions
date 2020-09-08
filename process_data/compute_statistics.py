import skmob
import numpy as np
import osmnx as ox
import matplotlib.pyplot as plt
from .all_methods import *
from .stats_utils import *

###### PARAMETERS
PATH_TO_INPUT_FILE = './output_files/'
NAME_OF_INPUT_FILE = 'italy_27837users_2017-01-01_2017-02-02__florence_4330users__filtered_3633users_120sec__WITH_EMISSIONS.csv'
PATH_TO_ROAD_NETWORKS = './input_files/road_nets/'
region = 'Florence'
subregion = None #'Quartiere XXXII Europa' #'Municipio Roma I' #'Rione XV Esquilino'
list_of_pollutants = ['CO_2', 'NO_x', 'PM', 'VOC']

# set to True/False w.r.t. what you want to obtain or not:
plot_distributions = False
fit_distributions = False   # in some cases, this can take time
compute_correlations = False
print_statistics = True
######

### Loading tdf
tdf = skmob.read(PATH_TO_INPUT_FILE + NAME_OF_INPUT_FILE)

###########################################################################################################
################################### PLOT and FIT DISTRIBUTIONS ############################################
###########################################################################################################

### Extract overall distribution of pollution
dict_pollutant_to_entire_distribution = {}
for c_pollutant in list_of_pollutants:
	dict_pollutant_to_entire_distribution[c_pollutant] = np.array(tdf[c_pollutant])

### Loading road network (needed to compute the distribution of pollutant per road)
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
###


for c_pollutant in list_of_pollutants:

	dict_road_to_emissions = map_road_to_emissions(tdf, road_network, name_of_pollutant=c_pollutant)
	list_emissions_per_road = []
	for c_array in dict_road_to_emissions.values():
		c_sum = np.sum(c_array)
		list_emissions_per_road.extend([c_sum])

	dict_vehicle_to_emissions = map_vehicle_to_emissions(tdf, name_of_pollutant=c_pollutant)
	list_emissions_per_vehicle = []
	for c_array in dict_vehicle_to_emissions.values():
		c_sum = np.sum(c_array)
		list_emissions_per_vehicle.extend([c_sum])

	#####################
	if plot_distributions:

		#
		def plot_loglog(values_to_plot, pollutant):
			fig = plt.figure()
			x, y = zip(*lbpdf(2.0, values_to_plot))
			plt.plot(x, y, marker='o')
			plt.grid(alpha=0.2)
			plt.xlabel('%s [g]' % pollutant, fontsize=12)
			plt.loglog()
			return fig
		#

		###########################################
		#### plot overall distribution of pollution:

		print('Plotting the distribution of overall emissions of %s...' % c_pollutant)
		c_array = dict_pollutant_to_entire_distribution[c_pollutant]
		#plt.hist(c_array, bins=20, range=(0, np.percentile(c_array,90)))
		#
		fig = plot_loglog(list(c_array), c_pollutant)
		#
		plt.title('Distribution of emissions of %s' % c_pollutant)
		filename = str('loglog_emissions_%s__%s.png' %(c_pollutant, region_name))
		fig.savefig(filename, format='png', bbox_inches='tight')
		plt.close()

		#################################################
		#### plot distribution of pollution per road link:

		print('Plotting the distribution of emissions of %s per road link...' % c_pollutant)
		#plt.hist(emissions, bins=20, range=(0, np.percentile(emissions,90))) ### cutting at 90th percentile
		#
		fig = plot_loglog(list_emissions_per_road, c_pollutant)
		#
		plt.title('Distribution of emissions of %s per road link' %c_pollutant)
		filename = str('loglog_emissions_per_road_link_%s__%s.png' %(c_pollutant, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()

		###############################################
		#### plot distribution of pollution per vehicle:

		print('Plotting the distribution of emissions of %s per vehicle...' % c_pollutant)
		#plt.hist(emissions, bins=20, range=(0, np.percentile(emissions,90))) ### cutting at 90th percentile
		#
		fig = plot_loglog(list_emissions_per_vehicle, c_pollutant)
		#
		plt.title('Distribution of emissions of %s per vehicle' %c_pollutant)
		filename = str('loglog_emissions_per_vehicle_%s__%s.png' %(c_pollutant, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()


	####################
	if fit_distributions:
		import powerlaw

		print('')
		print('Fitting the distribution of emissions of %s per road link...' % c_pollutant)
		fit = powerlaw.Fit(list_emissions_per_road)
		print('---------------')
		print('Fit parameters:')
		print('alpha ', fit.power_law.alpha)
		print('x_min ', fit.power_law.xmin)
		print('---------------')

		print('Comparing with truncated power law:')
		R, p = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)
		print('log-likelihood ratio ', R) # positive if the data is more likely in the first distribution, and negative if the data is more likely in the second distribution.
		print('p-val ', p)
		if p <= 0.05:
			if R > 0:
				print('=> A power law better fits the data.')
			if R < 0:
				print('=> A truncated power law better fits the data.')
		else:
			print('=> Neither distribution is a significantly stronger fit (p > 0.05).')
		print('---------------')

		#powerlaw.plot_pdf(list_emissions_per_road, color='b')
		fig = fit.plot_ccdf(color='navy', linewidth=2, label='Data')
		fit.power_law.plot_ccdf(color='b', linestyle='--', ax=fig, label=r'Power law fit, $\alpha$=%.2f' %fit.power_law.alpha)
		fit.truncated_power_law.plot_ccdf(color='lightblue', linestyle='--', ax=fig, label=r'Truncated pw law fit, $\alpha$=%.2f, $\lambda$=%.2f' %(fit.truncated_power_law.alpha, fit.truncated_power_law.parameter2)) # pars: alpha, lambda
		#fit.exponential.plot_ccdf(color='r', linestyle='--', ax=fig, label=r'Exponential fit, $\lambda$=%.2f' %fit.exponential.parameter1)
		#fit.lognormal.plot_ccdf(color='r', linestyle='--', ax=fig, label=r'Lognormal fit, $\mu$=%.2f, $\sigma$=%.2f' %(fit.lognormal.mu, fit.lognormal.sigma))
		plt.legend(loc="lower left", frameon=False)
		plt.xlabel('%s (g)' %c_pollutant)
		plt.ylabel(r'$P(X \geq x)$')
		plt.title('Fit emissions of %s per road link' % c_pollutant)
		filename = str('fit_emissions_per_road_%s__%s.png' %(c_pollutant, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()

		###
		print('')
		print('Fitting the distribution of emissions of %s per vehicle...' % c_pollutant)
		fit = powerlaw.Fit(list_emissions_per_vehicle)
		print('---------------')
		print('Fit parameters:')
		print('alpha ', fit.power_law.alpha)
		print('x_min ', fit.power_law.xmin)
		print('---------------')

		print('Comparing with truncated power law:')
		R, p = fit.distribution_compare('power_law', 'truncated_power_law', normalized_ratio=True)
		print('log-likelihood ratio ', R)  # positive if the data is more likely in the first distribution, and negative if the data is more likely in the second distribution.
		print('p-val ', p)
		if p <= 0.05:
			if R > 0:
				print('=> A power law better fits the data.')
			if R < 0:
				print('=> A truncated power law better fits the data.')
		else:
			print('=> Neither distribution is a significantly stronger fit (p > 0.05).')
		print('---------------')

		# powerlaw.plot_pdf(list_emissions_per_road, color='b')
		fig = fit.plot_ccdf(color='navy', linewidth=2, label='Data')
		fit.power_law.plot_ccdf(color='b', linestyle='--', ax=fig, label=r'Power law fit, $\alpha$=%.2f' %fit.power_law.alpha)
		fit.truncated_power_law.plot_ccdf(color='lightblue', linestyle='--', ax=fig, label=r'Truncated pw law fit, $\alpha$=%.2f, $\lambda$=%.2f' %(fit.truncated_power_law.alpha, fit.truncated_power_law.parameter2)) # pars: alpha, lambda
		#fit.exponential.plot_ccdf(color='r', linestyle='--', ax=fig, label=r'Exponential fit, $\lambda$=%.2f' %fit.exponential.parameter1)
		#fit.lognormal.plot_ccdf(color='r', linestyle='--', ax=fig, label=r'Lognormal fit, $\mu$=%.2f, $\sigma$=%.2f' %(fit.lognormal.mu, fit.lognormal.sigma))
		plt.legend(loc="lower left", frameon=False)
		plt.xlabel('%s (g)' % c_pollutant)
		plt.ylabel(r'$P(X \geq x)$')
		plt.title('Fit emissions of %s per vehicle' % c_pollutant)
		filename = str('fit_emissions_per_vehicle_%s__%s.png' %(c_pollutant, region_name))
		plt.savefig(filename, format='png', bbox_inches='tight')
		plt.close()


###########################################################################################################
##################################### COMPUTE CORRELATIONS ################################################
###########################################################################################################

if compute_correlations:
	df_corrs = compute_corrs(tdf, list_of_pollutants)
	df_corrs.to_csv(PATH_TO_INPUT_FILE+'/correlations__%s.csv' %region_name, index=True)

###########################################################################################################
###################################### COMPUTE STATISTICS #################################################
###########################################################################################################

if print_statistics:

	num_vehicles = len(set(tdf['uid']))
	num_roads = len(set(tuple(i) for i in tdf['road_link']))  ## road links are lists of two elements each (non-hashable, so it is necessary to create a set of tuples instead)

	for c_pollutant in list_of_pollutants:

		print('------------------------------')
		print('Some stats for %s emissions:' %c_pollutant)
		print('------------------------------')

		## stats distribution PER VEHICLE ##
		dict_vehicle_to_emissions = map_vehicle_to_emissions(tdf, name_of_pollutant=c_pollutant)
		dict_vehicle_to_total_emissions = {}
		emissions = []
		max_sum = 0
		min_sum = float('inf')
		for c_uid, c_array in dict_vehicle_to_emissions.items():
			c_sum = np.sum(c_array)
			dict_vehicle_to_total_emissions[c_uid] = c_sum
			if c_sum > max_sum:
				max_sum = c_sum
				uid_max = c_uid
			if c_sum < min_sum:
				min_sum = c_sum
				uid_min = c_uid
			emissions.extend([c_sum])

		print('----- stats per vehicle ------')

		overall_sum = np.sum(emissions)
		print('Overall value of emission: ', overall_sum)
		overall_mean = np.mean(emissions)
		print('Mean value of emissions per vehicle: %s (the %s of total emissions)' %(overall_mean, overall_mean/overall_sum))
		overall_median = np.median(emissions)
		print('Median value of emissions per vehicle: ', overall_median)
		rate_max = max_sum/overall_sum
		rate_min = min_sum/overall_sum
		print('User %s emitted the most (%s grams, the %s of total emissions in the network).' %(uid_max, max_sum, rate_max))
		print('User %s emitted the least (%s grams, the %s of total emissions in the network).' %(uid_min, min_sum, rate_min))
		num_vehicles_10_percent = int(num_vehicles / 100 * 10)
		sum_10_percent = 0
		for c_uid, c_tot in sorted(dict_vehicle_to_total_emissions.items(), key=lambda item: item[1], reverse=True)[0:num_vehicles_10_percent]:
			sum_10_percent += c_tot
		print('10%% of the vehicles are responsible for the %s of total emissions.' %(sum_10_percent/overall_sum))

		## stats distribution PER ROAD ##
		dict_road_to_emissions = map_road_to_emissions(tdf, road_network, name_of_pollutant=c_pollutant)
		dict_road_to_total_emissions = {}
		emissions = []
		max_sum = 0
		min_sum = float('inf')
		for c_rid, c_array in dict_road_to_emissions.items():
			c_sum = np.sum(c_array)
			dict_road_to_total_emissions[c_rid] = c_sum
			if c_sum > max_sum:
				max_sum = c_sum
				rid_max = c_rid
			if c_sum < min_sum:
				min_sum = c_sum
				rid_min = c_rid
			emissions.extend([c_sum])

		print('------- stats per road -------')

		overall_sum = np.sum(emissions)
		#print('Overall value of emission: ', overall_sum)
		overall_mean = np.mean(emissions)
		print('Mean value of emissions per road: %s (the %s of total emissions)' % (
		overall_mean, overall_mean / overall_sum))
		overall_median = np.median(emissions)
		print('Median value of emissions per road: ', overall_median)
		rate_max = max_sum / overall_sum
		rate_min = min_sum / overall_sum
		print('Road %s has the maximum quantity of emissions (%s grams, the %s of total emissions in the network).' % (
		rid_max, max_sum, rate_max))
		print('Road %s has the minimum quantity of emissions (%s grams, the %s of total emissions in the network).' % (
		rid_min, min_sum, rate_min))
		num_roads_10_percent = int(num_roads / 100 * 10)
		sum_10_percent = 0
		for c_rid, c_tot in sorted(dict_road_to_total_emissions.items(), key=lambda item: item[1], reverse=True)[
							0:num_roads_10_percent]:
			sum_10_percent += c_tot
		print('10%% of the roads have the %s of total emissions.' % (sum_10_percent / overall_sum))