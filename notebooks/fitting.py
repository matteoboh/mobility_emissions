import powerlaw
import matplotlib.pyplot as plt
import itertools

###### PARAMETERS
# fontisizes of plot elements:
title_size = 22
label_size = 21
ticks_size = 20
legend_size = 16
######

###########################################################################################################
###################################### DEFINITION OF METHODS ##############################################
###########################################################################################################

def fit_powerlaw(map__date__distrib, region_name, metric_name, xmin=None, sigma_threshold=None,
				 list_distributions_to_fit=['power_law'],
				 plot_ccdf=False, plot_pdf=False, plot_xmin=False, x_label=r'$x$',
				 single_fit_to_plot='truncated_power_law', save_figures=False):
	"""Uses powerlaw package to fit a power-law to the data and eventually compare more than one fit.

	Parameters:
	-----------
	map__date__distrib : dictionary
		a dictionary with the date as key and the list of points as value.

	region_name : str
		the name of the region/city to which the data belong. This is just used for (eventually) saving figures and results.

	metric_name : str
		the name of the pollutant.

	xmin : int or float
		The data value beyond which distributions should be fitted.
		If None an optimal one will be calculated.
		See powerlaw package for details.

	sigma_threshold : float
		Upper limit on the standard error of the power law fit. Used after fitting, when identifying valid xmin values.

	list_distributions_to_fit : list
		the list of names of distributions to fit and compare (if more than one). Must be non-empty.
		They can be chosen from: {'power_law','lognormal','exponential','truncated_power_law','stretched_exponential','lognormal_positive'}.

	plot_ccdf : bool
		whether to plot the Complementary Cumulative Distribution Function of the data together with the best fit(s) found (and parameters), or not.

	plot_pdf : bool
		whether to plot the Probability Density Function of the data together with the best fit(s) found (and parameters), or not.

	plot_xmin : bool
		whether to plot the value of the Kolmogorov-Smirnov distance obtained for each choice of xmin,
		together with sigma and the optimal xmin.

	x_label : str
		the x label for the plot (the y label is fixed and depends on the type of plot).

	single_fit_to_plot : str
		the name of one of the fitted model to plot alone with the data.
		Ignored if plot_ccdf = False.

	save_figures : bool
		whether to save or not the figures obtained with plot_ccdf, plot_pdf, plot_xmin.

	Returns:
	--------
	dictionary
		two dictionaries, both with the type of dstribution as key, and as values respectively:
		a dictionary collecting the results of the fitting procedure;
		a dictionary collecting the results of the loglikelihood tests between the fitted distributions.
	"""

	map__fit__params = {
		'power_law': ['alpha'],
		'lognormal': ['mu', 'sigma'],
		'exponential': ['Lambda'],
		'truncated_power_law': ['alpha', 'Lambda'],
		'stretched_exponential': ['Lambda', 'beta'],
		'lognormal_positive': ['mu', 'sigma']
	}

	def get_fit_params_(fit, fit_name):
		best_fit_features = getattr(fit, fit_name)
		map__par__value = {par: getattr(best_fit_features, par) for par in map__fit__params[fit_name]}
		map__par__value['xmin'] = getattr(best_fit_features, 'xmin')

		return map__par__value

	list_of_dates = sorted(map__date__distrib.keys())
	map__date__fitting_results = {date: {} for date in list_of_dates}
	map__date__comparison_results = {date: {} for date in list_of_dates}

	for date, distrib in sorted(map__date__distrib.items()):
		print('-- Fitting distribution %s --' % date)
		print('num points : ', len(distrib))
		fit = powerlaw.Fit(distrib, xmin=xmin, sigma_threshold=sigma_threshold, verbose=False)
		print('fixed xmin: ', fit.fixed_xmin)

		# 1. comparison of fits:
		if len(list_distributions_to_fit) > 1:
			map__distribution__n_best_fits = {dist: 0 for dist in list_distributions_to_fit}
			map__comparison__results = {}
			for (dist1, dist2) in itertools.combinations(list_distributions_to_fit, 2):
				best_fit, R, p = compare_fits_(fit, dist1, dist2)
				map__comparison__results[(dist1, dist2)] = {'R': R, 'p': p, 'best_fit': best_fit}
				if best_fit != None:
					map__distribution__n_best_fits[best_fit] += 1

			# 2. extracting the best fit (if any) resulting from the comparisons:
			list_best_fits_ever = [dist for dist in map__distribution__n_best_fits.keys() if
								   map__distribution__n_best_fits[dist] == max(map__distribution__n_best_fits.values())]
			# there could be more than one best fits:
			# e.g. power-law and lognormal "won" 2 comparisons, then take as the best fit the one which won the (power-law, lognormal) comparison (if any)
			if len(list_best_fits_ever) > 1:
				if len(list_best_fits_ever) == 2:
					winning_fit = [map__comparison__results[(dist1, dist2)]['best_fit'] for (dist1, dist2) in
								   itertools.combinations(list_best_fits_ever, 2)]
					if winning_fit != [None]:
						list_best_fits_ever = winning_fit
				else:
					results_of_comparison_between_best_fits = [map__comparison__results[(dist1, dist2)]['best_fit'] for
															   (dist1, dist2) in
															   itertools.combinations(list_best_fits_ever, 2)]
					results_of_comparison_between_best_fits_without_None = [el for el in
																			results_of_comparison_between_best_fits if
																			el != None]
					if not results_of_comparison_between_best_fits_without_None:
						list_best_fits_ever = [None]  # no winner
					else:
						list_best_fits_ever = max(results_of_comparison_between_best_fits_without_None,
												  key=results_of_comparison_between_best_fits_without_None.count)

			map__date__comparison_results[date] = map__comparison__results

		else:
			list_best_fits_ever = list_distributions_to_fit
			print('> No comparison has been made, assuming %s as best fit. <' % list_best_fits_ever)

		# 3. saving fitting (and comparison) results:
		for c_dist in list_distributions_to_fit:
			map__parameter__value = get_fit_params_(fit, c_dist)
			if c_dist in list_best_fits_ever:
				map__parameter__value['best_fit'] = True
			else:
				map__parameter__value['best_fit'] = False
			map__date__fitting_results[date][c_dist] = map__parameter__value

		print('Best fit(s) : ', list_best_fits_ever)
		print('-- end --')

		if plot_ccdf:
			plot_ccdf_with_fit(fit, distrib, map__date__fitting_results[date], date, metric_name, region_name, x_label, save_figures)
			plot_ccdf_with_one_fit(fit, distrib, map__date__fitting_results[date], single_fit_to_plot, date,
								   metric_name, region_name, x_label, save_figures)
		if plot_pdf:
			plot_pdf_with_fit(fit, distrib, map__date__fitting_results[date], date, metric_name, region_name, x_label, save_figures)
		if plot_xmin:
			plot_xmin_choice(fit, metric_name, date, region_name, save_figures)

	return map__date__fitting_results, map__date__comparison_results


###

def compare_fits_(fit, dist1, dist2):
	R, p = fit.distribution_compare(dist1, dist2, normalized_ratio=True)
	# The log-likelihood ratio R is
	#   - positive if the data is more likely in the first distribution,
	#   - negative if the data is more likely in the second distribution.

	# print('log-likelihood ratio ', R)
	# print('p-val ', p)
	if p <= 0.05:
		if R > 0:
			# print('=> A power law better fits the data.')
			best_fit = dist1
		if R < 0:
			# print('=> A %s better fits the data.' %fit_comparison_with_)
			best_fit = dist2
	else:
		# print('=> Neither distribution is a significantly stronger fit (p > 0.05).')
		best_fit = None

	return best_fit, R, p


###

def plot_ccdf_with_fit(fit, data, map__distribution__fit_results, date, metric_name, region_name, x_label,
					   save_fig=False):
	map__fit__plot_feat = {
		'power_law': {'legend_name': 'powerlaw', 'color': 'orange', 'lstyle': '--'},
		'lognormal': {'legend_name': 'lognormal', 'color': 'blue', 'lstyle': '-.'},
		'exponential': {'legend_name': 'exponential', 'color': 'green', 'lstyle': ':'},
		'truncated_power_law': {'legend_name': 'tr. powerlaw', 'color': 'red', 'lstyle': '-'},
		'stretched_exponential': {'legend_name': 'str. exponential', 'color': 'magenta', 'lstyle': ':'},
		'lognormal_positive': {'legend_name': 'lognormal pos.', 'color': 'salmon', 'lstyle': '-.'},
	}

	fig = plt.figure(figsize=(6, 6))
	ax = plt.axes()

	# plotting the ccdf of the data:
	# powerlaw.plot_ccdf(data, color='navy', label='data', ax=ax)
	from powerlaw import ccdf
	x, y = ccdf(data, linear_bins=False)
	ax.scatter(x, y, color='black', s=3, label='data')

	# selecting the best fit(s):
	best_fits_ever = [name_fit for name_fit in map__distribution__fit_results if
					  map__distribution__fit_results[name_fit]['best_fit'] == True]

	map__distribution_to_plot__fit_results = {dist: fit_results for dist, fit_results in
											  map__distribution__fit_results.items() if dist != 'exponential'}
	for dist, dict_fit_results in map__distribution_to_plot__fit_results.items():
		if dist in best_fits_ever:
			linewidth = 3.5
			label = r'$\bf{%s}$' % map__fit__plot_feat[dist]['legend_name'] + ', \n'
		else:
			linewidth = 2.5
			label = '%s, \n' % map__fit__plot_feat[dist]['legend_name']
		for c_par, c_val in dict_fit_results.items():
			if c_par not in ['xmin', 'best_fit']:
				c_val_approx = '%.2e' % c_val if abs(c_val) < 10 ** (-2) or abs(c_val) > 10 ** (2) else '%.2f' % c_val
				label += r'$\%s$=%s ' % (c_par.lower(), c_val_approx)
		getattr(fit, dist).plot_ccdf(color=map__fit__plot_feat[dist]['color'],
									 linestyle=map__fit__plot_feat[dist]['lstyle'], linewidth=linewidth, ax=ax,
									 label=label)

	# reordering legend labels
	handles, labels = plt.gca().get_legend_handles_labels()
	handles = handles[-1:] + handles[:-1]
	labels = labels[-1:] + labels[:-1]

	plt.legend(handles, labels, loc='lower left', frameon=False, fontsize=legend_size, scatterpoints=5,
			   markerscale=0.8, handlelength=1.5, handletextpad=0.6, borderpad=0.2, borderaxespad=0.2)

	plt.grid(alpha=0.2)
	plt.xlabel(x_label, fontsize=label_size)
	plt.ylabel(r'CCDF: $P(X>x)$', fontsize=label_size)

	plt.xticks(fontsize=ticks_size)
	plt.yticks(fontsize=ticks_size)

	plot_title = str(r'$%s$' % metric_name) + ' ' + date
	plt.title(plot_title, loc='center', pad=None, fontdict={'fontsize': title_size})

	if save_fig:
		plot_file_name = "plot_fit__" + metric_name + "__" + date.replace(' ', '_') + "__" + region_name + "__CCDF.png"
		plt.savefig(plot_file_name, dpi=300, bbox_inches='tight')
		plot_file_name_pdf = "plot_fit__" + metric_name + "__" + date.replace(' ', '_') + "__" + region_name + "__CCDF.pdf"
		plt.savefig(plot_file_name_pdf, dpi=300, bbox_inches='tight')
		plt.clf()
		plt.close()
	else:
		plt.show()

	return


###

def plot_pdf_with_fit(fit, data, map__distribution__fit_results, date, metric_name, region_name, x_label,
					  save_fig=False):
	map__fit__plot_feat = {
		'power_law': {'legend_name': 'powerlaw', 'color': 'orange', 'lstyle': '--'},
		'lognormal': {'legend_name': 'lognormal', 'color': 'blue', 'lstyle': '-.'},
		'exponential': {'legend_name': 'exponential', 'color': 'green', 'lstyle': ':'},
		'truncated_power_law': {'legend_name': 'tr. powerlaw', 'color': 'red', 'lstyle': '-'},
		'stretched_exponential': {'legend_name': 'str. exponential', 'color': 'magenta', 'lstyle': ':'},
		'lognormal_positive': {'legend_name': 'lognormal pos.', 'color': 'salmon', 'lstyle': '-.'},
	}

	fig = plt.figure(figsize=(6, 6))
	ax = plt.axes()

	# plotting the pdf of the data:
	# powerlaw.plot_pdf(data, color='navy', linear_bins=True, label='data')
	from powerlaw import pdf
	x, y = pdf(data, linear_bins=False)
	ind = y > 0
	y = y[ind]
	x = x[:-1]
	x = x[ind]
	ax.scatter(x, y, color='black', s=3, label='data')

	# selecting the best fit(s):
	best_fits_ever = [name_fit for name_fit in map__distribution__fit_results if
					  map__distribution__fit_results[name_fit]['best_fit'] == True]

	map__distribution_to_plot__fit_results = {dist: fit_results for dist, fit_results in
											  map__distribution__fit_results.items() if dist != 'exponential'}
	for dist, dict_fit_results in map__distribution_to_plot__fit_results.items():
		if dist in best_fits_ever:
			linewidth = 3.5
			label = r'$\bf{%s}$' % map__fit__plot_feat[dist]['legend_name'] + ', \n'
		else:
			linewidth = 2.5
			label = '%s, \n' % map__fit__plot_feat[dist]['legend_name']
		for c_par, c_val in dict_fit_results.items():
			if c_par not in ['xmin', 'best_fit']:
				c_val_approx = '%.2e' % c_val if abs(c_val) < 10 ** (-2) or abs(c_val) > 10 ** (2) else '%.2f' % c_val
				label += r'$\%s$=%s ' % (c_par.lower(), c_val_approx)
		getattr(fit, dist).plot_pdf(color=map__fit__plot_feat[dist]['color'],
									linestyle=map__fit__plot_feat[dist]['lstyle'], linewidth=linewidth, ax=ax,
									label=label)

	# reordering legend labels
	handles, labels = plt.gca().get_legend_handles_labels()
	handles = handles[-1:] + handles[:-1]
	labels = labels[-1:] + labels[:-1]

	plt.legend(handles, labels, loc='lower left', frameon=False, fontsize=legend_size, scatterpoints=5,
			   markerscale=0.8, handlelength=1.5, handletextpad=0.6, borderpad=0.2, borderaxespad=0.2)

	plt.grid(alpha=0.2)
	plt.xlabel(x_label, fontsize=label_size)
	plt.ylabel(r'$P(X)$', fontsize=label_size)

	plt.xticks(fontsize=ticks_size)
	plt.yticks(fontsize=ticks_size)

	plot_title = str(r'$%s$' % metric_name) + ' ' + date
	plt.title(plot_title, loc='center', pad=None, fontdict={'fontsize': title_size})

	if save_fig:
		plot_file_name = "plot_fit__" + metric_name + "__" + date.replace(' ', '_') + "__" + region_name + "__PDF.png"
		plt.savefig(plot_file_name, dpi=300, bbox_inches='tight')
		plot_file_name_pdf = "plot_fit__" + metric_name + "__" + date.replace(' ',
																			  '_') + "__" + region_name + "__PDF.pdf"
		plt.savefig(plot_file_name_pdf, dpi=300, bbox_inches='tight')
		plt.clf()
		plt.close()
	else:
		plt.show()

	return


###

def plot_ccdf_with_one_fit(fit, data, map__distribution__fit_results, fit_to_plot, date, metric_name, region_name,
						   x_label, save_fig=False):
	map__fit__plot_feat = {
		'power_law': {'legend_name': 'powerlaw', 'color': 'orange', 'lstyle': '--'},
		'lognormal': {'legend_name': 'lognormal', 'color': 'blue', 'lstyle': '-.'},
		'exponential': {'legend_name': 'exponential', 'color': 'green', 'lstyle': ':'},
		'truncated_power_law': {'legend_name': 'tr. powerlaw', 'color': 'red', 'lstyle': '-'},
		'stretched_exponential': {'legend_name': 'str. exponential', 'color': 'magenta', 'lstyle': ':'},
		'lognormal_positive': {'legend_name': 'lognormal pos.', 'color': 'salmon', 'lstyle': '-.'},
	}

	fig = plt.figure(figsize=(6, 6))
	ax = plt.axes()

	# plotting the ccdf of the data:
	# powerlaw.plot_ccdf(data, color='navy', label='data', ax=ax)
	from powerlaw import ccdf
	x, y = ccdf(data, linear_bins=False)
	ax.scatter(x, y, color='black', s=3, label='data')

	dict_fit_results = map__distribution__fit_results[fit_to_plot]
	label = '%s, \n' % map__fit__plot_feat[fit_to_plot]['legend_name']
	for c_par, c_val in dict_fit_results.items():
		if c_par not in ['xmin', 'best_fit']:
			c_val_approx = '%.2e' % c_val if abs(c_val) < 10 ** (-2) or abs(c_val) > 10 ** (2) else '%.2f' % c_val
			label += r'$\%s$=%s ' % (c_par.lower(), c_val_approx)
	getattr(fit, fit_to_plot).plot_ccdf(color=map__fit__plot_feat[fit_to_plot]['color'],
										linestyle=map__fit__plot_feat[fit_to_plot]['lstyle'],
										linewidth=2.5, ax=ax, label=label)

	# reordering legend labels
	handles, labels = plt.gca().get_legend_handles_labels()
	handles = handles[-1:] + handles[:-1]
	labels = labels[-1:] + labels[:-1]

	plt.legend(handles, labels, loc='lower left', frameon=False, fontsize=legend_size+6, scatterpoints=5,
			   markerscale=0.8, handlelength=1.5, handletextpad=0.6, borderpad=0.2, borderaxespad=0.2)

	plt.grid(alpha=0.2)
	plt.xlabel(x_label, fontsize=label_size)
	plt.ylabel(r'CCDF: $P(X>x)$', fontsize=label_size)

	plt.xticks(fontsize=ticks_size)
	plt.yticks(fontsize=ticks_size)

	plot_title = str(r'$%s$' % metric_name) + ' ' + date
	plt.title(plot_title, loc='center', pad=None, fontdict={'fontsize': title_size})

	if save_fig:
		plot_file_name = "plot_fit__" + metric_name + "__" + date.replace(' ',
																		  '_') + "__" + region_name + "__CCDF__" + fit_to_plot + ".png"
		plt.savefig(plot_file_name, dpi=300, bbox_inches='tight')
		plot_file_name_pdf = "plot_fit__" + metric_name + "__" + date.replace(' ',
																			  '_') + "__" + region_name + "__CCDF__" + fit_to_plot + ".pdf"
		plt.savefig(plot_file_name_pdf, dpi=300, bbox_inches='tight')
		plt.clf()
		plt.close()
	else:
		plt.show()

	return


###

def plot_xmin_choice(fit, metric_name, date, region_name, save_fig=False):
	fig = plt.figure(figsize=(6, 4))
	ax = plt.axes()

	ax.plot(fit.xmins, fit.Ds, label=r'$D$', color='navy', linewidth=2.5)
	ax.plot(fit.xmins, fit.sigmas, label=r'$\sigma$', linestyle='-.', color='black', linewidth=2.5)
	ax.vlines(fit.xmin, 0, 0.4, colors='red', linestyles='--', linewidth=3, label=r'$x_{min}$ = %.2e' %fit.xmin if abs(fit.xmin)< 10**(-2) else r'$x_{min}$ = %.2f' %fit.xmin)
	plt.ylim(0, 0.4)
	plt.legend(loc='lower right', fontsize=legend_size, markerscale=0.8, handlelength=1.5, handletextpad=0.6, borderpad=0.2, borderaxespad=0.2)
	plt.xlabel(r'$x_{min}$', fontsize=label_size)
	plt.ylabel(r'$D,\sigma$', fontsize=label_size)
	plt.xticks(fontsize=ticks_size-2, rotation=45)
	plt.yticks(fontsize=ticks_size-2)
	plt.title(r'$%s$ %s' %(metric_name,date), fontsize=title_size)
	plt.grid(alpha=0.2)

	if save_fig:
		plt.savefig('plot_xmin_D__%s_%s__%s.png' %(metric_name,date.replace(' ', '_'),region_name), dpi=300, bbox_inches='tight')
		plt.clf()
		plt.close()
	else:
		plt.show()

	return