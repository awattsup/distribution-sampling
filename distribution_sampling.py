import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from astropy.table import Table
from scipy.stats import ks_2samp, anderson_ksamp
from scipy.optimize import curve_fit
from matplotlib.lines import Line2D
import argparse


rng = np.random.default_rng()

def main():
	fits_filename = '/home/awatts/Adam_PhD/models_fitting/asymmetries/data/xGASS_asymmetries_catalogue.ascii'
	fits = Table.read(fits_filename,format = 'ascii')
	min_SN = 7
	# print('Number of fits',len(fits))
	goodfits = fits[np.where((fits['SN_HI'] >= min_SN) & (fits['HIconf_flag']  < 1))[0]]
	print('N fits with SN>{}'.format(min_SN), len(goodfits))
	# goodfits = goodfits[np.where(goodfits['HIconf_flag']  < 1)[0]]
	# print('Number of fits not confused', len(goodfits))

	dex_SN = 0.05
	SNbins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SNbins = np.append(SNbins,[np.log10(200)])
	Afr_bins = np.arange(1,2.2,0.05)


	satellite = goodfits[np.where(goodfits['env_code'] == 0)[0]]
	central = goodfits[np.where((goodfits['env_code'] == 1) |
		(		goodfits['env_code'] == 2))[0]]


	gass_low = goodfits[np.where(goodfits['GASS'] > 100000 )[0]]
	gass = goodfits[np.where(goodfits['GASS'] < 100000 )[0]]

 
	samples = [central['Afr_spec_HI'],satellite['Afr_spec_HI']]
	controls = [np.log10(central['SN_HI']),np.log10(satellite['SN_HI'])]
	names = ['central','satellite']
	colors = ['Blue','Red']
	legend_names = ['Centrals ({})'.format(len(central)),
					'Satellites ({})'.format(len(satellite))]



	samples_all_hists, samples_all_DAGJKsigma, samples_all_iter, controls_all_iter = \
									control_samples(samples,Afr_bins,controls,SNbins)


	plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,names,colors=colors,save=True)


#################################################################################
#		Sampling from a common 1D distribution & comparing properties
#################################################################################
def control_samples(samples, sample_bins, controls, control_bins, Niter = 1000, DAG_frac = 0.2):
	"""
	Resample two samples to conform to the same common parameter distribution

	Parameters
	----------
	samples : list, length 2
		List of the parameters that are being compared after the sampling
	sample_bins : array
		Bin edges for the parameters being compared
	controls : list, length 2
		List of the parameters of each sample that they are being drawn from to conform to 
	control_bins : array
		Bin edges for the control sample
	Niter : int
		Number of sampling iterations to perform
	DAG_frac : float, must be a fraction of 1
		Fraction of the sample to delete in each DAGJK iteration

	Returns
	-------
	samples_all_hists : list
		List of the cumulative histogram in each of Niter iterations in bins of sample_bins for each sample
	samples_all_DAGJKsigma : list
		List of the DAGJK uncertainty in each bin of the cumulative histograms for each Niter for each sample 
	indexes_all_iter : list
		List of the indicies selected for each sample in each Niter
	"""

	hist_control = np.zeros([len(samples), len(control_bins) - 1])				#Control sample histograms
	for ss in range(len(samples)):
		hist_control[ss,:] = np.histogram(controls[ss],
						 bins = control_bins, density = True)[0] 

	hist_common = np.min(hist_control, axis = 0)								#Common control histogram

	Rsamp = int(1. / DAG_frac)

	indexes_all_iter = []
	samples_all_hists = []
	samples_all_DAGJKsigma = []
	for ss in range(len(samples)):												#Iterate over provided samples

		sample = samples[ss]
		control = controls[ss]

		hist_ratio = hist_common / hist_control[ss,:]							#Fraction of sample to keep in each control histogram bin

		sampled_hists = np.zeros([Niter, len(sample_bins)-1])
		sampled_hists_DAGJKsigma = np.zeros([Niter, len(sample_bins)-1])

		index_all_iter = []
		for nn in range(Niter):													#Loop over iterations

			control_sample_index = []											
			for ii in range(len(control)):
				bincentre_diff = np.abs(control_bins[:-1] + 0.5*np.diff(control_bins) - control[ii])			#Find which control bin the data point lies in
				control_bin = np.where(bincentre_diff == np.min(bincentre_diff))[0]
				
				test = rnd.random()																				#Generate random number for keeping
				if test <= hist_ratio[control_bin]:																#If rand < fraction to keep in the bin, keep the point
					control_sample_index.extend([ii])															#Otherwise discard

			sample_control = np.array(sample[control_sample_index])												#Sample and control data points for sub-sample
			control_control = np.array(control[control_sample_index])
			
			index_all_iter.append([control_sample_index])

			control_hist, hist_inbin_indicies = histogram_indices(control_control, control_bins)				#Find data points in each bin of the control histogram

			histbin_Nsamples = np.zeros([Rsamp, len(control_bins)-1])

			for ii in range(len(control_bins) - 1):

				Ngals_inbin = int(control_hist[ii])
				jj = 0
				while(jj < Ngals_inbin):																			#Define the number of data points to select in each control histogram bin in each DAGJK iteration
					histbin_Nsamples[(jj + ii) % Rsamp, ii] += 1
					jj += 1

			DAG_indices = []
			for jj in range(Rsamp):
				DAG_sample = []
				for ii in range(len(control_bins) - 1):
					if histbin_Nsamples[jj, ii] != 0:
						indices_list = np.array(hist_inbin_indicies[ii])											#List of indices in the control distribution bin
						indices_sample = rnd.sample(range(len(indices_list)), int(histbin_Nsamples[jj,ii]))			#Randomly select the above^ defined number of datapoints 
						DAG_sample.extend(indices_list[indices_sample])												#Add to indices list to be deleted in the given DAGJK iteration
						
						indices_list = np.delete(indices_list,indices_sample)										#Remove selected indices so they can't be selected again
						hist_inbin_indicies[ii] = indices_list.tolist()												

				DAG_indices.append(DAG_sample)																		#List of indices selected in each control bin are to be deleted in the DAGJK iteration

			hist_sample_control_DAGJK = np.zeros([Rsamp,len(sample_bins)-1])
			for ii in range(len(DAG_indices)):
				sample_subset = np.delete(sample_control, DAG_indices[ii])											#Delete the selected indices ^ 
				sample_subset_hist = np.histogram(sample_subset, bins = sample_bins, density=True)[0]				#Calculate the parameter estimate
				hist_sample_control_DAGJK[ii,:] = np.cumsum(sample_subset_hist) / np.nansum(sample_subset_hist)
				
				# plt.hist(np.delete(np.log10(SNcontrol_central),del_index[ii]),bins=control_bins,histtype='step',density=True)
			#plt.show()
			Rsamp_scale = ((Rsamp - 1.) / Rsamp)																	#DAGJK parameter uncertainty estimate
			hist_sample_control_DAGJK_mean = np.mean(hist_sample_control_DAGJK, axis = 0)
			sampled_hists_DAGJKsigma[nn,:] = np.sqrt( Rsamp_scale * np.nansum(
								(hist_sample_control_DAGJK - hist_sample_control_DAGJK_mean) ** 2., axis = 0) )

			sample_control_hist = np.histogram(sample_control, bins = sample_bins, density = True)[0]				#Controlled sample cumulative histogram
			sampled_hists[nn,:] = np.cumsum(sample_control_hist) / np.nansum(sample_control_hist)


		indexes_all_iter.append(index_all_iter)																		#Lists of output for given sample
		samples_all_hists.append(sampled_hists)
		samples_all_DAGJKsigma.append(sampled_hists_DAGJKsigma)

	return samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter

def plot_controlled_cumulative_histograms(sampled_hists, sample_bins, sampled_hists_DAGJKsigma, names = None, colors = ['Orange','Green'], ls = ['-','-'], axis = None, save=None):
	"""
	Plot mean cumulative histograms of each sample, controlled to their common parameter distribution 

	Parameters
	----------
	sampled_hists : list
		List of the cumulative histogram in each of Niter iterations in bins of sample_bins for each sample
	sample_bins : array
		Bin edges 
	sampled_hists_DAGJKsigma : list
		List of the DAGJK uncertainty in each bin of the cumulative histograms for each Niter for each sample 
	names : list
		List of names of each sample for the legend
	colors : list
		Colors for the cumulative histograms
	ls : list
		List of linestyles
	axis : axis object
		Axis to plot the histogram on. Default is to define a new figure but this can be used to add it to an external figure
	save : String
		Filename to save the figure to
	"""
	if names == None:
		names = ['sample 1 ({})'.format(len(sample1)),'sample 2 ({})'.format(len(sample2))]

	if axis == None:
		fig = plt.figure(figsize = (14,9))
		gs = gridspec.GridSpec(1, 1, top = 0.98, right = 0.98, bottom  = 0.11, left = 0.08)
		axis = fig.add_subplot(gs[0,0])
	
	axis.tick_params(axis = 'both', which='both',direction = 'in', labelsize=16,length = 8, width = 1.25)
	axis.tick_params(axis = 'both', which='minor',direction = 'in', labelsize=16, length = 4, width=1.25)
	axis.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=16)
	axis.set_ylabel('Cumulative Histogram', fontsize=16)
	axis.set_ylim([0.1, 1])
	legend = []

	for ss in range(len(sampled_hists)):
		mean_sampled_hist = np.mean(sampled_hists[ss], axis=0)
		median_DAGJK_sigma = np.median(sampled_hists_DAGJKsigma[ss], axis=0)
		axis.errorbar(sample_bins[0:-1], mean_sampled_hist, yerr = median_DAGJK_sigma,
						color=colors[ss], ls = ls[ss], linewidth=3.5, capsize=6)

		legend.extend([Line2D([0], [0], color = colors[ss], ls = ls[ss], linewidth = 3)])
	axis.legend(legend,names,fontsize=16)

	if save == None and axis == None:
		plt.show()
	elif save != None:
		# fig.savefig('figures/controlled_hist_{}'.format('_'.join(names)), dpi = 200)
		fig.savefig(save, dpi = 200)

def plot_DAGJK_sigmas(samples_all_DAGJKsigma, sample_bins, save=None):
	"""
	Plot distribution of DAGJK uncertainties

	Parameters
	----------
	sample_bins : array
		Bin edges 
	sampled_hists_DAGJKsigma : list
		List of the DAGJK uncertainty in each bin of the cumulative histograms for each Niter for each sample 
	save : String
		Directory to save the  figure into
	"""
	for ss in range(len(samples_all_DAGJKsigma)):

		sample_DAGJKsigma = samples_all_DAGJKsigma[ss]

		fig = plt.figure(figsize = (12,8))
		gs = gridspec.GridSpec(4,6, hspace=0, wspace=0, top = 0.99, right = 0.99,
			 bottom  = 0.06, left = 0.06)
		for ii in range(len(sample_bins)-1):
			ax = fig.add_subplot(gs[int(ii/6),(ii)%6])
			ax.set_xlim([0,0.105])
			ax.tick_params(axis='x',which='both',direction = 'in', labelsize=12.)
			if ii%6 != 0:
				ax.tick_params(axis='y',which='both',direction = 'in', labelsize=0.)
			else:
				ax.tick_params(axis='y',which='both',direction = 'in', labelsize=12.)
				ax.set_ylabel('Histogram Density',fontsize=14)
			if ii < 18:
				ax.tick_params(axis='x',which='both',direction = 'in', labelsize=0.)
			else:
				ax.set_xlabel('$\sigma_{{JK}}$',fontsize=12)

			ax.text(0.05,0.9, '$A_{{fr}}$ = [{a:.2f},{b:.2f})'.format(
					a=sample_bins[ii],b=sample_bins[ii+1]),fontsize=12., 
					transform=ax.transAxes, zorder=1)
			ax.text(0.05,0.8, 'med($\sigma_{{JK}})$ = {:.2e}'.format(
					np.median(sample_DAGJKsigma[:,ii])), fontsize=12., transform=ax.transAxes, zorder=1)
			ax.hist(sample_DAGJKsigma[:,ii],bins=np.arange(0,0.10,0.005),density=True,alpha=0.9)
		
		if save == None:
			plt.show()
		else:
			fig.savefig(save, dpi = 200)

def plot_compare_DAGJK_Nsamp_sigmas(samples_all_DAGJKsigma, samples_all_hists, sample_bins, save=None):
	"""
	Plot distribution of DAGJK uncertainties relative to the uncertainty 
	due to sampling the samples to the same distribution

	sampled_hists : list
		List of the cumulative histogram in each of Niter iterations in bins of sample_bins for each sample
	sample_bins : array
		Bin edges 
	sampled_hists_DAGJKsigma : list
		List of the DAGJK uncertainty in each bin of the cumulative histograms for each Niter for each sample 
	

	Parameters
	----------
	sampled_hists_DAGJKsigma : list
		List of the DAGJK uncertainty in each bin of the cumulative histograms for each Niter for each sample 
	samples_all_hists : list
		List of the cumulative histogram in each of Niter iterations in bins of sample_bins for each sample
	sample_bins : array
		Bin edges 
	save : String
		Directory to save the  figure into
	"""
	for ss in range(len(samples_all_DAGJKsigma)):

		sample_DAGJKsigma = samples_all_DAGJKsigma[ss]
		sample_hists = samples_all_hists[ss]

		fig = plt.figure(figsize = (12,8))
		gs = gridspec.GridSpec(5,6, hspace=0, wspace=0, top = 0.99, right = 0.99,
			 bottom  = 0.06, left = 0.06)
		for ii in range(len(sample_bins)-1):
			ax = fig.add_subplot(gs[int(ii/6),(ii)%6])
			ax.set_xlim([0,6])
			ax.tick_params(axis='x',which='both',direction = 'in', labelsize=12.)
			if ii%6 != 0:
				ax.tick_params(axis='y',which='both',direction = 'in', labelsize=0.)
			else:
				ax.tick_params(axis='y',which='both',direction = 'in', labelsize=12.)
				ax.set_ylabel('Histogram Density',fontsize=14)
			if ii < 18:
				ax.tick_params(axis='x',which='both',direction = 'in', labelsize=0.)
			else:
				ax.set_xlabel('$\sigma_{{JK}} / \sigma_{{N}}$',fontsize=12)

			sigma_ratio_dist = sample_DAGJKsigma[:,ii] / np.std(sample_hists[:,ii])

			ax.text(0.05,0.9, '$A_{{fr}}$ = [{a:.2f},{b:.2f})'.format(
					a=sample_bins[ii],b=sample_bins[ii+1]),fontsize=12., 
					transform=ax.transAxes, zorder=1)
			ax.text(0.05,0.8, 'med($\sigma_{{JK}} / \sigma_{{N}}$) = {:.2f}'.format(
					np.median(sigma_ratio_dist)), fontsize=12., transform=ax.transAxes, zorder=1)
			ax.hist(sigma_ratio_dist,bins=10,density=True,alpha=0.9)
			
		if save == None:
			plt.show()
		else:
			save1 = '{dir}_samp{ss}.png'.format(dir=save.split('.png')[0],ss=ss+1)
			fig.savefig(save1, dpi = 200)

def plot_avg_KS_AD_test(samples_all_iter, save = None):
	"""
	Compute two-sample Kolmogorv-Smirnov and Anderson-Darling on each iteration of Niter 
	and compute the median test value for each.

	Parameters
	----------
	samples_all_iter : list
		List of controlled sub-samples in each Niter for each sample
	save : String
		Directory to save each test figure output to 
	"""

	samp1 = samples_all_iter[0]
	samp2 = samples_all_iter[1]
	Niter = len(samp1)

	KS_test_out = np.zeros([Niter,2])
	AD_test_out = np.zeros([Niter,2])
	for ii in range(Niter):
		KStest = ks_2samp(samp1[ii],samp2[ii])
		KS_test_out[ii,:] = KStest[0:2]
		ADtest = anderson_ksamp([samp1[ii],samp2[ii]])
		AD_test_out[ii,0] = ADtest[0]
		AD_test_out[ii,1] = ADtest[2]

	KStest_fig = plt.figure(figsize = (16,8))
	gs = gridspec.GridSpec(1,2, hspace=0, wspace=0, top = 0.99, right = 0.99,
		 bottom  = 0.1, left = 0.1)
	KSDval_ax = KStest_fig.add_subplot(gs[0,0])
	KSpval_ax = KStest_fig.add_subplot(gs[0,1],sharey = KSDval_ax)

	KSDval_ax.set_xlabel('KS D-value',fontsize=22)
	KSpval_ax.set_xlabel('KS p-value',fontsize=22)
	KSDval_ax.set_ylabel('Histogram Density',fontsize=22)
	KSDval_ax.tick_params(axis = 'both', which='both',direction = 'in', labelsize=22, length = 8, width=1.25)
	KSDval_ax.tick_params(axis = 'both', which='minor',direction = 'in', labelsize=22, length = 4, width=1.25)
	KSpval_ax.tick_params(axis = 'both', which='both',direction = 'in', labelsize=22, length = 8, width=1.25)
	KSpval_ax.tick_params(axis = 'y', which='both',direction = 'in', labelsize=0)

	KSDval_ax.hist(KS_test_out[:,0])
	KSpval_ax.hist(KS_test_out[:,1])
	KSDval_ax.text(0.05,0.8, 'med D-val = {:.2e}'.format(
				np.median(KS_test_out[:,0])), fontsize=12., transform=KSDval_ax.transAxes)
	KSpval_ax.text(0.05,0.8, 'med p-val = {:.4f}'.format(
				np.median(KS_test_out[:,1])), fontsize=12., transform=KSpval_ax.transAxes)


	ADtest_fig = plt.figure(figsize = (16,8))
	gs = gridspec.GridSpec(1,2, hspace=0, wspace=0, top = 0.99, right = 0.99,
		 bottom  = 0.1, left = 0.1)
	ADAval_ax = ADtest_fig.add_subplot(gs[0,0])
	ADpval_ax = ADtest_fig.add_subplot(gs[0,1],sharey = ADAval_ax)

	ADAval_ax.set_xlabel('AD A-value',fontsize=22)
	ADpval_ax.set_xlabel('AD p-value',fontsize=22)
	ADAval_ax.set_ylabel('Histogram Density',fontsize=22)
	ADAval_ax.tick_params(axis = 'both', which='both',direction = 'in', labelsize=22, length = 8, width=1.25)
	ADAval_ax.tick_params(axis = 'both', which='minor',direction = 'in', labelsize=22, length = 4, width=1.25)
	ADpval_ax.tick_params(axis = 'both', which='both',direction = 'in', labelsize=22, length = 8, width=1.25)
	ADpval_ax.tick_params(axis = 'y', which='both',direction = 'in', labelsize=0)

	ADAval_ax.hist(AD_test_out[:,0])
	ADpval_ax.hist(AD_test_out[:,1])
	ADAval_ax.text(0.05,0.8, 'med A-val = {:.2e}'.format(
				np.median(AD_test_out[:,0])), fontsize=12., transform=ADAval_ax.transAxes)
	ADpval_ax.text(0.05,0.8, 'med p-val = {:.4f}'.format(
				np.median(AD_test_out[:,1])), fontsize=12., transform=ADpval_ax.transAxes)

	if save == True:
		KStest_fig.savefig('{dir}_KStest.png'.format(save), dpi = 200)
		ADtest_fig.savefig('{dir}_ADtest.png'.format(save), dpi = 200)
	else:
		plt.show()

def histogram_indices(data,bins):
	"""
	Compute histogram of data and return the indices contributing to each bin
	Parameters
	----------
	data : array
		Data to be put into a histogram
	bins : array
		Bin edges 
	
	Returns
	-------
	hist : array
		number of samples in each bin
	indices : list, length bins - 1
		indices of data points in each bin 
	"""
	hist = np.zeros(len(bins) - 1)
	indices = []
	for ii in range(len(bins) - 1):
		bin_low = bins[ii]
		bin_high = bins[ii + 1]
		inbin = np.where((data >= bin_low) & (data < bin_high))[0]
		hist[ii] = len(inbin)
		indices.append(inbin.tolist())

	return hist, indices



def sample_to_common_dist(controls, bins, Niter = 1):
	"""
	Resample N samples to conform to the same common distribution

	Parameters
	----------
	controls : list, length Nsamp
		List of values for each sample that will used for common resampling
	bins : array
		Bin edges for the control sample
	Niter : int
		Number of sampling iterations to perform


	Returns
	-------
	indices_all : list, size [Nsamp][Niter]
		List of the indicies selected for each sample in each Niter
	indices_inbin_all : list, size [Nsamp][Niter][len(bins)]
		Number of, and values of, the indicies selected in each contol bin for each sample in each Niter
		NOTE - THESE ARE INDICES OF INDICES! (yeah, confusing, but needed I think)
	control_hists : array, size Nsamp x len(bins)
		Control histogram of each sample, and common histogram
	"""

	hist_control = np.zeros([len(controls), len(bins) - 1])				#Control sample histograms
	for ss in range(len(controls)):
		hist_control[ss,:] = np.histogram(controls[ss],
						 bins = bins, density = True)[0] 
	print('Created control histograms')
	hist_common = np.min(hist_control, axis = 0)								#Common control histogram
	# hist_common = hist_common /(np.sum(hist_common * np.diff(bins)))
	print('Defined common histogram')
	indices_all_samples = []														#List for all indices for all samples for all iter
	for ss in range(len(controls)):										#Iterate over provided samples

		control = controls[ss]

		hist_ratio =  hist_common / hist_control[ss,:]  					#Fraction of sample to keep in each control histogram bin
		indices_all = []
		for nn in range(Niter):													#Loop over iterations

			indices_iter = []													#List for all indices for sample
			for ii in range(len(control)):
				bincentre_diff = np.abs(bins[:-1] + 0.5*np.diff(bins) - control[ii])	#Find which control bin the data point lies in
				control_bin = np.where(bincentre_diff == np.min(bincentre_diff))[0]
				

				test = rng.uniform()											#Generate random number for keeping
				if test < hist_ratio[control_bin]:									#If rand < fraction to keep in the bin, keep the point														
					indices_iter.extend([ii])														#Otherwise discard

			control_hist_iter, indices_inbin_iter = histogram_indices(control[indices_iter], bins)				#Find data points in each bin of the control histogram

			indices_all.append([indices_iter,control_hist_iter,indices_inbin_iter])								#Record indices kept, corresponding control hist,
		print(f'Completed control of sample {ss+1}')																										#and indices in each bin of control hist  
		indices_all_samples.append(indices_all)
	control_hists = np.vstack([hist_control, hist_common])

	return indices_all_samples, control_hists 

def jackknife_sample_distributions(samples, bins, indices_all, DAG_frac = 0.2, cumulative = True):
	"""
	Jackknife distributions that have been controlled to a common distribution

	Parameters
	----------
	samples : list of samples, length Nsamp
		List of the sample values that are being compared
	bins : array
		Sample histogram bins
	indices_all : list, size [Nsamp][Niter]
		List of the indicies selected for each sample in each Niter
	indices_inbin_all : list
		Number of, and values of, the indicies selected in each contol bin for each sample in each Niter
	DAG_frac : float, None
		Fraction of the sample to delete in each DAGJK iteration, set to None for standard jackknife


	Returns
	-------
	samples_all_hists : list
		List of the cumulative histogram in each of Niter iterations in bins of sample_bins for each sample
	samples_all_DAGJKsigma : list
		List of the DAGJK uncertainty in each bin of the cumulative histograms for each Niter for each sample 
	indexes_all_iter : list
		List of the indicies selected for each sample in each Niter
	"""

	sample_hists_all = []
	samples_all_DAGJKsigma = []
	
	Niter = len(indices_all[0])

	for ss in range(len(samples)):												#Iterate over provided samples

		sample = samples[ss]
							
		sample_hists = np.zeros([Niter, len(bins)-1])
		sigma_DAGJK_hist = np.zeros([Niter, len(bins)-1])

		for nn in range(Niter):													#Loop over iterations

			indices_iter = indices_all[ss][nn][0]						#indicies of controlled sample in this iter
			control_hist_iter = indices_all[ss][nn][1]					#Number of points in each bin of control hist
			indices_inbin_iter = indices_all[ss][nn][2]					#Indices in each bin of control hist


			sample_iter = np.array(sample[indices_iter])						#sample values for control iteration
			sample_hist_iter = np.histogram(sample_iter, bins = bins, density = True)[0]				#Controlled sample cumulative histogram
			if cumulative == True:
				sample_hists[nn,:] = np.cumsum(sample_hist_iter) / np.nansum(sample_hist_iter)
			else:
				sample_hists[nn,:] = sample_hist_iter


			if DAG_frac != None:
				Rsamp = int(1. / DAG_frac)	
				histbin_Nsamples = np.zeros([Rsamp, len(control_hist_iter)])			#array for number of samples to be drawn from each control bin

				for ii in range(len(control_hist_iter)):							#Distritibute the number of samples in each control bin
					Ngals_inbin = int(control_hist_iter[ii])						#to define the number of data points to select
					jj = 0 															#in each DAGJK iteration
					while(jj < Ngals_inbin):										
						histbin_Nsamples[(jj + ii) % Rsamp, ii] += 1
						jj += 1

				DAG_indices = []									#Indices of values in all DAGJK iterations
				for jj in range(Rsamp):
					DAG_sample = []									#Indices of values in this DAGJK iteration
					for ii in range(len(control_hist_iter)):																		#for each bin
						if histbin_Nsamples[jj, ii] != 0:																			#if we are choosing a sample
							indices_list = np.array(indices_inbin_iter[ii])															#Get the list of indices in this bin
							
							indices_sample = rng.choice(range(len(indices_list)), int(histbin_Nsamples[jj,ii]),replace=False)		#Randomly choose the (indices of the) defined number of samples in this bin
							
							DAG_sample.extend(indices_list[indices_sample])															#Add to indices list to be deleted in the given DAGJK iteration
							
							indices_list = np.delete(indices_list,indices_sample)										#Remove selected indices so they can't be selected again
							
							indices_inbin_iter[ii] = indices_list.tolist()												

					DAG_indices.append(DAG_sample)																	

				DAGJK_hists = np.zeros([Rsamp,len(bins) - 1])
				for ii in range(len(DAG_indices)):
					subset = np.delete(sample_iter, DAG_indices[ii])											#Delete the indices  
					subset_hist = np.histogram(subset, bins = bins, density=True)[0]							#Calculate the parameter estimate (histogram, in this case)
					if cumulative == True:
						DAGJK_hists[ii,:] = np.cumsum(subset_hist) / np.nansum(subset_hist)
					else:
						DAGJK_hists[ii,:] = subset_hist

				Rsamp_scale = ((Rsamp - 1.) / Rsamp)																	#DAGJK parameter uncertainty estimate
				mean_DAGJK_hist = np.mean(DAGJK_hists, axis = 0)
				sigma_DAGJK_hist[nn,:] = np.sqrt(Rsamp_scale * 
									np.nansum((DAGJK_hists - mean_DAGJK_hist) ** 2., axis = 0) )

		sample_hists_all.append(sample_hists)
		samples_all_DAGJKsigma.append(sigma_DAGJK_hist)

	return sample_hists_all, samples_all_DAGJKsigma

#################################################################################
#		Sampling a population to conform to the 2D-distribution of another
#################################################################################

def sample_from_same_parameterspace(sample, control, xbins, ybins):
	#sample = sample being conformed to target
	#control = control population which sample is being conformed to
	#xbins = x-axis bins
	#ybins = y-axis bins

	dist_sample = np.histogram2d(sample[0],sample[1],bins=[xbins,ybins],density=True)[0]
	dist_control = np.histogram2d(control[0],control[1],bins=[xbins,ybins],density=True)[0]

	hist_ratio = dist_control / dist_sample 

	keep_indices = []
	for ii in range(len(sample[0])):
		xx = np.where(  np.abs(xbins[0:-1]+0.5*np.abs(np.diff(xbins)[0]) - sample[0][ii]) == np.min(np.abs(xbins[0:-1]+0.5*np.abs(np.diff(xbins)[0]) - sample[0][ii])) )[0][0]
		yy = np.where(  np.abs(ybins[0:-1]+0.5*np.abs(np.diff(ybins)[0]) - sample[1][ii]) == np.min(np.abs(ybins[0:-1]+0.5*np.abs(np.diff(ybins)[0]) - sample[1][ii])) )[0][0]


		if nprand.uniform() <= hist_ratio[xx,yy]:
			keep_indices.extend([ii])

	return keep_indices




#################################################################################
#		Matching a sample to a control sample & computing parameter offsets
#################################################################################
def match_populations(sample, control, dex_lims, Nmatch = 5):
	"""
	Match a sample to a control sample based on having similar properties

	Parameters
	----------
	sample : list length N
		Properties of the sample being matched
	control : list length N
		Properties of the control sample being matched to
	dex_lims : list shape N x [start, max, agnostic]
		List of tolerances for properties. Matches will be found within 'start' dex,
		and iteratively increased to 'max' dex if Nmatch aren't found. Matches are
		made between properties with value above 'agnostic' regardless of value
	Nmatch : int
		Minimum number of matched required to keep the datapoint
	

	Returns
	-------
	matched : list length len(sample[0])
		Indices of the rows in the control sample matched each object in sample being matched
	"""

	from functools import reduce

	print('Number in sample population', len(sample[0]))
	print('Number in control population', len(control[0]))

	successful = 0
	matched = []
	for ii in range(len(sample[0])):
		matches = []
		Niter = 0
		while(len(matches) < Nmatch and Niter <= 10):
			Niter += 1
			matches = []										#lists of indices in control population matched per-parameter 
			for jj in range(len(sample)):						#iterate over the number of matching properties
				step = (dex_lims[jj][1] - dex_lims[jj][0]) / 10
				# print(step)
				tol = dex_lims[jj][0] + (Niter-1) * step

				if sample[jj][ii] > dex_lims[jj][2]: 			#check if property is above 'agnostic' threshold
					match_low = dex_lims[jj][2]
					match_high = 1.e10
				else:
					match_low = sample[jj][ii] - tol
					match_high = sample[jj][ii] + tol

				match = np.where((control[jj] > match_low) & (control[jj] < match_high))[0]

				matches.append(match)
		
			matches = reduce(np.intersect1d,(matches))		#find only indices that match for all parameters
		if len(matches) < Nmatch:
			matched.append([-1])							#insufficient matches found
		else:
			matched.append(matches)
			successful += 1

			# print(matches)
	print(successful, 'matched out of ',len(sample[0]),': f = ', successful/len(sample[0]))

	return matched

def calculate_property_offset(sample1, sample2, matched):


	offsets_samp1 = []
	offsets_samp2_grouped = []
	matches_good = []
	for ii in range(len(sample1)):
		matches = matched[ii]
		# print(matches)
		if matches[0] != -1:
			
			matches_good.append(matches)
			
			delta_samp1 = sample1[ii] - np.median(sample2[matches])
			delta_samp2 = sample2[matches] - np.median(sample2[matches])

			offsets_samp1.extend([delta_samp1])
			offsets_samp2_grouped.append(delta_samp2)

	matches_unique = [value for sublist in matches_good for value in sublist]
	matches_unique = np.unique(matches_unique)

	offsets_samp2 = [value for sublist in offsets_samp2_grouped for value in sublist]

	median_offsets = []
	for ii in range(len(matches_unique)):
		offsets = []
		for jj in range(len(matches_good)):
			# print(matches_good)
			if any([index == matches_unique[ii] for index in matches_good[jj]]):
				ref = np.where(matches_good[jj] == matches_unique[ii])[0]
				offsets = offsets_samp2_grouped[jj][ref]
		median_offsets.extend([np.median(offsets)])

	median_offsets_samp2 = [matches_unique,median_offsets]

	return offsets_samp1, offsets_samp2, median_offsets_samp2

def bootstrap_median(sample):
	Nsamp = 10000
	medians = np.zeros(Nsamp)
	for ii in range(Nsamp):
		samp = np.random.choice(sample, len(sample), replace=True)
		medians[ii] = np.median(samp)

	BSuncert = np.std(medians)
	return BSuncert


if __name__ == '__main__':
	main()