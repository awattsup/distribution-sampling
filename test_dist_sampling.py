import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from astropy.convolution import convolve, Box1DKernel
import random

from astropy.table import Table

import distribution_sampling as ds

rng = np.random.default_rng()

from functools import reduce



def create_dists():

	Afr_bins = np.arange(1.e0,2.2e0,5.e-2)
	Afrs_high = [
	1*np.ones([250]),
	1.05*np.ones([245]),
	1.10*np.ones([240]),
	1.15*np.ones([235]),
	1.20*np.ones([230]),
	1.25*np.ones([220]),
	1.30*np.ones([210]),
	1.35*np.ones([190]),
	1.40*np.ones([180]),
	1.45*np.ones([170]),
	1.50*np.ones([160]),
	1.55*np.ones([150]),
	1.60*np.ones([135]),
	1.65*np.ones([110]),
	1.70*np.ones([95]),
	1.75*np.ones([80]),
	1.80*np.ones([65]),
	1.85*np.ones([50]),
	1.90*np.ones([35]),
	1.95*np.ones([20]),
	2*np.ones([5])]

	Afr_high_list =	reduce(np.append,(Afrs_high))
	print(len(Afr_high_list))

	Afr_high_prob = np.histogram(Afr_high_list,bins=Afr_bins,density=True)[0]
	Afr_high_cumsum = np.cumsum(Afr_high_prob) / np.nansum(Afr_high_prob)

	Afrs_low = [
	1*np.ones([1000]),
	1.05*np.ones([700]),
	1.10*np.ones([500]),
	1.15*np.ones([400]),
	1.20*np.ones([350]),
	1.25*np.ones([300]),
	1.30*np.ones([250]),
	1.35*np.ones([200]),
	1.40*np.ones([150]),
	1.45*np.ones([100]),
	1.50*np.ones([75]),
	1.55*np.ones([50]),
	1.60*np.ones([40]),
	1.65*np.ones([30]),
	1.70*np.ones([20]),
	1.75*np.ones([10]),
	1.80*np.ones([7]),
	1.85*np.ones([5]),
	1.90*np.ones([3]),
	1.95*np.ones([2]),
	2*np.ones([1])]


	Afr_low_list =	reduce(np.append,(Afrs_low))
	print(len(Afr_low_list))
	Afr_low_prob = np.histogram(Afr_low_list,bins=Afr_bins,density=True)[0]
	
	Afr_low_cumsum = np.cumsum(Afr_low_prob) / np.nansum(Afr_low_prob)

	prob1 = rng.uniform(size=100)
	Afr_low_sample = np.interp(prob1,Afr_low_cumsum,Afr_bins[0:-1]+0.025)
	Afr_high_sample = np.interp(prob1,Afr_high_cumsum,Afr_bins[0:-1]+0.025)

	Afr_low_sample = np.array([int(round(xx*20)/0.2)/100 for xx in Afr_low_sample])
	Afr_high_sample = np.array([int(round(xx*20)/0.2)/100 for xx in Afr_high_sample])
	# exit()


	# plt.hist(Afr_high_list,bins=Afr_bins,histtype = 'step',cumulative=True,density=True)
	# plt.hist(Afr_low_list,bins=Afr_bins,histtype = 'step',cumulative=True,density=True)
	plt.hist(Afr_high_sample,bins=Afr_bins,histtype = 'step',cumulative=True,density=True)
	plt.hist(Afr_low_sample,bins=Afr_bins,histtype = 'step',cumulative=True,density=True)

	plt.plot(Afr_bins[0:-1]+0.025,Afr_high_cumsum)
	plt.plot(Afr_bins[0:-1]+0.025,Afr_low_cumsum)
	plt.show()

	SNrange = np.arange(7,101,1)
	SNdist_low = norm_exp_dist(SNrange,1,1/10,1,7,100)
	SNdist_high = norm_exp_dist(SNrange,1,1/20,1,7,100)


	prob1 = rng.uniform(size=10000)
	print(prob1)

	SN1 = sample_SN(prob1,1,1/10,7,100)
	SN2 = sample_SN(prob1,1,1/20,7,100)


	plt.plot(SNrange,SNdist_low)
	plt.plot(SNrange,SNdist_high)
	plt.hist(SN1,bins=10**(np.arange(np.log10(7),np.log10(100),0.05)),density=True,histtype='step')
	plt.hist(SN2,bins=10**(np.arange(np.log10(7),np.log10(100),0.05)),density=True,histtype='step')
	plt.xscale('log')
	plt.show()















def test_recover_same_dist():


	data1 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/base_sample_highSN_measuements.ascii',format='ascii')
	data2 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/base_sample_lowSN_measuements.ascii',format='ascii')


	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	# plt.hist(nprand.choice(data1['Afr'],100),bins=Afr_bins,histtype='step',color='Green',density=True,cumulative=True)
	# plt.hist(nprand.choice(data2['Afr'],100),bins=Afr_bins,histtype='step',color='Orange',density=True,cumulative=True)
	# plt.show()

	# plt.hist(data1['SN'],histtype='step')
	# plt.hist(data2['SN'],histtype='step')
	# plt.show()

	Nsamp = 200

	for ii in range(10):
		fig = plt.figure(figsize = (10,12))

		gs = gridspec.GridSpec(2, 1, top = 0.99, right = 0.99, bottom  = 0.08, left = 0.12,hspace=0)

		ax1 = fig.add_subplot(gs[0,0])
		ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)


		index1 = nprand.choice(len(data1),Nsamp)
		index2 = nprand.choice(len(data2),Nsamp)


		hist1,edges = np.histogram(data1['Afr'][index1],bins=Afr_bins,density=True)
		hist2,edges = np.histogram(data2['Afr'][index2],bins=Afr_bins,density=True)
		hist1 = np.cumsum(hist1) / np.sum(hist1)
		hist2 = np.cumsum(hist2) / np.sum(hist2)
		ax1.plot(Afr_bins[0:-1],hist1,color='Green')
		ax1.plot(Afr_bins[0:-1],hist2,color='Orange')



		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
							ds.control_samples([data2['Afr'][index2],data1['Afr'][index1]],Afr_bins,
									[np.log10(data2['SN'][index2]),np.log10(data1['SN'][index1])],SN_bins,
									Niter=10000)

		ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
				 axis = ax2,names=['low SN','high SN'])


		ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
		ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
		ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		
		ax1.set_ylim([0.1, 1])
		ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

		ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
		ax1.set_ylabel('Cumulative Histogram', fontsize=27)
		ax2.set_ylabel('Cumulative Histogram', fontsize=27)

		fig.savefig('./figures/recover_same_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))

		# plt.show()


def test_recover_different_dist():


	data1 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/sym_sample_highSN_measuements.ascii',format='ascii')
	data2 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/asym_sample_lowSN_measuements.ascii',format='ascii')


	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	# plt.hist(nprand.choice(data1['Afr'],100),bins=Afr_bins,histtype='step',color='Green',density=True,cumulative=True)
	# plt.hist(nprand.choice(data2['Afr'],100),bins=Afr_bins,histtype='step',color='Orange',density=True,cumulative=True)
	# plt.show()

	# plt.hist(data1['SN'],histtype='step')
	# plt.hist(data2['SN'],histtype='step')
	# plt.show()

	Nsamp = 100

	for ii in range(10):
		fig = plt.figure(figsize = (10,12))

		gs = gridspec.GridSpec(2, 1, top = 0.99, right = 0.99, bottom  = 0.08, left = 0.12,hspace=0)

		ax1 = fig.add_subplot(gs[0,0])
		ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)


		index1 = nprand.choice(len(data1),Nsamp)
		index2 = nprand.choice(len(data2),Nsamp)


		hist1,edges = np.histogram(data1['Afr'][index1],bins=Afr_bins,density=True)
		hist2,edges = np.histogram(data2['Afr'][index2],bins=Afr_bins,density=True)
		hist1 = np.cumsum(hist1) / np.sum(hist1)
		hist2 = np.cumsum(hist2) / np.sum(hist2)
		ax1.plot(Afr_bins[0:-1],hist1,color='Green')
		ax1.plot(Afr_bins[0:-1],hist2,color='Orange')



		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
							ds.control_samples([data2['Afr'][index2],data1['Afr'][index1]],Afr_bins,
									[np.log10(data2['SN'][index2]),np.log10(data1['SN'][index1])],SN_bins,
									Niter=10000)

		ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
				 axis = ax2,names=['asym, low SN','sym, high SN'])


		ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
		ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
		ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		
		ax1.set_ylim([0.1, 1])
		ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

		ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
		ax1.set_ylabel('Cumulative Histogram', fontsize=27)
		ax2.set_ylabel('Cumulative Histogram', fontsize=27)

		fig.savefig('./figures/recover_diff_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))

		# plt.show()

def test_opposite_recover_different_dist():


	data2 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/asym_sample_highSN_measuements.ascii',format='ascii')
	data1 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/sym_sample_lowSN_measuements.ascii',format='ascii')


	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	# plt.hist(nprand.choice(data1['Afr'],100),bins=Afr_bins,histtype='step',color='Green',density=True,cumulative=True)
	# plt.hist(nprand.choice(data2['Afr'],100),bins=Afr_bins,histtype='step',color='Orange',density=True,cumulative=True)
	# plt.show()

	# plt.hist(data1['SN'],histtype='step',color='Green')
	# plt.hist(data2['SN'],histtype='step',color='Orange')
	# plt.show()

	Nsamp = 100

	for ii in range(10):
		fig = plt.figure(figsize = (10,12))

		gs = gridspec.GridSpec(2, 1, top = 0.99, right = 0.99, bottom  = 0.08, left = 0.12,hspace=0)

		ax1 = fig.add_subplot(gs[0,0])
		ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)


		index1 = nprand.choice(len(data1),Nsamp)
		index2 = nprand.choice(len(data2),Nsamp)


		hist1,edges = np.histogram(data1['Afr'][index1],bins=Afr_bins,density=True)
		hist2,edges = np.histogram(data2['Afr'][index2],bins=Afr_bins,density=True)
		hist1 = np.cumsum(hist1) / np.sum(hist1)
		hist2 = np.cumsum(hist2) / np.sum(hist2)
		ax1.plot(Afr_bins[0:-1],hist1,color='Green')
		ax1.plot(Afr_bins[0:-1],hist2,color='Orange')



		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
							ds.control_samples([data2['Afr'][index2],data1['Afr'][index1]],Afr_bins,
									[np.log10(data2['SN'][index2]),np.log10(data1['SN'][index1])],SN_bins,
									Niter=10000)

		ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
				 axis = ax2,names=['asym, high SN','sym, low SN'])


		ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
		ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
		ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		
		ax1.set_ylim([0.1, 1])
		ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

		ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
		ax1.set_ylabel('Cumulative Histogram', fontsize=27)
		ax2.set_ylabel('Cumulative Histogram', fontsize=27)

		fig.savefig('./figures/opposite_recover_diff_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))

		# plt.show()





def create_templates():
	"""
	Generates model HI spectrum and adds noise realisations

    Parameters
    ----------
    args : list
        List of input arguments and options
        	Full width for top-hat inputs 	[kms/s]
        	Std.dev for Gaussian inputs 	[km/s]
        	Inclincation, dispersion, HI and RC parameters for toy model generation inputs
        	S/N min, max and step

    Returns
    -------
    Files : named by directory and S/N
    	"_spectra.dat"
        header with spectrum shape inputs, RMS and smoothed resolution
        velocity channels, perfect spectrum and N model spectra
    """

	dim = 2000
	incl = 45
	model= 'FE'
	MHI = 1.e11
	Vdisp = 0
	dist = 150.e0
	rms_temp = -1
	Vres = 2.e0
	Vsm = 10
	

	HI_params = [1.e0, 1.65e0, 1.e0, 1.65e0]
	RC_params_list = [[200.e0, 0.164e0, 0.002e0, 200e0, 0.164e0, 0.002e0]
		,[200.e0, 0.164e0, 0.002e0, 215.5e0, 0.164e0, 0.002e0] #Afr=1.05
	  ,[200.e0, 0.164e0, 0.002e0, 231.2e0, 0.164e0, 0.002e0]# Afr=1.1
	  ,[200.e0, 0.164e0, 0.002e0, 248.e0, 0.164e0, 0.002e0]# Afr=1.15
	  ,[200.e0, 0.164e0, 0.002e0, 266e0, 0.164e0, 0.002e0] # Afr=1.2
	  ,[200.e0, 0.164e0, 0.002e0, 284e0, 0.164e0, 0.002e0] #Afr=1.25
	  ,[200.e0, 0.164e0, 0.002e0, 304e0, 0.164e0, 0.002e0] #Afr=1.3
	  ,[200.e0, 0.164e0, 0.002e0, 324.5e0, 0.164e0, 0.002e0] #Afr=1.35
	  ,[200.e0, 0.164e0, 0.002e0, 347e0, 0.164e0, 0.002e0] #Afr=1.4
	  ,[200.e0, 0.164e0, 0.002e0, 371e0, 0.164e0, 0.002e0] #Afr=1.45
	  ,[200.e0, 0.164e0, 0.002e0, 397.5e0, 0.164e0, 0.002e0] #Afr=1.50
	  ,[200.e0, 0.164e0, 0.002e0, 425.5e0, 0.164e0, 0.002e0] #Afr=1.55
	  ,[200.e0, 0.164e0, 0.002e0, 457e0, 0.164e0, 0.002e0] #Afr = 1.6
	  ,[200.e0, 0.164e0, 0.002e0, 492e0, 0.164e0, 0.002e0] #Afr=1.65
	  ,[200.e0, 0.164e0, 0.002e0, 531e0, 0.164e0, 0.002e0]  #Afr=1.7
	  ,[200.e0, 0.164e0, 0.002e0, 575e0, 0.164e0, 0.002e0] #Afr=1.75
	  ,[200.e0, 0.164e0, 0.002e0, 625e0, 0.164e0, 0.002e0] #Afr=1.8
	  ,[200.e0, 0.164e0, 0.002e0, 672.5e0, 0.164e0, 0.002e0] #Afr=1.85
	  ,[200.e0, 0.164e0, 0.002e0, 720e0, 0.164e0, 0.005e0] #Afr=1.9
	  ,[200.e0, 0.164e0, 0.002e0, 743e0, 0.164e0, 0.007e0]] #Afr=1.95

	for RC_params in RC_params_list:

		mjy_conv = 1.e3 / (2.356e5  * (dist ** 2.e0))
		Sint = mjy_conv * MHI 
		
		Vmin = -500.e0
		Vmax = 500.e0
			
		input_params = [incl, model,
						1, MHI, HI_params[0], HI_params[1], -1, -1 , HI_params[2], HI_params[3], -1, -1,
						1, RC_params[0], RC_params[1], RC_params[2], RC_params[3], RC_params[4], RC_params[5],
						Vdisp, dist, rms_temp, Vmin, Vmax, Vres, Vsm]		
		radius, costheta, R_opt = create_arrays(dim, input_params)
		obs_mom0, rad1d, input_HI = create_mom0(radius, costheta, input_params, R_opt)
		obs_mom1, input_RC  = create_mom1(radius, costheta, rad1d, input_params, R_opt)
		vel_bins, base_spectrum, Sint = hi_spectra(obs_mom0, obs_mom1, input_params)

		spectrum = smooth_spectra(vel_bins, base_spectrum, input_params)
		if len(np.where(spectrum ==  np.nanmax(spectrum))[0]) > 3:
			Peaks = [np.nanmax(spectrum), np.nanmax(spectrum)]
		else:
			Peaklocs = locate_peaks(spectrum)
			Peaks = spectrum[Peaklocs]
		width_full = locate_width(spectrum, Peaks, 0.2e0)
		width = (width_full[1] - width_full[0]) * Vres
		Sint, Afr = areal_asymmetry(spectrum, width_full, Vres)

		# plt.plot(spectrum)
		# plt.show()
		data = np.array([vel_bins,base_spectrum]).T
		print(Afr)
		np.savetxt('./template_spectra/template_Afr{Afr:.2f}'.format(Afr=Afr),data)







		# 	SN_range = np.arange(args.SN_range[0], args.SN_range[1] + args.SN_range[2], args.SN_range[2])
		# 	if args.PN:
		# 		RMS_sm = np.nanmax(spectrum) / SN_range
		# 	elif args.AA:
		# 		RMS_sm = rms_from_StN(SN_range, Sint, width, Vsm)
		# 	RMS_input = RMS_sm * np.sqrt(int(Vsm / Vres))

		# 	model_directory = './{base}{St}{option}_Afr{Afr:.2f}_{mt}/'.format(
		# 		base = base, Afr = Afr, St = SN_type, mt = model_type, option = args.option)
		# 	if len(glob.glob(model_directory)) == 0:
		# 		os.mkdir(model_directory)

		# N_SNvals = len(RMS_input)
		# spectra = np.zeros([len(spectrum), args.Nmodels[0] + 2])
		# spectra[:,0] = vel_bins
		# spectra[:,1] = spectrum






		# for ii in range(rank * int(N_SNvals / nproc),(rank + 1) * int(N_SNvals / nproc)):
		# 	print('proc', rank, 'Generating realisations for SN = ', int(SN_range[ii]))
		# 	input_params[21] = RMS_input[ii]
		# 	for n in range(args.Nmodels[0]):
		# 		obs_spectrum = add_noise(base_spectrum, input_params)
		# 		obs_spectrum = smooth_spectra(vel_bins, obs_spectrum, input_params)
		# 		spectra[:, n + 2] = obs_spectrum

		# 	filename = '{md}SN{SN}_spectra.dat'.format( 
		# 		md = model_directory, SN = int(SN_range[ii]))
			
		# 	if args.TH[0] != 0:
		# 		header = 'Tophat = {width}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
		# 				width = args.TH[0], rms = RMS_sm[ii], Vsm = Vsm)
		# 	elif args.GS[0] != 0:
		# 		header = 'Gaussian = {sigma} {alpha}\nrms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
		# 				sigma = args.GS[0], alpha = args.GS[1] , rms = RMS_sm[ii], Vsm = Vsm)
		# 	else:
		# 		header = 'HI = {model}, {H1}, {H2}, {H3}, {H4} \nRC = {R1}, {R2}, {R3}, {R4}, {R5}, {R6} \n' \
		# 				'rms = {rms:0.5f}\nVsm = {Vsm}\n'.format(
		# 			model = model, H1=args.HI[0], H2 = args.HI[1], H3=args.HI[2], H4 = args.HI[3], 
		# 			R1 = args.RC[0], R2 = args.RC[1], R3 = args.RC[2], R4 = args.RC[3], 
		# 			R5 = args.RC[4], R6 = args.RC[5], rms = RMS_sm[ii], Vsm = Vsm)
		# 	np.savetxt(filename, spectra, header = header,fmt = "%.4e")


def create_sample():


	Afr_vals = np.array([1.00,1.05,1.10,1.15,1.20,1.25,1.30,1.35,1.40,1.45,1.50,1.55,
				1.60,1.65,1.70,1.75,1.80,1.85,1.90,1.95])

	# Nspec_list = norm_exp_dist(Afr_vals,1,7,2000,1,2)*0.05		#sym sample
	# Nspec_list = norm_exp_dist(Afr_vals,1,3,2000,1,2)*0.05			#asym sample
	Nspec_list = norm_exp_dist(Afr_vals,1,5,2000,1,2)*0.05			#base sample

	plt.plot(Afr_vals,Nspec_list)
	plt.show()

	print(Nspec_list)

	for ii in range(len(Afr_vals)):
		template = './template_spectra/template_Afr{Afr:.2f}'.format(Afr=Afr_vals[ii])
		data = np.loadtxt(template)
		Nspec = int(Nspec_list[ii])

		if ii == 0:
			sample = np.tile(data[:,1],(Nspec,1))
		else:
			samp = np.tile(data[:,1],(Nspec,1))
			sample = np.append(sample,samp,axis=0)

	# reorder = np.arange(len(sample),dtype=int)

	reorder = random.sample(range(len(sample)),len(sample))
	sample = sample[reorder]

	sample = np.append([data[:,0]],sample,axis=0).T

	np.savetxt('./samples/base_sample.dat',sample)



def add_SN():

	sample = np.loadtxt('./samples/sym_sample.dat')
	vel = sample[:,0]
	spectra = sample[:,1::]


	Vres = np.abs(np.diff(vel)[0])
	Vsm = 10


	Nspec = len(spectra[0,:])

	prob = nprand.random(Nspec)

	SN_values = sample_SN(prob,7,0.01,7,100) #highSN
	# SN_values = sample_SN(prob,7,0.1,7,100)	#lowSN

	obs_spectra = np.zeros([len(vel),Nspec+1])
	obs_spectra[:,0] = vel

	measurements = np.zeros([Nspec,5])

	for ii in range(Nspec):
		base_spectrum = spectra[:,ii]
		box_channels = int(Vsm / Vres)
		smoothed_spectrum = convolve(base_spectrum, Box1DKernel(box_channels)) 


		if len(np.where(smoothed_spectrum ==  np.nanmax(smoothed_spectrum))[0]) > 3:
			Peaks = [np.nanmax(smoothed_spectrum), np.nanmax(smoothed_spectrum)]
		else:
			Peaklocs = locate_peaks(smoothed_spectrum)
			Peaks = smoothed_spectrum[Peaklocs]


		width_full = locate_width(smoothed_spectrum, Peaks, 0.2e0)
		width = (width_full[1] - width_full[0]) * Vres
		Sint_base, Afr_base = areal_asymmetry(smoothed_spectrum, width_full, Vres)

		

		RMS_sm = rms_from_StN(SN_values[ii], Sint_base, width, Vsm)
		RMS_input = RMS_sm * np.sqrt(int(Vsm / Vres))

		noise_arr = np.random.normal(np.zeros(len(base_spectrum)), RMS_input)
		obs_spectrum = base_spectrum + noise_arr
		obs_spectrum = convolve(obs_spectrum, Box1DKernel(box_channels)) 

		obs_spectra[:,ii+1] = obs_spectrum

		Sint, Afr = areal_asymmetry(obs_spectrum, width_full, Vres)

		SN = StN(Sint,width,RMS_sm,Vsm)

		measurements[ii,:] = np.array([Sint, width, RMS_sm, SN, Afr])


	np.savetxt('./samples/sym_sample_highSN.dat',obs_spectra)

	measurements = Table(measurements,names=('Sint','w20','rms','SN','Afr'))
	measurements.write('./measurements/sym_sample_highSN_measuements.ascii',format='ascii',overwrite=True)

	exit()






def norm_exp_dist(x,a,b,Ntot,lim1,lim2):
	
	A = (-b)*Ntot / ((np.exp((-b)*(lim2-a)) - np.exp((-b)*(lim1-a))))

	f = A*np.exp(-b*(x-a))

	return f



def sample_SN(prob,a,b,lim1,lim2):

	A = (-b) / ((np.exp((-b)*(lim2-a)) - np.exp((-b)*(lim1-a))))


	SN = (np.log(-b*prob/A + np.exp((-b)*(lim1-a)))/(-b)) + a

	return SN








if __name__ == '__main__':


	create_dists()

	# create_templates()
	# create_sample()
	# add_SN()

	# test_recover_same_dist()
	# test_recover_different_dist()
	# test_opposite_recover_different_dist()





