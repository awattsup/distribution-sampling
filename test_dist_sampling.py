import numpy as np 
import matplotlib.pyplot as plt 
import matplotlib.gridspec as gridspec
from astropy.convolution import convolve, Box1DKernel
from matplotlib.lines import Line2D

from astropy.table import Table

import distribution_sampling as ds

rng = np.random.default_rng()


import sys
sys.path.append('/home/awatts/programs/HI-spectrum-toy-model/')
import model_asymmetry_parameterspace as MAP

from functools import reduce




def test_dist_creation():
	
	Nsamp = 100

	Afr_lowAsym_lowSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
	Afr_lowAsym_lowSN[:,0] = sample_SN(Nsamp, -1000,1/5,7,100)
	Afr_lowAsym_lowSN[:,1] = sample_Afr_dist(Nsamp,asym_rate = 'low')
	Afr_lowAsym_lowSN[:,2::] = MAP.get_SN_Afr(Afr_lowAsym_lowSN[:,0],Afr = Afr_lowAsym_lowSN[:,1])
	np.savetxt(f'./samples/Afr_lowAsym_lowSN_N{Nsamp}.dat',Afr_lowAsym_lowSN)


	Afr_lowAsym_highSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
	Afr_lowAsym_highSN[:,0] = sample_SN(Nsamp, -1000,1/8,7,100)
	Afr_lowAsym_highSN[:,1] = sample_Afr_dist(Nsamp,asym_rate = 'low')
	Afr_lowAsym_highSN[:,2::] = MAP.get_SN_Afr(Afr_lowAsym_highSN[:,0],Afr = Afr_lowAsym_highSN[:,1])
	np.savetxt(f'./samples/Afr_lowAsym_highSN_N{Nsamp}.dat',Afr_lowAsym_highSN)
	

	Afr_highAsym_lowSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
	Afr_highAsym_lowSN[:,0] = sample_SN(Nsamp, -1000,1/5,7,100)
	Afr_highAsym_lowSN[:,1] = sample_Afr_dist(Nsamp,asym_rate = 'high')
	Afr_highAsym_lowSN[:,2::] = MAP.get_SN_Afr(Afr_highAsym_lowSN[:,0],Afr = Afr_highAsym_lowSN[:,1])
	np.savetxt(f'./samples/Afr_highAsym_lowSN_N{Nsamp}.dat',Afr_highAsym_lowSN)
	

	Afr_highAsym_highSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
	Afr_highAsym_highSN[:,0] = sample_SN(Nsamp, -1000,1/8,7,100)
	Afr_highAsym_highSN[:,1] = sample_Afr_dist(Nsamp,asym_rate = 'high')
	Afr_highAsym_highSN[:,2::] = MAP.get_SN_Afr(Afr_highAsym_highSN[:,0],Afr = Afr_highAsym_highSN[:,1])
	np.savetxt(f'./samples/Afr_highAsym_highSN_N{Nsamp}.dat',Afr_highAsym_highSN)
	

	# SNdist_low = norm_exp_dist(SNrange,-1000,1/5,1,7,100)				#low S/N dist
	# SNdist_high = norm_exp_dist(SNrange,-1000,1/8,1,7,100)				#high S/N dist

	# plt.hist(Afr_lowAsym_lowSN[:,1],bins=np.arange(1,2.2,0.05),density=True,histtype='step',cumulative='True')
	# plt.hist(Afr_lowAsym_lowSN[:,3],bins=np.arange(1,2.2,0.05),density=True,histtype='step',cumulative='True')
	# plt.show()

	# plt.hist(Afr_lowAsym_lowSN[:,2],bins=bins=10**(np.arange(np.log10(7),np.log10(100),0.05)),density=True,histtype='step',cumulative='True')
	# plt.hist(Afr_lowAsym_lowSN[:,4],bins=bins=10**(np.arange(np.log10(7),np.log10(100),0.05)),density=True,histtype='step',cumulative='True')
	# plt.show()

def SN_control_demo():

	samp1_base = './samples/Afr_lowAsym_lowSN'
	samp2_base = './samples/Afr_lowAsym_highSN'

	fig = plt.figure(figsize=(12,7))
	gs = gridspec.GridSpec(2,2,left=0.08,bottom=0.09,right=0.99,top=0.99,hspace=0.1)

	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])



	Nsamp = 1000
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	
	
	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,0]),np.log10(sample2[:,0])],
							SN_bins,
							Niter=10) 
	
	ax1 = fig.add_subplot(gs[0,0]) 
	ax2 = fig.add_subplot(gs[0,1],sharey=ax1,sharex=ax1) 


	ax1.hist(10**SN_bins[:-1],10**SN_bins,weights = control_hists[0,:],ls='-',histtype='step',color='Blue',lw=3)
	ax1.hist(10**SN_bins[:-1],10**SN_bins,weights = control_hists[1,:],ls='-',histtype='step',color='Red',lw=3)
	ax1.hist(10**SN_bins[:-1],10**SN_bins,weights = control_hists[2,:],ls='--',histtype='step',color='Black',lw=4)


	control_hist = control_hists[2,:] /(np.sum(control_hists[2,:] * np.diff(SN_bins))) #/ np.diff(SN_bins)
	ax2.hist(10**SN_bins[:-1],10**SN_bins,weights = control_hist,ls='--',histtype='step',color='Black',lw=5)

	for ii in range(5):
		hist1 = np.histogram(np.log10(sample1[np.array(indices_all[0][ii][0]),0]),bins=SN_bins,density=True)[0]
		ax2.hist(10**SN_bins[:-1],bins=10**SN_bins,weights=hist1,histtype='step',color='Blue')
		hist2 = np.histogram(np.log10(sample2[np.array(indices_all[1][ii][0]),0]),bins=SN_bins,density=True)[0]
		ax2.hist(10**SN_bins[:-1],bins=10**SN_bins,weights=hist2,histtype='step',color='Red')

	ax1.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax2.tick_params(axis='both',which='both',direction='in',labelsize=18)
	# ax1.set_xlabel('S/N',fontsize=20)
	ax1.set_ylabel('Probability density',fontsize=20)
	# ax2.set_xlabel('S/N',fontsize=20)
	ax2.set_ylabel('Probability density',fontsize=20)
	ax1.set_xscale('log')
	ax1.set_xticks([7,10,20,40,70,100])
	ax1.set_xticklabels([7,10,20,40,70,100])
	ax1.set_xlim([6,120])
	
	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Blue', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Red', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '--', linewidth = 3)]

	ax1.legend(legend1,[f'N = {Nsamp}','low S/N','high S/N', 'Common S/N'],fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Blue', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Red', ls = '-', linewidth = 3)
			]

	ax2.legend(legend2,[f'N = {Nsamp}','Common S/N','sampled low S/N','sampled high S/N'],fontsize=14)



	Nsamp = 100
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	
	
	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,0]),np.log10(sample2[:,0])],
							SN_bins,
							Niter=10) 
	
	ax3 = fig.add_subplot(gs[1,0]) 
	ax4 = fig.add_subplot(gs[1,1],sharey=ax1,sharex=ax1) 


	ax3.hist(10**SN_bins[:-1],10**SN_bins,weights = control_hists[0,:],ls='-',histtype='step',color='Blue',lw=3)
	ax3.hist(10**SN_bins[:-1],10**SN_bins,weights = control_hists[1,:],ls='-',histtype='step',color='Red',lw=3)
	ax3.hist(10**SN_bins[:-1],10**SN_bins,weights = control_hists[2,:],ls='--',histtype='step',color='Black',lw=4)


	control_hist = control_hists[2,:] /(np.sum(control_hists[2,:] * np.diff(SN_bins))) #/ np.diff(SN_bins)
	ax4.hist(10**SN_bins[:-1],10**SN_bins,weights = control_hist,ls='--',histtype='step',color='Black',lw=5)

	for ii in range(5):
		hist1 = np.histogram(np.log10(sample1[np.array(indices_all[0][ii][0]),0]),bins=SN_bins,density=True)[0]
		ax4.hist(10**SN_bins[:-1],bins=10**SN_bins,weights=hist1,histtype='step',color='Blue')
		hist2 = np.histogram(np.log10(sample2[np.array(indices_all[1][ii][0]),0]),bins=SN_bins,density=True)[0]
		ax4.hist(10**SN_bins[:-1],bins=10**SN_bins,weights=hist2,histtype='step',color='Red')

	ax3.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax4.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax3.set_xlabel('S/N',fontsize=20)
	ax3.set_ylabel('Probability density',fontsize=20)
	ax4.set_xlabel('S/N',fontsize=20)
	ax4.set_ylabel('Probability density',fontsize=20)
	ax3.set_xscale('log')
	ax3.set_xticks([7,10,20,40,70,100])
	ax3.set_xticklabels([7,10,20,40,70,100])
	ax3.set_xlim([6,120])
	
	legend3 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Blue', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Red', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '--', linewidth = 3)]

	ax3.legend(legend3,[f'N = {Nsamp}','low S/N','high S/N', 'Common S/N'],fontsize=14)

	legend4 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Blue', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Red', ls = '-', linewidth = 3)
			]

	ax4.legend(legend4,[f'N = {Nsamp}','Common S/N','sampled low S/N','sampled high S/N'],fontsize=14)

	plt.show()

def SNsampled_Afrhist_demo():
	
	samp1_base = './samples/Afr_lowAsym_lowSN'
	samp2_base = './samples/Afr_lowAsym_highSN'

	Niter = 1000

	fig = plt.figure(figsize=(12,7))
	gs = gridspec.GridSpec(2,2,left=0.08,bottom=0.09,right=0.98,top=0.99,hspace=0.1)

	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	Nsamp = 1000
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])


	noiseless_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless_hist = np.cumsum(noiseless_hist) / np.sum(noiseless_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	


	ax1.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax1.plot(Afr_bins[0:-1],sample2_hist,color='Green',ls='--')
	ax1.plot(Afr_bins[0:-1],sample1_hist,color='Orange',ls='--')


	ax2.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax2.plot(Afr_bins[0:-1],sample2_hist,color='Green',ls='--')
	ax2.plot(Afr_bins[0:-1],sample1_hist,color='Orange',ls='--')

	lowAsym_lowSN_mean_hist = np.mean(sample_hists_all[0], axis=0)
	lowAsym_lowSN_stddev_hist = np.std(sample_hists_all[0], axis=0)
	ax2.errorbar(Afr_bins[0:-1], lowAsym_lowSN_mean_hist, yerr = lowAsym_lowSN_stddev_hist,
					color='Orange', linewidth=2, capsize=6)
	
	lowAsym_highSN_mean_hist = np.mean(sample_hists_all[1], axis=0)
	lowAsym_highSN_stddev_hist = np.std(sample_hists_all[1], axis=0)

	ax2.errorbar(Afr_bins[0:-1], lowAsym_highSN_mean_hist, yerr = lowAsym_highSN_stddev_hist,
					color='Green', linewidth=2, capsize=6)


	ax1.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax2.tick_params(axis='both',which='both',direction='in',labelsize=18)
	# ax1.set_xlabel('S/N',fontsize=20)
	ax1.set_ylabel('Cumulative density',fontsize=20)
	# ax2.set_xlabel('S/N',fontsize=20)
	ax2.set_ylabel('Cumulative density',fontsize=20)
	ax1.set_xticks([1,1.2,1.4,1.6,1.8,2,2.2])
	


	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax1.legend(legend1,[f'N = {Nsamp}','Noiseless','high S/N', 'low S/N'],fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax2.legend(legend2,[f'N = {Nsamp}','sampled high S/N','sampled low S/N'],fontsize=14)

	Nsamp = 100
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax3 = fig.add_subplot(gs[1,0],sharex=ax1,sharey=ax1)
	ax4 = fig.add_subplot(gs[1,1],sharex=ax1,sharey=ax1)


	noiseless_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless_hist = np.cumsum(noiseless_hist) / np.sum(noiseless_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax3.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax3.plot(Afr_bins[0:-1],sample2_hist,color='Green',ls='--')
	ax3.plot(Afr_bins[0:-1],sample1_hist,color='Orange',ls='--')


	ax4.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax4.plot(Afr_bins[0:-1],sample2_hist,color='Green',ls='--')
	ax4.plot(Afr_bins[0:-1],sample1_hist,color='Orange',ls='--')

	lowAsym_lowSN_mean_hist = np.mean(sample_hists_all[0], axis=0)
	lowAsym_lowSN_stddev_hist = np.std(sample_hists_all[0], axis=0)
	ax4.errorbar(Afr_bins[0:-1], lowAsym_lowSN_mean_hist, yerr = lowAsym_lowSN_stddev_hist,
					color='Orange', linewidth=2, capsize=6)
	
	lowAsym_highSN_mean_hist = np.mean(sample_hists_all[1], axis=0)
	lowAsym_highSN_stddev_hist = np.std(sample_hists_all[1], axis=0)

	ax4.errorbar(Afr_bins[0:-1], lowAsym_highSN_mean_hist, yerr = lowAsym_highSN_stddev_hist,
					color='Green', linewidth=2, capsize=6)


	ax3.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax4.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax3.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax3.set_ylabel('Cumulative density',fontsize=20)
	ax4.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax4.set_ylabel('Cumulative density',fontsize=20)
	
	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax3.legend(legend1,[f'N = {Nsamp}','Noiseless','high S/N', 'low S/N'],fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax4.legend(legend2,[f'N = {Nsamp}','sampled high S/N','sampled low S/N'],fontsize=14)

	plt.show()


def DAGJK_sample_sigma_compare():
	
	samp1_base = './samples/Afr_lowAsym_lowSN'
	samp2_base = './samples/Afr_lowAsym_highSN'

	Niter = 1000

	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	Nsamp = 100
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)


	Nrow = 6
	Ncol = 4
	fig = plt.figure(figsize = (9,12))
	gs = gridspec.GridSpec(Nrow,Ncol, hspace=0.25, wspace=0.01, top = 0.99, right = 0.99,
		 bottom  = 0.06, left = 0.06)
	

	sample_hists = sample_hists_all[0]
	sample_DAGJKsigmas = samples_all_DAGJKsigma[0]


	for ii in range(len(Afr_bins) - 1):

		bin_value_dist = sample_hists[:,ii]
		bin_SNsample_sigma = np.std(bin_value_dist)

		bin_DAGJK_sigmas = sample_DAGJKsigmas[:,ii]

		bin_sigma_ratio = bin_DAGJK_sigmas / bin_SNsample_sigma
		median_sigma_ratio = np.median(bin_sigma_ratio)


		ax = fig.add_subplot(gs[int(ii/Ncol),(ii)%Ncol])

		ax.hist(bin_sigma_ratio,bins=20,density=True,alpha=1)
		ax.plot([median_sigma_ratio,median_sigma_ratio],[0,1.3],color='Black')
		ax.text(0.45,0.85, f'[{Afr_bins[ii]:.2f},{Afr_bins[ii+1]:.2f})',
			fontsize=12.,transform=ax.transAxes, zorder=1)
		ax.text(0.45,0.7,f'$\sigma_{{JK}} / \sigma_{{N}}$ = {median_sigma_ratio:.2f}',
			fontsize=12.,transform=ax.transAxes, zorder=1)

		ax.tick_params(axis='both',which='both',direction='in',labelsize=0)
		ax.tick_params(axis='x',which='both',direction='in',labelsize=14)
		ax.set_ylim([0,1.3])
		ax.set_yticks([0,0.5,1])
		ax.set_xticks([xx for xx in range(0,int(1.2*ax.get_xlim()[1]),2)])



		if ii%Ncol == 0:
			ax.set_ylabel('PDF',fontsize=18)
			ax.tick_params(axis='y',which='both',direction='in',labelsize=14)


		if int(ii/Ncol) == Nrow-1:

			ax.set_xlabel('$\sigma_{{JK}} / \sigma_{{N}}$',fontsize=18)

	plt.show()



def test_recover_same_dist_1():
	
	samp1_base = './samples/Afr_lowAsym_highSN'
	samp2_base = './samples/Afr_lowAsym_lowSN'


	Niter = 100

	fig = plt.figure(figsize=(12,7))
	gs = gridspec.GridSpec(2,2,left=0.08,bottom=0.09,right=0.98,top=0.99,hspace=0.1)

	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	Nsamp = 1000
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])


	noiseless_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless_hist = np.cumsum(noiseless_hist) / np.sum(noiseless_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax1.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax1.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax1.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')


	ax2.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax2.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax2.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')

	
	sample1_mean_hist = np.mean(sample_hists_all[0], axis=0)
	sample1_DAGJK_sigma = np.median(samples_all_DAGJKsigma[0], axis=0)
	ax2.errorbar(Afr_bins[0:-1], sample1_mean_hist, yerr = sample1_DAGJK_sigma,
					color='Green', linewidth=2, capsize=6)
	
	sample2_mean_hist = np.mean(sample_hists_all[1], axis=0)
	sample2_DAGJK_sigma = np.median(samples_all_DAGJKsigma[1], axis=0)
	ax2.errorbar(Afr_bins[0:-1], sample2_mean_hist, yerr = sample2_DAGJK_sigma,
					color='Orange', linewidth=2, capsize=6)


	ax1.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax2.tick_params(axis='both',which='both',direction='in',labelsize=18)
	# ax1.set_xlabel('S/N',fontsize=20)
	ax1.set_ylabel('Cumulative density',fontsize=20)
	# ax2.set_xlabel('S/N',fontsize=20)
	ax2.set_ylabel('Cumulative density',fontsize=20)
	ax1.set_xticks([1,1.2,1.4,1.6,1.8,2,2.2])
	


	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax1.legend(legend1,[f'N = {Nsamp}','Noiseless','high S/N', 'low S/N'],fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax2.legend(legend2,[f'N = {Nsamp}','final high S/N','final low S/N'],fontsize=14)

	Nsamp = 100
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax3 = fig.add_subplot(gs[1,0],sharex=ax1,sharey=ax1)
	ax4 = fig.add_subplot(gs[1,1],sharex=ax1,sharey=ax1)


	noiseless_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless_hist = np.cumsum(noiseless_hist) / np.sum(noiseless_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax3.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax3.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax3.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')


	ax4.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax4.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax4.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')

	sample1_mean_hist = np.mean(sample_hists_all[0], axis=0)
	sample1_DAGJK_sigma = np.median(samples_all_DAGJKsigma[0], axis=0)
	ax4.errorbar(Afr_bins[0:-1], sample1_mean_hist, yerr = sample1_DAGJK_sigma,
					color='Green', linewidth=2, capsize=6)
	
	sample2_mean_hist = np.mean(sample_hists_all[1], axis=0)
	sample2_DAGJK_sigma = np.median(samples_all_DAGJKsigma[1], axis=0)
	ax4.errorbar(Afr_bins[0:-1], sample2_mean_hist, yerr = sample2_DAGJK_sigma,
					color='Orange', linewidth=2, capsize=6)

	ax3.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax4.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax3.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax3.set_ylabel('Cumulative density',fontsize=20)
	ax4.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax4.set_ylabel('Cumulative density',fontsize=20)
	
	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax3.legend(legend1,[f'N = {Nsamp}','Noiseless','high S/N', 'low S/N'],fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax4.legend(legend2,[f'N = {Nsamp}','final high S/N','final low S/N'],fontsize=14)

	plt.show()

def test_recover_same_dist_2():

	# # Same v2 - more asymmetric distribution
	samp1_base = './samples/Afr_highAsym_highSN'
	samp2_base = './samples/Afr_highAsym_lowSN'

	Niter = 100

	fig = plt.figure(figsize=(12,7))
	gs = gridspec.GridSpec(2,2,left=0.08,bottom=0.09,right=0.98,top=0.99,hspace=0.1)

	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	Nsamp = 1000
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])


	noiseless_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless_hist = np.cumsum(noiseless_hist) / np.sum(noiseless_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax1.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax1.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax1.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')


	ax2.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax2.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax2.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')

	
	sample1_mean_hist = np.mean(sample_hists_all[0], axis=0)
	sample1_DAGJK_sigma = np.median(samples_all_DAGJKsigma[0], axis=0)
	ax2.errorbar(Afr_bins[0:-1], sample1_mean_hist, yerr = sample1_DAGJK_sigma,
					color='Green', linewidth=2, capsize=6)
	
	sample2_mean_hist = np.mean(sample_hists_all[1], axis=0)
	sample2_DAGJK_sigma = np.median(samples_all_DAGJKsigma[1], axis=0)
	ax2.errorbar(Afr_bins[0:-1], sample2_mean_hist, yerr = sample2_DAGJK_sigma,
					color='Orange', linewidth=2, capsize=6)


	ax1.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax2.tick_params(axis='both',which='both',direction='in',labelsize=18)
	# ax1.set_xlabel('S/N',fontsize=20)
	ax1.set_ylabel('Cumulative density',fontsize=20)
	# ax2.set_xlabel('S/N',fontsize=20)
	ax2.set_ylabel('Cumulative density',fontsize=20)
	ax1.set_xticks([1,1.2,1.4,1.6,1.8,2,2.2])
	


	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax1.legend(legend1,[f'N = {Nsamp}','Noiseless','high S/N', 'low S/N'],fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax2.legend(legend2,[f'N = {Nsamp}','final high S/N','final low S/N'],fontsize=14)

	Nsamp = 100
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax3 = fig.add_subplot(gs[1,0],sharex=ax1,sharey=ax1)
	ax4 = fig.add_subplot(gs[1,1],sharex=ax1,sharey=ax1)


	noiseless_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless_hist = np.cumsum(noiseless_hist) / np.sum(noiseless_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax3.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax3.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax3.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')


	ax4.plot(Afr_bins[0:-1],noiseless_hist,color='Black',ls='-')
	ax4.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax4.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')

	sample1_mean_hist = np.mean(sample_hists_all[0], axis=0)
	sample1_DAGJK_sigma = np.median(samples_all_DAGJKsigma[0], axis=0)
	ax4.errorbar(Afr_bins[0:-1], sample1_mean_hist, yerr = sample1_DAGJK_sigma,
					color='Green', linewidth=2, capsize=6)
	
	sample2_mean_hist = np.mean(sample_hists_all[1], axis=0)
	sample2_DAGJK_sigma = np.median(samples_all_DAGJKsigma[1], axis=0)
	ax4.errorbar(Afr_bins[0:-1], sample2_mean_hist, yerr = sample2_DAGJK_sigma,
					color='Orange', linewidth=2, capsize=6)

	ax3.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax4.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax3.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax3.set_ylabel('Cumulative density',fontsize=20)
	ax4.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax4.set_ylabel('Cumulative density',fontsize=20)
	
	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax3.legend(legend1,[f'N = {Nsamp}','Noiseless','high S/N', 'low S/N'],fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax4.legend(legend2,[f'N = {Nsamp}','final high S/N','final low S/N'],fontsize=14)

	plt.show()


def test_recover_different_dist_1():
	
	#more asym sample has more noise-induced asymmetry
	samp1_base = './samples/Afr_lowAsym_highSN'
	samp2_base = './samples/Afr_highAsym_lowSN'

	Niter = 100

	fig = plt.figure(figsize=(12,7))
	gs = gridspec.GridSpec(2,2,left=0.08,bottom=0.09,right=0.98,top=0.99,hspace=0.1)

	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	Nsamp = 1000
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])


	noiseless1_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	noiseless2_hist = np.histogram(sample2[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless1_hist = np.cumsum(noiseless1_hist) / np.sum(noiseless1_hist)
	noiseless2_hist = np.cumsum(noiseless2_hist) / np.sum(noiseless2_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax1.plot(Afr_bins[0:-1],noiseless1_hist,color='Black',ls='-')
	ax1.plot(Afr_bins[0:-1],noiseless2_hist,color='Black',ls=':')
	ax1.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax1.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')


	ax2.plot(Afr_bins[0:-1],noiseless1_hist,color='Black',ls='-')
	ax2.plot(Afr_bins[0:-1],noiseless2_hist,color='Black',ls=':')
	ax2.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax2.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')

	
	sample1_mean_hist = np.mean(sample_hists_all[0], axis=0)
	sample1_DAGJK_sigma = np.median(samples_all_DAGJKsigma[0], axis=0)
	ax2.errorbar(Afr_bins[0:-1], sample1_mean_hist, yerr = sample1_DAGJK_sigma,
					color='Green', linewidth=2, capsize=6)
	
	sample2_mean_hist = np.mean(sample_hists_all[1], axis=0)
	sample2_DAGJK_sigma = np.median(samples_all_DAGJKsigma[1], axis=0)
	ax2.errorbar(Afr_bins[0:-1], sample2_mean_hist, yerr = sample2_DAGJK_sigma,
					color='Orange', linewidth=2, capsize=6)


	ax1.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax2.tick_params(axis='both',which='both',direction='in',labelsize=18)
	# ax1.set_xlabel('S/N',fontsize=20)
	ax1.set_ylabel('Cumulative density',fontsize=20)
	# ax2.set_xlabel('S/N',fontsize=20)
	ax2.set_ylabel('Cumulative density',fontsize=20)
	ax1.set_xticks([1,1.2,1.4,1.6,1.8,2,2.2])
	


	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = ':', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax1.legend(legend1,
				[f'N = {Nsamp}','low Asym','high Asym','low Asym; high S/N', 'high Asym; low S/N'],
				fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax2.legend(legend2,
		[f'N = {Nsamp}','final low Asym; high S/N','final high Asym; low S/N'],
		fontsize=14)

	Nsamp = 100
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax3 = fig.add_subplot(gs[1,0],sharex=ax1,sharey=ax1)
	ax4 = fig.add_subplot(gs[1,1],sharex=ax1,sharey=ax1)


	noiseless1_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	noiseless2_hist = np.histogram(sample2[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless1_hist = np.cumsum(noiseless1_hist) / np.sum(noiseless1_hist)
	noiseless2_hist = np.cumsum(noiseless2_hist) / np.sum(noiseless2_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax3.plot(Afr_bins[0:-1],noiseless1_hist,color='Black',ls='-')
	ax3.plot(Afr_bins[0:-1],noiseless2_hist,color='Black',ls=':')
	ax3.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax3.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')


	ax4.plot(Afr_bins[0:-1],noiseless1_hist,color='Black',ls='-')
	ax4.plot(Afr_bins[0:-1],noiseless2_hist,color='Black',ls=':')
	ax4.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax4.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')

	sample1_mean_hist = np.mean(sample_hists_all[0], axis=0)
	sample1_DAGJK_sigma = np.median(samples_all_DAGJKsigma[0], axis=0)
	ax4.errorbar(Afr_bins[0:-1], sample1_mean_hist, yerr = sample1_DAGJK_sigma,
					color='Green', linewidth=2, capsize=6)
	
	sample2_mean_hist = np.mean(sample_hists_all[1], axis=0)
	sample2_DAGJK_sigma = np.median(samples_all_DAGJKsigma[1], axis=0)
	ax4.errorbar(Afr_bins[0:-1], sample2_mean_hist, yerr = sample2_DAGJK_sigma,
					color='Orange', linewidth=2, capsize=6)

	ax3.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax4.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax3.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax3.set_ylabel('Cumulative density',fontsize=20)
	ax4.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax4.set_ylabel('Cumulative density',fontsize=20)
	
	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = ':', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax3.legend(legend1,
				[f'N = {Nsamp}','low Asym','high Asym','low Asym; high S/N', 'high Asym; low S/N'],
				fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax4.legend(legend2,
		[f'N = {Nsamp}','final low Asym; high S/N','final high Asym; low S/N'],
		fontsize=14)

	plt.show()

def test_recover_different_dist_2():
	
	# #different v2 - less asym sample has more noise-induced asymmetry
	samp1_base = './samples/Afr_lowAsym_lowSN'
	samp2_base = './samples/Afr_highAsym_highSN'


	Niter = 100

	fig = plt.figure(figsize=(12,7))
	gs = gridspec.GridSpec(2,2,left=0.08,bottom=0.09,right=0.98,top=0.99,hspace=0.1)

	Afr_bins = np.arange(1,2.2,0.05)
	dex_SN = 0.05
	min_SN=7
	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
	SN_bins = np.append(SN_bins,[np.log10(200)])

	Nsamp = 1000
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax1 = fig.add_subplot(gs[0,0])
	ax2 = fig.add_subplot(gs[0,1])


	noiseless1_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	noiseless2_hist = np.histogram(sample2[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless1_hist = np.cumsum(noiseless1_hist) / np.sum(noiseless1_hist)
	noiseless2_hist = np.cumsum(noiseless2_hist) / np.sum(noiseless2_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax1.plot(Afr_bins[0:-1],noiseless1_hist,color='Black',ls='-')
	ax1.plot(Afr_bins[0:-1],noiseless2_hist,color='Black',ls=':')
	ax1.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax1.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')


	ax2.plot(Afr_bins[0:-1],noiseless1_hist,color='Black',ls='-')
	ax2.plot(Afr_bins[0:-1],noiseless2_hist,color='Black',ls=':')
	ax2.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax2.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')

	
	sample1_mean_hist = np.mean(sample_hists_all[0], axis=0)
	sample1_DAGJK_sigma = np.median(samples_all_DAGJKsigma[0], axis=0)
	ax2.errorbar(Afr_bins[0:-1], sample1_mean_hist, yerr = sample1_DAGJK_sigma,
					color='Green', linewidth=2, capsize=6)
	
	sample2_mean_hist = np.mean(sample_hists_all[1], axis=0)
	sample2_DAGJK_sigma = np.median(samples_all_DAGJKsigma[1], axis=0)
	ax2.errorbar(Afr_bins[0:-1], sample2_mean_hist, yerr = sample2_DAGJK_sigma,
					color='Orange', linewidth=2, capsize=6)


	ax1.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax2.tick_params(axis='both',which='both',direction='in',labelsize=18)
	# ax1.set_xlabel('S/N',fontsize=20)
	ax1.set_ylabel('Cumulative density',fontsize=20)
	# ax2.set_xlabel('S/N',fontsize=20)
	ax2.set_ylabel('Cumulative density',fontsize=20)
	ax1.set_xticks([1,1.2,1.4,1.6,1.8,2,2.2])
	


	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = ':', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax1.legend(legend1,
		[f'N = {Nsamp}','low Asym','high Asym','low Asym; low S/N', 'high Asym; high S/N'],
		fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax2.legend(legend2,
		[f'N = {Nsamp}','final low Asym; low S/N','high Asym; high S/N'],
		fontsize=14)

	Nsamp = 100
	sample1 = np.loadtxt(f'{samp1_base}_N{Nsamp}.dat')
	sample2 = np.loadtxt(f'{samp2_base}_N{Nsamp}.dat')
	

	indices_all, control_hists = ds.sample_to_common_dist(
							[np.log10(sample1[:,2]),np.log10(sample2[:,2])],
							SN_bins,
							Niter=Niter) 
	
	sample_hists_all, samples_all_DAGJKsigma =  ds.jackknife_sample_distributions(
								[sample1[:,3],sample2[:,3]],
								Afr_bins,indices_all)
	
	ax3 = fig.add_subplot(gs[1,0],sharex=ax1,sharey=ax1)
	ax4 = fig.add_subplot(gs[1,1],sharex=ax1,sharey=ax1)


	noiseless1_hist = np.histogram(sample1[:,1],bins=Afr_bins,density=True)[0]
	noiseless2_hist = np.histogram(sample2[:,1],bins=Afr_bins,density=True)[0]
	sample1_hist = np.histogram(sample1[:,3],bins=Afr_bins,density=True)[0]
	sample2_hist = np.histogram(sample2[:,3],bins=Afr_bins,density=True)[0]


	noiseless1_hist = np.cumsum(noiseless1_hist) / np.sum(noiseless1_hist)
	noiseless2_hist = np.cumsum(noiseless2_hist) / np.sum(noiseless2_hist)
	sample1_hist = np.cumsum(sample1_hist) / np.sum(sample1_hist)
	sample2_hist = np.cumsum(sample2_hist) / np.sum(sample2_hist)
	

	ax3.plot(Afr_bins[0:-1],noiseless1_hist,color='Black',ls='-')
	ax3.plot(Afr_bins[0:-1],noiseless2_hist,color='Black',ls=':')
	ax3.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax3.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')


	ax4.plot(Afr_bins[0:-1],noiseless1_hist,color='Black',ls='-')
	ax4.plot(Afr_bins[0:-1],noiseless2_hist,color='Black',ls=':')
	ax4.plot(Afr_bins[0:-1],sample1_hist,color='Green',ls='--')
	ax4.plot(Afr_bins[0:-1],sample2_hist,color='Orange',ls='--')

	sample1_mean_hist = np.mean(sample_hists_all[0], axis=0)
	sample1_DAGJK_sigma = np.median(samples_all_DAGJKsigma[0], axis=0)
	ax4.errorbar(Afr_bins[0:-1], sample1_mean_hist, yerr = sample1_DAGJK_sigma,
					color='Green', linewidth=2, capsize=6)
	
	sample2_mean_hist = np.mean(sample_hists_all[1], axis=0)
	sample2_DAGJK_sigma = np.median(samples_all_DAGJKsigma[1], axis=0)
	ax4.errorbar(Afr_bins[0:-1], sample2_mean_hist, yerr = sample2_DAGJK_sigma,
					color='Orange', linewidth=2, capsize=6)

	ax3.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax4.tick_params(axis='both',which='both',direction='in',labelsize=18)
	ax3.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax3.set_ylabel('Cumulative density',fontsize=20)
	ax4.set_xlabel('Asymmetry meausure A$_{fr}$',fontsize=20)
	ax4.set_ylabel('Cumulative density',fontsize=20)
	
	legend1 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Black', ls = ':', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '--', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '--', linewidth = 3)]

	ax3.legend(legend1,
		[f'N = {Nsamp}','low Asym','high Asym','low Asym; low S/N', 'high Asym; high S/N'],
		fontsize=14)

	legend2 = [Line2D([0], [0], color = 'White', ls = '-', linewidth = 3),
			# Line2D([0], [0], color = 'Black', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Green', ls = '-', linewidth = 3),
			Line2D([0], [0], color = 'Orange', ls = '-', linewidth = 3)
			]

	ax4.legend(legend2,
		[f'N = {Nsamp}','final low Asym; low S/N','high Asym; high S/N'],
		fontsize=14)

	plt.show()




# def test_recover_same_dist_2():

# 	Nsamp = 1000
# 	Afr_lowAsym_lowSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
# 	sample1[:,0] = sample_SN(Nsamp, 1,1/10,7,100)
# 	sample1[:,1] = sample_Afr_dist(Nsamp,asym_rate = 'high')
# 	sample1[:,2:] = MAP.get_SN_Afr(sample1[:,0],Afr = sample1[:,1])
	

# 	Afr_lowAsym_highSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
# 	sample2[:,0] = sample_SN(Nsamp, 1,1/25,7,100)
# 	sample2[:,1] = sample_Afr_dist(Nsamp, asym_rate = 'high')
# 	sample2[:,2:] = MAP.get_SN_Afr(sample2[:,0],Afr = sample2[:,1])

	

# 	Afr_bins = np.arange(1,2.2,0.05)
# 	dex_SN = 0.05
# 	min_SN=7
# 	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
# 	SN_bins = np.append(SN_bins,[np.log10(200)])


# 	Afr_lowAsym_lowSN_hist,edges = np.histogram(Afr_lowAsym_lowSN[:,3],bins=Afr_bins,density=True)
# 	Afr_lowAsym_highSN_hist,edges = np.histogram(sample2[:,3],bins=Afr_bins,density=True)
# 	Afr_lowAsym_lowSN_hist = np.cumsum(Afr_lowAsym_lowSN_hist) / np.sum(Afr_lowAsym_lowSN_hist)
# 	Afr_lowAsym_highSN_hist = np.cumsum(Afr_lowAsym_highSN_hist) / np.sum(Afr_lowAsym_highSN_hist)
# 	ax1.plot(Afr_bins[0:-1],Afr_lowAsym_lowSN_hist,color='Green')
# 	ax1.plot(Afr_bins[0:-1],Afr_lowAsym_highSN_hist,color='Orange')



# 	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
# 						ds.control_samples(
# 							[Afr_lowAsym_lowSN[:,3],Afr_lowAsym_highSN[:,3]],
# 							Afr_bins,
# 							[np.log10(Afr_lowAsym_lowSN[:,2]),np.log10(Afr_lowAsym_highSN[:,2])],
# 							SN_bins,
# 							Niter=1000)

# 	#plot cumulative controlled hists

# 	fig = plt.figure(figsize = (10,12))
# 	gs = gridspec.GridSpec(2, 1, top = 0.99, right = 0.99, bottom  = 0.08, left = 0.12,hspace=0)

# 	ax1 = fig.add_subplot(gs[0,0])
# 	ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)

# 	ax1.plot(Afr_bins[0:-1],hist1,color='Green')
# 	ax1.plot(Afr_bins[0:-1],hist2,color='Orange')

# 	ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
# 			 axis = ax2,names=['low SN','high SN'])


# 	ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	
# 	ax1.set_ylim([0.1, 1])
# 	ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

# 	ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
# 	ax1.set_ylabel('Cumulative Histogram', fontsize=27)
# 	ax2.set_ylabel('Cumulative Histogram', fontsize=27)

# 	# fig.savefig('./figures/recover_same_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))




# 	#plot sampling uncertaintiy vs DAGJK uncertainty
# 	ds.plot_compare_DAGJK_Nsamp_sigmas(samples_all_DAGJKsigma, samples_all_hists, sample_bins, save=None)







# 	# for ii in range(10):
		


# 	# 	index1 = nprand.choice(len(data1),Nsamp)
# 	# 	index2 = nprand.choice(len(data2),Nsamp)


# 	# 	hist1,edges = np.histogram(data1['Afr'][index1],bins=Afr_bins,density=True)
# 	# 	hist2,edges = np.histogram(data2['Afr'][index2],bins=Afr_bins,density=True)
# 	# 	hist1 = np.cumsum(hist1) / np.sum(hist1)
# 	# 	hist2 = np.cumsum(hist2) / np.sum(hist2)
# 	# 	ax1.plot(Afr_bins[0:-1],hist1,color='Green')
# 	# 	ax1.plot(Afr_bins[0:-1],hist2,color='Orange')



# 	# 	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
# 	# 						ds.control_samples([data2['Afr'][index2],data1['Afr'][index1]],Afr_bins,
# 	# 								[np.log10(data2['SN'][index2]),np.log10(data1['SN'][index1])],SN_bins,
# 	# 								Niter=10000)

# 	# 	ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
# 	# 			 axis = ax2,names=['low SN','high SN'])


# 	# 	ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	# 	ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	# 	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		
# 	# 	ax1.set_ylim([0.1, 1])
# 	# 	ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

# 	# 	ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
# 	# 	ax1.set_ylabel('Cumulative Histogram', fontsize=27)
# 	# 	ax2.set_ylabel('Cumulative Histogram', fontsize=27)

# 	# 	fig.savefig('./figures/recover_same_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))

# 	# 	# plt.show()

# def test_recover_different_dist_1():

# 	Nsamp = 1000
# 	Afr_lowAsym_highSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
# 	Afr_lowAsym_highSN[:,0] = sample_SN(Nsamp, 1,1/25,7,100)
# 	Afr_lowAsym_highSN[:,1] = sample_Afr_dist(Nsamp,asym_rate = 'low')
# 	Afr_lowAsym_highSN[:,2:] = MAP.get_SN_Afr(Afr_lowAsym_lowSN[:,0],Afr = Afr_lowAsym_lowSN[:,1])
	

# 	Afr_highAsym_lowSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
# 	Afr_highAsym_lowSN[:,0] = sample_SN(Nsamp, 1,1/10,7,100)
# 	Afr_highAsym_lowSN[:,1] = sample_Afr_dist(Nsamp, asym_rate = 'high')
# 	Afr_highAsym_lowSN[:,2:] = MAP.get_SN_Afr(Afr_lowAsym_highSN[:,0],Afr = Afr_lowAsym_highSN[:,1])

	

# 	Afr_bins = np.arange(1,2.2,0.05)
# 	dex_SN = 0.05
# 	min_SN=7
# 	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
# 	SN_bins = np.append(SN_bins,[np.log10(200)])


# 	Afr_lowAsym_lowSN_hist,edges = np.histogram(Afr_lowAsym_lowSN[:,3],bins=Afr_bins,density=True)
# 	Afr_lowAsym_highSN_hist,edges = np.histogram(Afr_lowAsym_highSN[:,3],bins=Afr_bins,density=True)
# 	Afr_lowAsym_lowSN_hist = np.cumsum(Afr_lowAsym_lowSN_hist) / np.sum(Afr_lowAsym_lowSN_hist)
# 	Afr_lowAsym_highSN_hist = np.cumsum(Afr_lowAsym_highSN_hist) / np.sum(Afr_lowAsym_highSN_hist)
# 	ax1.plot(Afr_bins[0:-1],Afr_lowAsym_lowSN_hist,color='Green')
# 	ax1.plot(Afr_bins[0:-1],Afr_lowAsym_highSN_hist,color='Orange')



# 	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
# 						ds.control_samples(
# 							[Afr_lowAsym_lowSN[:,3],Afr_lowAsym_highSN[:,3]],
# 							Afr_bins,
# 							[np.log10(Afr_lowAsym_lowSN[:,2]),np.log10(Afr_lowAsym_highSN[:,2])],
# 							SN_bins,
# 							Niter=1000)

# 	#plot cumulative controlled hists

# 	fig = plt.figure(figsize = (10,12))
# 	gs = gridspec.GridSpec(2, 1, top = 0.99, right = 0.99, bottom  = 0.08, left = 0.12,hspace=0)

# 	ax1 = fig.add_subplot(gs[0,0])
# 	ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)

# 	ax1.plot(Afr_bins[0:-1],hist1,color='Green')
# 	ax1.plot(Afr_bins[0:-1],hist2,color='Orange')

# 	ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
# 			 axis = ax2,names=['low SN','high SN'])


# 	ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	
# 	ax1.set_ylim([0.1, 1])
# 	ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

# 	ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
# 	ax1.set_ylabel('Cumulative Histogram', fontsize=27)
# 	ax2.set_ylabel('Cumulative Histogram', fontsize=27)

# 	# fig.savefig('./figures/recover_same_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))




# 	#plot sampling uncertaintiy vs DAGJK uncertainty
# 	ds.plot_compare_DAGJK_Nsamp_sigmas(samples_all_DAGJKsigma, samples_all_hists, sample_bins, save=None)







# 	# for ii in range(10):
		


# 	# 	index1 = nprand.choice(len(data1),Nsamp)
# 	# 	index2 = nprand.choice(len(data2),Nsamp)


# 	# 	hist1,edges = np.histogram(data1['Afr'][index1],bins=Afr_bins,density=True)
# 	# 	hist2,edges = np.histogram(data2['Afr'][index2],bins=Afr_bins,density=True)
# 	# 	hist1 = np.cumsum(hist1) / np.sum(hist1)
# 	# 	hist2 = np.cumsum(hist2) / np.sum(hist2)
# 	# 	ax1.plot(Afr_bins[0:-1],hist1,color='Green')
# 	# 	ax1.plot(Afr_bins[0:-1],hist2,color='Orange')



# 	# 	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
# 	# 						ds.control_samples([data2['Afr'][index2],data1['Afr'][index1]],Afr_bins,
# 	# 								[np.log10(data2['SN'][index2]),np.log10(data1['SN'][index1])],SN_bins,
# 	# 								Niter=10000)

# 	# 	ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
# 	# 			 axis = ax2,names=['low SN','high SN'])


# 	# 	ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	# 	ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	# 	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		
# 	# 	ax1.set_ylim([0.1, 1])
# 	# 	ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

# 	# 	ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
# 	# 	ax1.set_ylabel('Cumulative Histogram', fontsize=27)
# 	# 	ax2.set_ylabel('Cumulative Histogram', fontsize=27)

# 	# 	fig.savefig('./figures/recover_same_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))

# 	# 	# plt.show()

# def test_recover_different_dist_2():

# 	Nsamp = 1000
# 	Afr_lowAsym_lowSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
# 	Afr_lowAsym_lowSN[:,0] = sample_SN(Nsamp, 1,1/10,7,100)
# 	Afr_lowAsym_lowSN[:,1] = sample_Afr_dist(Nsamp,asym_rate = 'low')
# 	Afr_lowAsym_lowSN[:,2:] = MAP.get_SN_Afr(Afr_lowAsym_lowSN[:,0],Afr = Afr_lowAsym_lowSN[:,1])
	

# 	Afr_highAsym_highSN = np.zeros([Nsamp,4])			#SN, Afr_noiseless, SN_noise, Afr_noise
# 	Afr_highAsym_highSN[:,0] = sample_SN(Nsamp, 1,1/25,7,100)
# 	Afr_highAsym_highSN[:,1] = sample_Afr_dist(Nsamp, asym_rate = 'high')
# 	Afr_highAsym_highSN[:,2:] = MAP.get_SN_Afr(Afr_lowAsym_highSN[:,0],Afr = Afr_lowAsym_highSN[:,1])

	

# 	Afr_bins = np.arange(1,2.2,0.05)
# 	dex_SN = 0.05
# 	min_SN=7
# 	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
# 	SN_bins = np.append(SN_bins,[np.log10(200)])


# 	Afr_lowAsym_lowSN_hist,edges = np.histogram(Afr_lowAsym_lowSN[:,3],bins=Afr_bins,density=True)
# 	Afr_lowAsym_highSN_hist,edges = np.histogram(Afr_lowAsym_highSN[:,3],bins=Afr_bins,density=True)
# 	Afr_lowAsym_lowSN_hist = np.cumsum(Afr_lowAsym_lowSN_hist) / np.sum(Afr_lowAsym_lowSN_hist)
# 	Afr_lowAsym_highSN_hist = np.cumsum(Afr_lowAsym_highSN_hist) / np.sum(Afr_lowAsym_highSN_hist)
# 	ax1.plot(Afr_bins[0:-1],Afr_lowAsym_lowSN_hist,color='Green')
# 	ax1.plot(Afr_bins[0:-1],Afr_lowAsym_highSN_hist,color='Orange')



# 	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
# 						ds.control_samples(
# 							[Afr_lowAsym_lowSN[:,3],Afr_lowAsym_highSN[:,3]],
# 							Afr_bins,
# 							[np.log10(Afr_lowAsym_lowSN[:,2]),np.log10(Afr_lowAsym_highSN[:,2])],
# 							SN_bins,
# 							Niter=1000)

# 	#plot cumulative controlled hists

# 	fig = plt.figure(figsize = (10,12))
# 	gs = gridspec.GridSpec(2, 1, top = 0.99, right = 0.99, bottom  = 0.08, left = 0.12,hspace=0)

# 	ax1 = fig.add_subplot(gs[0,0])
# 	ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)

# 	ax1.plot(Afr_bins[0:-1],hist1,color='Green')
# 	ax1.plot(Afr_bins[0:-1],hist2,color='Orange')

# 	ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
# 			 axis = ax2,names=['low SN','high SN'])


# 	ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
	
# 	ax1.set_ylim([0.1, 1])
# 	ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

# 	ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
# 	ax1.set_ylabel('Cumulative Histogram', fontsize=27)
# 	ax2.set_ylabel('Cumulative Histogram', fontsize=27)

# 	# fig.savefig('./figures/recover_same_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))




# 	#plot sampling uncertaintiy vs DAGJK uncertainty
# 	ds.plot_compare_DAGJK_Nsamp_sigmas(samples_all_DAGJKsigma, samples_all_hists, sample_bins, save=None)







# 	# for ii in range(10):
		


# 	# 	index1 = nprand.choice(len(data1),Nsamp)
# 	# 	index2 = nprand.choice(len(data2),Nsamp)


# 	# 	hist1,edges = np.histogram(data1['Afr'][index1],bins=Afr_bins,density=True)
# 	# 	hist2,edges = np.histogram(data2['Afr'][index2],bins=Afr_bins,density=True)
# 	# 	hist1 = np.cumsum(hist1) / np.sum(hist1)
# 	# 	hist2 = np.cumsum(hist2) / np.sum(hist2)
# 	# 	ax1.plot(Afr_bins[0:-1],hist1,color='Green')
# 	# 	ax1.plot(Afr_bins[0:-1],hist2,color='Orange')



# 	# 	samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
# 	# 						ds.control_samples([data2['Afr'][index2],data1['Afr'][index1]],Afr_bins,
# 	# 								[np.log10(data2['SN'][index2]),np.log10(data1['SN'][index1])],SN_bins,
# 	# 								Niter=10000)

# 	# 	ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
# 	# 			 axis = ax2,names=['low SN','high SN'])


# 	# 	ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	# 	ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 	# 	ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		
# 	# 	ax1.set_ylim([0.1, 1])
# 	# 	ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

# 	# 	ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
# 	# 	ax1.set_ylabel('Cumulative Histogram', fontsize=27)
# 	# 	ax2.set_ylabel('Cumulative Histogram', fontsize=27)

# 	# 	fig.savefig('./figures/recover_same_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))

# 	# 	# plt.show()


# def test_recover_different_dist():


# 	data1 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/sym_sample_highSN_measuements.ascii',format='ascii')
# 	data2 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/asym_sample_lowSN_measuements.ascii',format='ascii')


# 	Afr_bins = np.arange(1,2.2,0.05)
# 	dex_SN = 0.05
# 	min_SN=7
# 	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
# 	SN_bins = np.append(SN_bins,[np.log10(200)])

# 	# plt.hist(nprand.choice(data1['Afr'],100),bins=Afr_bins,histtype='step',color='Green',density=True,cumulative=True)
# 	# plt.hist(nprand.choice(data2['Afr'],100),bins=Afr_bins,histtype='step',color='Orange',density=True,cumulative=True)
# 	# plt.show()

# 	# plt.hist(data1['SN'],histtype='step')
# 	# plt.hist(data2['SN'],histtype='step')
# 	# plt.show()

# 	Nsamp = 100

# 	for ii in range(10):
# 		fig = plt.figure(figsize = (10,12))

# 		gs = gridspec.GridSpec(2, 1, top = 0.99, right = 0.99, bottom  = 0.08, left = 0.12,hspace=0)

# 		ax1 = fig.add_subplot(gs[0,0])
# 		ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)


# 		index1 = nprand.choice(len(data1),Nsamp)
# 		index2 = nprand.choice(len(data2),Nsamp)


# 		hist1,edges = np.histogram(data1['Afr'][index1],bins=Afr_bins,density=True)
# 		hist2,edges = np.histogram(data2['Afr'][index2],bins=Afr_bins,density=True)
# 		hist1 = np.cumsum(hist1) / np.sum(hist1)
# 		hist2 = np.cumsum(hist2) / np.sum(hist2)
# 		ax1.plot(Afr_bins[0:-1],hist1,color='Green')
# 		ax1.plot(Afr_bins[0:-1],hist2,color='Orange')



# 		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
# 							ds.control_samples([data2['Afr'][index2],data1['Afr'][index1]],Afr_bins,
# 									[np.log10(data2['SN'][index2]),np.log10(data1['SN'][index1])],SN_bins,
# 									Niter=10000)

# 		ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
# 				 axis = ax2,names=['asym, low SN','sym, high SN'])


# 		ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 		ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 		ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		
# 		ax1.set_ylim([0.1, 1])
# 		ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

# 		ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
# 		ax1.set_ylabel('Cumulative Histogram', fontsize=27)
# 		ax2.set_ylabel('Cumulative Histogram', fontsize=27)

# 		fig.savefig('./figures/recover_diff_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))

# 		# plt.show()

# def test_opposite_recover_different_dist():


# 	data2 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/asym_sample_highSN_measuements.ascii',format='ascii')
# 	data1 = Table.read('/home/awatts/Adam_PhD/models_fitting/distribution-sampling/measurements/sym_sample_lowSN_measuements.ascii',format='ascii')


# 	Afr_bins = np.arange(1,2.2,0.05)
# 	dex_SN = 0.05
# 	min_SN=7
# 	SN_bins = np.arange(np.log10(min_SN),np.log10(50) + dex_SN,dex_SN)
# 	SN_bins = np.append(SN_bins,[np.log10(200)])

# 	# plt.hist(nprand.choice(data1['Afr'],100),bins=Afr_bins,histtype='step',color='Green',density=True,cumulative=True)
# 	# plt.hist(nprand.choice(data2['Afr'],100),bins=Afr_bins,histtype='step',color='Orange',density=True,cumulative=True)
# 	# plt.show()

# 	# plt.hist(data1['SN'],histtype='step',color='Green')
# 	# plt.hist(data2['SN'],histtype='step',color='Orange')
# 	# plt.show()

# 	Nsamp = 100

# 	for ii in range(10):
# 		fig = plt.figure(figsize = (10,12))

# 		gs = gridspec.GridSpec(2, 1, top = 0.99, right = 0.99, bottom  = 0.08, left = 0.12,hspace=0)

# 		ax1 = fig.add_subplot(gs[0,0])
# 		ax2 = fig.add_subplot(gs[1,0],sharex=ax1, sharey=ax1)


# 		index1 = nprand.choice(len(data1),Nsamp)
# 		index2 = nprand.choice(len(data2),Nsamp)


# 		hist1,edges = np.histogram(data1['Afr'][index1],bins=Afr_bins,density=True)
# 		hist2,edges = np.histogram(data2['Afr'][index2],bins=Afr_bins,density=True)
# 		hist1 = np.cumsum(hist1) / np.sum(hist1)
# 		hist2 = np.cumsum(hist2) / np.sum(hist2)
# 		ax1.plot(Afr_bins[0:-1],hist1,color='Green')
# 		ax1.plot(Afr_bins[0:-1],hist2,color='Orange')



# 		samples_all_hists, samples_all_DAGJKsigma, indexes_all_iter = \
# 							ds.control_samples([data2['Afr'][index2],data1['Afr'][index1]],Afr_bins,
# 									[np.log10(data2['SN'][index2]),np.log10(data1['SN'][index1])],SN_bins,
# 									Niter=10000)

# 		ds.plot_controlled_cumulative_histograms(samples_all_hists,Afr_bins,samples_all_DAGJKsigma,
# 				 axis = ax2,names=['asym, high SN','sym, low SN'])


# 		ax1.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 		ax2.tick_params(axis = 'both', which = 'both', direction = 'in', labelsize = 25)
# 		ax1.tick_params(axis = 'x', which = 'both', direction = 'in', labelsize = 0)
		
# 		ax1.set_ylim([0.1, 1])
# 		ax1.set_title('{Nsamp} samples'.format(Nsamp=Nsamp))

# 		ax2.set_xlabel('Asymmetry measure $A_{fr}$', fontsize=27)
# 		ax1.set_ylabel('Cumulative Histogram', fontsize=27)
# 		ax2.set_ylabel('Cumulative Histogram', fontsize=27)

# 		fig.savefig('./figures/opposite_recover_diff_dist_samp{Nsamp}_{ii}.png'.format(Nsamp=Nsamp,ii=ii))

# 		# plt.show()






def sample_SN(Nsamp,a,b,lim1,lim2):
	#sample Nsamp S/N values from an exponential S/N distribution


	prob = rng.uniform(size = Nsamp)
	A = (-b) / ((np.exp((-b)*(lim2-a)) - np.exp((-b)*(lim1-a))))
	SN = (np.log(-b*prob/A + np.exp((-b)*(lim1-a)))/(-b)) + a


	return SN

def sample_SN_test():
	
	SNrange = np.arange(7,101,1)
	SNdist_low = norm_exp_dist(SNrange,-1000,1/5,1,7,100)				#low S/N dist
	SNdist_high = norm_exp_dist(SNrange,-1000,1/8,1,7,100)				#high S/N dist


	SN1 = sample_SN(100000,-1000,1/5,7,100)			#lower SN distribution
	SN2 = sample_SN(100000,-1000,1/8,7,100)			#higher SN distribution

	plt.plot(np.log10(SNrange),SNdist_low)
	plt.plot(np.log10(SNrange),SNdist_high)
	plt.hist(np.log10(SN1),bins=(np.arange(np.log10(7),np.log10(100),0.05)),density=True,histtype='step')
	plt.hist(np.log10(SN2),bins=(np.arange(np.log10(7),np.log10(100),0.05)),density=True,histtype='step')
	# plt.xscale('log')
	plt.show()



def sample_Afr_dist(Nsamp=1000,asym_rate='high'):

	Afr_bins = np.arange(1.e0,2.2e0,0.05)

	if asym_rate == 'high':
		Afrs = [
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


	elif asym_rate == 'low':
		Afrs = [
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


	Afr_list =	reduce(np.append,(Afrs))
	Afr_prob = np.histogram(Afr_list,bins=Afr_bins,density=True)[0]
	Afr_cumsum = np.cumsum(Afr_prob) / np.nansum(Afr_prob)
	Afr_sample = np.interp(rng.uniform(size=Nsamp),Afr_cumsum,Afr_bins[0:-1]+0.025)
	Afr_sample = np.array([int(round(xx*20)/0.2)/100 for xx in Afr_sample])



	# plt.hist(Afr_sample,bins=Afr_bins,histtype = 'step',cumulative=True,density=True)
	# plt.plot(Afr_bins[0:-1]+0.025,Afr_cumsum)
	# plt.show()

	return Afr_sample











#### old stuff
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







if __name__ == '__main__':


	# test_dist_creation()
	# SN_control_demo()
	# SNsampled_Afrhist_demo()
	# DAGJK_sample_sigma_compare()
	test_recover_same_dist_1()
	test_recover_same_dist_2()
	test_recover_different_dist_1()
	test_recover_different_dist_2()

	# sample_SN_test()


	# test_recover_same_dist()
	# test_recover_different_dist()
	# test_opposite_recover_different_dist()

	

	# create_templates()
	# create_sample()
	# add_SN()



