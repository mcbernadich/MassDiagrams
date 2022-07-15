import numpy as np
from glob import glob
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
#from numba import jit
import argparse
import sys

def normal_distribution(y,nominal_y,dy):
	return np.exp(-((y-nominal_y)**2)/(2*(dy**2)))

def find_percentil(axis,cum_sum,cum_value):

	#Find closest value.
	difference=abs(cum_sum-cum_value)
	percentil_value=axis[np.argmin(difference)]
	return percentil_value

def find_sigma_values(axis,prob):

	#Compute marginalised probbilities.
	cum_sum=np.cumsum(prob)/np.sum(prob)

	percentiles=[]

	three_sigma_left=find_percentil(axis,cum_sum,(1-0.9973)/2)
	percentiles.append(three_sigma_left)
	two_sigma_left=find_percentil(axis,cum_sum,(1-0.9545)/2)
	percentiles.append(two_sigma_left)
	one_sigma_left=find_percentil(axis,cum_sum,(1-0.6827)/2)
	percentiles.append(one_sigma_left)
	median=find_percentil(axis,cum_sum,0.5)
	percentiles.append(median)
	one_sigma_right=find_percentil(axis,cum_sum,(1-0.6827)/2+0.6827)
	percentiles.append(one_sigma_right)
	two_sigma_right=find_percentil(axis,cum_sum,(1-0.9545)/2+0.9545)
	percentiles.append(two_sigma_right)
	three_sigma_right=find_percentil(axis,cum_sum,(1-0.9973)/2+0.9973)
	percentiles.append(three_sigma_right)

	return percentiles

def periastron_advance_constraint(p_orb,ecc,x,omdot,domdot):

	p_orb=p_orb*24*3600
	cnt=(1.9891e30**(2/3))*(180/np.pi)*(31557600)*3*(6.67430e-11/299792458**3)**(2/3)*(p_orb/(2*np.pi))**(-5/3)/(1-ecc**2)
	nominal_omdot=omdot
	domdot=domdot
	one_sigma=normal_distribution(nominal_omdot+domdot,nominal_omdot,domdot)

	#Initialize the axes.
	mtot_ax=np.arange(m_start,m_end,m_step)
	cosi_ax=np.arange(cosi_start,cosi_end,cosi_step)
	(cosi,mtot)=np.meshgrid(cosi_ax,mtot_ax)
	m2=(299792458*x/(1-cosi**2)**(1/2))*((mtot**2*((4*np.pi**2)/6.67430e-11)/p_orb**2)/1.9891e30)**(1/3)
	m1=mtot-m2	

	#Evaluate omdot at each point of the m1-m2 grid and compute the distribution.
	omdot=cnt*((m1+m2)**(2/3))
	distribution=normal_distribution(omdot,nominal_omdot,domdot)

	return distribution,one_sigma

def constraint_from_h3(p_orb,x,h3,dh3):

	p_orb=p_orb*24*3600
	nominal_h3=h3
	dh3=dh3
	one_sigma=normal_distribution(nominal_h3+dh3,nominal_h3,dh3)

	#Initialize the axes.
	mtot_ax=np.arange(m_start,m_end,m_step)
	cosi_ax=np.arange(cosi_start,cosi_end,cosi_step)
	(cosi,mtot)=np.meshgrid(cosi_ax,mtot_ax)
	m2=(299792458*x/(1-cosi**2)**(1/2))*((mtot**2*((4*np.pi**2)/6.67430e-11)/p_orb**2)/1.9891e30)**(1/3)

	#Evaluate h3 at each point of the m2-cosi grid and compute the distribution.
	h3=(6.67430e-11*1.9891e30*m2/(299792458**3))*(((1-cosi)/(1+cosi))**(3/2))
	h3=h3.astype("float")
	distribution=normal_distribution(h3,nominal_h3,dh3)

	return distribution,one_sigma

def constraint_from_stig(p_orb,x,stig,dstig):

	p_orb=p_orb*24*3600
	nominal_stig=stig
	dstig=dstig
	one_sigma=normal_distribution(nominal_stig+dstig,nominal_stig,dstig)

	#Initialize the axes.
	mtot_ax=np.arange(m_start,m_end,m_step)
	cosi_ax=np.arange(cosi_start,cosi_end,cosi_step)
	(cosi,mtot)=np.meshgrid(cosi_ax,mtot_ax)

	stig=((1-cosi)/(1+cosi))**(1/2)
	distribution=normal_distribution(stig,nominal_stig,dstig)

	return distribution,one_sigma

def constraint_from_gamma(p_orb,ecc,x,gamma,dgamma):

	p_orb=p_orb*24*3600
	cnt=ecc*(1.9891e30*6.67430e-11/299792458**3)**(2/3)*(p_orb/(2*np.pi))**(1/3)
	nominal_gamma=gamma
	dgamma=dgamma
	one_sigma=normal_distribution(nominal_gamma+dgamma,nominal_gamma,dgamma)

	#Initialize the axes.
	mtot_ax=np.arange(m_start,m_end,m_step)
	cosi_ax=np.arange(cosi_start,cosi_end,cosi_step)
	(cosi,mtot)=np.meshgrid(cosi_ax,mtot_ax)
	m2=(299792458*x/(1-cosi**2)**(1/2))*((mtot**2*((4*np.pi**2)/6.67430e-11)/p_orb**2)/1.9891e30)**(1/3)
	m1=mtot-m2	

	#Evaluate gamma at each point of the m1-m2 grid and compute the distribution.
	gamma=cnt*m2*(m1+2*m2)/((m1+m2)**(4/3))
	distribution=normal_distribution(gamma,nominal_gamma,dgamma)

	return distribution,one_sigma

def constraint_from_PBdecay(p_orb,ecc,x,pbdot,dpbdot):

	p_orb=p_orb*24*3600
	cnt=-(1.9891e30**(5/3))*(192*np.pi/5)*(6.67430e-11/299792458**3)**(5/3)*(p_orb/(2*np.pi))**(-5/3)*(1+(73/24)*ecc**2+(37/96)*ecc**4)/(1-ecc**2)**(7/2)
	nominal_pbdot=pbdot
	dpbdot=dpbdot
	one_sigma=normal_distribution(nominal_pbdot+dpbdot,nominal_pbdot,dpbdot)

	#Initialize the axes.
	mtot_ax=np.arange(m_start,m_end,m_step)
	cosi_ax=np.arange(cosi_start,cosi_end,cosi_step)
	(cosi,mtot)=np.meshgrid(cosi_ax,mtot_ax)
	m2=(299792458*x/(1-cosi**2)**(1/2))*((mtot**2*((4*np.pi**2)/6.67430e-11)/p_orb**2)/1.9891e30)**(1/3)
	m1=mtot-m2	

	#Evaluate gamma at each point of the m1-m2 grid and compute the distribution.
	pbdot=cnt*m1*m2/(mtot**(1/3))
	distribution=normal_distribution(pbdot,nominal_pbdot,dpbdot)

	return distribution,one_sigma

def constraint_from_DDGR(p_orb,x,files):

	p_orb=p_orb*24*3600

	cosi_ax=np.loadtxt(files[0],usecols=0,delimiter=",")
	mtot_ax=[]
	chi2_dist=[]

	for file in files:

		mtot_ax.append(float(file.split(".txt")[0].split("/")[-1]))
		chi2_dist.append(np.loadtxt(file,usecols=1,delimiter=","))

	mtot_ax=np.array(mtot_ax)
	(cosi,mtot)=np.meshgrid(cosi_ax,mtot_ax)
	m2=(299792458*x/(1-cosi**2)**(1/2))*((mtot**2*((4*np.pi**2)/6.67430e-11)/p_orb**2)/1.9891e30)**(1/3)
	m1=mtot-m2	

	chi2_dist=np.array(chi2_dist)
	nominal_chi2=np.min(chi2_dist)
	sigma_levels=[]
	sigma_levels.append(normal_distribution(nominal_chi2+4,nominal_chi2,1))
	sigma_levels.append(normal_distribution(nominal_chi2+3,nominal_chi2,1))
	sigma_levels.append(normal_distribution(nominal_chi2+2,nominal_chi2,1))
	sigma_levels.append(normal_distribution(nominal_chi2+1,nominal_chi2,1))
	chi2_dist=normal_distribution(chi2_dist,nominal_chi2,1)

	return cosi,mtot,m1,m2,chi2_dist,sigma_levels

#@jit(nopython=True)
def find_sigma_contours(array):

	#Compute the total contents of the array, and the maximum value.
	total=np.sum(array)
	max_val=np.max(array)

	#Find the 1-sigma, 2-sigma and 3-sigma contour values.
	one_sigma=total*(1-0.6827)
	two_sigma=total*(1-0.9545)
	three_sigma=total*(1-0.9973)
	sigmas=[three_sigma,two_sigma,one_sigma]

	#Loop over accumulaed values.
	fractions=10**np.arange(-0.5,3.003,0.002)
	i=1
	accumulated_value=np.zeros(shape=len(fractions))
	contour_values=[]
	print("")
	print("Summing the power of the array:")
	for fraction in fractions:
		accumulated_value[i-1] = np.sum(array[ array < (max_val/1000)*fraction ])
		if i/50==int(i/50):
			print(i,(max_val/1000)*fraction)
		i=i+1
	accumulated_value=np.array(accumulated_value)

	#Find the contour values.
	for sigma in sigmas:
		argument_of_closest=np.argmin(abs(accumulated_value-sigma))
		contour_values.append((max_val/1000)*fractions[argument_of_closest])

	print(contour_values)
	return contour_values

def read_ephemeris(file):
	ephemeris=open(file,"r")
	ecc=0
	omega=0
	for line in ephemeris:
		line=line.replace(" ","")
		line=line.replace("	","")
		if line[:2]=="F0":
			f0=float(line[2:])
		elif line[:2]=="PB":
			p_orb=float(line[2:])
		elif line[:2]=="A1":
			x=float(line[2:])
		elif line[:3]=="ECC":
			ecc=float(line[3:])
		elif line[:2]=="OM":
			omega=float(line[2:])
		elif line[:2]=="T0":
			periastron=float(line[2:])
	if f0 and p_orb and x and periastron:
		print("Pulsar parameters loaded from {}:".format(file))
		print("- Pulsar frequency: {} Hz".format(f0))
		print("- Orbital period: {} days".format(p_orb))
		print("- Projected axis: {} ls".format(x))
		print("- Eccentricity: {}".format(ecc))
		print("- Periastron angle: {} degrees".format(omega))
		print("- Periastron passage: {} (MJD)".format(periastron))
		print("")
	else:
		sys.exit("The ephemeris file doesn't include all the necessary parameters.")
	return f0,p_orb,x,ecc,omega,periastron

parser=argparse.ArgumentParser(description="Take in orbital parameters and post-Keplerian effects and constrain the mass and inclination angle.")
parser.add_argument("--ephemeris",help="Fitorbit-format ephemeris. If given all other inputs will be ignored.")
parser.add_argument("-p","--period",type=float,help="Orbital period in days")
parser.add_argument("-x","--axis",type=float,help="Projected semimajor axis in ls.")
parser.add_argument("-e","--eccentricity",type=float,help="Excentricity. If not given, assumed 0. Needed for higher order estimations.")
parser.add_argument("--omdot",help="Periastron advance (value+/-uncertainty) in º/yr")
parser.add_argument("--gamma",help="Einstein delay (value+/-uncertainty) in s.")
parser.add_argument("--pbdot",help="Spin-down (value+/-uncertainty) in s/s.")
parser.add_argument("-r","--range",help="Range of Shapiro delay in s. It supersedes h3.")
parser.add_argument("-s","--shape",help="Shape of Shapiro delay. It supersedes stig.")
parser.add_argument("--h3",help="Orthometric amplitude amplitude of saphiro delay (value+/-uncertainty) in s.")
parser.add_argument("--stig",help="Orthometric amplitude amplitude of saphiro delay (value+/-uncertainty).")
parser.add_argument("--m1",help="M1 range (computing and plotting, solar masses). Default: 0.005:2.58")
parser.add_argument("--m2",help="M2 range (computing and plotting, solar masses). Default: 0.005:2.58")
parser.add_argument("--cosi",help="cosi range (computing and plotting). Default: 0.001:1.0")
parser.add_argument("--ddgr",help="Path to DDGR chi2 map files. They should be named {mtot}.txt and have colums of (cosi,chi2). If this option is chose, individual PK parameters constraints can still be ploted, but no probabilities are computed from them.")
parser.add_argument("-v","--verbose",action="store_true")
args = parser.parse_args()

if args.ephemeris:
	(f0,p_orb,x,ecc,omega,periastron)=read_ephemeris(args.ephemeris)
else:
	if args.period:
		p_orb=args.period
		if args.verbose==True:
			print("Orbital period: {} days".format(p_orb))
			print(" ")
	else:
		sys.exit("Please specify orbital period in days with -p")
	if args.axis:
		x=args.axis
		if args.verbose==True:
			print("Projected axis: {} ls".format(x))
			print(" ")
	else:
		sys.exit("Please specify projected semimajor axis in days with -x")
	if args.eccentricity:
		ecc=args.eccentricity
		if args.verbose==True:
			print("Eccentricity: {} ls".format(ecc))
			print(" ")
	else:
		ecc=0
		if args.verbose==True:
			print("Eccentricity assumed to be 0")
			print(" ")
	if args.m1:
		m1_start=float(args.m1.split(":")[0])
		m1_end=float(args.m1.split(":")[1])
		m1_step=(m1_end-m1_start)/500
		if args.verbose==True:
			print("M1 range: {} solar masses".format(args.m1))
			print(" ")
	else:
		m1_start=0.005
		m1_end=2.58
		m1_step=0.005
		if args.verbose==True:
			print("M1 range: 0 to 2.6 solar masses")
			print(" ")
	if args.m2:
		m2_start=float(args.m2.split(":")[0])
		m2_end=float(args.m2.split(":")[1])
		m2_step=(m2_end-m2_start)/500
		if args.verbose==True:
			print("M2 range: {} solar masses".format(args.m2))
			print(" ")
	else:
		m2_start=0.005
		m2_end=2.58
		m2_step=0.005
		if args.verbose==True:
			print("M2 range: 0 to 2.6 solar masses")
			print(" ")
	if args.cosi:
		cosi_start=float(args.cosi.split(":")[0])
		cosi_end=float(args.cosi.split(":")[1])
		cosi_step=(cosi_end-cosi_start)/2000
		if args.verbose==True:
			print("cosi range: {} solar masses".format(args.cosi))
			print(" ")
	else:
		cosi_start=0.0005
		cosi_end=1.0
		cosi_step=0.0005
		if args.verbose==True:
			print("cosi range: 0 to 1 degrees")
			print(" ")


m_start=m1_start+m2_start
m_end=m1_end+m2_end
m_step=min(m1_step,m2_step)/10

significance=[]

if args.omdot:
	omdot=float(args.omdot.split("+/-")[0])
	domdot=float(args.omdot.split("+/-")[1])
	significance.append(omdot/domdot)
	if args.verbose==True:
		print("Constraining from omdot {} º/yr.".format(args.omdot))
	(omdot,omdot_one_sigma)=periastron_advance_constraint(p_orb,ecc,x,omdot,domdot)

if args.gamma:
	gamma=float(args.gamma.split("+/-")[0])
	dgamma=float(args.gamma.split("+/-")[1])
	significance.append(gamma/dgamma)
	if args.verbose==True:
		print("Constraining from gamma {} s.".format(args.gamma))
	(gamma,gamma_one_sigma)=constraint_from_gamma(p_orb,ecc,x,gamma,dgamma)

if args.pbdot:
	pbdot=float(args.pbdot.split("+/-")[0])
	dpbdot=float(args.pbdot.split("+/-")[1])
	significance.append(pbdot/dpbdot)
	if args.verbose==True:
		print("Constraining from pbdot {} s.".format(args.pbdot))
	(pbdot,pbdot_one_sigma)=constraint_from_PBdecay(p_orb,ecc,x,pbdot,dpbdot)

if args.h3:
	h3=float(args.h3.split("+/-")[0])
	dh3=float(args.h3.split("+/-")[1])
	significance.append(h3/dh3)
	if args.verbose==True:
		print("Constraining from Shapiro delay (h3).")
	(h3,h3_one_sigma)=constraint_from_h3(p_orb,x,h3,dh3)

if args.stig:
	stig=float(args.stig.split("+/-")[0])
	dstig=float(args.stig.split("+/-")[1])
	significance.append(stig/dstig)
	if args.verbose==True:
		print("Constraining from Shapiro delay (stig).")
	(stig,stig_one_sigma)=constraint_from_stig(p_orb,x,stig,dstig)

if args.ddgr:
	files=sorted(glob(args.ddgr))
	if args.verbose==True:
		print("Connstarining the chi2 space from the DDGR grid.")
	(DDGR_cosi,DDGR_mtot,DDGR_m1,DDGR_m2,DDGR_dist,DDGR_sigmas)=constraint_from_DDGR(p_orb,x,files)

if args.omdot or args.gamma or args.h3 or args.stig:
	significance=np.array(significance)
	priority=np.zeros(len(significance),dtype=int)
	ordered_significance=np.sort(significance)
	i=0
	for element in ordered_significance:
		index=np.where(significance==element)
		priority[index]=i+1
		i=i+1
	colours=["c","c","c","b"]
else:
	priority=[0]
	colours=["magenta","magenta","magenta","purple"]

# Start plotting constraints like a madman:

AX = gridspec.GridSpec(3,5)
AX.update(wspace = 0.1, hspace = 0.1)

ax_m2m1=plt.subplot(AX[1:3,2:4])

#Initialize all axis.
mtot_ax=np.arange(m_start,m_end,m_step)
cosi_ax=np.arange(cosi_start,cosi_end,cosi_step)
(cosi,mtot)=np.meshgrid(cosi_ax,mtot_ax)
sini=(1-cosi**2)**(1/2)
m2=(299792458*x/(1-cosi**2)**(1/2))*((mtot**2*((4*np.pi**2)/6.67430e-11)/(p_orb*24*3600)**2)/1.9891e30)**(1/3)
m1=mtot-m2

#Other DNS systems.
dns_list=np.loadtxt("../J1208-5936/paper_plots/DNS_list.txt",dtype=str).T
nature=dns_list[18]
m_t=dns_list[19]
m_comp=dns_list[20]
m_puls=dns_list[21]

#i=0
#for element in m_comp:

#	if (element[0]!=">") and ("NS[" in nature[i]):
#		plt.scatter([float(m_puls[i].split("(")[0])],[float(element.split("(")[0])],color="r",marker="*",s=100,zorder=100)

#	if (element[0]!=">") and ("(?)" in nature[i]):
#		plt.scatter([float(m_puls[i].split("(")[0])],[float(element.split("(")[0])],color="g",marker="*",s=100,zorder=100)

#	i=i+1

total=(mtot/mtot)

total[ (m1 <= m1_start) & (m1 > m1_end) & (m2 <= m2_start) & (m2 > m2_end) ] = 0

i=0

if args.omdot:

	plt.contour(m1,m2,omdot,[omdot_one_sigma],colors=['y'],zorder=priority[i])
	total=total*omdot
	i=i+1

if args.gamma:

	plt.contour(m1,m2,gamma,[gamma_one_sigma],colors=['r'],zorder=priority[i])
	total=total*gamma
	i=i+1

if args.pbdot:

	plt.contour(m1,m2,pbdot,[pbdot_one_sigma],colors=['c'],zorder=priority[i])
	total=total*pbdot
	i=i+1

if args.h3:

	plt.contour(m1,m2,h3,[h3_one_sigma],colors=['b'],zorder=priority[i])
	total=total*h3
	i=i+1

if args.stig:

	plt.contour(m1,m2,stig,[stig_one_sigma],colors=['g'],zorder=priority[i])
	total=total*stig
	i=i+1

if args.ddgr:

	#plt.contour(DDGR_m1,DDGR_m2,DDGR_dist,DDGR_sigmas,colors=['purple'],linewidths=[0.8],zorder=0)
	i=i+1

# Plot the total mass distribution
plt.xlabel("Pulsar mass, $M_1$ (M$_\odot$)")
plt.yticks([])
#plt.xlim((0.0,m1_end))
#plt.ylim((0.0,m2_end))
plt.xlim((m1_start,m1_end))
plt.ylim((m2_start,m2_end))

# Plot the colormap of the probability density
if args.ddgr:
	plt.pcolormesh(DDGR_m1, DDGR_m2, DDGR_dist, cmap="Purples",zorder=0)
else:
#	total_omdot=total*omdot
	plt.pcolormesh(m1[::10,::10], m2[::10,::10], total[::10,::10], cmap="Blues",zorder=0)
#plt.contourf(m1, m2, total, np.arange(0.1,1.1,0.1)*np.max(total),colors=["c"],zorder=0)

# Plot the mass function limit and sini lines.
plt.contour(m1,m2,sini,[np.sin(10*np.pi/180),np.sin(20*np.pi/180),np.sin(30*np.pi/180),np.sin(40*np.pi/180),np.sin(50*np.pi/180),np.sin(60*np.pi/180),np.sin(70*np.pi/180),np.sin(80*np.pi/180)],colors=['darkgray'],linestyles=["dashed"],linewidths=[0.8],zorder=0)
m2_ax=299792458*x*((mtot_ax**2*((4*np.pi**2)/6.67430e-11)/(p_orb*24*3600)**2)/1.9891e30)**(1/3)
m1_ax=mtot_ax-m2_ax
plt.fill_between(m1_ax,m2_ax,color="darkgray",zorder=np.max(priority)+1)

#Plot corner distribution for M2.
ax_m2=plt.subplot(AX[1:3,4])
m2_ax=np.arange(m2_start,m2_end,m2_step)
m2_ax_sum=[]
i=0
print("")
print("Computing corner plot for M2:")
for value in m2_ax[:-1]:
	if args.ddgr:
		m2_ax_sum.append(np.sum(DDGR_dist[ (DDGR_m2 >= value) & (DDGR_m2 < value+m2_step) ]))	
	else:
		m2_ax_sum.append(np.sum(total[ (m2 >= value) & (m2 < value+m2_step) ]))
	if i/500==int(i/500):
		print(value)
	i=i+1
m2_ax_sum=m2_ax_sum/np.sum(m2_ax_sum)
m2_dist_ax=m2_ax[:-1]+m2_step/2
percentile_borders=find_sigma_values(m2_dist_ax,m2_ax_sum)
m2_value=percentile_borders[3]
m2_right=percentile_borders[4]
m2_left=percentile_borders[2]
print("")
print("M2= {} +{} -{} solar masses".format(round(percentile_borders[3],3),round(percentile_borders[4]-percentile_borders[3],3),round(percentile_borders[3]-percentile_borders[2],3)))
plt.fill_betweenx(m2_dist_ax,m2_ax_sum,color=colours[0],alpha=0.2,zorder=0)
plt.fill_betweenx(m2_dist_ax[ (m2_dist_ax>percentile_borders[0]) & (m2_dist_ax<percentile_borders[6]) ],m2_ax_sum[ (m2_dist_ax>percentile_borders[0]) & (m2_dist_ax<percentile_borders[6]) ],color=colours[1],alpha=0.5,zorder=0)
plt.fill_betweenx(m2_dist_ax[ (m2_dist_ax>percentile_borders[1]) & (m2_dist_ax<percentile_borders[5]) ],m2_ax_sum[ (m2_dist_ax>percentile_borders[1]) & (m2_dist_ax<percentile_borders[5]) ],color=colours[2],zorder=0)
plt.fill_betweenx(m2_dist_ax[ (m2_dist_ax>percentile_borders[2]) & (m2_dist_ax<percentile_borders[4]) ],m2_ax_sum[ (m2_dist_ax>percentile_borders[2]) & (m2_dist_ax<percentile_borders[4]) ],color=colours[3],zorder=0)
plt.hlines(percentile_borders[3],0,m2_ax_sum[m2_dist_ax==percentile_borders[3]],colors="k",zorder=1)
plt.xlim((0.0,1.1*np.max(m2_ax_sum)))
#plt.ylim((0.0,m2_end))
plt.ylim((m2_start,m2_end))
ax_m2.yaxis.tick_right()
plt.xticks([])
plt.yticks([])
plt.xlabel("Probability density")
#plt.ylabel("$M_2= "+"^{+"++"}_{-"++"}$ (M$_\odot$)".format(round(percentile_borders[3],2),round(percentile_borders[4]-percentile_borders[3],2),round(percentile_borders[3]-percentile_borders[2],2)))

#Plot corner distribution for M1.
ax_m1=plt.subplot(AX[0,2:4])
m1_ax=np.arange(m1_start,m1_end,m1_step)
m1_ax_sum=[]
i=0
print("")
print("Computing corner plot for M1:")
for value in m1_ax[:-1]:
	if args.ddgr:
		m1_ax_sum.append(np.sum(DDGR_dist[ (DDGR_m1 >= value) & (DDGR_m1 < value+m1_step) ]))
	else:
		m1_ax_sum.append(np.sum(total[ (m1 >= value) & (m1 < value+m1_step) ]))
	if i/500==int(i/500):
		print(value)
	i=i+1
m1_ax_sum=m1_ax_sum/np.sum(m1_ax_sum)
m1_dist_ax=m1_ax[:-1]+m1_step/2
percentile_borders=find_sigma_values(m1_dist_ax,m1_ax_sum)
m1_value=percentile_borders[3]
m1_right=percentile_borders[4]
m1_left=percentile_borders[2]
print("")
print("M1= {} +{} -{} solar masses".format(round(percentile_borders[3],3),round(percentile_borders[4]-percentile_borders[3],3),round(percentile_borders[3]-percentile_borders[2],3)))
plt.fill_between(m1_dist_ax,m1_ax_sum,color=colours[0],alpha=0.2)
plt.fill_between(m1_dist_ax[ (m1_dist_ax>percentile_borders[0]) & (m1_dist_ax<percentile_borders[6]) ],m1_ax_sum[ (m1_dist_ax>percentile_borders[0]) & (m1_dist_ax<percentile_borders[6]) ],color=colours[1],alpha=0.5,zorder=0)
plt.fill_between(m1_dist_ax[ (m1_dist_ax>percentile_borders[1]) & (m1_dist_ax<percentile_borders[5]) ],m1_ax_sum[ (m1_dist_ax>percentile_borders[1]) & (m1_dist_ax<percentile_borders[5]) ],color=colours[2],zorder=0)
plt.fill_between(m1_dist_ax[ (m1_dist_ax>percentile_borders[2]) & (m1_dist_ax<percentile_borders[4]) ],m1_ax_sum[ (m1_dist_ax>percentile_borders[2]) & (m1_dist_ax<percentile_borders[4]) ],color=colours[3],zorder=0)
plt.vlines(percentile_borders[3],0,m1_ax_sum[m1_dist_ax==percentile_borders[3]],colors="k",zorder=1)
plt.ylim((0.0,1.1*np.max(m1_ax_sum)))
#plt.xlim((0.0,m1_end))
plt.xlim((m1_start,m1_end))
plt.xticks([])
plt.yticks([])
#plt.title("$M_1= {}^+^{}_-_{}$ (M$_\odot$)".format(round(percentile_borders[3],2),round(percentile_borders[4]-percentile_borders[3],2),round(percentile_borders[3]-percentile_borders[2],2)))

# Start plotting constraints like a madman:

ax_m2cosi=plt.subplot(AX[1:3,0:2])

#i=0
#for element in m_comp:

#	if (element[0]!=">") and ("NS[" in nature[i]):
#		sini_p=299792458*x*((1/(3600*24*p_orb)**2)*(4*np.pi**2/6.67430e-11)*((float(m_t[i].split("(")[0])**2)/(1.9891e30*float(element.split("(")[0])**3)))**(1/3)
#		plt.scatter([np.sqrt(1-sini_p**2)],[float(element.split("(")[0])],color="r",marker="*",s=100,zorder=100)

#	if (element[0]!=">") and ("(?)" in nature[i]):
#		sini_p=299792458*x*((1/(3600*24*p_orb)**2)*(4*np.pi**2/6.67430e-11)*((float(m_t[i].split("(")[0])**2)/(1.9891e30*float(element.split("(")[0])**3)))**(1/3)
#		plt.scatter([np.sqrt(1-sini_p**2)],[float(element.split("(")[0])],color="g",marker="*",s=100,zorder=100)

#	i=i+1

i=0

if args.omdot:

	plt.contour(cosi,m2,omdot,[omdot_one_sigma],colors=['y'],zorder=priority[i])
	i=i+1

if args.gamma:

	plt.contour(cosi,m2,gamma,[gamma_one_sigma],colors=['r'],zorder=priority[i])
	i=i+1

if args.pbdot:

	plt.contour(cosi,m2,pbdot,[pbdot_one_sigma],colors=['c'],zorder=priority[i])
	i=i+1

if args.h3:

	plt.contour(cosi,m2,h3,[h3_one_sigma],colors=['b'],zorder=priority[i])
	i=i+1

if args.stig:

	plt.contour(cosi,m2,stig,[stig_one_sigma],colors=['g'],zorder=priority[i])
	i=i+1

if args.ddgr:

#	plt.contour(DDGR_cosi,DDGR_m2,DDGR_dist,DDGR_sigmas,colors=['purple'],linewidths=[0.8],zorder=0)
	i=i+1


m2_ax=mtot_ax-m1_end
mtot_ax_for_limit=mtot_ax[ (m2_ax > m2_start) & (m2_ax < m2_end) ]
m2_ax=m2_ax[ (m2_ax > m2_start) & (m2_ax < m2_end) ]
sini_ax=(299792458*x/m2_ax)*(mtot_ax_for_limit**2*(((4*np.pi**2)/6.67430e-11)/(p_orb*24*3600)**2)/1.9891e30)**(1/3)
m2_ax=m2_ax[ sini_ax < 1.0 ]
sini_ax=sini_ax[ sini_ax < 1.0 ]
sini_ax=np.insert(sini_ax,0,1)
m2_ax=np.insert(m2_ax,0,m2_ax[0])
cosi_ax_for_limit=(1-sini_ax**2)**(1/2)
plt.fill_between(cosi_ax_for_limit,m2_ax,m2_end,color="darkgray",zorder=np.max(priority)+1)

m2_ax=(299792458*x/(1-cosi_ax**2)**(1/2))**3*(((4*np.pi**2)/6.67430e-11)/(p_orb*24*3600)**2)/1.9891e30
plt.fill_between(cosi_ax,m2_ax,color="darkgray",zorder=np.max(priority)+1)

plt.xlabel("Cosine of inclination angle, cos(i)")
plt.ylabel("Companion mass, $M_2$ (M$_\odot$)")
#plt.xlim((0,0.99))
#plt.ylim((0.0,m2_end))
plt.xlim((cosi_start,cosi_end))
plt.ylim((m2_start,m2_end))
plt.xticks(zorder=np.max(priority)+2)
plt.yticks(zorder=np.max(priority)+2)

# Plot the colormap of the probability density
if args.ddgr:
	plt.pcolormesh(DDGR_cosi, DDGR_m2, DDGR_dist, cmap="Purples",zorder=0)
else:
	plt.pcolormesh(cosi[::10,::10], m2[::10,::10], total[::10,::10], cmap="Blues",zorder=0)
#plt.contourf(cosi, m2, total, np.arange(0.1,1.1,0.1)*np.max(total),colors=["c"],zorder=0)

# Plot the mass function limit, max mass limit, and m1 lines.
plt.contour(cosi,m2,m1,np.arange(0.5,m1_end,0.5),colors=["darkgray"],linestyles=["dashed"],linewidths=[0.8],zorder=0)

#Plot corner distribution for cosi.
ax_cosi=plt.subplot(AX[0,0:2])
if args.ddgr:
	total_cosi=np.sum(DDGR_dist,axis=0)
	cosi_ax=DDGR_cosi[0]
else:
	total_cosi=np.sum(total,axis=0)
total_cosi=total_cosi/np.sum(total_cosi)
percentile_borders=find_sigma_values(cosi_ax,total_cosi)
center=np.arccos(percentile_borders[3])*180/np.pi
right=np.arccos(percentile_borders[2])*180/np.pi-np.arccos(percentile_borders[3])*180/np.pi
left=np.arccos(percentile_borders[3])*180/np.pi-np.arccos(percentile_borders[4])*180/np.pi
cosi_value=percentile_borders[3]
cosi_right=percentile_borders[4]
cosi_left=percentile_borders[2]
print("")
print("i= {} +{} -{} degrees".format(round(center,3),round(right,3),round(left,3)))
plt.fill_between(cosi_ax,total_cosi,color=colours[0],alpha=0.2,zorder=0)
plt.fill_between(cosi_ax[ (cosi_ax>percentile_borders[0]) & (cosi_ax<percentile_borders[6]) ],total_cosi[ (cosi_ax>percentile_borders[0]) & (cosi_ax<percentile_borders[6]) ],color=colours[1],alpha=0.5,zorder=0)
plt.fill_between(cosi_ax[ (cosi_ax>percentile_borders[1]) & (cosi_ax<percentile_borders[5]) ],total_cosi[ (cosi_ax>percentile_borders[1]) & (cosi_ax<percentile_borders[5]) ],color=colours[2],zorder=0)
plt.fill_between(cosi_ax[ (cosi_ax>percentile_borders[2]) & (cosi_ax<percentile_borders[4]) ],total_cosi[ (cosi_ax>percentile_borders[2]) & (cosi_ax<percentile_borders[4]) ],color=colours[3],zorder=0)
plt.vlines(percentile_borders[3],0,total_cosi[cosi_ax==percentile_borders[3]],colors="k",zorder=1)
plt.xticks([])
#plt.xlim((0,0.99))
plt.xlim((cosi_start,cosi_end))
plt.ylim((0.0,1.1*np.max(total_cosi)))
plt.ylabel("Probability density")
plt.yticks([])
#plt.title("$i= {}^+^{}_-_{}$ (degrees)".format(round(center,2),round(right,2),round(left,2)))

#Get total system mass intervals:
if args.ddgr:
	total_mtot=np.sum(DDGR_mtot,axis=1)
	mtot_ax=DDGR_mtot[:,0]
else:
	total_mtot=np.sum(total,axis=1)
percentile_borders=find_sigma_values(mtot_ax,total_mtot)
center=percentile_borders[3]
right=percentile_borders[4]-percentile_borders[3]
left=percentile_borders[3]-percentile_borders[2]
print("")
print("MTOT= {} +{} -{} solar masses".format(center,right,left))



cnt=(1.9891e30**(2/3))*(180/np.pi)*(31557600)*3*(6.67430e-11/299792458**3)**(2/3)*(p_orb*24*3600/(2*np.pi))**(-5/3)/(1-ecc**2)

omdot_ax=cnt*(mtot_ax**(2/3))

percentile_borders=find_sigma_values(omdot_ax,total_mtot)
center=percentile_borders[3]
right=percentile_borders[4]-percentile_borders[3]
left=percentile_borders[3]-percentile_borders[2]

print("")
print("OMDOT= {} +{} -{} deg/yr".format(center,right,left))

cnt=ecc*(1.9891e30*6.67430e-11/299792458**3)**(2/3)*(p_orb*24*3600/(2*np.pi))**(1/3)

gamma=cnt*m2_value*(m1_value+2*m2_value)/((m1_value+m2_value)**(4/3))
dgamma=abs(cnt*(m2_right*(m1_left+2*m2_right)/((m1_left+m2_right)**(4/3))-m2_left*(m1_right+2*m2_left)/((m1_right+m2_left)**(4/3)))/2)

gamma_ax=np.arange(gamma-5*dgamma,gamma+5*dgamma,dgamma/50)

gamma=cnt*DDGR_m2*(DDGR_m2+DDGR_mtot)/((DDGR_mtot)**(4/3))
gamma_ax_sum=[]

for value in gamma_ax[:-1]:
	gamma_ax_sum.append(np.sum(DDGR_dist[ (gamma >= value) & (gamma < value+dgamma/50) ]))

gamma_dist_ax=gamma_ax[:-1]+dgamma/100
percentile_borders=find_sigma_values(gamma_dist_ax,gamma_ax_sum)
center=percentile_borders[3]
right=percentile_borders[4]-percentile_borders[3]
left=percentile_borders[3]-percentile_borders[2]

print("")
print("GAMMA= {} +{} -{} s".format(center,right,left))

h3=(6.67430e-11*1.9891e30*m2_value/(299792458**3))*(((1-cosi_value)/(1+cosi_value))**(3/2))
dh3=abs((6.67430e-11*1.9891e30*m2_right/(299792458**3))*(((1-cosi_right)/(1+cosi_right))**(3/2))-(6.67430e-11*1.9891e30*m2_left/(299792458**3))*(((1-cosi_left)/(1+cosi_left))**(3/2)))/2

h3_ax=np.arange(h3-5*dh3,h3+5*dh3,dh3/50)
h3=(6.67430e-11*1.9891e30*DDGR_m2/(299792458**3))*(((1-DDGR_cosi)/(1+DDGR_cosi))**(3/2))
h3_ax_sum=[]

for value in h3_ax[:-1]:
	h3_ax_sum.append(np.sum(DDGR_dist[ (h3 >= value) & (h3 < value+dh3/50) ]))

h3_dist_ax=h3_ax[:-1]+dh3/100
percentile_borders=find_sigma_values(h3_dist_ax,h3_ax_sum)
center=percentile_borders[3]
right=percentile_borders[4]-percentile_borders[3]
left=percentile_borders[3]-percentile_borders[2]

print("")
print("H3= {} +{} -{} s".format(center,right,left))

stig=((1-cosi_value)/(1+cosi_value))**(1/2)
dstig=abs(((1-cosi_right)/(1+cosi_right))**(1/2)-((1-cosi_left)/(1+cosi_left))**(1/2))/2

stig_ax=np.arange(stig-5*dstig,stig+5*dstig,dstig/50)
stig=((1-DDGR_cosi)/(1+DDGR_cosi))**(1/2)
stig_ax_sum=[]

for value in stig_ax[:-1]:
	stig_ax_sum.append(np.sum(DDGR_dist[ (stig >= value) & (stig < value+dstig/50) ]))

stig_dist_ax=stig_ax[:-1]+dstig/100
percentile_borders=find_sigma_values(stig_dist_ax,stig_ax_sum)
center=percentile_borders[3]
right=percentile_borders[4]-percentile_borders[3]
left=percentile_borders[3]-percentile_borders[2]

print("")
print("STIG= {} +{} -{}".format(center,right,left))

cnt=-(1.9891e30**(5/3))*(192*np.pi/5)*(6.67430e-11/299792458**3)**(5/3)*(p_orb*24*3600/(2*np.pi))**(-5/3)*(1+(73/24)*ecc**2+(37/96)*ecc**4)/(1-ecc**2)**(7/2)

pbdot=cnt*m1_value*m2_value/(m1_value+m2_value)**(1/3)
dpbdot=abs(cnt*(m1_left*m2_right/(m1_left+m2_right)**(1/3)-m1_right*m2_left/(m1_right+m2_left)**(1/3))/2)
xdot=(2/3)*x*pbdot/p_orb
dxdot=(2/3)*x*dpbdot/p_orb
xdot_ax=np.arange(xdot-5*dxdot,xdot+5*dxdot,dxdot/500)

pbdot_ax=np.arange(pbdot-5*dpbdot,pbdot+5*dpbdot,dpbdot/500)

pbdot=cnt*DDGR_m1*DDGR_m2/DDGR_mtot**(1/3)
pbdot_ax_sum=[]

for value in pbdot_ax[:-1]:
pbdot_ax_sum.append(np.sum(DDGR_dist[ (pbdot >= value) & (pbdot < value+dpbdot/500) ]))

pbdot_dist_ax=pbdot_ax[:-1]+dpbdot/1000
percentile_borders=find_sigma_values(pbdot_dist_ax,pbdot_ax_sum)
center=percentile_borders[3]
right=percentile_borders[4]-percentile_borders[3]
left=percentile_borders[3]-percentile_borders[2]

print("")
print("PBDOT= {} +{} -{} s/s".format(center,right,left))

xdot=(2/3)*x*pbdot/p_orb

xdot_ax_sum=[]

for value in xdot_ax[:-1]:
	xdot_ax_sum.append(np.sum(DDGR_dist[ (xdot >= value) & (xdot < value+dxdot/500) ]))

xdot_dist_ax=xdot_ax[:-1]+dxdot/1000
percentile_borders=find_sigma_values(xdot_dist_ax,xdot_ax_sum)
center=percentile_borders[3]
right=percentile_borders[4]-percentile_borders[3]
left=percentile_borders[3]-percentile_borders[2]

print("")
print("XDOT= {} +{} -{} ls/s".format(center,right,left))

ax_legend=plt.subplot(AX[0,4])

if args.omdot:

	plt.plot([],[],"yo",label="$\dot\omega$")

if args.gamma:

	plt.plot([],[],"ro",label="$\gamma$")

if args.gamma:

	plt.plot([],[],"co",label="$\dot P_B$")

if args.h3:

	plt.plot([],[],"bo",label="$h_3$")	

if args.stig:

	plt.plot([],[],"go",label="$\\varsigma$")

#plt.scatter([],[],color="r",marker="*",s=100,label="Known DNS",zorder=100)
#plt.scatter([],[],color="g",marker="*",s=100,label="Possible DNS",zorder=100)

ax_legend.spines['top'].set_visible(False)
ax_legend.spines['bottom'].set_visible(False)
ax_legend.spines['left'].set_visible(False)
ax_legend.spines['right'].set_visible(False)
plt.legend()
plt.xticks([])
plt.yticks([])

plt.tight_layout()
plt.show()

#plt.pcolormesh(cosi[::10,::10], mtot[::10,::10], total[::10,::10],cmap="Blues")
#plt.xlabel("Cosine of inclination angle, cos(i)")
#plt.ylabel("Total system mass, $M_T$ (M$_\odot$)")
#plt.show()

#plt.pcolormesh(DDGR_cosi,DDGR_mtot,DDGR_dist,cmap="Purples")
#plt.xlabel("Cosine of inclination angle, cos(i)")
#plt.ylabel("Total system mass, $M_T$ (M$_\odot$)")
#plt.show()

#plt.plot(gamma_dist_ax,gamma_ax_sum,"c-")
#plt.xlabel("Einstein delay (s/s)")
#plt.ylabel("Probability density")
#plt.show()

#plt.plot(pbdot_dist_ax,pbdot_ax_sum,"c-")
#plt.xlabel("Derivative of orbital period (s/s)")
#plt.ylabel("Probability density")
#plt.show()

#plt.plot(xdot_dist_ax,xdot_ax_sum,"c-")
#plt.xlabel("Derivative of semimajor axis (ls/s)")
#plt.ylabel("Probability density")
#plt.show()

#plt.plot(h3_dist_ax,h3_ax_sum,"c-")
#plt.xlabel("Amplitude of Shapiro Delay (s/s)")
#plt.ylabel("Probability density")
#plt.show()

#plt.plot(stig_dist_ax,stig_ax_sum,"c-")
#plt.xlabel("Shape of Shapiro delay (s/s)")
#plt.ylabel("Probability density")
#plt.show()