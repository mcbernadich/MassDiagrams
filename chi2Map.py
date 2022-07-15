import numpy as np
import subprocess as sub
import multiprocessing as multi
from functools import partial
from os.path import exists
import glob
import argparse

def read_parameters(parFile):

	par_read=open(parFile,"r")

	for line in par_read:
		chunks = line.strip().split()
		if chunks[0]=="MTOT":
			mtot=float(chunks[1])
			dmtot=float(chunks[3])
		if chunks[0]=="PB":
			pb=float(chunks[1])
		if chunks[0]=="A1":
			x=float(chunks[1])

	return mtot,dmtot,pb,x

def read_chi2r(tempo2logs):
	chi2r=99999999999
	npoints=2
	nfit=1
	for line in tempo2logs.split("\n"):
		if line[0:11]=="Fit Chisq =":
			chi2r=float(line.split("=")[3].split("	")[0])
		if line[0:25]=="Number of fit parameters:":
			nfit=int(line.split(":")[1])
		if line[0:25]=="Number of points in fit =":
			npoints=int(line.split("=")[1])
	return chi2r,npoints-nfit

def change_inclination(parFile,MTOT,M2):

	par_read=open(parFile,"r")
	parFile_new=parFile.split(".")[0]+"_"+str(MTOT)+"_"+str(M2)+".par"
	par_write=open(parFile_new,"w")
	for line in par_read:
		chunks = line.strip().split()
		if chunks[0]=="M2":
			par_write.write("M2 "+str(M2)+"\n")
		if chunks[0]=="MTOT":
			par_write.write("MTOT "+str(MTOT)+"\n")
		elif (chunks[0]!="M2" and chunks[0]!="MTOT"):
			par_write.write(line)

	return parFile_new

def fit_run(mtot,timFile,parFile,pb,x,cosi_ax):

	chi2_file=open("chi2_files/"+str(mtot)+".txt","w")

	for cosi in cosi_ax:

		m2=(299792458*x/(1-cosi**2)**(1/2))*((mtot**2*((4*np.pi**2)/6.67430e-11)/(pb*24*3600)**2)/1.9891e30)**(1/3)
		m1=mtot-m2

		if m2>=mtot or m1>=mtot or m2<0 or m1<0:

			chi2_file.write("{},{}\n".format(cosi,99999999999))

		else:

			parFile_new=change_inclination(parFile,mtot,m2)
			fit=sub.run(["tempo2","-f",parFile_new,timFile,"-outpar",parFile_new,"-nits","4"],stdout=sub.PIPE)
			(chi2r,nfree)=read_chi2r(fit.stdout.decode())
			sub.run(["rm",parFile_new],stdout=sub.DEVNULL)

			chi2_file.write("{},{}\n".format(cosi,chi2r*nfree))

	chi2_file.close()

	return 1

parser=argparse.ArgumentParser(description="Take in a DDGR file and make e a chi2r map on uniform cosi and around 5 sima of MTOT.")
parser.add_argument("-p","--parameter",help="Tempo2 parameter file.")
parser.add_argument("-t","--tim",help="Tempo2 tim file.")
parser.add_argument("--mtot",help="total mass (mass+/-uncertainty, solar masses). Search will go through 5 uniformly-spaced sigmas. If not given, read fro the parameter file.")
parser.add_argument("--cosi",help="cosi range. Default: 0.001:1.0")
parser.add_argument("--nthread",type=int,help="Number of inclinations to explore at the same time.",default=8)
args = parser.parse_args()

if args.parameter:
	parFile=args.parameter
if args.tim:
	timFile=args.tim

if args.cosi:
	cosi_start=float(args.cosi.split(":")[0])
	cosi_end=float(args.cosi.split(":")[1])
	cosi_step=(cosi_end-cosi_start)/2000
else:
	cosi_start=0.0005
	cosi_end=1.0
	cosi_step=0.0005
cosi_ax=np.arange(cosi_start,cosi_end,cosi_step)

(mtot,dmtot,pb,x)=read_parameters(parFile)
if args.mtot:
	mtot=float(args.mtot.split(":")[0])
	dmtot=float(args.mtot.split(":")[1])
mtot_ax=np.arange(mtot-5*dmtot,mtot+5*dmtot,dmtot/50)

sub.run(["mkdir","chi2_files"],stdout=sub.PIPE)

nMTOTs=len(mtot_ax)
j=0

while j < nMTOTs:

	multiprocesses=multi.Pool(processes=args.nthreads)
	dummy_array=multiprocesses.map(partial(fit_run,timFile=timFile,parFile=parFile,pb=pb,x=x,cosi_ax=cosi_ax),mtot_ax[j:j+args.nthreads])
	j=j+args.nthreads

	multiprocesses.close()
	multiprocesses.join()

j=j+1