# DiagramsForLife

Hello! This software has been writen to get realiable mass measurments from pulsar timing of binary systems. In general, the system needs a good timing solution and at least 2 detected Post-Keplerian (PK) parameters to get nice 2D m2-cosi, m2-m1 and 1D cosi, m2 and m1 distributions. So far, it is only compatible with DDH and DDGR.

## chi2Map.py

This code is designed to run a DDGR chi2 map in the MTOT - COSI space. Results will be stored in ```chi2_files```. Write ```>python3 chi2Map.py -h``` for more info!

## massDiagram.py

This code makes the actual plots and computes the 2D and marginal 1D probablity distributions. It can read either inputed Keplerian and PK parameters given by the user, or Keplerian parameters plus a ```chi2_files``` folder of results from ```chi2Map.py```. It may take a long to read, and the produced images can be very heavy, so storing them as png instead of pdf is recommended. Write ```>python3 massDiagram.py -h``` for more info!

## Requirements

python3, numpy, matplotlib. A tempo2 installation if you use ```chi2Map.py```.

## Questions, inquiries and feedback

Just write to mcbernadich@mpifr-bonn.mpg.de if you have any.