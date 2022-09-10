#!/usr/bin/env python
# convert_notebook.py
#Programmer: Tim Tyree
#Date: 9.9.2022
import os

# convert empty IPython notebook to a sphinx doc page
def convert_nb(nbname):
	os.system("runipy --o %s.ipynb --matplotlib --quiet" % nbname)
	os.system("ipython nbconvert --to rst %s.ipynb" % nbname)
	os.system("tools/nbstripout %s.ipynb" % nbname)

# a CLI for convert_nb arguments are paths to .ipynb files
if __name__ == "__main__":
	for nbname in sys.argv[1:]:
		convert_nb(nbname)
