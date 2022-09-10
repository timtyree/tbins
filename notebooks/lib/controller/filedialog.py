# filedialog.py
#Programmer: Tim Tyree
#Date: 9.9.2022
import os
from tkinter import Tk,filedialog

def search_for_file (currdir = None):
	'''use a widget dialog to selecting a file.
	Increasing the default fontsize seems too involved for right now.

	Example Usage:
dirname = search_for_file (currdir = os.getcwd())
	'''
	if currdir is None:
		currdir = os.getcwd()
	root = Tk()
	# root.config(font=("Courier", 44))
	tempdir = filedialog.askopenfilename(parent=root,
										 initialdir=currdir,
										 title="Please select a file")#,
										 # filetypes = (("all files","*.*")))
	root.destroy()
	if len(tempdir) > 0:
		print ("File: %s" % tempdir)
	return tempdir
