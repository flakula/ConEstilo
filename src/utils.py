from pprint import pprint
import json
import math
import matplotlib as mpl
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import spacy
import sys
import threading
import time
import streamlit as st

def read_file(folder, year, file):
	with open(os.path.join(folder, year, file), encoding="utf-8") as f:
		t = f.read()
	return t
def read_json(file):
	with open(file, encoding="utf-8") as f:
		j = json.load(f)
	return j

# @st.cache
def get_df():
	try: df = pd.read_csv("data/df.csv")
	except Exception as e: df = pd.DataFrame()
	try: del df['Unnamed: 0']
	except Exception as e: pass
	try: df = df.set_index('title')
	except Exception as e: df = df.set_index('year')
	return df

# @st.cache
def get_json():
	try:
		with open("data/d.json", encoding="utf-8") as f:
			j = json.load(f)
		pass
	except Exception as e:
		# raise e
		j = {}
	return j

def iterate(corpus_folder, verbose=False):
	file_no = 0
	for year in os.listdir(corpus_folder):
		for file in os.listdir(corpus_folder+year):
			if verbose:
				print(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()), file_no, corpus_folder+year+'/'+file)
			file_no += 1
			yield year, file, file_no

def debug(*arg, **kwargs):
	for key in kwargs:
		print(key)
	for key in arg:
		print(key)
	input()

# def threaddecorator(funcion):
# 	def wrapper(*arg, **kwargs):
# 		print("thread started:", funcion.__name__) # this isn't printing properly
# 		res = threading.Thread(target=funcion, args=arg, kwargs=kwargs)
# 		res.start()
# 		return res
# 	return wrapper

# @threaddecorator
# def wait_n_print(text):
# 	time.sleep(1)
# 	print(text)

def chrono(func):
	def wrapper(*arg, **kwargs):
		# st.write(time.strftime("%Y-%m-%d %I:%M:%S",(time.localtime())))
		t = time.time()
		h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
		# h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
		place = st.text("%s F: %s" % (h, func.__name__,))
		# print("%s F: %s" % (h, func.__name__))

		# try:
		# 	s= ""
		# 	print(**kwargs)
		# 	print(*arg)
		# 	for key in arg:
		# 		s += str(key)
		# 	print("%s F: %s args: %s" % (h, func.__name__, s))
		# except Exception as e:
		# 	raise e
		# 	pass
		# thread = wait_n_print("%s F: %s args: %s %s" % (h, func.__name__, *arg, **kwargs))

		res = func(*arg, **kwargs)
		if time.time()-t>1:
			h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
			elapsed_time = str(time.strftime("%H:%M:%S", time.gmtime(time.time()-t)))
			output_preview = str(res)[:min(len(str(res)),20)]
			place.text("%s F: %s Time: %s Output: %s"  % (h, func.__name__, elapsed_time, output_preview))
			print("%s F: %s Time: %s Output: %s"  % (h, func.__name__, elapsed_time, output_preview))
			st.success("Función "+func.__name__+" completada con éxito.")
		else:
			place.text("")
			del place
		return res
	wrapper.__doc__ = func.__doc__
	__doc__ = func.__doc__ 
	return wrapper

def fix_name(name):
	name = name.replace("_", ":")
	name = name.replace("=", "?")
	if name.startswith("!"):
		name = name.replace("!", "¿")
	return name

def conds(token):
	if token.pos_ == "PUNCT":
		return False
	if token.is_stop:
		return False
	return True


corpus_folder	= "data/corpus/"
feature_folder	= "data/features/"
freeling_folder	= "data/freeling/"
spacy_folder	= "data/spacy/"

preprocess_file = 'data/opt_preprocess.json'
features_file = 'data/opt_features.json'
# method_file = 'data/opt_method.json'

if not os.path.exists("data/"):
	os.makedirs("data/corpus")
	with open(preprocess_file, 'w', encoding='utf-8') as f:
		json.dump({"lower": True, "agrupar": False, "eliminar": False, "tag_lib": "freeling", "lemmatizacion": False, "pos": True}, f)
	with open(features_file, 'w', encoding='utf-8') as f:
		json.dump({"d": {"wl": ["mean"], "sl": ["mean", "std"], "pf": ["mean"]}, "n_grams": ["char", "word", "POS"], "n": [1, 2, 3], "k": 30}, f)

preprocess = read_json(preprocess_file)
features = read_json(features_file)
# method = read_json(method_file)

tokens_stream_folder = "data/tokens_stream_folder/"
n_grams_folder = "data/n_grams/"
cuestioned_index = "Sentido de Navidad"

nlp = spacy.load('es')

# df = get_df()# TODO load() method, cache etc