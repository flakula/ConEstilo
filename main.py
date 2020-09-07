# from mlxtend.plotting import *
from mpl_toolkits.mplot3d import Axes3D
# from nltk.stem import WordNetLemmatizer
# from nltk.tokenize import sent_tokenize, word_tokenize
# from pandas.tools.plotting import scatter_matrix
# import glob
# import matplotlib as mpl
# import pathlib
# import scipy
import seaborn as sns
# import streamlit as st
from nltk.corpus import stopwords
from nltk.util import ngrams
from operator import indexOf
from scipy.cluster import hierarchy
from sklearn.preprocessing import StandardScaler
import json
import matplotlib.pyplot as plt
import nltk
import numpy as np
import os
import pandas as pd
import streamlit as st
import threading
import time

# wlem = WordNetLemmatizer()

stopWords = set(stopwords.words('spanish'))

from src.understanding_delta import *
from src.utils import *
from src.spaghetti import *

web = True

df = get_df()
d = get_json()

total_files = sum([len(os.listdir(os.path.join(corpus_folder, y))) for y in os.listdir(corpus_folder)])

def log(arg):
	# t = time.time()
	h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
	st.write("%27s" % (h))
	st.write(arg)
	print("%27s" % (h))
	print(arg)

tokens_stream_folder = "data/tokens_stream_folder/"
n_grams_folder = "data/n_grams/"

cuestioned_index = "Sentido de Navidad"

def _sentences(file):
	for sentence in file.split("\n\n"):
		if len(sentence.split())>0:
			yield sentence

def _split(line):
	items = line.split()
	word 	= items[1]
	lexema 	= items[2]
	POS_ext = items[3]
	POS 	= items[4]
	info 	= items[5]
	return word, lexema, POS_ext, POS, info, items

def _get_sentence(sentence):
	return " ".join([tagged_word.split()[1] for tagged_word in sentence.split("\n")])

def preprocess_freeling(year, file):
	# log("preprocess_freeling")
	freeling_file = read_file(freeling_folder, year, file)	
	# log(freeling_folder, year, file)
	# log(freeling_file)
	sents = []
	for sentence_raw in _sentences(freeling_file):
		sentence = _get_sentence(sentence_raw)
		tokens = []
		for tagged_word in sentence_raw.split("\n"):
			word, lexema, POS_ext, POS, info, items = _split(tagged_word)
			if not preprocess["eliminar"] or word.lower() not in stopWords:
				if preprocess["lower"]:
					word = word.lower()
				if preprocess["lemmatizacion"]:
					word = lexema
				tokens.append({ "word":word, 
								"lexema":lexema, 
								"POS":POS, 
								"is_punct": POS.startswith("F"),
								"frl_token":items })
		sents.append({"tokens": tokens, "sentence": sentence})
		# yield {"tokens": tokens, "sentence": sentence}
	return sents
	
# def preprocess_nltk(year, file):
# 	# log("preprocess_nltk")
# 	text = read_file(corpus_folder, year, file)	
# 	sentences = []
# 	for sent in sent_tokenize(text):
# 		tokens = []
# 		for (word, tag) in pos_tag(word_tokenize(sent)):
# 			if not preprocess["eliminar"] or word.lower() not in stopWords:
# 				lexema = wlem.lemmatize(word)
# 				if preprocess["lower"]:
# 					word = word.lower()
# 				if preprocess["lemmatizacion"]:
# 					word = lexema
# 				tokens.append({ "word":word, 
# 								"lexema":lexema, 
# 								"POS":tag })
# 		sentences.append({"tokens": tokens, "sentence": sent})
# 		# yield {"tokens": tokens, "sentence": sentence}
# 	return sentences

def preprocess_spacy(year, file):
	# jl = read_json(spacy_folder, year, file)	# this is not sentence separated.. fix that
	text = read_file(corpus_folder, year, file)	
	# log("preprocess_spacy")
	doc = nlp(text)
	sentences = []
	for sentence in doc.sents:
		tokens = []
		for token in sentence:
			word = token.text
			
			# log(not preprocess["eliminar"], not token.is_stop)
			
			if not preprocess["eliminar"] or not token.is_stop:
				# log("entre!!")
				if preprocess["lower"]:
						word = word.lower()
				if preprocess["lemmatizacion"]:
					word = token.lemma_
				tokens.append({
							"word": word,
							"lexema": token.lemma_,
							"POS": token.pos_,
							"spc_token": token.tag_,
							"is_punct": token.pos_=="PUNCT"
							})
		sentences.append({"tokens": tokens, "sentence": str(sentence)})
		# yield {"tokens": tokens, "sentence": sentence}
	return sentences

# def _extract_n_grams(tokens_stream, n, k, kind):
# 	n_grams = []
		
# 	if kind=="char":
# 		only_words = filter(lambda x: not x["is_punct"], tokens_stream)
# 		char_stream = " ".join([token['word'] for token in only_words])
# 		for i in range(len(char_stream)-2):
# 			n_grams.append((char_stream[i], char_stream[i+1], char_stream[i+2]))
# 	else:
# 		for i in range(len(tokens_stream)-2):
# 			# log(tokens_stream[i][kind], tokens_stream[i+1][kind], tokens_stream[i+2][kind])
# 			n_grams.append((tokens_stream[i][kind], tokens_stream[i+1][kind], tokens_stream[i+2][kind]))
	
# 	# log("_extract_n_grams", n, kind, str(len(n_grams)))
# 	return n_grams
	
def _stylemas(preprocess_data):
	tokens_stream = []
	for sentence in preprocess_data:
		for token in sentence['tokens']:
			tokens_stream.append(token)

	vocabulary = nltk.FreqDist([token['word'] for token in tokens_stream])
	d = {}
	for token in vocabulary:
		# log(token, vocabulary[token])
		if not vocabulary[token] in d:
			d[vocabulary[token]] = []	
		d[vocabulary[token]].append(token)

	N = len([token for token in tokens_stream if not token["is_punct"]])

	V  = len(vocabulary)
	V1 = len(d[1])
	V2 = len(d[2])

	type_token_ratio = (V / N) * 100
	R = 100 * math.log(N)/((1-V1)/V)
	S = V2/V					
	W = N**(V**-0.17)

	M = sum([(i)**2 * len(d[i]) for i in sorted(d)]) 	
	K = 10000*(M - N)/N**2

	sl = [len([token for token in sent["tokens"] if not token["is_punct"] ]) for sent in preprocess_data]

	wl = []
	for sentence in preprocess_data:
		for token in sentence["tokens"]:
			if not token["is_punct"]:
				wl.append(len(token['word']))
				# if len(token['word'])>30:
				# 	log(token['word'])

	i=0
	pfreq={}
	for token in tokens_stream:
		if token["is_punct"]:
			if i not in pfreq: pfreq[i] = 1
			else: pfreq[i] += 1
			i=0
		i+=1
	# pf = [y for y in pfreq.values()]
	pf = []
	for i in pfreq:
		pf+=[i]*pfreq[i]
	# st.write(tokens_stream)
	# st.write(pf)
	array_features = {}
	for feat in features['d']: # wl, sl, pf
		for function in features['d'][feat]:
			exec("array_features[\'{0}_{1}\'] = float(np.{1}({0}))".format(feat, function))

	del preprocess_data, sentence, i, token, vocabulary, d, feat, function, pfreq, M
	return locals()

def _make_df(features, year, file):
	df_d = {}
	for key in features:
		df_d[key] = [features[key]]

	for key in features["array_features"]:
		df_d[key] = [features["array_features"][key]]

	del df_d["array_features"]
	del df_d["sl"]
	del df_d["wl"]
	del df_d["pf"]
	del df_d["tokens_stream"]
	df = pd.DataFrame(df_d)
	return df

def _plot_year_arrays(sls, wls, pfs, year):
	# TODO THIS IS BROKEN
	fig = plt.figure(year)
	fig.suptitle(year, fontsize=12)
	# fig.title(year)

	# plt.title("Sentences Length\n"+ year)
	ax1 = fig.add_subplot(311)
	fd1 = sorted(nltk.FreqDist(sls).items())
	ax1.bar([x for x, y in fd1],[y for x, y in fd1], label="Sentences Lengths") # c='r', color=1
	ax1.legend(loc='best')


	# plt.title("Words Length\n"+ year)
	ax2 = fig.add_subplot(312)
	fd2 = sorted(nltk.FreqDist(wls).items())
	ax2.bar([x for x, y in fd2],[y for x, y in fd2], label="Words Lengths") # c='g', color=2
	ax2.legend(loc='best')


	# plt.title("Punctuation Frequency\n"+ year)
	ax3 = fig.add_subplot(313)
	fd3 = sorted(nltk.FreqDist(pfs).items())
	ax3.bar([x for x, y in fd3],[y for x, y in fd3], label="Punctuations Frequencies") # c='b', color=3
	ax3.legend(loc='best')


	# plt.savefig("data/pngs/all/"+year+'.png')
	if not web:
		plt.savefig("data/_plot_year_arrays.png")
		plt.show()
	else:
		plt.savefig("data/_plot_year_arrays.png")
		st.pyplot()

def _plot_file_arrays(sls, wls, pfs, year, file):
	plt.title("Sentences Length\n"+ year +"\n"+ file)
	fd = sorted(nltk.FreqDist(sls).items())
	plt.plot([x for x, y in fd],[y for x, y in fd])
	# plt.savefig("data/pngs/all/"+"Sentences Length-"+ year +"-"+ file+'.png')
	if not web:
		plt.savefig("data/Sentences Length.png")
		plt.show()
	else:
		plt.savefig("data/Sentences Length.png")
		st.pyplot()

	plt.title("Words Length\n"+ year +"\n"+ file)
	fd = sorted(nltk.FreqDist(wls).items())
	plt.plot([x for x, y in fd],[y for x, y in fd])
	# plt.savefig("data/pngs/all/"+"Words Length-"+ year +"-"+ file+'.png')
	if not web:
		plt.savefig("data/Words Length.png")
		plt.show()
	else:
		plt.savefig("data/Words Length.png")
		st.pyplot()

	plt.title("Punctuation Frequency\n"+ year +"\n"+ file)
	fd = sorted(nltk.FreqDist(pfs).items())
	plt.plot([x for x, y in fd],[y for x, y in fd])
	# plt.savefig("data/pngs/all/"+"Punctuation Frequency-"+ year +"-"+ file+'.png')
	if not web:
		plt.savefig("data/Punctuation Frequency.png")
		plt.show()
	else:
		plt.savefig("data/Punctuation Frequency.png")
		st.pyplot()

def log_df(df, name):
	h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
	print("%27s" % (h), name)
	st.write("%27s" % (h), name)
	st.write(df)
	df.to_csv("data/"+name+".csv")

def save(df, name):
	h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
	print("%27s" % (h), name)
	st.write("%27s" % (h), name)
	st.write(df)
	df.to_csv("data/"+name+".csv")

#################################################################

@chrono
def plot_pca(df_saved, plot=True, plot_var=True, percent=.9, plot_cuestioned_index=False, plot_name="PCA"):
	df = df_saved
	columns = filter(lambda x: x!="title" and x!="year", df.columns)
	l = list(columns)
	X_scaled = StandardScaler().fit_transform(df.loc[:,l])
	features = X_scaled.T
	cov_matrix = np.cov(features)
	values, vectors = np.linalg.eig(cov_matrix)
	x=0
	explained_variances = []
	for i in range(len(values)):
		explained_variances.append(values[i] / np.sum(values))

	explained_var = pd.DataFrame()
	explained_var['i'] = [i for i in range(1, len(explained_variances)+1)]
	explained_var['Var acum'] = [np.sum(explained_variances[0:i]) for i in range(1, len(explained_variances)+1)]
	explained_var['Var i-esima'] = [explained_variances[i-1] for i in range(1, len(explained_variances)+1)]

	projected_1 = X_scaled.dot(vectors.T[0])
	res = pd.DataFrame(projected_1, columns=['PC1'])
	for i in range(1, len(vectors.T)):
		res['PC'+str(i+1)] = X_scaled.dot(vectors.T[i])

	try:
		res['year'] 	= [str(1940+int((int(y)-1940)/5)*5) for y in df_saved['year']]
		plot_by="year"
	except Exception as e:
		# st.error(e)
		res['title'] 	= df_saved.index
		plot_by="title"

	for i in range(1, len(explained_variances)+1):
		su = np.sum(explained_variances[0:i])
		if su > percent:
			break
	
	st.write("Con ", i, "componentes se obtiene una varianza acumulada de ", round(su,2))

	if plot_var:
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.set_xlabel('Cantidad de Componentes.')
		ax.set_ylabel('Varianza')
		ax.set_title('Varianzas por componentes hasta '+str(round(su,2)), fontsize = 20)
		plt.bar(range(1,i+1), explained_variances[:i])
		plt.savefig("data/pca_var.png")
		st.pyplot()

		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.set_xlabel('Cantidad de Componentes.')
		ax.set_ylabel('Varianza')
		ax.set_title('Varianza acumulada por componente')
		plt.bar(
				range(1,len(explained_variances)+1), 
				[np.sum(explained_variances[0:i]) for i in range(1, len(explained_variances)+1)]
				)
		plt.savefig("data/pca_var_accum.png")

		st.pyplot()

	if plot:
		fig = plt.figure()
		ax = fig.add_subplot(1,1,1)
		ax.set_xlabel('PC1')
		ax.set_ylabel('PC2')
		ax.set_title(plot_name)
		if plot_by =="year":
			l = sorted(list(set(res[plot_by])))
			for x in range(len(l)):
				mask = res[plot_by]==l[x]
				res1 = res.loc[mask,:]
				plt.scatter(res1['PC1'],
							res1['PC2'], 
							# c=[(int(l[x])-1940)/72]*len(res1.index), 
							c='b', 
							# norm=matplotlib.colors.Normalize(),
							alpha=(x+1)/(len(l)+1),
							label=l[x]
							)
		else:
			plt.scatter(res['PC1'], res['PC2'])

		if plot_cuestioned_index:
			try:
				idx= 0
				for x in range(len(df_saved.index)):
					if df_saved.index[x]==cuestioned_index:
						idx=x
						break
				# mask = df_saved.loc[:] == df_saved.loc[cuestioned_index]
				# mask = [x for x in mask[mask.columns[0]]]

				plt.scatter(res.loc[idx,'PC1'], 
							res.loc[idx,'PC2'], 
							label=cuestioned_index,
							s=50,
							c='r',
							marker='x',
							)

			except Exception as e:
				st.error(e)
				# raise e
			

		plt.legend(loc='best')
		plt.savefig("data/pca2d.png")
		st.pyplot()
	return res, explained_var # for you to select

@chrono
def plot_dendrogram_euclidean(df, year, corpus=False, groupby=False):
	log_df(df, "df")
	columns = filter(lambda x: x not in ["N", "V", "V1", "V2"], df.columns)
	l = list(columns)
	if not corpus:
		mask = df["year"]==year
		new_df = df.loc[mask, l]
		# new_df = df.loc[mask, ["R","S","W","K",'wl_mean','sl_mean','sl_std','pf_mean','title']]
	else:
		if groupby:
			# new_df = df.groupby('year').mean().loc[:, ["R","S","W","K",'wl_mean','sl_mean','sl_std','pf_mean','title']]
			new_df = df.groupby('year').mean().loc[:, l]
		else:
			new_df = df.loc[:, l]
	log_df(new_df, "new_df")
	try: new_df = new_df.set_index('title')
	except Exception as e: pass
	if len(new_df)<2:
		log("\t> Muy pocos archivos.")
		return
	# t = new_df['title']
	# del new_df['title']
	# del new_df['Unnamed: 0']
	# log(new_df)
	# Calculate the distance between each sample
	Z = hierarchy.linkage(new_df, 'ward')
	P = hierarchy.dendrogram(Z,  leaf_font_size=8, orientation="left", labels=new_df.index)
	if not web:
		plt.show()
	else:
		st.pyplot()
	return Z, P

@chrono
def plot_dendrogram_delta(df, inpath):
	inpath = os.path.join(os.getcwd(), inpath+"\\")
	# log(inpath)
	if len(glob.glob(inpath+"*.txt"))<2:
		log("\t> Only one file this year... no dendrogram for you.")
		return
	freqmatrix = countWordfrequencies(inpath) # TODO fix this
	zscorematrix = getZscore(freqmatrix)
	delta_matrix = delta(zscorematrix)

	plt.xticks(fontsize=8)
	plt.yticks(fontsize=8)
	linkage_object = linkage(delta_matrix, method='ward')
	visualize_dend = sch.dendrogram(Z=linkage_object, labels = delta_matrix.index, orientation='left')

	if not web:
		plt.show()
	else:
		st.pyplot()
	return delta_matrix, linkage_object, visualize_dend

@chrono
def full_dendrogram(df):
	# try: input("Abort now if you don't know what you're doing")
	# except KeyboardInterrupt as e: log("good boy!")
	# return plot_dendrogram_delta(df, "data/all/")
	log("plot_dendrogram_delta(df, \"data/all/\")")

@chrono
def plot_array_features(years=[str(x) for x in range(1940, 2013)], corpus=False):
	# TODO fix this method
	arrays = read_json('data/arrays.json')
	gsls= []
	gwls= []
	gpfs= []
	for year in os.listdir(corpus_folder): 
		if len(years)>0:
			if year not in years:
				continue
		sls = []
		wls = []
		pfs = []
		for file in os.listdir(corpus_folder+year):
			sls += arrays[year][file]['sl']
			wls += arrays[year][file]['wl']
			pfs += arrays[year][file]['pf']

		if year in years and not corpus:
			_plot_year_arrays(sls, wls, pfs, year)

		if corpus:
			gsls += sls
			gwls += wls
			gpfs += pfs # this is wrong

	if corpus:
		_plot_year_arrays(gsls, gwls, gpfs, "V. Piñera") # TODO

@chrono
def extract_features(years):
	# global df
	# s = df.index

	file_on_process = st.text('')
	my_bar = st.progress(0)
	file_no = 1
	for year in os.listdir(corpus_folder): 
		if len(years)>0:
			if not year in years:
				# st.write("no", year)
				continue
		
		# st.write("yes", year)

		d[year] = {}		
		for file in os.listdir(corpus_folder+year):
			h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
			file_on_process.text(" | ".join([h, str(file_no), year, fix_name(file[:-4])]))
			print(h, file_no, year, file)
			d[year][file] = {}
			exec("d[year][file][\"preprocess\"] = preprocess_{0}(year, file)".format(preprocess["tag_lib"]))
			d[year][file]['features'] 	= _stylemas(d[year][file]["preprocess"])
			
			# st.write(d[year][file]['features']['pf'])
			# st.write(read_file(corpus_folder, year, file))

			tsfolder = os.path.join(tokens_stream_folder,year+'/')
			if not os.path.exists(tsfolder ):
				os.makedirs(tsfolder)

			with open(os.path.join(tsfolder, file), 'w', encoding='utf-8') as f:	
				json.dump(d[year][file]['features']['tokens_stream'], f)
			
			d[year][file]['df'] 		= _make_df(d[year][file]["features"], year, file)
			d[year][file]['df']['year'] = [int(year)]
			d[year][file]['df']['title'] = [fix_name(file)[:-4]]
			my_bar.progress(int(file_no/(total_files)*100))
			file_no += 1

	asd = []
	file_on_process.text("Guardando csv")
	# [d[year][file]['df'] for year, file, _ in iterate(corpus_folder)]
	for year, file, file_no in iterate(corpus_folder):	
		my_bar.progress(int(file_no/(total_files)*100))
		if len(years)>0:
			if not year in years:
				continue
		asd.append(d[year][file]['df'])

	df = pd.concat(asd, ignore_index=True)
	df.to_csv("data/df.csv")

	file_on_process.text("Eliminando datos innecesarios")
	for year, file, file_no in iterate(corpus_folder):	
		my_bar.progress(int(file_no/(total_files)*100))
		if len(years)>0:
			if not year in years:
				continue
		del d[year][file]['df']

	arrays = {}
	file_on_process.text("Guardando wl, sl, pf")
	for year, file, file_no in iterate(corpus_folder):
		my_bar.progress(int(file_no/(total_files)*100))
		if len(years)>0:
			if not year in years:
				continue
		if year not in arrays:
			arrays[year]={}
		arrays[year][file] = {}
		arrays[year][file]['sl'] = d[year][file]['features']['sl']
		arrays[year][file]['wl'] = d[year][file]['features']['wl']
		arrays[year][file]['pf'] = d[year][file]['features']['pf']

	with open("data/arrays.json", 'w', encoding='utf-8') as f:	
		json.dump(arrays,f)

	file_on_process.text("Eliminando más datos innecesarios")
	for year, file, file_no in iterate(corpus_folder):	
		my_bar.progress(int(file_no/(total_files)*100))
		if len(years)>0:
			if not year in years:
				continue
		del d[year][file]['features']['sl']
		del d[year][file]['features']['wl']
		del d[year][file]['features']['pf']
		del d[year][file]['features']
		del d[year][file]['preprocess']

	file_on_process.text("Guardando json")
	with open("data/d.json", 'w', encoding='utf-8') as f:
		json.dump(d,f)

	file_on_process.text("Estilemas estraídos con éxito.")
	return df

@chrono
def pos_tagging(freeling=False, spcy=False, years=[]):
	file_no = 0

	file_on_process = st.text('')
	my_bar = st.progress(0)

	for year, file, file_no in iterate(corpus_folder): 
		if len(years)>0:
			if not year in years:
				continue

		h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
		file_on_process.text(" | ".join([h, str(file_no), year, fix_name(file[:-4])]))
		print(h, file_no, year, file)

		p 	= os.path.join(spacy_folder, year)
		pf 	= os.path.join(freeling_folder, year)
		if not os.path.exists(p):
			os.makedirs(p)
		if not os.path.exists(pf):
			os.makedirs(pf)

		folder = os.path.join(corpus_folder, year)
		old_file = os.path.join(folder, file)

		# freeling se demoro poco mas de 2h
		if freeling:
			ffile = os.path.join(pf, file)
			# log("analyzer.bat -f config.cfg < \"%s\" > \"%s\"" % (old_file, ffile)) 
			if not os.path.exists(ffile):
				print("analyzer.bat -f config.cfg < \"%s\" > \"%s\"" % (old_file, ffile))
				os.system("analyzer.bat -f config.cfg < \"%s\" > \"%s\"" % (old_file, ffile)) 

		# spacy se demora menos
		if spcy:
			sfile = os.path.join(pf, file)
			if not os.path.exists(sfile):
				# log(sfile)
				with open(old_file, encoding="utf8") as f:
					text = f.read()
					doc = nlp(text)

				tokens = []
				for token in doc:
					tokens.append(
									{
										"text": token.text, 
										"lemma_": token.lemma_, 
										"pos_": token.pos_, 
										"tag_": token.tag_, 
										"dep_": token.dep_, 
										"shape_": token.shape_, 
										"is_alpha": token.is_alpha, 
										"is_stop": token.is_stop,
										"sentiment": token.sentiment,
										# "vocab": token.vocab,
										# "head": token.head
									}
							)

				with open(sfile, 'w', encoding="utf8") as f:
					json.dump(tokens, f)
		my_bar.progress(int(file_no/(total_files)*100))	
	file_on_process.text("")

@chrono
def dend(df, name):
	st.write("Dendrograma de ", name)
	place = st.text("Obteniendo los z-scores...")
	zscorematrix = getZscore(df)
	st.write(zscorematrix)
	place.text("Calculando la matriz de distancias Delta...")
	delta_matrix = delta(zscorematrix)
	st.write(delta_matrix)
	place.text("Conectando los documentos segun el metodo ward...")
	linkage_object = linkage(delta_matrix, method='ward')
	# st.write(linkage_object)
	place.text("Construyendo el dendrograma...")
	x = len(delta_matrix)
	fig = plt.figure(figsize=(8,x/5))
	visualize_dend = sch.dendrogram(Z=linkage_object, labels = delta_matrix.index, orientation='left')
	# st.write(visualize_dend)
	place.text("Terminado el dendrograma con éxito.")
	# ax.set_title("Dendrograma: "+ name)
	plt.savefig("data/{}.png".format(name))
	st.pyplot()
	return delta_matrix, linkage_object, visualize_dend

@chrono
def plot_corr_matrix(df, name):
		cor_matrix = df.corr().round(2)
		log_df(cor_matrix, "Matriz de covarianzas")

		fig = plt.figure(figsize=(12,12));
		sns.heatmap(cor_matrix, annot=True, center=0, 
					# cmap=sns.diverging_palette(250, 10, as_cmap=True), 
					ax=plt.subplot(111));
		# plt.show()
		plt.savefig("data/{}corr_matrix.png".format(name))
		st.pyplot()

@chrono
def pca_df(df):
	X_scaled = StandardScaler().fit_transform(df)
	features = X_scaled.T
	cov_matrix = np.cov(features)
	values, vectors = np.linalg.eig(cov_matrix)
	x=0
	explained_variances = []
	for i in range(len(values)):
		explained_variances.append(values[i] / np.sum(values))

	explained_var = pd.DataFrame()
	explained_var['i'] = [i for i in range(1, len(explained_variances)+1)]
	explained_var['Var acum'] = [np.sum(explained_variances[0:i]) for i in range(1, len(explained_variances)+1)]
	explained_var['Var i-esima'] = [explained_variances[i-1] for i in range(1, len(explained_variances)+1)]
	# log("2")

	projected_1 = X_scaled.dot(vectors.T[0])
	res = pd.DataFrame(projected_1, columns=['PC1'])
	for i in range(1, len(vectors.T)):
		res['PC'+str(i+1)] = X_scaled.dot(vectors.T[i])

	return res, explained_var
	
@chrono
def xtract_n_grmas(df, years):
	# n_grams": ["char", "word", "POS"], "n": [1, 2, 3], "k": 30

	k = features["k"]
	n_grams_kind = [x for x in features['n_grams'] if "char"!=x]

	matrix = {}
	n_grams = {}

	total_iters = len(features["n"]) * len(n_grams_kind) * total_files
	file_no = 1
	file_on_process = st.text('Extrayendo los n-gramas más frecuentes')
	my_bar = st.progress(0)
	
	for n in features["n"]:
		for kind in n_grams_kind:
			ts_corpus = []
			# tokens_stream_per_year = []
			for year in os.listdir(corpus_folder): 	

				ts_year = []
				# log(year)
				for file in os.listdir(corpus_folder+year):
					my_bar.progress(int(file_no/(total_iters)*100))
					file_no+=1
					if len(years)>0:
						if not year in years:
							continue

					h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))
					# file_on_process.text(" | ".join([h, str(file_no), str(n), kind, year, fix_name(file[:-4])]))
					file_on_process.text(" | ".join([h, 'Extrayendo los n-gramas más frecuentes', str(n), kind, year]))

					if fix_name(file[:-4]) not in matrix:
						matrix[fix_name(file[:-4])] = {}
						matrix[fix_name(file[:-4])]['year'] = year
					matrix[fix_name(file[:-4])][kind] = [ts[kind] for ts in read_json(os.path.join(tokens_stream_folder, year, file))]
					ts_year += matrix[fix_name(file[:-4])][kind] 

				ts_corpus += ts_year
				# tokens_stream_per_year.append(ts_year)
			# log(ts_corpus)
			# log(n, kind, len(ts_corpus))
			ll = nltk.FreqDist(ngrams(ts_corpus, n)).most_common(k)
			n_grams["|".join([str(n), str(k), kind])] = ll
	
	# log(n_grams)
	# file_on_process.text("Guardando n-gramas")
	n_df = pd.DataFrame(n_grams)
	# n_df.to_csv("data/n_df.csv")
	# log(n_df)
	file_on_process.text("Extraídos con éxito los n-gramas más frecuentes.")

	# construir df de 30x6 columnas y 151 filass
	# log(columns)

	# doc x ngrama
	# df_doc_vs_freq = pd.DataFrame()

	file_on_process1= st.text("Actualizando df")
	file_no = 1
	total_iters = len(matrix) * len(features["n"]) * len(n_grams_kind)
	my_bar1 = st.progress(0)

	for title in matrix:
		h = str(time.strftime("%Y-%m-%d %I:%M:%S", time.localtime()))	
		file_on_process1.text(" ".join(["Actualizando n-gramas de",title]))
		# file_on_process1.text(" | ".join([title,str(n),kind]))
		df.loc[title, "year"]=int(matrix[title]['year'])
		for n in features["n"]:
			for kind in n_grams_kind:
				ln = ngrams(matrix[title][kind], n)
				fd = nltk.FreqDist(["|".join(x) for x in list(ln)])

				columns = []
				i = "|".join([str(n), str(k), kind])
				for n_gr, freq in n_grams[i]:
					columns.append("|".join(n_gr))

				for col in columns:
					if col in fd:
						# df.loc[title, col] = fd[col] * 100 / len(matrix[title][kind])
						df.loc[title, col] = fd[col] * 100 / 100000
						# df.loc[title, col] = fd[col]
					else:
						df.loc[title, col] = 0 
						#This means that the n_gram 
						#(que esta entre los top k) 
						#no se encuentra en title
						# log(title + ' ' + col)
				my_bar1.progress(int(file_no/(total_iters)*100))

				file_no+=1


	file_on_process1.text("Actualizados los n-gramas de cada archivo.")
	# df.to_csv("data/df.csv")
	# log(df)
	# log_df(df, "df")
	# hacer grafo ver en gephi
	return n_df, df

#################################################################
def run_pos_tagging(df):
	"""Ejecutar la extraccion de etiquetas POS, segun spacy o freeling"""
	freeling=st.sidebar.checkbox("freeling")
	spcy=st.sidebar.checkbox("spacy")
	if st.sidebar.button("Empezar"):
		pos_tagging(
					freeling=freeling, 
					spcy=spcy,
					years=[]
					)

def run_extract_features(df):
	"""Extraer las caracteristicas del corpus."""
	# global df
	if st.sidebar.button("Empezar"):
		df = extract_features([])
	return df

def run_plot_pca(df):
	"""Extraer Componentes Principales"""
	pca, expl_var = plot_pca(df, 
							plot=st.sidebar.checkbox("Graficar CP"), 
							plot_var=st.sidebar.checkbox("Graficar varianzas CP"), 
							# plot_by="year"# TODO allow plot by file?
							)
	pca.to_csv("data/pca.csv")
	expl_var.to_csv("data/expl_var.csv")
	log(pca)
	log(expl_var)

def run_plot_corr_matrix(df):
	"""Mostrar el heatmap o mapa de calor de la matriz de covarianzas"""
	df = get_df()
	plot_corr_matrix(df, "DataFrame")

def run_plot_array_features(df):
	"""Graficar longitud de las palabras, oraciones y frecuencias de puntuación"""
	corpus = st.sidebar.checkbox("Todo el corpus")
	if not corpus:
		year = st.sidebar.selectbox("Seleccione el año.", list(set(df["year"])))
		plot_array_features(years=[str(year)], corpus=corpus)
	else:
		plot_array_features(years=[], corpus=corpus)

def run_dendrogram(df):
	"""Construir Dendrograma"""
	dist = st.sidebar.selectbox("Distancia", ["Delta", "Euclideana"])
	folder = st.sidebar.selectbox("Carpeta", ["Año específico", "Corpus"], 0)
	if folder == "Año específico":
		year = st.sidebar.selectbox("Año", [folder for folder in os.listdir(corpus_folder)])
		if st.sidebar.button("Empezar"):
			if dist=="Euclideana":
				Z	= plot_dendrogram_euclidean(df, int(year))
				# log(Z)
			else:
				loc = plot_dendrogram_delta(df, os.path.join(corpus_folder, str(year)))
	else:
		if st.sidebar.button("Empezar"):
			if dist=="Euclideana":
				Z = plot_dendrogram_euclidean(df, 0, corpus=True)
				# log(Z)
			else:
				loc = full_dendrogram(df)

def run_extract_n_grams(df_saved):
	"""Extraer los n-gramas"""
	corpus 			=	st.sidebar.checkbox("Corpus entero")
	if not corpus:
		years 		=	[st.sidebar.selectbox("Seleccione el año.", list(os.listdir(corpus_folder)))]
		st.write("En ", years[0], " hay ", len(list(os.listdir(os.path.join(corpus_folder, years[0])))), "documentos.")
		# groupbyyear = False
	else:
		years 		= list(os.listdir(corpus_folder))
		st.write("En todo el corpus hay ", total_files, "documentos")

	if st.sidebar.button("Empezar"):
		n_df, df_doc_vs_freq 							= xtract_n_grmas(pd.DataFrame(), years)	
		del df_doc_vs_freq['year']
		log_df(n_df, "n_df")
		log_df(df_doc_vs_freq, "df_doc_vs_freq")
		
		delta_matrix, linkage_object, visualize_dend	= dend(df_doc_vs_freq, "n-gramas")
		log_df(delta_matrix, "delta_matrix")
		
		try:
			# pca_n, expl_var_n = pca_df(df_doc_vs_freq)
			pca_n, expl_var_n 	= plot_pca(df_doc_vs_freq, 
										plot=True, 
										plot_var=True, 
										plot_cuestioned_index=False,
										plot_name="PCA - n-gramas"
										)

			# log_df(pca_n, "pca_n")
			# log_df(expl_var_n, "expl_var_n")
		except Exception as e:
			raise e
			pass

def run_eval_feature(df):
	""" Evaluar el desempeño de una característica específica en el corpus.
	"""
	n = st.sidebar.selectbox("Analizar la característica a nivel de:", ("Documento", "Año")) #, "Corpus"

	df = get_df()
	# st.write(df)
	if n=="Documento":
		g_df = df
	if n=="Año":
		g_df = df.groupby('year').mean()
		# if "promedio"==st.sidebar.selectbox("Analizar la característica teniendo en cuenta el:", ("promedio", "total")):
		# 	g_df = df.groupby('year').mean()[s_feat]
		# else:
		# 	g_df = df.groupby('year').sum()[s_feat]

	s,w,p,f = [], [], [], []
	for feat in list(g_df.columns):
		if feat.startswith("sent"):
			s.append(feat)
		elif feat.startswith("punct"):
			p.append(feat)
		elif feat.startswith("word"):
			w.append(feat)
		else:
			f.append(feat)	

	s_feat = st.sidebar.multiselect('Seleccione las características generales a evaluar', f, f)
	g_df = g_df[s_feat]
	# if st.sidebar.checkbox("Más características"):
	# 	s_feat2 = st.sidebar.multiselect('Seleccione las características relativas a oraciones a evaluar', s, s)
	# 	s_feat3 = st.sidebar.multiselect('Seleccione las características relativas a la longitud de palabras a evaluar', w, w)
	# 	s_feat4 = st.sidebar.multiselect('Seleccione las características relativas a la distancia entre signos de puntuación a evaluar', p, p)

	trans = st.sidebar.radio("Transformar",["No", "Normalizar", "Escalar (0-1)", 'Normalizar y escalar (a 0-1)'])
	if trans != "No":
		if trans=='Normalizar':
			f_ = lambda grp: (grp - grp.mean()) / grp.var()
		if trans=='Escalar (0-1)':
			f_ = lambda grp: (grp.abs() / grp.abs().max())
		if trans=='Normalizar y escalar (a 0-1)':
			f_ = lambda grp: ((grp - grp.mean()) / grp.var() ) / ((grp - grp.mean()) / grp.var() ).max()

		for col in g_df:
			g_df[col] = g_df[col].transform(f_)
	
	if len(g_df.columns)==1:
		sort = st.sidebar.checkbox("Ordenar")
		if sort:
			g_df = g_df.sort_values(g_df.columns[0], ascending=False) # 


	# exec("st."+st.sidebar.radio("Biblioteca gráfica:",("pyplot","plotly_chart"))+"(fig)")
	plot_lib = st.sidebar.radio("Biblioteca gráfica:",("line_chart", "bar_chart", "pyplot", "plotly_chart"))
	if plot_lib=="pyplot":
		fig = plt.figure(figsize=(20,5))
		ax1 = fig.add_subplot(111)
		# fs = mpl.rcParams['font.size']
		# mpl.rcParams['font.size'] = 12
		if st.sidebar.checkbox("Leyenda"):
			ax1.legend(loc='best')
		xticks = g_df.index
		ax1.set_xticklabels(xticks, rotation=-90)
		for col in g_df:
			ax1.plot(g_df[col])
		st.pyplot(fig)
		# mpl.rcParams['font.size'] = fs
	if plot_lib=="line_chart":
		if len(g_df.columns)==1:
			if sort:
				st.line_chart(list(g_df[g_df.columns[0]]))
			else:
				st.line_chart(g_df)
		else:
			st.line_chart(g_df)

	if plot_lib=="bar_chart":
		if len(g_df.columns)==1:
			if sort:
				st.bar_chart(list(g_df[g_df.columns[0]]))
			else:
				st.bar_chart(g_df)
		else:
			st.bar_chart(g_df)

	if plot_lib=="plotly_chart":
		# fig = plt.figure(figsize=(20,5))
		fig = plt.figure()
		ax1 = fig.add_subplot(111)
		xticks = g_df.index
		ax1.set_xticklabels(xticks, rotation=-90)
		for col in g_df:
			ax1.plot(g_df[col])
		st.plotly_chart(fig)

	st.write(pd.DataFrame(g_df))
	try:
		st.write(g_df.describe())
		
	except Exception as e:
		st.error(e)

def run_plot_features(df):
	"""Graficar una característica contra otra"""
	# df = pd.read_csv("df.csv")
	df=get_df()

	opt = st.sidebar.selectbox("¿Cuántas variables plotear?",["2D","3D", "multiselect"])
	cols = [col for col in df.columns if col!="title"]
	if opt=="multiselect":
		features = st.sidebar.multiselect("Selecciona las características:", cols)
		if len(features)>0:
			if st.sidebar.checkbox("subplots"):
				ax = df[features].plot(subplots=True) # esto devuelve un array
			else:
				ax = df[features].plot()
	# if opt=="1D":
	# 	# x = st.sidebar.selectbox("x",df.columns)
	# 	features = st.sidebar.multiselect("Selecciona las características:", df.columns)
	# 	df.plot.bar(features[0w]) 
	if opt=="2D":
		x = st.sidebar.selectbox("x", cols)
		y = st.sidebar.selectbox("y", cols)
		ax = df.plot.scatter(x,y)
		ax.set_xlabel(x)
		ax.set_ylabel(y)
	if opt=="3D":
		fig = plt.figure()
		ax = fig.add_subplot(111, projection='3d')
		x = st.sidebar.selectbox("x", cols)
		y = st.sidebar.selectbox("y", cols)
		z = st.sidebar.selectbox("z", cols)
		ax.scatter(df[x],df[y],df[z])
		ax.set_xlabel(x)
		ax.set_ylabel(y)
		ax.set_ylabel(z)

	# scatter_matrix(df, alpha=0.5, diagonal='hist')
	st.pyplot()

	if st.sidebar.button("Mostrar DataFrame"):
		st.write(df)

def run_all(df):
	"""Ejecute todos los pasos para el corpus"""

	corpus 			=	st.sidebar.checkbox("Corpus entero")
	if not corpus:
		years 		=	[st.sidebar.selectbox("Seleccione el año.", list(os.listdir(corpus_folder)))]
		st.write("En ", years[0], " hay ", len(list(os.listdir(os.path.join(corpus_folder, str(years[0]))))), "documentos.")
		groupbyyear = False
	else:
		years 		= list(os.listdir(corpus_folder))

		groupbyyear	=	st.sidebar.checkbox("Agrupar por años")
		if groupbyyear:
			st.write("En todo el corpus hay ", len(list(os.listdir(corpus_folder))), "años.")
		else:
			st.write("En todo el corpus hay ", total_files, "documentos")
	pos_chbx 		=  st.sidebar.checkbox("Etiquetar POS")
	if pos_chbx:
		freeling 	=	st.sidebar.checkbox("freeling")
		spcy 		=	st.sidebar.checkbox("spacy")
	xtract_features =	st.sidebar.checkbox("Extraer estilemas")
	extract_n_grams =	st.sidebar.checkbox("Extraer n-gramas")
	corr_matrix 	=	st.sidebar.checkbox("Mostrar matriz de correlaciones")
	plot_arrays 	=	st.sidebar.checkbox("Graficar longitud de palabras, oraciones y frecuencia de puntuación")
	pca_chbx 		=	st.sidebar.checkbox("Analisis de Componentes Principales")
	if pca_chbx:
		plot_var	=	st.sidebar.checkbox("Graficar varianzas CP")
		plot 		=	st.sidebar.checkbox("Graficar CP")
	rdendrogram		=	st.sidebar.checkbox("Construir Dendrograma")

	if st.sidebar.button("Empezar"):
		# run_pos_tagging(df)
		if pos_chbx:
			pos_tagging(
						freeling=freeling, 
						spcy=spcy,
						years=years
						)
		
		# run_extract_features(df)
		if xtract_features:
			df 			= extract_features(years)# stylemas
			df 			= df.set_index('title')
			if corpus and groupbyyear:
				df = df.groupby('year').mean()
			# log_df(df, "df")
		else:
			df = get_df()
			
			if not corpus:# only one year or corpus
				mask = df['year'] == years[0]
				# st.write(df)
				# st.write(years)
				df = df.loc[mask,:]
				st.write(df)
			else:
				if groupbyyear:
					df = df.groupby('year').mean()

		# columns = filter(lambda x: x not in ["N", "V", "V1", "V2"], df.columns)
		# l = list(columns)
		# df = df.loc[:,l]
		log_df(df, "df")

		if corr_matrix:
			plot_corr_matrix(df,"estilemas")

		if extract_n_grams:
			n_df, n_grams	= xtract_n_grmas(pd.DataFrame(), years)

			if groupbyyear:
				n_grams = n_grams.groupby('year').mean()

			log_df(n_grams, "n-grams")

			full_df = pd.merge(df, n_grams, how='inner', left_index=True, right_index=True)

			for row in full_df.index:
				if groupbyyear:
					full_df['year'] = n_grams.index
					n_grams['year'] = n_grams.index
					df['year'] 		= n_grams.index
				else:
					full_df.loc[row,'year'] = n_grams.loc[row]['year']

			log_df(full_df, "full_df")
		
		# plot_corr_matrix(n_grams,"n_grams")
		
		# plot_corr_matrix(full_df,"full_df")

		# run_plot_pca(df)
		if pca_chbx:
			pca, expl_var 			= plot_pca(df, 
										plot 	 =plot, 
										plot_var =plot_var, 
										plot_cuestioned_index=not groupbyyear,
										plot_name="PCA - estilemas",
										)

			pca, expl_var 			= plot_pca(n_grams, 
										plot 	 =plot, 
										plot_var =plot_var, 
										plot_cuestioned_index=not groupbyyear,
										plot_name="PCA - n-gramas",
										)
			
			pca, expl_var 			= plot_pca(full_df, 
										plot 	 =plot, 
										plot_var =plot_var, 
										plot_cuestioned_index=not groupbyyear,
										plot_name="PCA - estilemas y n-gramas",
										)
			
			
			# pca_n, expl_var_n 		= pca_df(n_grams)
			# log_df(pca, "pca")
			# log_df(expl_var, "expl_var")
			# NotImplementedError: Dtype complex128 not understood.

		# run_plot_array_features(df)
		if plot_arrays:
			plot_array_features(years=years, corpus=corpus)

		# run_dendrogram(df)
		if rdendrogram:
			cols = []
			cols += [col for col in df.columns]
			cols += [col for col in n_grams.columns]

			delta_matrix = dend(	 df.loc[:,[col for col in		 df.columns if col!='year']], "Estilemas")
			delta_matrix = dend(n_grams.loc[:,[col for col in	n_grams.columns if col!='year']], "N-gramas")
			delta_matrix = dend(full_df.loc[:,[col for col in	list(set(cols)) if col!='year']], "Estilemas y n-gramas")
			
			# log_df(delta_matrix, "delta_matrix")
	
		if st.sidebar.button("Guardar"):
			save(df, "df")
			save(n_grams, "n_grams")
			save(full_df, "full_df")

#################################################################

def opt_preprocess(l):
	"""Opciones de preprocesamiento."""
	# log("preprocess")
	lower = st.checkbox("LLevar todo a minúsculas", l["lower"])
	agrupar = st.checkbox("Agrupar por años", l["agrupar"])
	eliminar = st.checkbox("Eliminar stopwords", l["eliminar"])
	tag_lib = st.selectbox("herramienta de etiquetado:",["freeling", "spacy", "nltk"], indexOf(["freeling", "spacy", "nltk"], l["tag_lib"]))
	lemmatizacion = st.checkbox("lemmatizacion", l["lemmatizacion"])
	pos = st.checkbox("POS-tagging", l["pos"])
	del l
	log(locals())
	return locals()

def opt_features(l):
	"""Marcadores estilométricos."""
	# log("marcadores_estilometricos")
	opts = ["TTR", "W", "R", "S", "K", "N", "V", "V1", "V2", "Vi"]
	pd_f = ["count", "sum", "mean", "median", "min", "max", "std", "var"]
	esp = ["wl", "sl", "pf"]
	d = {}
	for x in esp:
		d[x] = st.multiselect(x, pd_f, l["d"][x])
	n_grams = st.multiselect("N-gramas",["word", "lemma", "POS", "char"], l["n_grams"])
	if len(n_grams) > 0:
		n = st.multiselect('n', [1, 2, 3], l["n"])
	else:
		n = []

	if st.checkbox("seleccionar los k n-gramas mas frecuentes", l["k"]!=0):
		k = st.slider('k', 10, 100, l["k"], step=10)
	else:
		k = 0
	del opts, pd_f, esp, x, l
	log(locals())
	return locals()
	
#################################################################

#TODO!!
def normalize(df):
	pass
def scale(df):
	pass

#################################################################

def leer():
	""" Leer los documentos originales de Virgilio Piñera 
	"""
	# if st.sidebar.checkbox("Analizar año específico"):
	folders = ["data/corpus/", "data/features/", "data/freeling/", "data/spacy/", "backups/all-content/"]
	# input_folder = st.sidebar.selectbox("Seleccione carpeta", folders)
	input_folder = "data/corpus/"

	years = [folder for folder in os.listdir(input_folder)]
	
	year = st.sidebar.selectbox("Seleccione año", years)

	files = [file for file in os.listdir(input_folder+year+"/")]

	file = st.sidebar.selectbox("Seleccione archivo", files)

	st.write(fix_name(file[:-4]))
	with open(input_folder+year+"/"+file, encoding="utf8") as f:
		st.write(f.read())
def man():
	with open("README.md") as f:
		t = f.read()[18:]
	st.write(t)
def sidebar():
	st.sidebar.header("Controles")
	do = st.sidebar.selectbox("",[ "Manual", "Leer", "Configurar", "Ejecutar"]) # Mostrar
	if do=="Configurar":
		opts = {}
		for key, value in globals().items():
			if key.startswith("opt_"):
				opts[value.__doc__] = value
		option = st.sidebar.selectbox('Seleccionar:', list(opts))
		try:
			with open("data/"+str(opts[option].__name__)+".json", encoding='utf-8') as f:
				l = json.load(f)
		except Exception as e:
			st.error(e)
			l = {}
		l = opts[option](l)		
		with open("data/"+str(opts[option].__name__)+".json", 'w', encoding='utf-8') as f:
			json.dump(l, f)
	if do=="Ejecutar":
		opts = {}
		for key, value in globals().items():
			if key.startswith("run_"):
				# exec("if st.sidebar.button(\""+value.__doc__+"\"): "+key+"(df)")
				opts[value.__doc__] = value
		option = st.sidebar.selectbox('Seleccionar:', list(opts))
		l = opts[option](df)		
	if do=="Leer":
		leer()
	if do=="Manual":
		man()
@chrono
def delta(zscorematrix):
	"""
		This function calculate delta for the whole corpus
	"""
	# F: delta Time: 01:28:29
	# We take names of the columns of the dataframe, that means, the tokens
	tokens = list(zscorematrix.columns.values)
	# print(tokens)
	
	# We take names of the rows of the dataframe, that means, the names of the files
	indexs = list(zscorematrix.index)
	# print(indexs)
	
	# We creata an empty dataframe whose columns and rows are the names of the files
	delta_matrix = pd.DataFrame(columns=indexs,index=indexs)
	file_no = 1
	place = st.text("")
	my_bar = st.progress(0)
	# We take a text
	for index1 in indexs:
		# print (index1)
		# We take another text
		for index2 in indexs:
			place.text(" | ".join([index1, index2]))
			# We create a variable for saving the distance between the texts
			text_distance = 0
			# print(index2)
			my_bar.progress(int(file_no/(len(indexs)**2)*100))
			file_no+=1
			# time.sleep(3)
			try:
				if str(delta_matrix.at[index1,index2]) !="nan":
					continue
			except Exception as e:
				# st.error(e)
				pass

			# Now that we have two texts, we take a token
			for token in tokens:
				# And we see the value of this token in both texts
				value1=max(0, zscorematrix.loc[index1,token])
				value2=max(0, zscorematrix.loc[index2,token])
				# We calculate the distance between them. The form would be |text_value_1 - text_value_2|
				text_distance = text_distance+abs(value1-value2)
				# st.write(token, value1, value2, text_distance)
			# We sum all the values for every peer of texts in order to get the distance between two texts in all the dimensions, in all the words 
			delta_matrix.loc[index1,index2] = text_distance
			delta_matrix.loc[index2,index1] = text_distance
			# st.write(delta_matrix)


			# print(text_distance)
	# print("\n\n\n\nHere it ist, the delta matrix for the corpus!")
	# print(delta_matrix)
	return delta_matrix



st.title('ConEstilo ✒️') # 📝 ✏
sidebar()
