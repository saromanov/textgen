import argparse
from keras.models import Sequential
from keras.layers.recurrent import LSTM
from keras.layers.core import Activation, Dense, Dropout
from keras.datasets.data_utils import get_file

def build_sentences(text, maxlen):
	if maxlen == 0:
		raise Exception("maxlen must be > 0")
	if not text:
		raise Exception("Text data must be non-empty")

	sents = []
	for i in range(maxlen):
		sents.append(text[i:i + maxlen])
	return sents

def start(values):
	url = values["url"]
	path = "deafult"
	if url:
		get_file(path, url=url)

	if values["path"]:
		path = values["path"]
	data = open(path).read.lower()

	maxlen = values["maxlen"]
	characters = set(data)
	sentences = build_sentences(data, maxlen)
	model = Sequential()
	model.add(LSTM(512, return_sequence=False))
	model.add(Dropout(0.3))
	model.add(Dense(len(characters)))
	model.add(Activation('softmax'))
	model.compile(loss="categorical_crossentropy", optimizer="adadelta")


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument('url', help="Url to data")
	parser.add_argument('maxlen', help="Maximum length of sequence", type=int, default=30)
	parser.add_argument('path', help="Path to data")
	start(parser.parse_args())