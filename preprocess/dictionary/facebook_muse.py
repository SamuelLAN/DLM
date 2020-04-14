import json
import codecs

dataFile = codecs.open('../../Facebook_Data/de-en.txt', 'r', 'utf-8')
Lines = dataFile.readlines()

Dict = {}

for line in Lines:
    words = line.split()
    Dict[words[1]] = {"translation": [words[0]]}
dataFile.close()

with codecs.open('../../Facebook_Data/de-en.json', 'w', 'utf-8') as outfile:
    json.dump(Dict, outfile)