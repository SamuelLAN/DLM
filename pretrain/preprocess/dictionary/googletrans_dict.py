import sys
import argparse
sys.path.append('/Users/jiayonglin/Desktop/828B/DLM/')
from googletrans import Translator
import os
from nltk.corpus import wordnet as wn
from lib.utils import write_json
from fake_useragent import UserAgent
import random
from urllib.request import Request, urlopen
from bs4 import BeautifulSoup
from pretrain.preprocess.config import dictionary_dir
google_trans_dir = os.path.join(dictionary_dir, 'google')

my_parser = argparse.ArgumentParser(description='config')
my_parser.add_argument('-i', metavar='start index', type=int, help='The start Index of the word in wn.words()')
my_parser.add_argument('-f', metavar='Saved Json File Name', type=str, help='The name of saved json file')
args = my_parser.parse_args()
i = args.i
file_name = args.f

if not os.path.exists(google_trans_dir):
    os.mkdir(google_trans_dir)
write_dict_path = os.path.join(google_trans_dir, file_name)

# 'en_ro_dict_google_trans_4.json'
def random_proxy():
  return random.randint(0, len(proxies) - 1)

ua = UserAgent() # From here we generate a random user agent
proxies = [] #
proxies_req = Request('https://www.sslproxies.org/')
proxies_req.add_header('User-Agent', ua.random)
proxies_doc = urlopen(proxies_req).read().decode('utf8')
soup = BeautifulSoup(proxies_doc, 'html.parser')
proxies_table = soup.find(id='proxylisttable')

# Save proxies in the array
for row in proxies_table.tbody.find_all('tr'):
    proxies.append({
    'ip':   row.find_all('td')[0].string,
    'port': row.find_all('td')[1].string
  })

proxy_index = random_proxy()
proxy = proxies[proxy_index]
p = {
 "http": "http://"+proxy['ip']+":"+proxy['port'],
 "https": "https://"+proxy['ip']+":"+proxy['port'],
}

translator = Translator(proxies = p, timeout = 3)


dictionary = dict()

all_words = list(wn.words())

while i < len(all_words):
#for word in list(wn.words()):
    word  = all_words[i]
    print(word)
    if i != 0 and i % 40 ==0:
        proxy_index = random_proxy()
        proxy = proxies[proxy_index]
        p = {
         "http": "http://"+proxy['ip']+":"+proxy['port'],
         "https": "https://"+proxy['ip']+":"+proxy['port'],
        }
        translator = Translator(proxies = p, timeout = 3)
        
    try:
        word_details = dict()
        
        trans = translator.translate(word,dest= "ro", src='en').text
        if trans != word:
            word_details["translation"] = [trans]
            dictionary[word] = word_details


        if i != 0 and i % 1000 == 0:
            print("write to json")
            write_json(write_dict_path, dictionary)
            print("--------")
        
        i+=1
        
        
    except:
        del proxies[proxy_index]
        print('Proxy ' + proxy['ip'] + ':' + proxy['port'] + ' deleted.')
        if len(proxies) == 0:
            proxies = [] #
            proxies_req = Request('https://www.sslproxies.org/')
            proxies_req.add_header('User-Agent', ua.random)
            proxies_doc = urlopen(proxies_req).read().decode('utf8')
            soup = BeautifulSoup(proxies_doc, 'html.parser')
            proxies_table = soup.find(id='proxylisttable')

# Save proxies in the array
            for row in proxies_table.tbody.find_all('tr'):
                proxies.append({
                'ip':   row.find_all('td')[0].string,
                'port': row.find_all('td')[1].string
              })
        
        
        proxy_index = random_proxy()
        proxy = proxies[proxy_index]
        p = {
         "http": "http://"+proxy['ip']+":"+proxy['port'],
         "https": "https://"+proxy['ip']+":"+proxy['port'],
        }
        translator = Translator(proxies = p, timeout = 3)


write_json(write_dict_path, dictionary)
        