#!/usr/bin/env python3

import requests
from bs4 import BeautifulSoup
import json
import urllib

headers = {'user-agent': 'Mozilla/5.0 (X11; Linux x86_64; rv:50.0) Gecko/20100101 Firefox/50.0'}


queries = ["Happy human face", "Sad human face", "Angry human face", "Disgusted human face", "Surprised human face", "Fearful human face",
        "Счастливое лицо", "Печальное лицо", "Злое лицо", "Лицо отвращение", "Удивленное лицо", "Испуганное лицо"
        ]

class ImagesUrlFetcher (object):
    def generateRequest(self, query, from_, num):
        pass

    def getLinks(self, query, count):
        pass


class GoogleImagesUrlFetcher(ImagesUrlFetcher):
    request = "https://www.google.ru/search?async=_id:rg_s,_pms:s&ei=35miWKGKF8KKsAGipoCABQ&yv=2&q={query}&start={start}&asearch=ichunk&newwindow=1&tbm=isch&ijn=3"
    num_per_request = 100

    def generateRequest(self, query, from_, num=0):
        return self.request.format(query=query, start=from_, yv=from_//self.num_per_request)
    
    def getLinks(self, query, count):
        links = set()
        q = urllib.parse.quote_plus(query)
        for i in range(count // self.num_per_request):
            r = self.request.format(query=q, start=i * self.num_per_request, yv=i)
            response = requests.get(r, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.json()[1][1], 'html.parser')
                divs = soup.find_all('div', class_ = 'rg_meta')
                links.update(map(lambda div: json.loads(div.text)['ou'], divs))
        return links


class YandexImagesUrlFetcher(ImagesUrlFetcher):
    request = "https://yandex.ru/images/search?text={query}&p={p}"
    num_per_request = 30

    def generateRequest(self, query, from_, num=0):
        return self.request.format(query=query, p=from_)

    def getLinks(self, query, count):
        links = set()
        q = urllib.parse.quote_plus(query)
        for i in range(count // self.num_per_request):
            print(i)
            r = self.generateRequest(q, i)
            response = requests.get(r, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                divs = soup.find_all('div', class_='serp-item')
                links.update(map(lambda div: json.loads(div.attrs['data-bem'])['serp-item']['img_href'], divs))
                print (len(links))
        return links


class BingImagesUrlFetcher(ImagesUrlFetcher):
    request = "http://www.bing.com/images/async?q={query}&first={start}&count=100"
    num_per_request = 100

    def generateRequest(self, query, from_, num=0 ):
        return self.request.format(query=query, start = from_)

    def getLinks(self, query, count):
        links = set()
        q = urllib.parse.quote_plus(query)
        for i in range(count // self.num_per_request):
            print (i)
            r = self.generateRequest(q, i*self.num_per_request)
            response = requests.get(r, headers=headers)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                a = soup.find_all('a', class_='iusc')
                links.update(map(lambda div: json.loads(div.attrs['m'])['murl'], a))
        return links

if __name__ == '__main__':
    count = 1000
    giuf = GoogleImagesUrlFetcher()
    yiuf = YandexImagesUrlFetcher()
    biuf = BingImagesUrlFetcher()
    for query in queries:
        links = set()
        with open (query+".urls", "w") as fout:
            print (query)
            links.update(yiuf.getLinks (query, count))
            print (len(links))
            links.update (giuf.getLinks(query, count))
            print (len(links))
            links.update (biuf.getLinks(query, count))
            print (len(links))
            for link in links:
                fout.write(link)
                fout.write("\n")
