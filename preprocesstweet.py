# coding: utf8
from __future__ import unicode_literals

import redis
import sys
import re
import json



def json2redis(filename,database):
    with open(filename, 'r') as f:
        for line in f:
            m = re.search(".",line) # Permet D'éviter le bug lorsqu'il y a un saut de ligne    
            if m != None :
                try :                
                    tweet = json.loads(line)
                    date = json.dumps(tweet['created_at'])
                    database.hset(filename,date,tweet)
                except:
                    print('failed try')
                    pass

def getTweetText(tweet):
    print('getTweetText')
    #tweet = json.loads(tweet)
    text = json.dumps(tweet['text']) # récupere le texte du tweet
    return text

def redis2json(hashname, database):
    jsonfile = database.hvals(hashname)
    return jsonfile

def json2tweet(line):
    print('json2tweet')
    tweet = json.loads(line)
    return tweet

basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)
print(sys.argv[1])  
filename = sys.argv[1]
json2redis(filename,basejson)
'''print('scan')
print(basejson.hscan(filename))
print('getall')
print(basejson.hgetall(filename))
'''
jsonfile = redis2json(filename,basejson)

for line in jsonfile:
    
    print('line')
    print(line)
    # tweet = json2tweet(line)
    text = getTweetText(line)
    print(text)
    
    
