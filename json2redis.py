# coding: utf8
from __future__ import unicode_literals

import redis
import sys
import re
import json

basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)

print(sys.argv[1])
filename = sys.argv[1]

count = 0
with open(filename, 'r') as f:
    for line in f:
        m = re.search(".",line) # Permet D'éviter le bug lorsqu'il y a un saut de ligne    
        if m != None :
            try :
                count = count + 1
                '''
                print(count)
                print(type(tweet))
                print(tweet)
                print(tweet.keys())
                print(tweet['id_str'])
                print(tweet['text'])                    
                print('ok')'''
                tweet = json.loads(line)
                basejson.hset(filename,tweet['id_str'],json.dumps(tweet))
            except:
                print('failed try')
                pass
            

print('%s tweets uploads' %(count))
