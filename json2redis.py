# coding: utf8
from __future__ import unicode_literals

import redis
import sys
import re

basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)

print(sys.argv[1])
filename = sys.argv[1]

with open(filename, 'r') as f:
    for line in f:
        m = re.search(".",line) # Permet D'éviter le bug lorsqu'il y a un saut de ligne    
        if m != None :
            try :
                tweet = json.loads(line)
                print('tweet')
                date = json.dumps(tweet['created_at'])
                basejson.hset(filename,date,tweet)
            except:
                print('failed try')
                pass
print(basejson.hscan(filename))


