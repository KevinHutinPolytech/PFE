# coding: utf8
from __future__ import unicode_literals

import redis
import sys
import re
import json

basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)

print(sys.argv[1])
filename = sys.argv[1]
#filename = 'stream__Macron.json'

#### VERSION 1
count = 0
file = {}
with open(filename, 'r') as f:
    for line in f:
        m = re.search(".",line) # Permet D'éviter le bug lorsqu'il y a un saut de ligne    
        if m != None :
            try :
                count = count + 1
                countstr = '%s'  %count
                file[count] = line
               # tweet = json.loads(line)
               # date = json.dumps(tweet['created_at'])
                
            except:
                print('failed try')
                pass
#print(basejson.hscan(file))
print(file)
print('/n')
print('/n')

print(file[1])
print('/n')
print('/n')

tweet = json.loads(file[1])
print(json.dumps(tweet['text']))

basejson.set(filename,file)
print(basejson.get(filename))
'''
#### Version 2
f=open(filename,'r')
s=f.read()
pyObject = json.loads(s)
basejson.set(filename,pyObject)
'''
file = basejson.get(filename)
tweet = json.loads(file[1])
print(json.dumps(tweet['text']))

for line in file:
    print('line')
    print (line)
    tweetText= json.dumps(line['text'])
    print('tweetText')
    print(tweetText)
    
