import redis
import sys

basejson = redis.StrictRedis(host='127.0.0.1',port=6379,db=0)

filename = sys.argv[1]

with open(filename, 'r') as f:
    try :
        tweet = json.loads(line)
        print('tweet')
        date = json.dumps(tweet['created_at'])
        basejson.hset(filename,date,tweet)
    except:
        print('failed try')
        pass
print(basejson.hscan(filename))


