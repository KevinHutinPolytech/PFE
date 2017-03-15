# coding: utf8
from __future__ import unicode_literals
############################################################################
# RECUPERER DES TWEET
#########################################################################

from __future__ import absolute_import, print_function

import tweepy
import sys
import string
import time
from tweepy import Stream
from tweepy.streaming import StreamListener

 
consumer_key = 'STuwMlRcOAM4x11tvFnhrNfov'
consumer_secret = 'Ow12PHjNB5IkErNB6PrIaYynqIwk9Z4XkRCRlcmXqaLlA19NUd'
access_token = '828603299993645057-pTEQy5rv2ZnOSjvTnnDLLED4KRPkEb7'
access_secret = 'XX4GnbeXW9RSfNjqyDYu9hXkr1ZgKojEaCu0BUMPQh6To'
 
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.secure = True
auth.set_access_token(access_token, access_secret)
 
api = tweepy.API(auth)


#############################################################################"""
#STREAMING DE TWEET
#############################################################################

from tweepy import Stream
from tweepy.streaming import StreamListener
 
class MyListener(StreamListener):

    def __init__(self, fname):
        safe_fname = format_filename(fname)
        self.outfile = "%s.json" % safe_fname
        self.count = 0
    def on_data(self, data):
        try:
            with open(self.outfile, 'a') as f:
                self.count = self.count + 1
                print(data)
                print(type(data))
                f.write(data)
                print(self.count)
                return True
        except BaseException as e:
            print("quot;Error on_data: %s&quot;" % str(e))
            return True
 
    def on_error(self, status):
        print(status)
        return True


def format_filename(fname):
    """Convert fname into a safe string for a file name.
    Return: string
    """
    return ''.join(convert_valid(one_char) for one_char in fname)


def convert_valid(one_char):
    """Convert a character into '_' if "invalid".
    Return: string
    """
    valid_chars = "-_.%s%s" % (string.ascii_letters, string.digits)
    if one_char in valid_chars:
        return one_char
    else:
        return ''


query = sys.argv[1:] # list of CLI arguments
#query = "Hamon"
print(query)
print(type(query))
if query == '-h' :
    print('Passez en paramètre la clé à tracker sur tweeter.')
    print("Le fichier de sorti s'appellera stream_'lenomdelaclé'")
    print("(les caractère spéciaux sont convertit en '_' dans le nom de fichier)")
    
query_fname = ' '.join(query) # string
twitter_stream = Stream(auth, MyListener(query_fname))
twitter_stream.filter(track=query, async=True)
