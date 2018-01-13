
# coding: utf-8

# In[1]:

#prepare netflix data as an input to to cuMF
#data should be in ./data/netflix/
#assume input is given in text format
#each line is like 
#"user_id item_id rating"
import os
import pandas as pd
from six.moves import urllib
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import csc_matrix
from scipy.sparse import coo_matrix
from scipy import sparse
from sklearn.cross_validation import train_test_split


# In[2]:

# Step 1: Download the data.
url = 'http://files.grouplens.org/datasets/movielens/'

def maybe_download(filename, expected_bytes):
  """Download a file if not present, and make sure it's the right size."""
  if not os.path.exists(filename):
    filename, _ = urllib.request.urlretrieve(url + filename, filename)
  statinfo = os.stat(filename)
  if statinfo.st_size == expected_bytes:
    print('Found and verified', filename)
  else:
    print(statinfo.st_size)
    raise Exception(
        'Failed to verify ' + filename + '. Can you get to it with a browser?')
  return filename

data_file = maybe_download('ml-10m.zip', 65566137)


# In[3]:

# ADRIAN: UNCOMMENT THIS LINE
#os.system(u'unzip -o ml-10m.zip')

#!cd ./ml-10M100K && ./split_ratings.sh


# In[4]:

#file look like
'''
1::122::5::838985046
1::185::5::838983525
1::231::5::838983392
1::292::5::838983421
1::316::5::838983392
1::329::5::838983392
1::355::5::838984474
1::356::5::838983653
1::362::5::838984885
1::364::5::838983707
'''
m = 71567
n = 65133


# In[5]:

user,item,rating, ts = np.loadtxt('ml-10M100K/ratings.dat', delimiter='::', dtype=np.int32,unpack=True)
print(user)
print(item)
print(rating)
print(np.max(user))
print(np.max(item))
print(np.max(rating))
print(user.size)


# In[6]:

user_item = np.vstack((user, item))


# In[7]:

user_item_train, user_item_test, rating_train, rating_test = train_test_split(user_item.T, rating, test_size=1000006, random_state=42)
nnz_train = 9000048
nnz_test = 1000006
print(user_item_train.shape)
print(user_item_test.shape)
print(rating_train.shape)
print(rating_test.shape)

result_train = np.vstack((user_item_train.T, rating_train))
result_test = np.vstack((user_item_test.T, rating_test))
np.savetxt('ml10M_train', result_train.T, delimiter=' ', newline='\n', fmt='%i')
np.savetxt('ml10M_test', result_test.T, delimiter=' ', newline='\n', fmt='%i')
