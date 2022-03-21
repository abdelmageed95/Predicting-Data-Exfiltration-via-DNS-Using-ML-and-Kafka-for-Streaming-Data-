
#!pip install tldextract
import tldextract
from collections import Counter
import numpy as np
import pandas as pd 

# generate features from url 

# 1 - Total count of characters in FQDN
def get_FQDN(url):
  count= 0
  for elem in url:
      if elem==".":  
          continue
      else: 
          count += 1
  return count
  
# 2 - Count of characters in subdomain
def generate_subdomain_length (url):
    subdomain, _ , __ =tldextract.extract(url)
    return get_FQDN(subdomain)

# 3 - Count of uppercase characters
def generate_upper(url):
    count = 0
    for elem in url:
        if elem.isupper():
            count += 1
    return count

# 4 - Count the lowercase characters
def generate_lower(url):
    count = 0
    for elem in url:
        if (elem.islower()==True) and (elem.isdigit()==False) :
            count += 1
    return count

# 5 - Count of numerical characters
def generate_numeric(url):
    count = 0
    for elem in url:
        if elem.isnumeric():
            count += 1
    return count

# 6 - get entropy
def entropy(url):
  p, lens = Counter(url), float(len(url))
  return - sum( count/lens * np.log2(count/lens) for count in p.values())

# 7 - Number of special characters; special characters such as dash, underscore, equal sign, space, tab
def generate_special(url): 
   count= 0
   for elem in url:
        if (elem.isalpha()) or (elem.isdigit() or elem == "."):
            continue
        else: 
            count += 1
   return count

# 8 - Number of labels; e.g., in the query name "www.scholar.google.com", there are four labels separated by dots
def generate_labels(url):
  N =len(url.split('.'))
  return N

# 9 - Maximum label length
def generate_labels_max(url):
  return max(get_len_labels(url))

# 10 - Average label length 
def generate_labels_average(url):
  le=get_len_labels(url)
  return sum(le)/len(le)

# 11 - Longest meaningful word over domain length average
def generate_longest_word(url):
    M = generate_labels_max(url)
    lens = get_len_labels(url)
    return url.split('.')[lens.index(max(lens))]

# 12 - Second level domain
def generate_sld(url):
  subdomain,sld,suffix=tldextract.extract(url)
  return  sld 

# 13 - subdomain 
def generate_subdomain(url):
  subdomain,sld,suffix=tldextract.extract(url)
  if len(subdomain) != 0:
      return 1
  else :
      return 0 

def get_len_labels(url):
  return  [len(l) for l in url.split('.')]

# 14 - lengths of sudomain and domain 
def get_len(url):
  subdomain,sld,__=tldextract.extract(url)
  return get_FQDN(subdomain)+get_FQDN(sld)

# construct the dataframe from url (domain)
def construct_df(url):
  d = { "FQDN": get_FQDN(url),
       "subdomain_length" : generate_subdomain_length(url),
       "upper": generate_upper(url),
       "lower": generate_lower(url),
       "numeric": generate_numeric(url),
       "entropy" : entropy(url),
       "special": generate_special(url),
       "labels": generate_labels(url),
       "labels_max": generate_labels_max(url),
       "labels_average" : generate_labels_average(url),
       "longest_word": generate_longest_word(url) , 
       "sld" : generate_sld(url),
       "len" :get_len(url) ,
       "subdomain" : generate_subdomain(url)
  }
  df = pd.DataFrame(d , index = [0])
  return df
