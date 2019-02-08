"""
@author Lobarr
"""

import os
import urllib.request as request

def parse(url, output_fp, labels):
  input_fp = './temp.data'
  request.urlretrieve(url, input_fp)
  with open(input_fp, 'r') as input_f, open(output_fp, 'w') as output_f:
    output_f.write(labels)
    lines = input_f.readlines()
    for line in lines:
      parsed = ','.join([str(x) for x in [line.split('\t')[0].split(' ')][0] if x is not ''] + [line.split('\t')[1].rstrip('\n').replace('"', '')])+'\n'
      output_f.write(parsed)
  os.remove(input_fp)

if __name__ == "__main__":
  url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/auto-mpg/auto-mpg.data'
  output_fp = './auto-mpg.csv'
  labels = 'mpg,cylinders,displacement,horsepower,weight,acceleration,model_year,origin,car_name\n'
  parse(url, output_fp, labels)