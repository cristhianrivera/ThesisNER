from __future__ import print_function
import argparse
import os


def remove_crap(input_file):
    f = open(input_file)
    lines = f.readlines()
    start_docs = 0
    l = list()
    for line in lines:
        #line = line.decode('latin-1')
        if "-DOCSTART-" in line:
            start_docs = start_docs +1
            pass
        else:  
            l.append(line)
    ff = open('temp.txt', 'w')
    ff.writelines(l)
    ff.close()



def modify_data_size(output_file, trim):
    final_list = list()
    l = list()
    temp_len = 0
    count = 0
    for line in open('temp.txt', 'r'):
        if line in ['\n', '\r\n']:
            line = line.decode('latin-1')
            if temp_len == 0:
                l = []
            elif temp_len > trim:
                count += 1
                l = []
                temp_len = 0
            else:
                l.append(line)
                final_list.append(l)
                l = []
                temp_len = 0
        else:
            l.append(line)
            temp_len += 1
    f = open(output_file, 'w')
    for i in final_list:
        f.writelines(i)
    f.close()
    print('%d sentences trimmed out of %d total sentences' % (count, len(final_list)))
    os.system('rm temp.txt')

"""
parser = argparse.ArgumentParser()
parser.add_argument('--input', type=str, help='input file location', required=True)
parser.add_argument('--output', type=str, help='output file location', required=True)
parser.add_argument('--trim', type=int, help='trimmed sentence length', required=True)
args = parser.parse_args()
"""



