from bs4 import BeautifulSoup
from collections import Counter
import os

fn_path_train = "./data/training/"
files_train = [f for f in os.listdir(fn_path_train) if os.path.isfile(os.path.join(fn_path_train, f))]
file_tags = {}
tags_set = set()
tags_list = list()

for fn in files_train:
    with open(fn_path_train + fn) as file:
        content = file.read()
    soup = BeautifulSoup(content, "xml")
    tags_type = [tag["TYPE"] for tag in soup.TAGS.find_all()]
    file_tags[fn] = tags_type
    tags_set.update(tags_type)
    tags_list.extend(tags_type)

print(tags_set)
print("Number of unique tags in train data: " + str(len(tags_set)))
print("Number of PHI instances in train data: " + str(len(tags_list)))
print(Counter(tags_list))

fn_path_test = "./data/test-gold/"
files_test = [f for f in os.listdir(fn_path_test) if os.path.isfile(os.path.join(fn_path_test, f))]
file_tags_t = {}
tags_set_t = set()
tags_list_t = list()

for fn in files_test:
    with open(fn_path_test + fn) as file:
        content = file.read()
    soup = BeautifulSoup(content, "xml")
    tags_type_t = [tag["TYPE"] for tag in soup.TAGS.find_all()]
    file_tags_t[fn] = tags_type_t
    tags_set_t.update(tags_type_t)
    tags_list_t.extend(tags_type_t)

print()
print(tags_set_t)
print("Number of unique tags in test data: " + str(len(tags_set_t)))
print("Number of PHI instances in test data: " + str(len(tags_list_t)))
print(Counter(tags_list_t))