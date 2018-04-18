import os
import csv
from itertools import chain
import pprint


"""
William Murphy
4/18/2018
"""

# using  sequential decorators to mimic a data pipeline structure
# this acts as a (extract->transform->load) process
# So far, this really only demonstrates the extract portion
# I'm avoiding pandas on purpose so as to demonstrate what can be done with pure python

def extract(f):
    def wrapper(path, *args, **kwargs):

        files = f(path, *args, **kwargs)

        dirname = os.path.dirname(path)
        if os.getcwd() is not dirname:
            try:
                os.chdir(dirname)
                print(os.getcwd())
            except FileExistsError as e:
                print(str(e))
            except FileNotFoundError as e:
                print(str(e))
            except IOError as e:
                print(str(e))
            except Exception as e:
                print(str(e))

        # --- walk through files to find files of interest
        dir_files = [l for _, _, l in os.walk(dirname) if len(l) > 0]
        # --- flatten dir_files to a single list, rather than nested lists
        flattened_files = list(chain(*dir_files))
        reduce_dir_files = [a for a in args if a in flattened_files]
        return reduce_dir_files
    return wrapper

def extract_to_csv(f):
    def wrapper(*args, **kwargs):

        files = f(*args, **kwargs)
        CSV = list(map(lambda x: str(x), files))
        CSV_FILES = list()
        header = 0
        for _csv in CSV:
            with open(str(_csv), 'r') as file:
                reader = csv.reader(file)
                for row in reader:
                   CSV_FILES.append(row)

        return CSV_FILES
    return wrapper


@extract_to_csv
@extract
def set_files(path, *args):
    p = path
    fl = [str(a) for a in args]
    return (p, fl)

pprint.pprint(set_files("E:\\", "^OVX.csv", "^VIX.csv", "^SP500TR.csv"))
