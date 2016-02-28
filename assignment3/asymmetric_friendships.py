import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # key: person A
    # value: person B
    key = record[0]
    value = record[1]
    # attibute = value.split()
    mr.emit_intermediate(key, value)

def reducer(key, list_of_values):
    # key: person A
    # list_of_values: friend lists of person A
    for so_called_friend in list_of_values:
      if so_called_friend not in mr.intermediate.keys() or key not in mr.intermediate[so_called_friend]:
        mr.emit((key, so_called_friend))
        mr.emit((so_called_friend, key))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)