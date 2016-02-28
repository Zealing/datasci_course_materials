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
    # value: the number of friend --> 1 all the way
    key = record[1]
    value = record
    # attibute = value.split()
    # -------- for every tuple in the same record, put all tuple attributs as value and order id as key
    # for w in words:
    mr.emit_intermediate(key, value)

def reducer(key, list_of_values):
    # key: order id
    # value: list of all attribues
    # set the total as an empty array
    l = len(list_of_values)
    for i in range(1, l):
      v = []
      v += list_of_values[0]
      v += list_of_values[i]
      mr.emit(v)

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)