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
    key = record[0]
    value = 1
    # attibute = value.split()
    mr.emit_intermediate(key, value)

def reducer(key, list_of_values):
    # key: person A
    # value: list of 1s
    total = 0
    for i in list_of_values:
      total += i
    
    mr.emit((key, total))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)