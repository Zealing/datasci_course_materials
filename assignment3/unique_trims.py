import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line

def mapper(record):
    # Since we're only using the sequence, we'll just trim the last ten
    # characters 
    trimmed_sequence = (record[1])[:-10]
    mr.emit_intermediate(trimmed_sequence, 0)

def reducer(sequence, list_of_values):
    mr.emit((sequence))  # No, really, that's it.


# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)