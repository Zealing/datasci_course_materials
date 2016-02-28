import MapReduce
import sys

"""
Word Count Example in the Simple Python MapReduce Framework
"""

mr = MapReduce.MapReduce()

# =============================
# Do not modify above this line


MATDIM = 5

def mapper(record):
    key     = record[0]
    i       = record[1]
    j       = record[2]
    value   = record[3]

    if key == "a":
        mr.emit_intermediate(key, [i,j,value])
    elif key == "b":
        mr.emit_intermediate(key, [j,i,value])
    else:
        print "Error."

def reducer(key, list_of_values):
    A = {}
    B = {}
    result = 0
    if key == "a":
       for a in list_of_values:
           A[(a[0], a[1])] = a[2]
       for b in mr.intermediate["b"]:
           B[(b[0], b[1])] = b[2]
       # fill in zeros
       for i in range(0,MATDIM):
           for j in range(0,MATDIM):
               k = (i,j) 
               if k not in A.keys():
                  A[k] = 0
               if k not in B.keys():
                  B[k] = 0
       # now do the multiply.
       for i in range(0,MATDIM):
         for j in range(0,MATDIM):
           result = 0
           for k in range(0,MATDIM):
               result += A[(i,k)] * B[(j,k)]
           mr.emit((i,j,result))

# Do not modify below this line
# =============================
if __name__ == '__main__':
  inputdata = open(sys.argv[1])
  mr.execute(inputdata, mapper, reducer)