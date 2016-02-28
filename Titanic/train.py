import csv as csv
import numpy as np 

csv_file_object = csv.reader(open('./train.csv', 'rb'))
header = csv_file_object.next()

data = []
for row in csv_file_object:
	data.append(row)
data = np.array(data)

print header

num_Passenger = data[0::,1].astype(np.float).size
num_survived = np.sum(data[0::,1].astype(np.float))
prop = num_survived / num_Passenger

women_only_stats = data[0::, 4] == 'female'
men_only_stats = data[0::, 4] == 'male'

# women_stat = data[women_mask, 1].astype(np.float)
# women_survived = np.sum(women_stat)
# total_women = women_stat.size

# men_stat = data[men_mask,1].astype(np.float)
# men_survived = np.sum(men_stat)
# total_men = men_stat.size

# survived_prop_women = women_survived / total_women
# survived_prop_men = men_survived / total_men

fare_ceiling = 40
data [ data[0::, 9].astype(np.float) >= fare_ceiling, 9 ] = fare_ceiling - 1.0

fare_bracket_size = 10
num_price_brackets = fare_ceiling / fare_bracket_size

num_classes = len(np.unique(data[0::,2]))

# only care about shape: 2 dummy sex * 3 classes * 4 brackets
survival_table = np.zeros((2, num_classes, num_price_brackets))

for i in xrange(num_classes):
	for j in xrange(num_price_brackets):

		women_only_stats = data[
								(data[0::,4] == 'female')
								& (data[0::,2].astype(np.float) == i + 1)
								& (data[0::,9].astype(np.float) >= j*fare_bracket_size)
								& (data[0::,9].astype(np.float) < (j+1)*fare_bracket_size)
								, 1]

		men_only_stats = data[
								(data[0::,4] == 'male')
								& (data[0::,2].astype(np.float) == i + 1)
								& (data[0::,9].astype(np.float) >= j*fare_bracket_size)
								& (data[0::,9].astype(np.float) < (j+1)*fare_bracket_size)
								, 1]

		survival_table[0,i,j] = np.mean(women_only_stats.astype(np.float))
		survival_table[1,i,j] = np.mean(men_only_stats.astype(np.float))

# deal with nan problem
survival_table[ survival_table != survival_table ] = 0
survival_table[ survival_table < 0.5] = 0
survival_table [ survival_table >= 0.5] = 1

test_file = open('./test.csv', 'rb')
test_file_object = csv.reader(test_file)
header = test_file_object.next()

pred_file = open('genderclassmodel.csv', 'wb')
p = csv.writer(pred_file)
p.writerow(["PassengerId", "Survived"])

# go through each passenger in the test set to find its survival label
for row in test_file_object:
	for j in xrange(num_price_brackets):
		try:
			row[8] = float(row[8])
		except:
			bin_fare = 3 - float(row[1])
			break
		if row[8] > fare_ceiling:
			bin_fare = num_price_brackets - 1
			break
		if row[8] >= j * fare_bracket_size and row[8] < (j + 1) * fare_bracket_size:
			bin_fare = j
			break

	if row[3] == 'female':
		p.writerow([row[0], "%d" %int(survival_table[0,float(row[1])-1, bin_fare ])])
	else:
		p.writerow([row[0], "%d" %int(survival_table[0,float(row[1])-1, bin_fare ])])

test_file.close()
pred_file.close()

# print survival_table








