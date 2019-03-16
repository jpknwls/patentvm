import numpy as np
import optparse

parser = optparse.OptionParser()
parser.add_option('-t', '--test',
    action="store", dest="test",
    help="to run test", default="toy")
parser.add_option('-m', '--metric',
    action="store", dest="metric",
    help="metric to calculate", default="mrr")


# calculates the inverse rank for single query
def inv_rank(node, labels, predictions):
	labels = set(labels)
	for index, node_id in enumerate(predictions):
		if node_id in labels:
			return float(1)/(float(index)+1)
	print("Didn't find any true neighbors in predictions for node id {}".format(node))
	return -1.0

# calculates mean reciprical rank for query set
def MRR(nodes, labels, predictions):
	Q = len(labels)
	RR = 0.0
	for i in range(Q):
		curr_rank = inv_rank(nodes[i], labels[i], predictions[i]) 
		#print(curr_rank)
		RR += curr_rank

	return (1.0/Q)*RR

# toy example from https://en.wikipedia.org/wiki/Mean_reciprocal_rank
def test():
	CATS = "cats"
	TORI = "tori"
	VIRUSES = "viruses"

	q1 = ["catten", "cati",CATS]
	q2 = ["torii", TORI, "toruses"]
	q3 = [VIRUSES, "virii", "viri"]
	predictions = [q1,q2,q3]
	labels = [[CATS],[TORI],[VIRUSES]]
	nodes = [CATS, TORI, VIRUSES]
	mrr = MRR(nodes, labels, predictions)
	print("MRR: {}".format(mrr))


if __name__ == "__main__":
	options, args = parser.parse_args()
	if options.test == "toy":
		test()