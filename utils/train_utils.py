
def aggregate_loss_dict(agg_loss_dict):
	mean_vals = {}
	for output in agg_loss_dict:
		for key in output:
			mean_vals[key] = mean_vals.setdefault(key, []) + [output[key]]
	for key in mean_vals:
		if len(mean_vals[key]) > 0:
			mean_vals[key] = sum(mean_vals[key]) / len(mean_vals[key])
		else:
			print('{} has no value'.format(key))
			mean_vals[key] = 0
	return mean_vals


def add_dict(dictionary, key, value):
	for i, v in enumerate(value):
		dictionary[i][key] = v
	return dictionary

def add_dict_from_args(dictionary, key, value):
	dictionary[key] = value
	return dictionary