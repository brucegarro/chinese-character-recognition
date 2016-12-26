from load import load_hsk_data, reformat

data = load_hsk_data()

train_data, train_labels = reformat(data["train_data"], data["train_labels"])
valid_data, valid_labels = reformat(data["valid_data"], data["valid_labels"])
test_data, test_labels = reformat(data["test_data"], data["test_labels"])

print "Training set: %s, %s" % (train_data.shape, train_labels.shape)
print "Validation set: %s, %s" % (valid_data.shape, valid_labels.shape)
print "Test set: %s, %s" % (test_data.shape, test_labels.shape)