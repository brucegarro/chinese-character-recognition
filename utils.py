def conv_output_width(input_size, kernel_size, stride, padding):
	output_size = (((input_size + 2*padding - kernel_size) / float(stride)) + 1)

	try:
		assert int(output_size) == output_size
	except AssertionError:
		raise ValueError("output_size: %s is not valid int" % output_size)

	return int(output_size)


def pool_output_width(input_size, kernel_size, stride):
	output_size = ((input_size - kernel_size) / stride) + 1

	try:
		assert int(output_size) == output_size
	except AssertionError:
		raise ValueError("output_size: %s is not valid int" % output_size)

	return int(output_size)