import numpy

#activation function
def sigmoid(x):
    return 1 / (1 + numpy.exp(-x))

Input_neyrons = numpy.array([[0,0,1],
                      [1,1,1],
                      [1,0,1],
                      [0,1,1]])
# changing wide and level
Output_Neyrons = numpy.array([[0,1,1,0]]).T

numpy.random.seed(1)
#level , wide , 0 to 1
synaptic_weights = 2 * numpy.random.random((3,1)) - 1

print('Random synaptic weights: ')
print(synaptic_weights)

# reverse teaching
for  i in range(40000):
    input_layer = Input_neyrons
    # Скалярное произведение двух массивов(Матриц)
    outputs = sigmoid( numpy.dot(input_layer, synaptic_weights))
    er = Output_Neyrons - outputs

    adjustments = numpy.dot(input_layer.T, er * (outputs * (1 - outputs)))
    synaptic_weights += adjustments

print('Weight after teaching: ')
print(synaptic_weights)

print('Result after teaching ')
print(outputs)