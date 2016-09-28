# Model Session: an Object Pattern for TensorFlow

Model session is an object-oriented pattern that makes it easier to run
and serialize TensorFlow models.
It streamlines graph and session management and serialization and
deserialization, allowing you to focus on the details of your model.


## Running the Example

This project contains a `ModelSession` base class that defines the pattern
and an example `MNISTModelSession` that implements it for an MNIST digits classifier.
To train an MNIST model run

    python mnist_model.py train

This will download MNIST data and train a model, serializing it in a
`checkpoint` directory.
If you run this command again, it will reload the serialized model and
continue training from where it left off.

Once a model has been trained, you can run it on a test set with

    python mnist_model.py test

See `python mnist_model.py --help` for more options.


## Discussion of the Example

The MNIST model is a two-layer convolutional network with dropout
based on the example in the [TensorFlow tutorial](https://www.tensorflow.org/versions/r0.10/tutorials/mnist/pros/index.html).
It has four parameters:

* The sizes of the first and second convolutional layers
* The learning rate
* The keep probability for the fully-connected layer

We wish to specify each parameter only once and have its value read back
transparently during deserialization.
The model session pattern allows each to be specified at the point
in the model's lifecycle when it is required.

The sizes of the two convolutional layers are passed to the `ModelSession.create` 
method that is used to create a new model.
These become part of the the model's structure and do not need to be
specified again when the model is deserialized.
The learning rate, on the other hand, only needs to be specified at
training time.
Furthermore, we would like to be able to change its value during training,
so it is a parameter of the `MNISTModelSession.train` method and passed
into the model via a feed dictionary.
Similarly, the keep probability takes on a user-specified value during training
and the value 1 during testing, so it too is a parameter of the `train`
method and passed in to the model via a feed dictionary.


## Using the Model Session Pattern

To use the model session pattern, inherit from the `ModelSession` base
class and create your graph in a method overriding its static `create_graph` method.
You should never call a `ModelSession` constructor directly.
Instead call the `create` and `restore` class methods when you are creating
a new model or reloading a serialized one, respectively.
Code in your model should refer to elements of your model's graph by name,
since this information is reloaded during deserialization.
If your graph contains a variable called `iteration` its value will be
used as the `global_step` decoration of the checkpoint file whenever a
model is saved.
