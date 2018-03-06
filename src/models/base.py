from theanolayer.utils import scope


class Model(object):

    def __init__(self, parameters, prefix):

        self.parameters = parameters
        self.prefix = prefix

    def model_scope(self, name):
        return scope(self.prefix, name)