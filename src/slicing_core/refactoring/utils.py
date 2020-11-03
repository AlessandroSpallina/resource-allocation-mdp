class Config:
    def __init__(self, ):
        self._config = conf # set it to conf

    def get_property(self, property_name):
        if property_name not in self._config.keys(): # we don't want KeyError
            return None  # just return None if not found
        return self._config[property_name]