import confuse
import time

CONFIG_FILE_PATH = "config.yaml"
EXPORTED_FILES_PATH = f"./exported/results/{int(time.time())}/"
LOG_FILE_PATH = f"{EXPORTED_FILES_PATH}report.log"
RESULTS_FILE_PATH = f"{EXPORTED_FILES_PATH}results.data"

template = {
    'immediate_action': confuse.OneOf([bool]),
    'arrival_processing_phase': confuse.OneOf([bool]),
    'mdp': {
        'algorithm': confuse.OneOf(['vi', 'fh']),
        'discount_factor': float
        # 'discount_factors': confuse.Sequence(float)
    },
    'simulation': {
        'runs': confuse.Integer(),
        'timeslots': confuse.Integer()
    },
    'server_max_cap': confuse.Integer(),
    'slices': confuse.Sequence(
        {
            'arrivals_histogram': confuse.Sequence(float),
            'server_capacity_histogram': confuse.Sequence(float),
            'queue_size': confuse.Integer(),
            'alpha': float,
            'beta': float,
            'gamma': float,
            'c_job': confuse.Integer(),
            'c_server': confuse.Integer(),
            'c_lost': confuse.Integer()
        }
    )
}


class Config:
    def __init__(self):
        self._config = confuse.Configuration('SlicingCore')
        self._config.set_file(CONFIG_FILE_PATH)
        self._validated = self._config.get(template)

    def get_property(self, property_name):
        path = property_name.split("/")
        path.reverse()
        value = self._validated[path.pop()]
        while len(path) > 0:
            value = value[path.pop()]
        return value


class PolicyConfig(Config):
    @property
    def algorithm(self):
        return self.get_property('mdp/algorithm')

    @property
    def discount_factor(self):
        return self.get_property('mdp/discount_factor')

    @property
    def immediate_action(self):
        return self.get_property('immediate_action')

    @property
    def arrival_processing_phase(self):
        return self.get_property('arrival_processing_phase')

    @property
    def timeslots(self):
        return self.get_property('simulation/timeslots')

    @property
    def slice_count(self):
        return len(self.get_property('slices'))

    @property
    def server_max_cap(self):
        return self.get_property('server_max_cap')

    @property
    def slices(self):
        return self.get_property('slices')


class EnvironmentConfig(Config):
    @property
    def immediate_action(self):
        return self.get_property('immediate_action')

    @property
    def arrival_processing_phase(self):
        return self.get_property('arrival_processing_phase')

    @property
    def timeslots(self):
        return self.get_property('simulation/timeslots')

    @property
    def runs(self):
        return self.get_property('simulation/runs')

    @property
    def slice_count(self):
        return len(self.get_property('slices'))

    @property
    def server_max_cap(self):
        return self.get_property('server_max_cap')

    @property
    def slices(self):
        return self.get_property('slices')

