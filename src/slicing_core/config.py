import confuse
import time
from copy import copy
import json
import hashlib

CONFIG_FILE_PATH = "config.yaml"
EXPORTED_FILES_PATH = f"./exported/results/{int(time.time())}/"
POLICY_CACHE_FILES_PATH = "./exported/policy_cache/"
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
            'alpha': int,
            'beta': int,
            'gamma': int,
            'c_job': float,
            'c_server': float,
            'c_lost': float
        }
    )
}

# for scientific notation issues see https://github.com/beetbox/confuse/issues/91


class Config:
    def __init__(self, custom_path="", config_processor=None):
        config = confuse.Configuration('SlicingCore')
        config.set_file(CONFIG_FILE_PATH if len(custom_path) == 0 else custom_path)
        self._validated = config.get(template)
        if config_processor is not None:
            config_processor()

    @property
    def hash(self):
        tmp = vars(self)
        tmp.pop('_validated', None)
        return hashlib.sha256(json.dumps(tmp).encode('UTF8')).hexdigest()

    def get_property(self, property_name):
        path = property_name.split("/")
        path.reverse()
        value = self._validated[path.pop()]
        while len(path) > 0:
            value = value[path.pop()]
        return value


class PolicyConfig(Config):
    """ Configuration parameters of a Multi-Slice System """
    def __init__(self, custom_path=""):
        super().__init__(custom_path)
        self._algorithm = self.get_property('mdp/algorithm')
        self._discount_factor = self.get_property('mdp/discount_factor')
        self._immediate_action = self.get_property('immediate_action')
        self._arrival_processing_phase = self.get_property('arrival_processing_phase')
        self._timeslots = self.get_property('simulation/timeslots')
        self._slice_count = len(self.get_property('slices'))
        self._server_max_cap = self.get_property('server_max_cap')
        self._slices = self.get_property('slices')

        self._normalize_alpha_beta_gamma()

    @property
    def algorithm(self):
        return self._algorithm

    @property
    def discount_factor(self):
        return self._discount_factor

    @property
    def immediate_action(self):
        return self._immediate_action

    @property
    def arrival_processing_phase(self):
        return self._arrival_processing_phase

    @property
    def timeslots(self):
        return self._timeslots

    @property
    def slice_count(self):
        return self._slice_count

    @property
    def server_max_cap(self):
        return self._server_max_cap

    @server_max_cap.setter
    def server_max_cap(self, n):
        self._server_max_cap = n

    @property
    def slices(self):
        return self._slices

    def slice(self, index):
        """Returns the config parameters of a slice (single-slice policy)"""
        to_ret = copy(self)
        for key in self._slices[index]:
            setattr(to_ret, key, self._slices[index][key])
        to_ret._slices = None
        return to_ret

    def _normalize_alpha_beta_gamma(self):
        for i in range(self._slice_count):
            alpha = self._validated['slices'][i]['alpha']
            beta = self._validated['slices'][i]['beta']
            gamma = self._validated['slices'][i]['gamma']

            self._validated['slices'][i]['alpha'] = alpha / (alpha + beta + gamma)
            self._validated['slices'][i]['beta'] = beta / (alpha + beta + gamma)
            self._validated['slices'][i]['gamma'] = gamma / (alpha + beta + gamma)


class SimulationConfig(Config):
    def __init__(self, custom_path=""):
        super().__init__(custom_path)
        self._immediate_action = self.get_property('immediate_action')
        self._arrival_processing_phase = self.get_property('arrival_processing_phase')
        self._timeslots = self.get_property('simulation/timeslots')
        self._runs = self.get_property('simulation/runs')
        self._slice_count = len(self.get_property('slices'))
        self._server_max_cap = self.get_property('server_max_cap')
        self._slices = self.get_property('slices')

        self._normalize_alpha_beta_gamma()

    @property
    def immediate_action(self):
        return self._immediate_action

    @property
    def arrival_processing_phase(self):
        return self._arrival_processing_phase

    @property
    def timeslots(self):
        return self._timeslots

    @property
    def runs(self):
        return self._runs

    @property
    def slice_count(self):
        return self._slice_count

    @property
    def server_max_cap(self):
        return self._server_max_cap

    @property
    def slices(self):
        return self._slices

    def slice(self, index):
        """Returns the config parameters of a slice (single-slice env)"""
        to_ret = copy(self)
        for key in self._slices[index]:
            setattr(to_ret, key, self._slices[index][key])
        to_ret._slices = None
        return to_ret

    def _normalize_alpha_beta_gamma(self):
        for i in range(self._slice_count):
            alpha = self._validated['slices'][i]['alpha']
            beta = self._validated['slices'][i]['beta']
            gamma = self._validated['slices'][i]['gamma']

            self._validated['slices'][i]['alpha'] = alpha / (alpha + beta + gamma)
            self._validated['slices'][i]['beta'] = beta / (alpha + beta + gamma)
            self._validated['slices'][i]['gamma'] = gamma / (alpha + beta + gamma)
