import confuse
import time
from copy import copy
import json
import hashlib

CONFIG_FILE_PATH = "config.yaml"
POLICY_CACHE_FILES_PATH = "./exported/policy_cache/"

EXPORTED_FILES_PATH = f"./exported/results/{int(time.time())}/"
LOG_FILENAME = "report.log"
RESULTS_FILENAME = "results.data"


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


class SlicingConfig(Config):
    def __init__(self, custom_path=""):
        super().__init__(custom_path)
        self.slice_count = len(self.get_property('slices'))
        self.server_max_cap = self.get_property('server_max_cap')
        self.slices = self.get_property('slices')

        self._normalize_alpha_beta_gamma()

    def slice(self, index):
        """Returns the config parameters of a slice (single-slice policy)"""
        to_ret = copy(self)
        for key in self.slices[index]:
            setattr(to_ret, key, self.slices[index][key])
        to_ret.slices = None
        to_ret.slice_count = None
        return to_ret

    def _normalize_alpha_beta_gamma(self):
        for i in range(self.slice_count):
            alpha = self._validated['slices'][i]['alpha']
            beta = self._validated['slices'][i]['beta']
            gamma = self._validated['slices'][i]['gamma']

            self._validated['slices'][i]['alpha'] = alpha / (alpha + beta + gamma)
            self._validated['slices'][i]['beta'] = beta / (alpha + beta + gamma)
            self._validated['slices'][i]['gamma'] = gamma / (alpha + beta + gamma)


class MdpPolicyConfig(SlicingConfig):
    def __init__(self, custom_path=""):
        super().__init__(custom_path)
        self.algorithm = self.get_property('mdp/algorithm')
        self.discount_factor = self.get_property('mdp/discount_factor')
        self.immediate_action = self.get_property('immediate_action')
        self.arrival_processing_phase = self.get_property('arrival_processing_phase')
        if self.algorithm == 'fh':
            self.timeslots = self.get_property('simulation/timeslots')


class StaticPolicyConfig(SlicingConfig):
    def __init__(self, custom_path=""):
        super().__init__(custom_path)

        allocations = self._eq_div(self.server_max_cap, self.slice_count)
        allocations.reverse()

        for slice_i in self.slices:
            slice_i['allocation'] = allocations.pop()

    def _eq_div(self, what, who):
        return [] if who <= 0 else [what // who + 1] * (what % who) + [what // who] * (who - what % who)


class SimulatorConfig(SlicingConfig):
    def __init__(self, custom_path=""):
        super().__init__(custom_path)
        self.immediate_action = self.get_property('immediate_action')
        self.arrival_processing_phase = self.get_property('arrival_processing_phase')
        self.timeslots = self.get_property('simulation/timeslots')
        self.runs = self.get_property('simulation/runs')


