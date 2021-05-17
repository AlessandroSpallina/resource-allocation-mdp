import confuse
import time
from copy import copy
import json
import hashlib

CONFIG_FILE_PATH = "config.yaml"
POLICY_CACHE_FILES_PATH = "./exported/policy_cache/"
SIMULATION_CACHE_FILES_PATH = "./exported/simulation_cache/"

EXPORTED_FILES_PATH = f"./exported/results/{int(time.time())}/"
LOG_FILENAME = "report.log"
RESULTS_FILENAME = "results.data"


template = {
    'immediate_action': confuse.OneOf([bool]),
    'arrival_processing_phase': confuse.OneOf([bool]),
    'mdp': {
        'algorithm': confuse.OneOf(['vi', 'rvi', 'fh']),
        'discount_factor': float,
        'queue_scaling': confuse.Integer(),
        'normalize_reward_matrix': confuse.OneOf([bool]),
        'loss_expected_pessimistic': confuse.Sequence(bool)
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
            'delta': int,
            'epsilon': int,
            'c_job': float,
            'c_server': float,
            'c_lost': float,
            'c_alloc': float,
            'c_dealloc': float
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
        tmp = copy(self.__dict__)
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

        self._normalize_alpha_beta_gamma_delta_epsilon()

    def slice(self, index):
        """Returns the config parameters of a slice (single-slice policy)"""
        to_ret = copy(self)
        for key in self.slices[index]:
            setattr(to_ret, key, self.slices[index][key])
        # setattr(to_ret, 'loss_expected_pessimistic', self.get_property('mdp/loss_expected_pessimistic')[index])
        del to_ret.slices
        del to_ret.slice_count
        # to_ret.slices = None
        # to_ret.slice_count = None

        return to_ret

    def _normalize_alpha_beta_gamma_delta_epsilon(self):
        for i in range(self.slice_count):
            alpha = self._validated['slices'][i]['alpha']
            beta = self._validated['slices'][i]['beta']
            gamma = self._validated['slices'][i]['gamma']
            delta = self._validated['slices'][i]['delta']
            epsilon = self._validated['slices'][i]['epsilon']

            self._validated['slices'][i]['alpha'] = alpha / (alpha + beta + gamma + delta + epsilon)
            self._validated['slices'][i]['beta'] = beta / (alpha + beta + gamma + delta + epsilon)
            self._validated['slices'][i]['gamma'] = gamma / (alpha + beta + gamma + delta + epsilon)
            self._validated['slices'][i]['delta'] = delta / (alpha + beta + gamma + delta + epsilon)
            self._validated['slices'][i]['epsilon'] = epsilon / (alpha + beta + gamma + delta + epsilon)


class MdpPolicyConfig(SlicingConfig):
    def __init__(self, custom_path=""):
        super().__init__(custom_path)
        self.algorithm = self.get_property('mdp/algorithm')
        self.immediate_action = self.get_property('immediate_action')
        self.arrival_processing_phase = self.get_property('arrival_processing_phase')
        self.queue_scaling = self.get_property('mdp/queue_scaling')
        self.normalize_reward_matrix = self.get_property('mdp/normalize_reward_matrix')
        if self.algorithm == 'fh':
            self.timeslots = self.get_property('simulation/timeslots')
        if self.algorithm != 'rvi':
            self.discount_factor = self.get_property('mdp/discount_factor')

    def slice(self, index):
        to_ret = SlicingConfig.slice(self, index)
        setattr(to_ret, 'loss_expected_pessimistic', self.get_property('mdp/loss_expected_pessimistic')[index])
        return to_ret


class StaticPolicyConfig(SlicingConfig):
    def __init__(self, custom_path=""):
        super().__init__(custom_path)

        allocations = self._eq_div(self.server_max_cap, self.slice_count)
        allocations.reverse()

        for slice_i in self.slices:
            slice_i['allocation'] = allocations.pop()

    def _eq_div(self, what, who):
        return [] if who <= 0 else [what // who + 1] * (what % who) + [what // who] * (who - what % who)

    # very important to use this function from 0 to N, with order!
    def set_allocation(self, slice_id, allocation):
        allocable_servers = self.server_max_cap
        for i in range(slice_id):
            allocable_servers -= self.slices[i].allocation
        if allocation <= allocable_servers:
            self.slices[slice_id]['allocation'] = allocation
        else:
            self.slices[slice_id]['allocation'] = allocable_servers
        remaining_servers = self.server_max_cap - allocation
        remaining_slices = self.slice_count - (slice_id + 1)
        new_allocations = self._eq_div(remaining_servers, remaining_slices)
        new_allocations.reverse()
        for slice_i in range(slice_id + 1, self.slice_count):
            self.slices[slice_i]['allocation'] = new_allocations.pop()


class SimulatorConfig(SlicingConfig):
    def __init__(self, custom_path=""):
        super().__init__(custom_path)
        self.immediate_action = self.get_property('immediate_action')
        self.arrival_processing_phase = self.get_property('arrival_processing_phase')
        self.timeslots = self.get_property('simulation/timeslots')
        self.runs = self.get_property('simulation/runs')


