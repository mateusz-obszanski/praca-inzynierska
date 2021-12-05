from marshmallow_dataclass import class_schema
from .experiment_base import ExperimentBase
from .experiment_tsp import ExperimentConfigTSP as _ExperimentTSP


ExperimentTSPSchema = class_schema(_ExperimentTSP)
