from marshmallow_dataclass import class_schema
from .experiment_base import ExperimentBase
from .experiment_tsp import ExperimentConfigTSP as _ExperimentTSP
from .experiment_vrp import ExperimentConfigVRP as _ExperimentVRP
from .experiment_vrpp import ExperimentConfigVRPP as _ExperimentVRPP
from .experiment_irp import ExperimentConfigIRP as _ExperimentIRP


ExperimentTSPSchema = class_schema(_ExperimentTSP)
ExperimentVRPSchema = class_schema(_ExperimentVRP)
ExperimentVRPPSchema = class_schema(_ExperimentVRPP)
ExperimentIRPSchema = class_schema(_ExperimentIRP)
