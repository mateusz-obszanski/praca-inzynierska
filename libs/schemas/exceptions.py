from marshmallow.exceptions import ValidationError


class ExperimentBaseError(ValidationError):
    ...


class MapDataError(ExperimentBaseError):
    ...


class PopulationDataError(ExperimentBaseError):
    ...


class LoadCallablesError(ExperimentBaseError):
    ...
