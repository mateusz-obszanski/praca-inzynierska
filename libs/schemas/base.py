from dataclasses import dataclass, fields
from enum import Enum
from typing import Any, Optional
from inspect import isclass
from pathlib import Path

from marshmallow import RAISE, ValidationError

from ..types.type_registry import Registry


@dataclass
class BaseDataclass:
    class Meta:
        """
        Behaviour on loading unknown fields.
        Available: `RAISE` (default), `INCLUDE`, `EXCLUDE`.
        """

        unknown = RAISE

    @classmethod
    def load_enums(
        cls,
        data: dict[str, Any],
        as_str: bool = False,
        enum_fields: Optional[dict[str, type[Enum]]] = None,
    ) -> dict[str, Any]:
        if enum_fields is None:
            enum_fields = cls.get_enum_fields()

        # look for enum names in data
        for name, field_type in enum_fields.items():
            val = data.get(name)
            if val is None:
                continue
            enum_val: Enum = (
                field_type[val] if isinstance(val, str) else field_type(val)  # type: ignore
            )
            data[name] = enum_val.name if as_str else val

        return data

    @classmethod
    def check_numeric_fields(cls, data: dict[str, Any]) -> None:
        """
        Checks bounds specified in metadata:
        - `"positive"`,
        - `"negative"`,
        - `"nonpositive"`,
        - `"nonnegative"`,
        - `"gt"`: `float`,
        - `"lt"`: `float`,
        - `"ge"`: float,
        - `"le"`: float,
        - `"range"`: 2-element iterable or dict with keys:
            - `"left_inclusive"`: `bool`,
            - `"right_inclusive": `bool``,
            - `"bounds"` - 2-element iterable
        - `"values"` - possible values,
        - `"excluded"` - excluded values,
        - `"even"`: `bool`,
        - `"odd"`: `bool`,
        - `divisible_by`: `float`
        """

        for field in fields(cls):
            metadata = field.metadata
            try:
                val = data[field.name]
            except KeyError as e:
                raise ValidationError(f"{field.name} not present in data", field.name)
            get_metadata = metadata.get
            if get_metadata("positive") and val <= 0:
                raise ValidationError(f"{field.name} must be positive", val)
            if get_metadata("negative") and val >= 0:
                raise ValidationError(f"{field.name} must be negative", val)
            if get_metadata("nonnegative") and val < 0:
                raise ValidationError(f"{field.name} must be nonnegative", val)
            if get_metadata("nonpositive") and val > 0:
                raise ValidationError(f"{field.name} must be nonpositive", val)
            gt_val = get_metadata("gt")
            if gt_val is not None and val <= gt_val:
                raise ValidationError(
                    f"{field.name} must be greater than `{gt_val}`", val, gt_val
                )
            ge_val = get_metadata("ge")
            if ge_val is not None and val < ge_val:
                raise ValidationError(
                    f"{field.name} must be greater or equal `{ge_val}`", val, ge_val
                )
            lt_val = get_metadata("lt")
            if lt_val is not None and val >= lt_val:
                raise ValidationError(
                    f"{field.name} must be less than `{lt_val}`", val, lt_val
                )
            le_val = get_metadata("le")
            if le_val is not None and val > le_val:
                raise ValidationError(
                    f"{field.name} must be less or equal `{le_val}`", val, le_val
                )
            val_range_data = get_metadata("range")
            if val_range_data is not None:
                if isinstance(val_range_data, dict):
                    val_range = val_range_data["bounds"]
                    left_inclusive = val_range_data.get("left_inclusive")
                    right_inclusive = val_range_data.get("right_inclusive")
                    if left_inclusive is None:
                        left_inclusive = True
                        if right_inclusive is None:
                            right_inclusive = True
                    elif right_inclusive is None:
                        right_inclusive = True
                else:
                    val_range = val_range_data

                vmin, vmax = val_range
                if (
                    not (vmin <= val <= vmax)
                    or (val == vmin and not left_inclusive)
                    or (val == vmax and not right_inclusive)
                ):
                    lbracket = "[" if left_inclusive else "("
                    rbracket = "]" if right_inclusive else ")"
                    range_str = f"{lbracket}{vmin}, {vmax}{rbracket}"
                    raise ValidationError(
                        f"{field.name} not in specified range {range_str}",
                        val,
                        {"range": val_range},
                    )

            allowed_values = get_metadata("values")
            if allowed_values is not None and val not in allowed_values:
                raise ValidationError(
                    f"{field.name}'s val not allowed", val, {"allowed": allowed_values}
                )
            excluded_values = get_metadata("excluded")
            if excluded_values is not None and val in excluded_values:
                raise ValidationError(
                    f"{field.name}'s value not allowed",
                    val,
                    {"excluded": excluded_values},
                )
            if get_metadata("even") and val % 2 != 0:
                raise ValidationError(f"{field.name} must be even", val)
            if get_metadata("odd") and not val % 2 != 1:
                raise ValidationError(f"{field.name} must be odd", val)
            divisor = get_metadata("divisible_by")
            if divisor is not None and val % divisor != 0:
                raise ValidationError(
                    f"{field.name} must be divisible by `divisor`",
                    val,
                    {"divisor": divisor},
                )

    @classmethod
    def get_enum_fields(cls) -> dict[str, type[Enum]]:
        # enum_fields: dict[str, Union[type[Enum], EnumMeta]] = {}
        enum_fields: dict[str, type[Enum]] = {}
        for f in fields(cls):
            f_type = f.type
            f_name = f.name
            if isclass(f) and issubclass(f_type, Enum):
                enum_fields[f_name] = f_type
            #     continue
            # for arg in get_args(f_type):
            #     if isinstance(arg, EnumMeta):
            #         enum_fields[f_name] = arg

        return enum_fields

    @classmethod
    def validate_paths(cls, data: dict[str, Any]) -> None:
        for field in fields(cls):
            field_name = field.name
            if field.metadata.get("is_path"):
                try:
                    data_path = Path(data[field_name])
                    if not data_path.exists():
                        raise FileNotFoundError(data_path)
                except FileNotFoundError as e:
                    raise ValidationError(
                        f"path `{data_path}` does not exist", field_name
                    ) from e
                except Exception as e:
                    raise ValidationError(
                        f"could not validate `{field_name}`", field_name
                    ) from e
                if field_name not in data:
                    raise ValidationError(
                        f"path `{field_name}` not present in data", field_name
                    )

    @classmethod
    def convert_paths(cls, data: dict[str, Any]) -> dict[str, Any]:
        for field in fields(cls):
            if not field.metadata.get("is_path"):
                continue
            field_name = field.name
            data[field_name] = Path(data[field_name])
        return data

    @classmethod
    def load_from_registers(cls, data: dict[str, Any]) -> dict[str, Any]:
        """
        Maps data fields if schame field is a `Registry`.
        """

        for field in fields(cls):
            field_type = field.type
            if not issubclass(field_type, Registry):
                continue
            field_name = field.name
            registry = field_type.get_registry()
            try:
                data[field_name] = registry[field_name]
            except KeyError as e:
                raise ValidationError(
                    f"`{field_name}` not registered at registry",
                    field_name,
                    {"registry": registry},
                ) from e
        return data
