"""Model configurations and native parameter state."""

from __future__ import annotations

from datetime import date
from enum import StrEnum

from pydantic import Field, JsonValue, model_validator

from .common import Frozen


class Track(StrEnum):
    CLOUD = "cloud"
    LOCAL = "local"
    FAKE = "fake"


class ParameterState(StrEnum):
    EXPLICIT = "explicit"  # submitted with a value
    OMITTED = "omitted"  # deliberately not submitted; provider default applies
    UNSUPPORTED = "unsupported"  # provider/engine does not accept it
    IGNORED = "ignored"  # submitted but documented or observed as having no effect
    UNKNOWN = "unknown"  # cannot be determined


class ParameterSetting(Frozen):
    """Parameter state is separate from value. Documented defaults are separate from observed values."""

    state: ParameterState
    value: JsonValue = None
    documented_default: JsonValue = None
    observed_effective: JsonValue = None
    source: str | None = None
    verified_on: date | None = None

    @model_validator(mode="after")
    def _consistent(self) -> ParameterSetting:
        if self.state is ParameterState.EXPLICIT and self.value is None:
            raise ValueError("an explicit parameter requires a submitted value")
        if (
            self.state in (ParameterState.OMITTED, ParameterState.UNSUPPORTED, ParameterState.UNKNOWN)
            and self.value is not None
        ):
            raise ValueError(f"a parameter in state '{self.state}' must not carry a submitted value")
        return self


class ModelConfiguration(Frozen):
    configuration_id: str = Field(pattern=r"^[a-z0-9][a-z0-9._-]*$")
    revision: int = Field(ge=1)
    track: Track
    adapter: str
    provider: str
    model: str
    profile: str = "native_recommended"
    context_limit: int | None = Field(default=None, ge=1)
    parameters: dict[str, ParameterSetting] = Field(default_factory=dict)
    adapter_options: dict[str, JsonValue] = Field(default_factory=dict)

    @property
    def key(self) -> str:
        return f"{self.configuration_id}@{self.revision}"

    def explicit_int(self, name: str) -> int | None:
        setting = self.parameters.get(name)
        if setting is None or setting.state is not ParameterState.EXPLICIT:
            return None
        if not isinstance(setting.value, int):
            raise ValueError(f"parameter {name} is not an integer")
        return setting.value
