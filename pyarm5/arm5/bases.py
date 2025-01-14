import pydantic


class FrozenModel(pydantic.BaseModel):
    """Pydantic BaseModel that is frozen"""

    model_config = pydantic.ConfigDict(frozen=True)
