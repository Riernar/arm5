import enum
from typing import Any, Self


class ArtEnumBase(enum.Enum):
    """
    Base enum class for Arts enum

    It implements the following functionnality:
    - enum.auto() uses the lower-case name of the enum member
    - Enum member can have alias(es), specified after the main value in the class body
    - ignore case when converting a str to the Enum class
    """

    _aliases_: frozenset[str]

    # We need to defined __new__ to change the value of the enum, see
    # https://docs.python.org/3/howto/enum.html#when-to-use-new-vs-init

    def __new__(cls, value: str, *aliases: str):
        obj = object.__new__(cls)
        obj._value_ = str(value).lower()
        obj._aliases_ = frozenset({str(a).lower() for a in aliases})
        return obj

    @classmethod
    def _missing_(cls, value: Any) -> Self:
        if isinstance(value, str):
            value = value.lower()
            for member in cls:
                if member._value_ == value:
                    return member
                elif value in member._aliases_:
                    return member
        return super()._missing_(value)


@enum.unique
class HermeticTechniques(ArtEnumBase):
    CREO = "creo", "cr"
    INTELLEGO = "intellego", "in"
    MUTO = "muto", "mu"
    PERDO = "perdo", "pe"
    REGO = "rego", "re"


@enum.unique
class HermeticForms(ArtEnumBase):
    ANIMAL = "animal", "an"
    AQUAM = "aquam", "aq"
    AURAM = "auram", "au"
    CORPUS = "corpus", "co"
    HERBAM = "herbam", "he"
    IGNEM = "ignem", "ig"
    IMAGINEM = "imaginem", "im"
    MENTEM = "mentem", "me"
    TERRAM = "terram", "te"
    VIM = "vim", "vi"


HermeticArts = HermeticTechniques | HermeticForms
Arts = HermeticArts
