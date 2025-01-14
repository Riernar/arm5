import pydantic

from arm5.datamodels import arts


def test_hermetic_techniques():
    for string, art in [
        ("creo", arts.HermeticTechniques.CREO),
        ("intellego", arts.HermeticTechniques.INTELLEGO),
        ("muto", arts.HermeticTechniques.MUTO),
        ("perdo", arts.HermeticTechniques.PERDO),
        ("rego", arts.HermeticTechniques.REGO),
    ]:
        assert arts.HermeticTechniques(string) is art
        assert arts.HermeticTechniques(string.title()) is art
        # assert arts.HermeticTechniques(string[:2]) is art
        assert pydantic.TypeAdapter(arts.HermeticTechniques).validate_python(string) is art
        # assert pydantic.TypeAdapter(arts.HermeticTechniques).validate_python(string[:2]) is art


def test_hermetic_forms():
    for string, art in [
        ("animal", arts.HermeticForms.ANIMAL),
        ("aquam", arts.HermeticForms.AQUAM),
        ("auram", arts.HermeticForms.AURAM),
        ("corpus", arts.HermeticForms.CORPUS),
        ("herbam", arts.HermeticForms.HERBAM),
        ("ignem", arts.HermeticForms.IGNEM),
        ("imaginem", arts.HermeticForms.IMAGINEM),
        ("mentem", arts.HermeticForms.MENTEM),
        ("terram", arts.HermeticForms.TERRAM),
        ("vim", arts.HermeticForms.VIM),
    ]:
        assert arts.HermeticForms(string) is art
        assert arts.HermeticForms(string.title()) is art
        # assert arts.HermeticForms(string[:2]) is art
        assert pydantic.TypeAdapter(arts.Arts).validate_python(string) is art
    # assert pydantic.TypeAdapter(arts.Arts).validate_python(string[:2]) is art
