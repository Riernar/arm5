from arm5.datamodels import times


def test_seasons():
    # Check base member instanciation
    assert times.Seasons("spring") is times.Seasons.SPRING
    assert times.Seasons("summer") is times.Seasons.SUMMER
    assert times.Seasons("autumn") is times.Seasons.AUTUMN
    assert times.Seasons("winter") is times.Seasons.WINTER

    # Test aliases
    assert times.Seasons.FALL is times.Seasons.AUTUMN
    assert times.Seasons("fall") is times.Seasons.AUTUMN


def test_season_order():
    assert times.Seasons.SPRING._enumeration_order == 0
    assert times.Seasons.SUMMER._enumeration_order == 1
    assert times.Seasons.AUTUMN._enumeration_order == 2
    assert times.Seasons.WINTER._enumeration_order == 3

    assert times.Seasons.SPRING < times.Seasons.SUMMER
    assert times.Seasons.SUMMER < times.Seasons.AUTUMN
    assert times.Seasons.AUTUMN < times.Seasons.WINTER

    assert times.Seasons.SUMMER > times.Seasons.SPRING
    assert times.Seasons.AUTUMN > times.Seasons.SUMMER
    assert times.Seasons.WINTER > times.Seasons.AUTUMN

    assert list(times.Seasons) == sorted(times.Seasons)


def test_year_season():
    start_of_year = times.YearSeason(year=1, season=times.Seasons.SPRING)
    assert times.YearSeason.start_of_year(1) == start_of_year
    assert times.YearSeason(year=1, season=times.Seasons.WINTER) == times.YearSeason.end_of_year(1)

    start = times.YearSeason(year=2, season=times.Seasons.AUTUMN)
    end = times.YearSeason(year=4, season=times.Seasons.SUMMER)
    assert start < end
    assert not (end < start)
    assert list(times.YearSeason.iter_between(start, end)) == [
        times.YearSeason(year=2, season=times.Seasons.AUTUMN),
        times.YearSeason(year=2, season=times.Seasons.WINTER),
        times.YearSeason(year=3, season=times.Seasons.SPRING),
        times.YearSeason(year=3, season=times.Seasons.SUMMER),
        times.YearSeason(year=3, season=times.Seasons.AUTUMN),
        times.YearSeason(year=3, season=times.Seasons.WINTER),
        times.YearSeason(year=4, season=times.Seasons.SPRING),
        times.YearSeason(year=4, season=times.Seasons.SUMMER),
    ]
