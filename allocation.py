import pandas as pd

IDX_LVL_DATE = 0  # The dates will be at the first level of the index
IDX_LVL_SERIE = 1  # and the series at the second level of the index


class AllocationMixin:
    def allocation(self, dates) -> pd.Series:
        # This method will not be reimplemented later, _allocation will be reimplemented instead
        # Here we do a lot of the cleaning

        if isinstance(
            dates, (str, pd.Timestamp)
        ):  # if dates is a single date  '2000-01-01'
            dates = [dates]  # we put it in a list because we expect multiple dates

        # we transform all strings to real dates
        dates = [pd.to_datetime(d) if isinstance(d, str) else d for d in dates]
        dates = sorted(set(dates))  # we remove duplicate dates

        return self._allocation(dates=dates)

    def _allocation(self, dates) -> pd.Series:
        # This is the method that we have to reimplement each time
        # We expect allocation to return a Series with a multiple index
        # - the first level of the index is the date
        # - the second level of the index is the code for the stock

        raise NotImplementedError()


class OneAllocationMixin(AllocationMixin):
    # This one is to simplify the implementation when we want to deal with only on date at a time

    def _allocation(self, dates) -> pd.Series:
        # we define _allocation as a loop on another method
        return pd.concat({date: self._one_allocation(date) for date in dates})

    def _one_allocation(self, date) -> pd.Series:
        # we expect this to return a series, with only one level of index with the stock code
        raise NotImplementedError()


class OneStaticAllocation(OneAllocationMixin):
    """
    Example for OneAllocationMixin that returns the same result everytime
    """

    def _one_allocation(self, date) -> pd.Series:
        return pd.Series({"a": 0.2, "b": 0.8})


class EqualWeight(AllocationMixin):
    """
    Example of AllocationMixin, that transforms to Equal Weight
    """

    def __init__(self, previous_step: AllocationMixin):
        self.previous_step = previous_step

    def _allocation(self, dates) -> pd.Series:
        # because EW is very simple, we can do operation on all the dates at the same time
        prev = self.previous_step.allocation(dates)

        def _ew(group: pd.Series):
            return pd.Series(1 / len(group), index=group.index)

        return prev.groupby(level=IDX_LVL_DATE).apply(_ew)


def demo():
    uv = OneStaticAllocation()
    ew = EqualWeight(previous_step=uv)

    dates = ["2022-08-01", "2022-08-02"]

    # I unstack because everything comes in Series format, not DataFrame
    print(uv.allocation(dates).unstack())
    print(ew.allocation(dates).unstack())


if __name__ == "__main__":
    demo()
