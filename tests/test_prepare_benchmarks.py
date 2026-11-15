import io
import math

import pytest

from reproduction.prepare.prepare_lcbench import iter_datasets


@pytest.mark.parametrize("chunk_size", [1, 7, 16, 65536])
def test_lcbench_stream_preserves_nonfinite_values_and_string_boundaries(chunk_size):
    source = (
        ' {"a": {"metric": [1, Infinity, -Infinity, NaN], '
        '"label": "Infinity, escaped \\" quote and } brace"}, "b": {"c": [0.2]}}  '
    )
    datasets = dict(iter_datasets(io.StringIO(source), chunk_size))
    assert datasets["b"] == {"c": [0.2]}
    assert datasets["a"]["metric"][:3] == [1, float("inf"), float("-inf")]
    assert math.isnan(datasets["a"]["metric"][3])
    assert datasets["a"]["label"] == 'Infinity, escaped " quote and } brace'


@pytest.mark.parametrize("source", ["{} extra", '{"a":{},}', '{"a":', '{"a":2}', "[]"])
def test_lcbench_stream_rejects_malformed_dataset_mappings(source):
    with pytest.raises(ValueError):
        list(iter_datasets(io.StringIO(source), chunk_size=2))
