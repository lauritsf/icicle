"""Required command inputs and removed evaluation options."""

import sys

import pytest
import tyro

from experiments import eval, hpo_regret


@pytest.mark.parametrize('command,base', [
    (eval, ['checkpoint', '--out', 'eval.parquet']),
    (hpo_regret, ['--method', 'asha', '--out', 'hpo.parquet']),
])
def test_explicit_benchmark_required(command, base):
    with pytest.raises(SystemExit) as error:
        tyro.cli(command.Args, args=base)
    assert error.value.code != 0
    args = tyro.cli(command.Args, args=[*base, '--benchmark', 'pd1_ifbo'])
    assert args.benchmark == 'pd1_ifbo'


@pytest.mark.parametrize('option', [
    ['--mode', 'ifbo-parity'], ['--context-size', '800'], ['--protocol-version', '2'],
])
def test_removed_eval_options(option):
    with pytest.raises(SystemExit) as error:
        tyro.cli(eval.Args, args=['checkpoint', '--benchmark', 'pd1_ifbo', '--out', 'eval.parquet', *option])
    assert error.value.code != 0


@pytest.mark.parametrize('command,base', [
    (eval, ['checkpoint', '--out', 'eval.parquet']),
    (hpo_regret, ['--method', 'asha', '--out', 'hpo.parquet']),
])
def test_all_rejected_before_loading(command, base, monkeypatch):
    monkeypatch.setattr(sys, 'argv', ['command', *base, '--benchmark', 'all'])
    with pytest.raises(SystemExit, match='Choose one benchmark'):
        command.main()
