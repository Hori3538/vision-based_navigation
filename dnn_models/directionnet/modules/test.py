import argparse
import smartargparse.smartargparse 
from smartargparse import BaseConfig
from smartargparse import parse_args
from argparse import ArgumentParser
from typing import Type, TypeVar
import dataclasses

def test() -> None:

    @dataclasses.dataclass(frozen=True)
    class Config(BaseConfig):
        int_wo_default: int
        float_wo_default: float
        str_wo_default: str
        bool_wo_default: bool
        int_w_default: int = 1
        float_w_default: float = 1.0
        str_w_default: str = "foo"
        bool_w_default: bool = True

    config = parse_args(Config)
    print(vars(config))


if __name__ == "__main__":
    test()

