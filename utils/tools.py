import importlib
import os
import sys


def get_dataloader(loader):
    Loader = importlib.import_module('dataset_loader.{}'.format(loader)).Loader
    return Loader


def get_model(model):
    Model = importlib.import_module('models.{}'.format(model)).Model
    return Model


def get_constructor(constructor):
    Constructor = importlib.import_module('{}'.format(constructor)).Constructor
    return Constructor


def get_evaluator(evaluate_tool):
    EvaluateTool = importlib.import_module('{}'.format(evaluate_tool)).EvaluateTool
    return EvaluateTool

import nltk
import ssl

def download_nltk():
    try:
        _create_unverified_https_context = ssl._create_unverified_context
    except AttributeError:
        pass
    else:
        ssl._create_default_https_context = _create_unverified_https_context

    nltk.download("punkt")


def add_sys_path():
    curPath = os.path.abspath(os.path.dirname(__file__))
    rootPath = os.path.split(curPath)[0]
    sys.path.append(rootPath)