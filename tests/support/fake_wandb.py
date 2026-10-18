"""In-memory stand-in for the parts of `wandb` the trainer calls.

Installed as sys.modules["wandb"] before the trainer is imported. Every call
is appended to CALLS so two trees can be compared call by call.
"""

import os

CALLS = []
run = None


def _record(_call, *args, **kwargs):
    CALLS.append((_call, args, kwargs))


class _Summary(dict):
    def update(self, *args, **kwargs):
        _record("run.summary.update", *args, **kwargs)
        super().update(*args, **kwargs)


class _Config(dict):
    def update(self, values, *args, **kwargs):
        _record("config.update", dict(values))
        super().update(values)


class _Uploaded:
    def wait(self):
        _record("artifact.wait")


class Artifact:
    def __init__(self, name, type, metadata=None):
        _record("Artifact", name=name, type=type, metadata=dict(metadata or {}))
        self.name = name

    def add_file(self, path, name=None):
        _record("Artifact.add_file", os.path.basename(str(path)), name=name)


class _Run:
    def __init__(self, directory):
        self.dir = directory
        self.id = "fakerun1"
        self.offline = False
        self.summary = _Summary()
        self._name = None

    @property
    def name(self):
        return self._name

    @name.setter
    def name(self, value):
        _record("run.name", value)
        self._name = value

    def log_artifact(self, artifact, aliases=None):
        _record("run.log_artifact", artifact.name, aliases=list(aliases or []))
        return _Uploaded()


config = _Config()


def login(*args, **kwargs):
    _record("login")


def init(project=None, entity=None, **kwargs):
    global run
    _record("init", project=project, entity=entity)
    run = _Run(os.environ["FAKE_WANDB_DIR"])
    return run


def define_metric(name, step_metric=None, **kwargs):
    _record("define_metric", name, step_metric=step_metric)


def log(values, *args, **kwargs):
    _record("log", dict(values))


def finish(*args, **kwargs):
    _record("finish")
