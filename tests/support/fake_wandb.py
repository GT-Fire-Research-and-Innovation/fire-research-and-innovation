"""In-memory stand-in for the parts of `wandb` the trainer calls.

Installed as sys.modules["wandb"] before the trainer is imported. Every call
is appended to CALLS so two trees can be compared call by call.
"""

import os

CALLS = []
run = None


def _record(_call, *args, **kwargs):
    """Log one call in CALLS."""
    CALLS.append((_call, args, kwargs))


class _Summary(dict):
    """run.summary: a dict whose updates are logged."""

    def update(self, *args, **kwargs):
        """Log, then apply the update."""
        _record("run.summary.update", *args, **kwargs)
        super().update(*args, **kwargs)


class _Config(dict):
    """wandb.config: a dict whose updates are logged."""

    def update(self, values, *args, **kwargs):
        """Log a copy of `values`, then apply them."""
        _record("config.update", dict(values))
        super().update(values)


class _Uploaded:
    """What run.log_artifact returns."""

    def wait(self):
        """Log that the caller waited for the upload."""
        _record("artifact.wait")


class Artifact:
    """wandb.Artifact: logs its name, type and metadata."""

    def __init__(self, name, type, metadata=None):
        """Log the new artifact."""
        _record("Artifact", name=name, type=type, metadata=dict(metadata or {}))
        self.name = name

    def add_file(self, path, name=None):
        """Log the added file's base name."""
        _record("Artifact.add_file", os.path.basename(str(path)), name=name)


class _Run:
    """wandb.run: run directory, id, offline flag, summary, logged name changes."""

    def __init__(self, directory):
        """A run whose files go to `directory` (FAKE_WANDB_DIR)."""
        self.dir = directory
        self.id = "fakerun1"
        self.offline = False
        self.summary = _Summary()
        self._name = None

    @property
    def name(self):
        """The run name set so far."""
        return self._name

    @name.setter
    def name(self, value):
        """Log and store a new run name."""
        _record("run.name", value)
        self._name = value

    def log_artifact(self, artifact, aliases=None):
        """Log the artifact and its aliases; return an object with wait()."""
        _record("run.log_artifact", artifact.name, aliases=list(aliases or []))
        return _Uploaded()


config = _Config()


def login(*args, **kwargs):
    """Log a login."""
    _record("login")


def init(project=None, entity=None, **kwargs):
    """Log the project/entity and start a run in $FAKE_WANDB_DIR."""
    global run
    _record("init", project=project, entity=entity)
    run = _Run(os.environ["FAKE_WANDB_DIR"])
    return run


def define_metric(name, step_metric=None, **kwargs):
    """Log a chart definition."""
    _record("define_metric", name, step_metric=step_metric)


def log(values, *args, **kwargs):
    """Log a copy of the logged values."""
    _record("log", dict(values))


def finish(*args, **kwargs):
    """Log the end of the run."""
    _record("finish")
