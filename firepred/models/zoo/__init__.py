"""One file per model. Every module here that does not start with "_" is
imported by firepred.models.registry, so its @register_model builder becomes
available as `-m <name>`."""
