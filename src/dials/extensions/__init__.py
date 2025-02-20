from __future__ import annotations

import importlib.metadata


class _Extension:
    """A base class for extension groups.
    This contains a common lookup mechanism and phil scope generator.
    """

    entry_point = None

    @classmethod
    def extensions(cls):
        """Return a list of all registered extension classes."""
        return [
            entry_point.load()
            for entry_point in importlib.metadata.entry_points(group=cls.entry_point)
        ]

    @classmethod
    def load(cls, name):
        """Get the requested extension class by name.

        :param name: The name of the extension
        :returns: The extension class
        """
        for entry_point in importlib.metadata.entry_points(group=cls.entry_point):
            if entry_point.name == name:
                return entry_point.load()

    @classmethod
    def phil_scope(cls):
        """Get the phil scope for the interface or extension.

        :returns: The phil scope for the interface or extension
        """
        from libtbx.phil import parse

        if cls == _Extension:
            raise RuntimeError("Extension has no phil parameters")
        doc = "\n".join('"%s"' % d for d in cls.__doc__.strip().splitlines())
        master_scope = parse(f"{cls.name} .help={doc} {{}}")
        main_scope = master_scope.get_without_substitution(cls.name)
        assert len(main_scope) == 1
        main_scope = main_scope[0]
        if "phil" in cls.__dict__:
            main_scope.adopt_scope(cls.phil())

        def ext_names(extensions):
            names = []
            default_index = -1
            for ext in extensions:
                if "default" in ext.__dict__:
                    default_index = len(names)
                names.append(ext.name)
            if default_index < 0:
                default_index = 0
            if names:
                names[default_index] = "*" + str(names[default_index])
            return names

        exts = cls.extensions()
        if exts:
            algorithm = parse(
                f"""
        algorithm = {' '.join(ext_names(exts))}
          .help = "The choice of algorithm"
          .type = choice
      """
            )
            main_scope.adopt_scope(algorithm)
            for ext in exts:
                if hasattr(ext, "phil"):
                    help_str = "\n".join(
                        ['"%s"' % line for line in ext.__doc__.split()]
                    )
                    ext_master_scope = parse(f"{ext.name} .help={help_str} {{}}")
                    ext_phil_scope = ext_master_scope.get_without_substitution(ext.name)
                    assert len(ext_phil_scope) == 1
                    ext_phil_scope = ext_phil_scope[0]
                    ext_phil = ext.phil()
                    if ext_phil is not None:
                        ext_phil_scope.adopt_scope(ext.phil())
                        main_scope.adopt_scope(ext_master_scope)
        return master_scope


class SpotFinderThreshold(_Extension):
    """Extensions for threshold algorithms to be used in spot finding."""

    entry_point = "dials.spotfinder.threshold"
    name = "threshold"
    scope = "spotfinder"


class ProfileModel(_Extension):
    """
    The interface definition for a profile model.
    """

    entry_point = "dxtbx.profile_model"
    name = "profile"
    scope = "profile"


class Centroid(_Extension):
    """Extensions for centroid algorithms."""

    entry_point = "dials.integration.centroid"
    name = "centroid"
    scope = "integration"


class Background(_Extension):
    """Extensions for background algorithms."""

    entry_point = "dials.integration.background"
    name = "background"
    scope = "integration"
