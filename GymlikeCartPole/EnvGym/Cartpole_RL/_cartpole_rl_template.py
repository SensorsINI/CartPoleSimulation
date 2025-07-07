"""
Required and AttributeCheckMeta are helper classes to check that all required attributes are set in the subclass.
If you are reading this, you probably want to change the CartPoleRLTemplate at the bottom of the file.
"""

import functools

from abc import ABC, ABCMeta, abstractmethod
from typing import Annotated, get_args, get_origin
from gymnasium import spaces


# Marker class to indicate required attributes
class Required:
    """Marker class to indicate required attributes."""
    pass


class AttributeCheckMeta(ABCMeta):
    """
    Metaclass that enforces that attributes marked as 'Required' in the base class
    are not None after initialization of any subclass.
    """

    def __new__(mcls, name, bases, namespace):
        cls = super().__new__(mcls, name, bases, namespace)

        # Collect required attributes from all base classes
        required_attributes = set()
        for base in bases:
            if hasattr(base, '__required_attributes__'):
                required_attributes.update(base.__required_attributes__)

        # Check for annotations in the current class
        annotations = namespace.get('__annotations__', {})
        for attr, attr_type in annotations.items():
            if get_origin(attr_type) is Annotated:
                args = get_args(attr_type)
                if Required in args:
                    required_attributes.add(attr)

        # Store the required attributes in the class
        cls.__required_attributes__ = required_attributes

        original_init = cls.__init__

        @functools.wraps(original_init)
        def wrapped_init(self, *args, **kwargs):
            # Call the original __init__
            original_init(self, *args, **kwargs)
            # Check required attributes
            missing_attrs = [
                attr for attr in cls.__required_attributes__
                if getattr(self, attr, None) is None
            ]
            if missing_attrs:
                raise ValueError(
                    f"The following attributes must be set in '{cls.__name__}.__init__': {', '.join(missing_attrs)}"
                )

        cls.__init__ = wrapped_init
        return cls


class CartPoleRLTemplate(ABC, metaclass=AttributeCheckMeta):
    # Define required attributes using Annotated with their actual types and Required marker
    x_threshold: Annotated[float, Required]
    pole_length_rendering: Annotated[float, Required]
    angle_rotation_direction_rendering: Annotated[int, Required]
    action_space: Annotated[spaces.Box, Required]
    observation_space: Annotated[spaces.Box, Required]

    @abstractmethod
    def __init__(self):
        # No need to set attributes to None
        pass

    @abstractmethod
    def get_next_state(self, state, action):
        pass

    @abstractmethod
    def reward_assignment(self, state, action, terminated):
        pass

    @abstractmethod
    def termination_condition(self, state):
        pass

    @abstractmethod
    def reset(self):
        pass
