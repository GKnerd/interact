"""
Constants of the interact package.

Only plain, immutable values, and no imports from ``interact``, so every module of the package can
import this one without an import cycle. ``tests/test_constants.py`` checks both rules.
"""

# Model classes a checkpoint config may name in `model_class`
MODEL_CLASS_NAMES = ('IntentInformedForecaster', 'IntentInformedHRForecaster')

# Robot points are centred on the person's L_Hand and R_Hand (last two joints, x/y/z flattened),
# as in training
ROBOT_CENTRE = slice(-6, None)
