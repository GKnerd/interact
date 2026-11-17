"""
Constants of the interact package.

Only plain, immutable values, and no imports from ``interact``, so every module of the package can
import this one without an import cycle. ``tests/test_constants.py`` checks both rules.
"""

# Model classes a checkpoint config may name in `model_class`
MODEL_CLASS_NAMES = ('IntentInformedForecaster', 'IntentInformedHRForecaster', 'TimeDCTForecaster')

# Robot points are centred on the person's L_Hand and R_Hand (last two joints, x/y/z flattened),
# as in training
ROBOT_CENTRE = slice(-6, None)

# Frames the AMASS dataset (interact/utils/amass.py) can return windows in: 'window' relative to the
# pelvis and heading at the last history frame, 'world' as converted (z up)
AMASS_FRAMES = ('window', 'world')

# Column of the SMPL pelvis rotation that points to the body's left: the hip line. SMPL's rest pose
# has x left, y up, z forward.
SMPL_HIP_AXIS = 0

# Shortest projection of the unit hip line onto the floor that still defines a heading. 0.5: the hip
# line is tilted at most 60 degrees from horizontal.
MIN_HIP_LINE_LENGTH = 0.5
