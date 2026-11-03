"""Default: 15 basic actions, as in the old course without its two Ryu specials.
A basic key is held for six emulator frames; consecutive C means continuous C.
Use NO_OP (or another action) to release it. This is not automatic repeated tapping.
Students may append short combinations/sequences totaling exactly six frames.
No complete fighting strategy, auto-facing or long charge macro is provided.
"""
CUSTOM_ACTIONS = []

# Optional student exercise (OFF by default): a short, explicit press/release.
# CUSTOM_ACTIONS = [
#     {'name': 'tap_C', 'segments': [
#         {'keys': ['C'], 'frames': 3}, {'keys': [], 'frames': 3}]},
# ]
