"""Fixed six-frame actions. Ordinary PPO: no persistent queues or action masks."""
from __future__ import annotations
import hashlib, json, re
import numpy as np

QUANTUM = 6
KEYS = frozenset(('LEFT','RIGHT','UP','DOWN','A','B','C','X','Y','Z'))
BASIC_NAMES = ('NO_OP','LEFT','RIGHT','DOWN','UP','UP_LEFT','UP_RIGHT',
               'DOWN_LEFT','DOWN_RIGHT','A','B','C','X','Y','Z')
BASIC_KEYS = ((),('LEFT',),('RIGHT',),('DOWN',),('UP',),('UP','LEFT'),
              ('UP','RIGHT'),('DOWN','LEFT'),('DOWN','RIGHT'),
              ('A',),('B',),('C',),('X',),('Y',),('Z',))

def validate_library(actions):
    if not isinstance(actions, list) or not 1 <= len(actions) <= 64:
        raise ValueError('Provide 1..64 actions.')
    names = set()
    for action in actions:
        if set(action) != {'name','segments'}:
            raise ValueError('An action needs exactly name and segments.')
        name = action['name']
        if not isinstance(name, str) or not re.fullmatch(r'[A-Za-z][A-Za-z0-9_]{0,47}',name) or name in names:
            raise ValueError('Action names must be unique short English identifiers.')
        names.add(name)
        if not isinstance(action['segments'],list) or not action['segments']:
            raise ValueError('Empty action sequence.')
        total = 0
        for part in action['segments']:
            if set(part) != {'keys','frames'}:
                raise ValueError('Each segment needs keys and frames.')
            k = part['keys']; duration = part['frames']
            if not isinstance(k,list) or any(not isinstance(x,str) for x in k):
                raise ValueError('keys must be a list of controller button names.')
            if len(set(k)) != len(k) or set(k)-KEYS:
                raise ValueError('Only combat keys are supported; no START, MODE, FWD/BACK or RAM writes.')
            if {'LEFT','RIGHT'} <= set(k) or {'UP','DOWN'} <= set(k):
                raise ValueError('Conflicting directions.')
            if type(duration) is not int or not 1 <= duration <= QUANTUM:
                raise ValueError('Each segment lasts 1..6 native frames.')
            total += duration
        if total != QUANTUM:
            raise ValueError('Every action must finish in exactly 6 frames. This demo has no long-macro queue.')
    return actions

def library(custom=()):
    """The 15 course basics, optionally followed by student-defined short actions."""
    if not isinstance(custom,(list,tuple)):
        raise ValueError('CUSTOM_ACTIONS must be a list.')
    basic = [{'name':n,'segments':[{'keys':list(k),'frames':QUANTUM}]}
             for n,k in zip(BASIC_NAMES,BASIC_KEYS)]
    return validate_library(basic + json.loads(json.dumps(list(custom))))

def compiled(actions):
    result = []
    for a in validate_library(actions):
        result.append(tuple(tuple(p['keys']) for p in a['segments'] for _ in range(p['frames'])))
    return tuple(result)

def action_index(action,n):
    a = np.asarray(action)
    if a.size != 1 or a.dtype.kind not in ('i','u'):
        raise ValueError('The policy must return one integer action ID.')
    value = int(a.item())
    if not 0 <= value < n:
        raise ValueError('Action ID is outside this agent action table.')
    return value

def signature(actions):
    data = {'protocol':'fixed-sixframe-v2','actions':validate_library(actions)}
    return hashlib.sha256(json.dumps(data,sort_keys=True,separators=(',',':')).encode()).hexdigest()
