"""One reviewed agent in a separate CPU process. JSON/base64 RPC, no pickle IPC.
This is process separation for compatibility/diagnostics, NOT a security sandbox.
The actor receives RGB + four approved coordinates, never an emulator handle.
"""
from __future__ import annotations
import argparse
import base64
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys
import traceback

MAX_REQUEST = 512 * 1024


def main():
    p=argparse.ArgumentParser();p.add_argument('--agent',required=True);p.add_argument('--work',required=True)
    args=p.parse_args();work=Path(args.work);work.mkdir(parents=True,exist_ok=True)
    wire=sys.stdout
    def reply(value):
        wire.write(json.dumps(value,separators=(',',':'),allow_nan=False)+'\n');wire.flush()
    actor=None;side=None;initial_hash=None
    try:
        with redirect_stdout(sys.stderr):
            import numpy as np
            import torch
            import cv2
            torch.set_num_threads(1);cv2.setNumThreads(1)
            from sf2_bundle import LoadedAgent
            from sf2_observation import Side, display_frames
            from sf2_common import parameter_hash,runtime_versions
            from sf2_actions import action_index
            actor=LoadedAgent(args.agent,trusted=True,device='cpu')
            actor.model.policy.set_training_mode(False)
            for t in actor.model.policy.state_dict().values():
                if t.is_floating_point() and not torch.isfinite(t).all():
                    raise ValueError('Non-finite model weights.')
            initial_hash=parameter_hash(actor.model)
            # Warm up preprocessing and inference, not a game acceptance test.
            probe=Side(actor.spec,1,actor.common['position_scale'])
            probe.reset(np.zeros((224,320,3),np.uint8),frame_id=0)
            raw={'p1_x':100,'p1_y':192,'p2_x':200,'p2_y':192}
            with torch.inference_mode():
                pred,_=actor.model.predict(probe.observe(raw),deterministic=True)
                action_index(pred,len(actor.spec['actions']))
            if parameter_hash(actor.model)!=initial_hash:
                raise RuntimeError('Warm-up changed model parameters/buffers.')
        reply({'status':'ready','parameter_hash':initial_hash,'runtime_versions':runtime_versions(),
               'image_shape':list(probe.observe(raw)['image'].shape) if isinstance(probe.observe(raw),dict) else list(probe.observe(raw).shape),
               'action_count':len(actor.spec['actions'])})
        for _ in range(100000):
            line=sys.stdin.buffer.readline(MAX_REQUEST+1)
            if not line:break
            if len(line)>MAX_REQUEST or not line.endswith(b'\n'):
                raise ValueError('Oversized/malformed request.')
            req=json.loads(line);op=req.get('op');seq=req.get('seq')
            with redirect_stdout(sys.stderr):
                if op=='close':
                    reply({'seq':seq,'status':'closed'});break
                if op=='finish':
                    h=parameter_hash(actor.model)
                    if h!=initial_hash:raise RuntimeError('Inference changed policy parameters/buffers.')
                    reply({'seq':seq,'status':'unchanged','parameter_hash':h});continue
                if op not in ('reset','act'):raise ValueError('Unknown actor operation.')
                blob=base64.b64decode(req['rgb_b64'],validate=True)
                if len(blob)!=224*320*3:raise ValueError('Wrong raw RGB byte count.')
                rgb=np.frombuffer(blob,np.uint8).reshape(224,320,3).copy()
                raw=req['coordinates']
                if set(raw)!={'p1_x','p1_y','p2_x','p2_y'}:raise ValueError('Only declared coordinates are accepted.')
                if any(type(v) is not int or not 0<=v<=65535 for v in raw.values()):
                    raise ValueError('Invalid coordinates.')
                frame=req['frame']
                if type(frame) is not int or frame<0:raise ValueError('Invalid frame ID.')
                if op=='reset':
                    player=req['player'];seed=req['seed']
                    if player not in (1,2):raise ValueError('Invalid controller port.')
                    import random
                    random.seed(seed);np.random.seed(seed);torch.manual_seed(seed)
                    side=Side(actor.spec,player,actor.common['position_scale']);side.reset(rgb,frame_id=frame)
                    save_index=0
                    reply({'seq':seq,'status':'reset','player':player,'frame':frame});continue
                if side is None:raise RuntimeError('Reset the actor before requesting an action.')
                if frame>side.frame_ids[-1]:side.update(rgb,frame_id=frame)
                elif frame!=side.frame_ids[-1]:raise ValueError('Frame IDs moved backwards.')
                observation=side.observe(raw)
                # Save early post-reset and real motion-history inputs for student diagnostics.
                if save_index in (0,3):
                    image=observation['image'] if isinstance(observation,dict) else observation
                    label=f'P{side.player}_{req.get("round_id",0):02d}_{save_index:02d}'
                    np.savez_compressed(work/(label+'.npz'), image=image,
                        position=observation.get('position',np.zeros(0,np.float32)) if isinstance(observation,dict) else np.zeros(0,np.float32),
                        frame_ids=np.array(side.frame_ids))
                    tiles=display_frames(image,actor.spec)
                    tiles=[np.repeat(t[:,:,None],3,axis=2) if t.ndim==2 else t for t in tiles]
                    cv2.imwrite(str(work/(label+'.png')),cv2.cvtColor(np.concatenate(tiles,axis=1),cv2.COLOR_RGB2BGR))
                save_index+=1
                with torch.inference_mode():
                    pred,_=actor.model.predict(observation,deterministic=True)
                    index=action_index(pred,len(actor.spec['actions']))
                reply({'seq':seq,'status':'action','action':index,'player':side.player,'frame':frame})
        else:raise RuntimeError('Actor request budget exhausted.')
    except BaseException as exc:
        traceback.print_exc(file=sys.stderr)
        try:reply({'status':'error','error':type(exc).__name__+': '+str(exc)})
        except Exception:pass
        return 1
    finally:
        if actor is not None:actor.close()
    return 0

if __name__=='__main__':raise SystemExit(main())
