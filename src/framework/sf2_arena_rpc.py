"""Bounded JSON pipe transport for inference subprocesses (Linux/Kaggle).
Processes are NOT a security sandbox. No pickle objects are received from agents.
"""
from __future__ import annotations
import json
import os
from pathlib import Path
import selectors
import signal
import subprocess
import sys
import time

class ActorProcess:
    def __init__(self,agent,work,limits,*,command=None):
        work=Path(work);work.mkdir(parents=True,exist_ok=True)
        self.proc=None;self.seq=0;self.buffer=b'';self.times=[];self.limits=limits
        self.log=(work/'actor_stderr.log').open('wb')
        env=os.environ.copy()
        env.update(CUDA_VISIBLE_DEVICES='',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1')
        cmd=command or [sys.executable,'-u',str(Path(__file__).with_name('sf2_arena_actor.py')),
                       '--agent',str(Path(agent).resolve()),'--work',str(work.resolve())]
        try:
            self.proc=subprocess.Popen(cmd,cwd=work,env=env,stdin=subprocess.PIPE,stdout=subprocess.PIPE,
                stderr=self.log,start_new_session=True,bufsize=0)
            os.set_blocking(self.proc.stdin.fileno(),False);os.set_blocking(self.proc.stdout.fileno(),False)
            self.ready=self._read(time.monotonic()+float(limits['load_timeout_seconds']))
            if self.ready.get('status')!='ready':raise RuntimeError('Actor failed loading: '+str(self.ready))
        except BaseException:
            self.close();raise

    def _wait(self,fd,event,deadline):
        remaining=deadline-time.monotonic()
        if remaining<=0:raise TimeoutError('Actor RPC watchdog expired.')
        with selectors.DefaultSelector() as s:
            s.register(fd,event)
            if not s.select(remaining):raise TimeoutError('Actor RPC watchdog expired.')

    def _write(self,data,deadline):
        view=memoryview(data)
        while view:
            self._wait(self.proc.stdin.fileno(),selectors.EVENT_WRITE,deadline)
            try:n=os.write(self.proc.stdin.fileno(),view)
            except BlockingIOError:continue
            if n<=0:raise RuntimeError('Actor input pipe closed.')
            view=view[n:]

    def _read(self,deadline):
        while b'\n' not in self.buffer:
            self._wait(self.proc.stdout.fileno(),selectors.EVENT_READ,deadline)
            try:b=os.read(self.proc.stdout.fileno(),65536)
            except BlockingIOError:continue
            if not b:raise RuntimeError('Actor exited before returning a result; see actor_stderr.log.')
            self.buffer+=b
            if len(self.buffer)>65536:raise RuntimeError('Actor response exceeds 64 KiB.')
        line,self.buffer=self.buffer.split(b'\n',1)
        value=json.loads(line)
        if not isinstance(value,dict):raise ValueError('Actor response must be a JSON object.')
        if value.get('status')=='error':raise RuntimeError(value.get('error','Actor error'))
        return value

    def request(self,op,**payload):
        self.seq+=1;start=time.monotonic()
        deadline=start+float(self.limits['load_timeout_seconds'] if op in ('reset','finish') else self.limits['decision_timeout_seconds'])
        req=json.dumps({'seq':self.seq,'op':op,**payload},separators=(',',':'),allow_nan=False).encode()+b'\n'
        if len(req)>512*1024:raise ValueError('Actor request exceeds 512 KiB.')
        self._write(req,deadline);result=self._read(deadline)
        elapsed=time.monotonic()-start
        if result.get('seq')!=self.seq:raise RuntimeError('Out-of-order actor response.')
        if op=='act':self.times.append(elapsed)
        return result

    def close(self):
        proc=getattr(self,'proc',None)
        if proc is not None:
            if proc.poll() is None:
                try:os.killpg(proc.pid,signal.SIGTERM)
                except ProcessLookupError:pass
                try:proc.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    try:os.killpg(proc.pid,signal.SIGKILL)
                    except ProcessLookupError:pass
                    proc.wait()
            for pipe in (proc.stdin,proc.stdout):
                if pipe is not None:pipe.close()
            self.proc=None
        if getattr(self,'log',None):self.log.close()
