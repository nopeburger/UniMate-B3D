"""Regression tests for neural timeline postprocessing (no model load)."""
import importlib.util
import json
import sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/"backend"))
from timeline import retime, forward_kinematics, to_local, native_index, smooth_motion
spec = importlib.util.spec_from_file_location("geometry", ROOT/"addon/unimate_motion/motion.py")
geo = importlib.util.module_from_spec(spec)
spec.loader.exec_module(geo)
folder = ROOT/"tests/fixtures/seated-reference"
request = json.loads((folder/"request.json").read_text(encoding="utf-8"))
skel = request["skeleton"]
with np.load(folder/"motion.npz") as data:
    raw_pos, raw_rot = geo.decode_features(data["features"], geo.canonicalize(skel))
    old_pos, old_rot = data["positions"], data["rotations"]
args=(raw_pos,raw_rot,[(0,60),(60,110)],request["clips"],skel)
pos,rot=retime(*args)
assert len(pos)==120
assert np.allclose(rot[-1],raw_rot[-1],atol=1e-8), "Endpoint rotations changed"
assert np.allclose(pos[-1],raw_pos[-1],atol=1e-8), "Endpoint pose changed"
assert np.allclose(pos[:60],raw_pos[:60],atol=1e-8), "Earlier prompt changed"
assert np.allclose(rot.swapaxes(-1,-2)@rot,np.eye(3),atol=1e-8)
heads=np.array(skel["heads"])
def fixed_offsets(p,r):
    for j,parent in enumerate(skel["parents"][1:],1):
        offset=np.einsum("tji,tj->ti",r[:,parent],p[:,j]-p[:,parent])
        assert np.allclose(offset,heads[j]-heads[parent],atol=1e-8), j
fixed_offsets(pos,rot)
step=np.degrees(Rotation.from_matrix((rot[:-1].swapaxes(-1,-2)@rot[1:]).reshape(-1,3,3)).magnitude()).reshape(119,-1)
assert step[59:].max()<5, "Sit contains a rotational spike"
assert step[-1].max()<.1, "Final frame snaps to reference"
assert np.linalg.norm(np.diff(pos[59:,0],axis=0),axis=-1).max()<.03, "Root snaps during sit"
# Turning both editing controls off still guarantees FK-correct resampling.
p0,r0=retime(*args,transition_frames=0,pose_approach_frames=0)
fixed_offsets(p0,r0)
assert np.allclose(p0[:,0],old_pos[:,0])
assert np.allclose(r0[[0,-1]],old_rot[[0,-1]])
# General tree, multiple references, first-frame target and an interior release.
parents=[-1,0,0,1,2]
synthetic={"parents":parents,"heads":[[0,0,1],[1,0,1],[-1,0,1],[2,0,1],[-2,0,1]]}
local=np.tile(np.eye(3),(60,5,1,1))
for j in range(5):
    local[:,j]=Rotation.from_rotvec(np.outer(np.linspace(0,.8,60),[0,j/5,0])).as_matrix()
root=np.column_stack([np.linspace(0,1,60),np.zeros(60),np.ones(60)])
sp,sr=forward_kinematics(root,local,synthetic)
testclips=[dict(start=5,end=64,references=[dict(frame=i) for i in [5,25,45]])]
pp,rr=retime(sp,sr,[(0,60)],testclips,synthetic,12,15)
for index in [0,20,40,59]:
    assert np.allclose(pp[index],sp[index])
    assert np.allclose(rr[index],sr[index])
assert np.isfinite(pp).all()
# Prompt join without pose references, short and long destination ranges.
for duration in [2,60,300]:
    plain=[dict(start=1,end=60,references=[]),dict(start=61,end=60+duration,references=[])]
    pp,rr=retime(raw_pos,raw_rot,[(0,60),(60,110)],plain,skel,12,0)
    assert len(pp)==60+duration
    fixed_offsets(pp,rr)
    # The join continues the previous clip's motion: no frozen frame (the old
    # blend repeated the last pose) and no jump in rotation or root speed.
    def turn(a,b):
        return np.degrees(Rotation.from_matrix((a.swapaxes(-1,-2)@b).reshape(-1,3,3)).magnitude())
    before,at=turn(rr[58],rr[59]),turn(rr[59],rr[60])
    assert np.abs(at-before).max()<1.5, ("Join changes rotation speed", np.abs(at-before).max())
    v_before,v_at=pp[59,0]-pp[58,0],pp[60,0]-pp[59,0]
    assert np.linalg.norm(v_at-v_before)<.2*np.linalg.norm(v_before)+1e-4, "Join changes root speed"
# A jump at a chained-window join inside a clip, and right after the first of
# two references, is smoothed; references stay exact.
def stepped(jump_at, frames=110):
    local=np.tile(np.eye(3),(frames,5,1,1))
    for t in range(frames):
        angle=.01*t+(.6 if t>=jump_at else 0)
        local[t,1]=Rotation.from_rotvec([0,0,angle]).as_matrix()
    root=np.column_stack([np.linspace(0,1,frames),np.zeros(frames),np.ones(frames)])
    return forward_kinematics(root,local,synthetic)
def worst_jerk(p,lo,hi):
    return np.linalg.norm(np.diff(p[lo:hi],2,axis=0),axis=-1).max()
sp,sr=stepped(60)
clip=[dict(start=1,end=90,references=[])]
smoothed,_=retime(sp,sr,[(0,110)],clip,synthetic,12,60,seams=[[60]])
unsmoothed,_=retime(sp,sr,[(0,110)],clip,synthetic,12,60)
assert worst_jerk(unsmoothed,40,60)>.3 and worst_jerk(smoothed,40,60)<.05, (worst_jerk(unsmoothed,40,60),worst_jerk(smoothed,40,60))
sp,sr=stepped(22,60)
two=[dict(start=1,end=120,references=[dict(frame=40),dict(frame=120)])]
p2,r2=retime(sp,sr,[(0,60)],two,synthetic,12,30)
assert worst_jerk(p2,38,60)<.05, worst_jerk(p2,38,60)
source_local=to_local(sr,synthetic["parents"])
assert np.allclose(to_local(r2,synthetic["parents"])[39],source_local[native_index(two[0],two[0]["references"][0],60)],atol=1e-8)
# References are put back exactly after a later edit (cleanup), fading out
# around them and never past a neighbouring reference.
from timeline import restore_references
edited_local=to_local(r2,synthetic["parents"]).copy()
edited_local[:,1]=Rotation.from_rotvec([0,.3,0]).as_matrix()@edited_local[:,1]
edited_p,edited_r=forward_kinematics(p2[:,0]+[0,0,.05],edited_local,synthetic)
back_p,back_r=restore_references(edited_p,edited_r,p2,r2,[39,119],synthetic)
for f in (39,119):
    assert np.allclose(back_r[f],r2[f],atol=1e-8) and np.allclose(back_p[f],p2[f],atol=1e-8)
assert np.allclose(back_r[70],edited_r[70],atol=1e-8), "Restore reached past its falloff"
# Window planning: one window reproduces the old single-window mapping; long
# clips chain windows and keep each reference on its proportional frame.
from timeline import plan_windows
plan=plan_windows(request["clips"],60,10)
assert [c for c,_ in plan]==[0,1] and sorted(plan[1][1])==[59]
assert plan_windows(request["clips"],60,10,extend=False)==plan
long=[dict(start=1,end=180,references=[dict(frame=1),dict(frame=90),dict(frame=180)]),
      dict(start=181,end=300,references=[dict(frame=240)])]
plan=plan_windows(long,60,10)
assert [c for c,_ in plan]==[0,0,0,1,1], plan
assert sum(60 if i==0 else 50 for i in range(3))==160
slots=[(c,s) for c,refs in plan for s in refs]
assert slots[0]==(0,0) and slots[-1][0]==1
for w,(c,refs) in enumerate(plan):
    assert all((0 if w==0 else 10)<=s<60 for s in refs), (w,refs)
assert [c for c,_ in plan_windows(long,60,10,extend=False)]==[0,1]
try:
    plan_windows([dict(start=1,end=600,references=[dict(frame=301),dict(frame=305)])],60,10,extend=False)
    raise AssertionError("Colliding references accepted")
except ValueError:
    pass
plan_windows([dict(start=1,end=600,references=[dict(frame=301),dict(frame=305)])],60,10)
# retime maps references through the same native index for chained spans.
chained=[dict(start=1,end=180,references=[dict(frame=90)])]
assert native_index(chained[0],chained[0]["references"][0],160)==79
# Smoothing: 0 is the identity; otherwise rough rotations calm down and bone lengths hold.
_rng=np.random.default_rng(3)
_sk=dict(parents=[-1,0,1],heads=[[0,0,0],[0,.5,0],[0,1,0]])
_loc=np.tile(np.eye(3),(40,3,1,1))
_loc[:,1:]=Rotation.from_rotvec(_rng.normal(0,.3,(80,3))).as_matrix().reshape(40,2,3,3)
_root=np.zeros((40,3))
_p,_r=forward_kinematics(_root,_loc,_sk)
_p0,_r0=smooth_motion(_p,_r,_sk,0)
assert _p0 is _p and _r0 is _r
_ps,_rs=smooth_motion(_p,_r,_sk,2.0)
assert np.abs(np.diff(_ps,2,axis=0)).mean()<np.abs(np.diff(_p,2,axis=0)).mean()*.5
assert np.allclose(np.linalg.norm(_ps[:,1]-_ps[:,0],axis=-1),.5,atol=1e-5)
# Contact limbs keep most of their swing; the rest of the body is smoothed harder.
_sk2=dict(parents=[-1,0,1,2],heads=[[0,0,0],[0,.5,0],[0,1,0],[0,1.5,0]],foot_profiles=[dict(joint=3,parent=2,upper=1)])
_t=np.arange(60)
_swing=Rotation.from_rotvec(np.stack([np.sin(_t*.9)*.6,0*_t,0*_t],1)).as_matrix()   # fast swing, 7 frames per cycle
_loc2=np.tile(np.eye(3),(60,4,1,1)); _loc2[:,1]=_swing; _loc2[:,3]=_swing
_p2,_r2=forward_kinematics(np.zeros((60,3)),_loc2,_sk2)
_ps2,_rs2=smooth_motion(_p2,_r2,_sk2,3.0)
_free=dict(_sk2,foot_profiles=[])
_pu,_ru=smooth_motion(_p2,_r2,_free,3.0)
_amp=lambda p:np.ptp(p[10:-10,3]-p[10:-10,0],axis=0).max()
assert _amp(_ps2)>_amp(_pu)*1.5 and _amp(_ps2)>_amp(_p2)*.5
report=dict(passed=["references restored after edits","window joins smoothed","reference exits smoothed","window planning","chained reference slots","exact reference frame","earlier prompt preserved","no sit rotation spike",
                    "no endpoint root snap","FK bone lengths","disable editing",
                    "multiple and first-frame references","interior reference release",
                    "short and stretched clips","motion smoothing","limbs keep their swing"],
            approach_max_degrees=float(step[59:].max()))
(ROOT/"tests/artifacts/timeline-regressions.json").write_text(json.dumps(report,indent=2))
print(json.dumps(report,indent=2))
