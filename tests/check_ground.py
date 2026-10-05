"""Ground contact regression tests using a small articulated fixture."""
import json,sys
from pathlib import Path
import numpy as np
from scipy.spatial.transform import Rotation
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/"backend"))
from ground import Surface,plant,profiles_for,foot_capsules,sole_points,stabilize_feet,preserve_bend
from timeline import forward_kinematics,to_local
skeleton=dict(parents=[-1,0,1,2,3],
 heads=[[0,0,1],[.1,0,1],[.1,0,.6],[.1,0,.15],[.1,-.2,.1]],
 bone_names=["hips","thigh.left","shin.left","foot.left",None],
 labels=["hips","thigh left","shin left","foot left","foot end"],
 rest_matrices=[np.eye(4).tolist() for _ in range(4)]+[None],
 collision_capsules=[dict(joint=3,a=[.1,-.03,.13],b=[.1,-.17,.1],radius=.08)],
 foot_profiles=[dict(joint=3,parent=2,upper=1,leg_length=.85,stance_tilt=28,swing_tilt=50)])
frames=40
local=np.tile(np.eye(3),(frames,5,1,1))
root=np.column_stack([np.linspace(0,.23,frames),np.zeros(frames),np.ones(frames)])
positions,rotations=forward_kinematics(root,local,skeleton)
ground=dict(normal=[0,0,1],height=0.,triangles=[])
corrected,orient,report=plant(positions,rotations,skeleton,ground)
assert report["active_frames"]>=25
assert report["median_planted_step"]<report["median_planted_step_before"]*.25
assert np.max(np.abs(corrected[:,0,:2]-positions[:,0,:2]))<1e-9
reconstructed,_=forward_kinematics(corrected[:,0],to_local(orient,skeleton["parents"]),skeleton)
assert np.allclose(reconstructed,corrected,atol=1e-8)
# A tilted static mesh returns its interpolated height and normal.
triangles=[
 [[0,0,0],[1,0,.1],[1,1,.1]],
 [[0,0,0],[1,1,.1],[0,1,0]]]
surface=Surface(dict(normal=[0,0,1],height=0.,triangles=triangles),skeleton)
height,normal,hit=surface.sample([.5,.5,.3])
assert hit and abs(height-.05)<1e-8 and normal[2]>.9
_,_,hit=surface.sample([3,3,0])
assert not hit and surface.misses==1
# The same foot may bend further in a swing than while planted.
tilt=np.radians(70)
turn=Rotation.from_rotvec([tilt,0,0]).as_matrix()
twist=np.tile(np.eye(3),(2,5,1,1))
twist[:,3]=turn
posed,rotated=forward_kinematics(np.array([[0,0,1],[0,0,1]]),twist,skeleton)
foot=foot_capsules(skeleton,profiles_for(skeleton))
a,b,limits=stabilize_feet(posed,rotated,skeleton,foot,Surface(ground,skeleton),[[(0,1)]])
assert limits[0]["stance_limit"]==28 and limits[0]["swing_limit"]==50
assert np.allclose(a[:,0],posed[:,0])
# A planted foot may not pull a turning knee through a straight-leg reversal.
bend_rig=dict(skeleton)
bend_rig["heads"]=[[0,0,1],[.1,0,1],[.1,-.15,.6],[.1,0,.2],[.1,-.2,.15]]
source_local=np.tile(np.eye(3),(1,5,1,1))
source_p,source_r=forward_kinematics(np.array([[0,0,1]]),source_local,bend_rig)
flipped_local=source_local.copy()
flipped_local[0,1]=Rotation.from_euler("z",180,degrees=True).as_matrix()
flipped_p,flipped_r=forward_kinematics(np.array([[0,0,1]]),flipped_local,bend_rig)
restored_p,restored_r,bend_counts=preserve_bend(flipped_p,flipped_r,source_p,bend_rig,[dict(profile=bend_rig["foot_profiles"][0])])
assert bend_counts["foot.left"]==1
assert restored_p[0,2,1]<0 and np.allclose(restored_p[0,3],flipped_p[0,3],atol=1e-7)
rebuilt,_=forward_kinematics(restored_p[:,0],to_local(restored_r,bend_rig["parents"]),bend_rig)
assert np.allclose(rebuilt,restored_p,atol=1e-7)
# A knee swivelling gradually away from the generated side is held there on
# every frame; a threshold-based fix popped it back on single frames.
steps=30
swing_local=np.tile(np.eye(3),(steps,5,1,1))
for t in range(steps):
    swing_local[t,1]=Rotation.from_euler("z",4*t,degrees=True).as_matrix()
source_many=np.repeat(source_p,steps,axis=0)
swung_p,swung_r=forward_kinematics(np.tile([0,0,1.],(steps,1)),swing_local,bend_rig)
held_p,_,_=preserve_bend(swung_p,swung_r,source_many,bend_rig,[dict(profile=bend_rig["foot_profiles"][0])])
knee_jerk=np.linalg.norm(np.diff(held_p[:,2],2,axis=0),axis=1).max()
assert knee_jerk<.01, knee_jerk
assert held_p[:,2,1].max()<0, "Knee left the generated side"
# Kneeling: the knee rests on the floor and the foot behind it is tucked into
# the ground. Cleanup keeps the knee down (no body lift), brings the foot out
# by bending the knee further, and never bends it backwards.
kneel_rig=dict(skeleton)
kneel_rig["collision_capsules"]=skeleton["collision_capsules"]+[dict(joint=2,a=[.1,0,.55],b=[.1,0,.2],radius=.05)]
kneel_local=np.tile(np.eye(3),(20,5,1,1))
kneel_local[:,2]=Rotation.from_rotvec([np.pi/2,0,0]).as_matrix()   # shin swung back to the floor
kneel_p,kneel_r=forward_kinematics(np.tile([0,0,.45],(20,1)),kneel_local,kneel_rig)
assert kneel_p[:,4,2].min()<-.05, "Fixture foot should start in the ground"
knelt_p,knelt_r,kneel_report=plant(kneel_p,kneel_r,kneel_rig,ground)
assert kneel_report["floor_contact_frames"]==20
assert abs(knelt_p[:,2,2]-kneel_p[:,2,2]).max()<.01, "Knee lifted off the floor"
assert knelt_p[:,3:,2].min()>-.01, knelt_p[:,3:,2].min()
def flexion(p):
    return np.degrees(np.arccos(np.clip(np.einsum("ti,ti->t",unit_rows(p[:,2]-p[:,1]),unit_rows(p[:,3]-p[:,2])),-1,1)))
def unit_rows(v):
    return v/np.linalg.norm(v,axis=1,keepdims=True)
assert (flexion(knelt_p)>=flexion(kneel_p)-1e-6).all(), "Knee straightened or bent backwards"
# A low-slung many-legged body (crab, spider): the belly touches the ground and
# every foot is far below it. The feet stay in charge, so the body is lifted
# until they stand on the ground; the kneeling logic must not take over.
legs=3
joints=1+4*legs
parents=[-1]
heads=[[0,0,1.0]]
names=["body"]
capsules=[dict(joint=0,a=[0,-.05,1.0],b=[0,.05,1.0],radius=.1)]
profiles=[]
for k in range(legs):
    x=(k-1)*.3
    base=len(parents)
    parents+= [0,base,base+1,base+2]
    heads+= [[x,0,1.0],[x,0,.6],[x,0,.15],[x,-.2,.1]]
    names+= [f"thigh{k}",f"shin{k}",f"foot{k}",None]
    capsules.append(dict(joint=base+2,a=[x,-.03,.13],b=[x,-.17,.1],radius=.05))
    profiles.append(dict(joint=base+2,parent=base+1,upper=base,leg_length=.85,stance_tilt=28,swing_tilt=50))
crab=dict(parents=parents,heads=heads,bone_names=names,labels=[n or "end" for n in names],
          rest_matrices=[np.eye(4).tolist() if n else None for n in names],
          collision_capsules=capsules,foot_profiles=profiles)
crab_local=np.tile(np.eye(3),(20,len(parents),1,1))
crab_p,crab_r=forward_kinematics(np.tile([0,0,.1],(20,1)),crab_local,crab)   # body on the ground, feet 0.9 m below it
assert crab_p[:,[3,7,11],2].max()<-.5
landed_p,landed_r,landed_report=plant(crab_p,crab_r,crab,ground)
assert landed_report["floor_contact_frames"]==0, landed_report["floor_contact_frames"]
assert landed_p[:,[3,7,11],2].min()>-.02, landed_p[:,[3,7,11],2].min()
result=dict(passed=["low-slung many-legged body lifted onto its feet","kneeling knee stays down, tucked foot raised","swivelling knee held smoothly","grounded support reduces slide","root path preserved","bone lengths preserved",
                    "inclined mesh height and normal","outside-mesh fallback","phase-aware limits",
                    "planted knee keeps generated bend side"],
            planted_step_before=report["median_planted_step_before"],
            planted_step_after=report["median_planted_step"])
(ROOT/"tests/artifacts/ground-regressions.json").write_text(json.dumps(result,indent=2))
print(json.dumps(result))
