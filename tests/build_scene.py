"""Create two small rigged mannequins for integration testing; preserve existing objects."""
import bpy
import math
from mathutils import Vector
from pathlib import Path

def material(name, color, metallic=.1):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = (*color, 1)
    mat.use_nodes = True
    shader = mat.node_tree.nodes.get("Principled BSDF")
    shader.inputs["Base Color"].default_value = (*color, 1)
    shader.inputs["Metallic"].default_value = metallic
    shader.inputs["Roughness"].default_value = .38
    return mat

def human_bones():
    bones = [
        ("hips", (0,0,1.0), (0,0,1.15), None, .17),
        ("spine", (0,0,1.15), (0,0,1.35), "hips", .17),
        ("chest", (0,0,1.35), (0,0,1.5), "spine", .23),
        ("neck", (0,0,1.5), (0,0,1.62), "chest", .075),
        ("head", (0,0,1.62), (0,-.015,1.86), "neck", .15),
    ]
    for s, side in ((1, "left"), (-1, "right")):
        bones += [
            ("upper_arm."+side, (s*.22,0,1.46), (s*.52,0,1.46), "chest", .07),
            ("forearm."+side, (s*.52,0,1.46), (s*.78,0,1.46), "upper_arm."+side, .055),
            ("hand."+side, (s*.78,0,1.46), (s*.91,0,1.46), "forearm."+side, .045),
            ("thigh."+side, (s*.105,0,1.02), (s*.12,-.012,.57), "hips", .09),
            ("shin."+side, (s*.12,-.012,.57), (s*.12,0,.13), "thigh."+side, .065),
            ("foot."+side, (s*.12,0,.13), (s*.12,-.20,.08), "shin."+side, .07),
        ]
    return bones

def creature_bones():
    bones = [
        ("pelvis", (0,.5,.82), (0,.28,.87), None, .24),
        ("spine", (0,.28,.87), (0,-.1,.90), "pelvis", .27),
        ("chest", (0,-.1,.90), (0,-.38,.94), "spine", .26),
        ("neck", (0,-.38,.94), (0,-.58,1.15), "chest", .15),
        ("head", (0,-.58,1.15), (0,-.90,1.13), "neck", .20),
        ("tail_base", (0,.57,.85), (0,.92,.92), "pelvis", .07),
        ("tail_tip", (0,.92,.92), (0,1.25,.80), "tail_base", .045),
    ]
    for s, side in ((1,"left"), (-1,"right")):
        bones += [
            ("front_upper_leg."+side, (s*.22,-.32,.9), (s*.24,-.25,.48), "chest", .075),
            ("front_lower_leg."+side, (s*.24,-.25,.48), (s*.24,-.38,.12), "front_upper_leg."+side, .055),
            ("front_paw."+side, (s*.24,-.38,.12), (s*.24,-.55,.075), "front_lower_leg."+side, .065),
            ("hind_thigh."+side, (s*.22,.48,.83), (s*.25,.28,.51), "pelvis", .1),
            ("hind_shin."+side, (s*.25,.28,.51), (s*.25,.55,.21), "hind_thigh."+side, .055),
            ("hind_paw."+side, (s*.25,.55,.21), (s*.25,.31,.075), "hind_shin."+side, .065),
        ]
    return bones

def spider_bones():
    """Eight legs (coxa, femur, tibia, tarsus); the tarsus tips touch the ground."""
    bones = [("body", (0,.05,.30), (0,-.12,.31), None, .14),
             ("abdomen", (0,.05,.32), (0,.50,.40), "body", .20),
             ("head", (0,-.12,.31), (0,-.24,.30), "body", .08)]
    for s, side in ((1, "left"), (-1, "right")):
        for i, angle in enumerate((42, 14, -14, -42)):
            dx, dy = math.cos(math.radians(angle)), -math.sin(math.radians(angle))
            def at(reach, z, s=s, dx=dx, dy=dy):
                return (s*dx*reach, .05 + dy*reach, z)
            tag = "leg%d_" % (i + 1)
            bones += [
                (tag+"coxa."+side, at(.10, .30), at(.20, .34), "body", .035),
                (tag+"femur."+side, at(.20, .34), at(.42, .56), tag+"coxa."+side, .03),
                (tag+"tibia."+side, at(.42, .56), at(.72, .20), tag+"femur."+side, .022),
                (tag+"tarsus."+side, at(.72, .20), at(.88, .03), tag+"tibia."+side, .015),
            ]
    return bones

def crab_bones():
    """Six walking legs (upper, lower, dactyl) and two claws held above the ground."""
    bones = [("carapace", (0,.1,.22), (0,-.15,.22), None, (.30,.12))]
    for s, side in ((1, "left"), (-1, "right")):
        for i, y in enumerate((.08, -.02, -.12)):
            tag = "leg%d_" % (i + 1)
            bones += [
                (tag+"upper."+side, (s*.25,y,.22), (s*.52,y,.40), "carapace", .03),
                (tag+"lower."+side, (s*.52,y,.40), (s*.85,y,.12), tag+"upper."+side, .025),
                (tag+"dactyl."+side, (s*.85,y,.12), (s*.95,y,.02), tag+"lower."+side, .015),
            ]
        bones += [
            ("claw_arm."+side, (s*.20,-.2,.24), (s*.40,-.42,.30), "carapace", .05),
            ("claw_fore."+side, (s*.40,-.42,.30), (s*.50,-.70,.28), "claw_arm."+side, .045),
            ("claw_a."+side, (s*.50,-.70,.28), (s*.38,-.92,.26), "claw_fore."+side, .03),
            ("claw_b."+side, (s*.50,-.70,.28), (s*.64,-.92,.26), "claw_fore."+side, .03),
        ]
    return bones

def bird_bones():
    """Two legs ending in 'foot' bones, two spread wings, a tail and a neck."""
    bones = [
        ("pelvis", (0,.12,.36), (0,0,.40), None, .12),
        ("chest", (0,0,.40), (0,-.15,.44), "pelvis", .14),
        ("neck", (0,-.15,.44), (0,-.22,.60), "chest", .05),
        ("head", (0,-.22,.60), (0,-.34,.63), "neck", .06),
        ("tail", (0,.12,.36), (0,.45,.32), "pelvis", (.10,.015)),
        ("tail_tip", (0,.45,.32), (0,.70,.28), "tail", (.12,.01)),
    ]
    for s, side in ((1, "left"), (-1, "right")):
        bones += [
            ("wing_shoulder."+side, (s*.08,-.05,.46), (s*.30,-.05,.50), "chest", (.05,.03)),
            ("wing_upper."+side, (s*.30,-.05,.50), (s*.60,-.02,.52), "wing_shoulder."+side, (.10,.012)),
            ("wing_fore."+side, (s*.60,-.02,.52), (s*.95,.05,.50), "wing_upper."+side, (.12,.012)),
            ("wing_hand."+side, (s*.95,.05,.50), (s*1.30,.12,.47), "wing_fore."+side, (.14,.01)),
            ("thigh."+side, (s*.07,.08,.33), (s*.07,.02,.20), "pelvis", .05),
            ("shin."+side, (s*.07,.02,.20), (s*.07,.10,.06), "thigh."+side, .025),
            ("foot."+side, (s*.07,.10,.06), (s*.07,-.04,.015), "shin."+side, .03),
        ]
    return bones

def dragon_bones():
    """Four legs ending in claws, two three-finger wings, a neck with a jaw and a long tail."""
    bones = [
        ("pelvis", (0,.5,.9), (0,.2,.95), None, .28),
        ("spine", (0,.2,.95), (0,-.2,1.0), "pelvis", .30),
        ("chest", (0,-.2,1.0), (0,-.6,1.05), "spine", .34),
        ("neck1", (0,-.6,1.05), (0,-.85,1.3), "chest", .14),
        ("neck2", (0,-.85,1.3), (0,-1.0,1.55), "neck1", .12),
        ("head", (0,-1.0,1.55), (0,-1.35,1.55), "neck2", .14),
        ("jaw", (0,-1.0,1.5), (0,-1.3,1.38), "head", .08),
        ("tail1", (0,.5,.9), (0,.9,.8), "pelvis", .15),
        ("tail2", (0,.9,.8), (0,1.4,.6), "tail1", .11),
        ("tail3", (0,1.4,.6), (0,1.9,.4), "tail2", .08),
        ("tail4", (0,1.9,.4), (0,2.3,.25), "tail3", .05),
    ]
    for s, side in ((1, "left"), (-1, "right")):
        bones += [
            ("front_upper."+side, (s*.3,-.5,.95), (s*.35,-.4,.6), "chest", .09),
            ("front_lower."+side, (s*.35,-.4,.6), (s*.35,-.55,.3), "front_upper."+side, .07),
            ("front_meta."+side, (s*.35,-.55,.3), (s*.35,-.5,.1), "front_lower."+side, .05),
            ("front_claw."+side, (s*.35,-.5,.1), (s*.35,-.7,.03), "front_meta."+side, .045),
            ("hind_upper."+side, (s*.3,.45,.9), (s*.38,.3,.55), "pelvis", .13),
            ("hind_lower."+side, (s*.38,.3,.55), (s*.38,.55,.3), "hind_upper."+side, .09),
            ("hind_meta."+side, (s*.38,.55,.3), (s*.38,.5,.1), "hind_lower."+side, .06),
            ("hind_claw."+side, (s*.38,.5,.1), (s*.38,.3,.03), "hind_meta."+side, .05),
            ("wing_shoulder."+side, (s*.25,-.3,1.1), (s*.7,-.3,1.4), "chest", (.12,.05)),
            ("wing_arm."+side, (s*.7,-.3,1.4), (s*1.3,-.2,1.7), "wing_shoulder."+side, (.16,.025)),
            ("wing_fore."+side, (s*1.3,-.2,1.7), (s*2.0,0,1.8), "wing_arm."+side, (.18,.02)),
            ("wing_finger1."+side, (s*2.0,0,1.8), (s*2.8,.3,1.7), "wing_fore."+side, (.15,.015)),
            ("wing_finger2."+side, (s*2.0,0,1.8), (s*2.4,.6,1.6), "wing_fore."+side, (.15,.015)),
            ("wing_finger3."+side, (s*2.0,0,1.8), (s*2.1,.8,1.5), "wing_fore."+side, (.15,.015)),
        ]
    return bones

def snake_bones():
    """Serpentine: a chain along the ground in both directions from a middle root. No limbs."""
    bones = [("mid", (0,0,.06), (0,-.12,.06), None, .05)]
    for i in range(7):
        parent = "mid" if i == 0 else "front%d" % i
        bones.append(("front%d" % (i + 1), (0,-.12*(i+1),.06), (0,-.12*(i+2),.06), parent, max(.05-.004*i, .02)))
        parent = "mid" if i == 0 else "back%d" % i
        bones.append(("back%d" % (i + 1), (0,.12*i,.06), (0,.12*(i+1),.06), parent, max(.05-.004*i, .015)))
    return bones

def fish_bones():
    """Marine: spine, tail fin and fins, all in the air. No limbs."""
    bones = [("body", (0,0,.5), (0,-.2,.5), None, (.10,.14)),
             ("head", (0,-.2,.5), (0,-.5,.5), "body", (.08,.10)),
             ("tail1", (0,0,.5), (0,.3,.5), "body", (.07,.10)),
             ("tail2", (0,.3,.5), (0,.6,.5), "tail1", (.05,.07)),
             ("tail3", (0,.6,.5), (0,.85,.5), "tail2", (.03,.05)),
             ("tail_fin", (0,.85,.5), (0,1.0,.5), "tail3", (.01,.2)),
             ("dorsal_fin", (0,-.05,.62), (0,.15,.78), "body", (.01,.1))]
    for s, side in ((1, "left"), (-1, "right")):
        bones += [("fin_a."+side, (s*.09,-.15,.48), (s*.3,-.1,.42), "body", (.08,.008)),
                  ("fin_b."+side, (s*.3,-.1,.42), (s*.5,0,.38), "fin_a."+side, (.07,.006))]
    return bones

CREATURES = (("UniMate Spider", spider_bones, (.35,.30,.28)),
             ("UniMate Crab", crab_bones, (.75,.25,.18)),
             ("UniMate Bird", bird_bones, (.20,.45,.75)),
             ("UniMate Dragon", dragon_bones, (.55,.18,.45)),
             ("UniMate Snake", snake_bones, (.25,.55,.20)),
             ("UniMate Fish", fish_bones, (.85,.62,.15)))

def build_creatures(spacing=6.0):
    """Create the multi-legged, winged, serpentine and marine test rigs in a row along +X."""
    collection = bpy.data.collections.get("UniMate Test Rigs")
    if not collection:
        collection = bpy.data.collections.new("UniMate Test Rigs")
        bpy.context.scene.collection.children.link(collection)
    rigs = {}
    for index, (name, builder, color) in enumerate(CREATURES):
        rigs[name.split()[-1].lower()] = make_rig(
            name, builder(), (index*spacing + 6, 8, 0), material(name + " Mat", color), collection)
    return rigs

def make_rig(name, bones, location, mat, collection):
    existing = bpy.data.objects.get(name)
    if existing:
        return existing
    armature = bpy.data.armatures.new(name + " Skeleton")
    rig = bpy.data.objects.new(name, armature)
    collection.objects.link(rig)
    rig.location = location
    rig.show_in_front = True
    rig.display_type = "WIRE"
    armature.display_type = "STICK"
    bpy.ops.object.select_all(action="DESELECT")
    rig.select_set(True)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    for bone_name, head, tail, parent, radius in bones:
        bone = armature.edit_bones.new(bone_name)
        bone.head, bone.tail = head, tail
        if parent:
            bone.parent = armature.edit_bones[parent]
        # Deliberately vary roll: import must work for arbitrary bone frames.
        bone.roll = .17 if "left" in bone_name else -.11
    bpy.ops.object.mode_set(mode="OBJECT")
    pieces = []
    for bone_name, head, tail, parent, radius in bones:
        head, tail = Vector(head), Vector(tail)
        direction = tail - head
        bpy.ops.mesh.primitive_uv_sphere_add(segments=16, ring_count=10)
        obj = bpy.context.object
        obj.name = name + " | " + bone_name
        rot = direction.to_track_quat("Z", "Y")
        center = (head + tail) * .5
        for vertex in obj.data.vertices:
            point = vertex.co.copy()
            rx, ry = radius if isinstance(radius, tuple) else (radius, radius)
            point.x *= rx  # across the bone, horizontal
            point.y *= ry  # across the bone, vertical (flat wings and fins use a small ry)
            point.z *= direction.length * .58 + max(rx, ry) * .25
            vertex.co = rot @ point + center
        obj.location = location
        obj.data.materials.append(mat)
        for poly in obj.data.polygons:
            poly.use_smooth = True
        group = obj.vertex_groups.new(name=bone_name)
        group.add(list(range(len(obj.data.vertices))), 1., "REPLACE")
        modifier = obj.modifiers.new("Deform", "ARMATURE")
        modifier.object = rig
        for coll in list(obj.users_collection):
            coll.objects.unlink(obj)
        collection.objects.link(obj)
        pieces.append(obj)
    rig["unimate_test_rig"] = True
    rig["unimate_description"] = "Simple weighted test mannequin, facing local -Y"
    return rig

def build():
    collection = bpy.data.collections.get("UniMate Test Rigs")
    if not collection:
        collection = bpy.data.collections.new("UniMate Test Rigs")
        bpy.context.scene.collection.children.link(collection)
    human = make_rig("UniMate Human", human_bones(), (-1.35,0,0),
                     material("UniMate Teal", (.045,.47,.44)), collection)
    creature = make_rig("UniMate Creature", creature_bones(), (1.35,0,0),
                        material("UniMate Copper", (.62,.22,.07)), collection)
    bpy.ops.object.select_all(action="DESELECT")
    human.select_set(True)
    bpy.context.view_layer.objects.active = human
    scene = bpy.context.scene
    scene.render.fps = 30
    scene.frame_start, scene.frame_end = 1, 60
    if hasattr(scene, "unimate_motion"):
        scene.unimate_motion.rig = human
    for screen in bpy.data.screens:
        for area in screen.areas:
            if area.type == "VIEW_3D":
                space = area.spaces.active
                space.shading.type = "MATERIAL"
                space.region_3d.view_distance = 6.3
                space.region_3d.view_location = (0,0,.85)
                from mathutils import Quaternion
                from math import radians
                space.region_3d.view_rotation = (Vector((0,-6,3)).to_track_quat("Z","Y"))
                space.show_region_ui = True
    return human, creature

if __name__ == "__main__":
    build()
