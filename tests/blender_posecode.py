"""Posecode import inside Blender: imports twice without a dialog in background mode."""
import importlib.util
import json
import sys
import tempfile
from pathlib import Path
import bpy

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / "addon"), str(ROOT / "tests")]
spec = importlib.util.spec_from_file_location("test_posecode", ROOT / "tests" / "test_posecode.py")
test_posecode = importlib.util.module_from_spec(spec)
spec.loader.exec_module(test_posecode)
import unimate_motion

bpy.ops.wm.read_factory_settings(use_empty=True)
unimate_motion.register()

def mixamo_rig():
    """A bare armature with the fixture's Mixamo bone names, positions and hierarchy."""
    armature = bpy.data.armatures.new("Posecode Rig")
    rig = bpy.data.objects.new("Posecode Rig", armature)
    bpy.context.scene.collection.objects.link(rig)
    bpy.context.view_layer.objects.active = rig
    bpy.ops.object.mode_set(mode="EDIT")
    names, parents, heads = test_posecode.NAMES, test_posecode.PARENTS, test_posecode.HEADS
    for name, head in zip(names, heads):
        bone = armature.edit_bones.new(name)
        bone.head = head
        children = [j for j, p in enumerate(parents) if p == names.index(name)]
        bone.tail = heads[children[0]] if children else (head[0], head[1], head[2] + .1)
    for name, parent in zip(names, parents):
        if parent >= 0:
            armature.edit_bones[name].parent = armature.edit_bones[names[parent]]
    bpy.ops.object.mode_set(mode="OBJECT")
    return rig

human = mixamo_rig()
settings = bpy.context.scene.unimate_motion
settings.rig, settings.family, settings.forward = human, "mixamo", "-Y"
bpy.context.view_layer.objects.active = human
manifest = test_posecode.manifest()
bpy.context.scene.render.fps = manifest["timing"]["fps"]
path = Path(tempfile.mkdtemp()) / "manifest.json"
path.write_text(json.dumps(manifest))

assert bpy.ops.unimate.import_posecode(filepath=str(path)) == {"FINISHED"}, settings.status
count = len(settings.clips)
assert count > 0
# A second import finds existing clips. With a window it asks first; in background Blender
# there is no dialog (opening one crashes Blender), so the timeline is replaced.
assert bpy.ops.unimate.import_posecode(filepath=str(path)) == {"FINISHED"}, settings.status
assert len(settings.clips) == count
report = dict(passed=["import creates clips", "second import replaces the timeline without a dialog in background mode"], clips=count)
(ROOT / "tests" / "artifacts").mkdir(exist_ok=True)
(ROOT / "tests" / "artifacts" / "posecode-blender.json").write_text(json.dumps(report, indent=2))
print("POSECODE_BLENDER_PASSED", json.dumps(report))
unimate_motion.unregister()
