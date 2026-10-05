"""Minimal bpy-free GLB reader: node transforms, skin joints, skinned rest geometry."""

import json
import struct

import numpy as np

_COMPONENTS = {5120: np.int8, 5121: np.uint8, 5122: np.int16, 5123: np.uint16,
               5125: np.uint32, 5126: np.float32}
_SIZES = {'SCALAR': 1, 'VEC2': 2, 'VEC3': 3, 'VEC4': 4, 'MAT4': 16}
_JSON_CHUNK, _BIN_CHUNK = 0x4E4F534A, 0x004E4942


class Glb:
    """One GLB file: its glTF JSON (``json``), binary chunk and node list."""

    def __init__(self, path):
        with open(path, 'rb') as f:
            data = f.read()
        self.json, self.bin = {}, b''
        offset = 12
        while offset < len(data):
            length, kind = struct.unpack_from('<II', data, offset)
            chunk = data[offset + 8:offset + 8 + length]
            if kind == _JSON_CHUNK:
                self.json = json.loads(chunk)
            elif kind == _BIN_CHUNK:
                self.bin = chunk
            offset += 8 + length
        self.nodes = self.json.get('nodes', [])

    def accessor(self, index):
        """Accessor *index* as a float64 ``(count, components)`` array."""
        acc = self.json['accessors'][index]
        view = self.json['bufferViews'][acc['bufferView']]
        dtype = np.dtype(_COMPONENTS[acc['componentType']])
        width = _SIZES[acc['type']]
        start = view.get('byteOffset', 0) + acc.get('byteOffset', 0)
        stride = view.get('byteStride', 0) or dtype.itemsize * width
        raw = np.frombuffer(self.bin, dtype=np.uint8, offset=start,
                            count=stride * (acc['count'] - 1) + dtype.itemsize * width)
        rows = np.lib.stride_tricks.as_strided(raw, (acc['count'], dtype.itemsize * width),
                                               (stride, 1))
        arr = np.ascontiguousarray(rows).view(dtype).reshape(acc['count'], width)
        arr = arr.astype(np.float64)
        if acc.get('normalized'):
            arr /= np.iinfo(dtype).max
        return arr

    @staticmethod
    def _local(node):
        if 'matrix' in node:
            return np.array(node['matrix'], dtype=np.float64).reshape(4, 4).T
        x, y, z, w = node.get('rotation', [0, 0, 0, 1])
        rot = np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])
        mat = np.eye(4)
        mat[:3, :3] = rot * np.array(node.get('scale', [1, 1, 1]))
        mat[:3, 3] = node.get('translation', [0, 0, 0])
        return mat

    def world(self):
        """World matrices of every node, ``(N, 4, 4)``."""
        parent = {c: i for i, n in enumerate(self.nodes) for c in n.get('children', [])}
        out = [None] * len(self.nodes)

        def get(i):
            if out[i] is None:
                local = self._local(self.nodes[i])
                out[i] = local if i not in parent else get(parent[i]) @ local
            return out[i]
        return np.stack([get(i) for i in range(len(self.nodes))]) if self.nodes \
            else np.zeros((0, 4, 4))

    def joints(self):
        """``{name: world 4x4}`` of every skin joint."""
        world = self.world()
        return {self.nodes[j].get('name'): world[j]
                for skin in self.json.get('skins', []) for j in skin['joints']}

    def skinned_triangles(self):
        """Rest-pose triangles of every skinned primitive as ``(T, 9)``, and the
        dominant joint name of each of their corners, ``(T, 3)``."""
        world = self.world()
        tris, dominant = [], []
        for node in self.nodes:
            if 'mesh' not in node or 'skin' not in node:
                continue
            skin = self.json['skins'][node['skin']]
            joints = skin['joints']
            ibm = self.accessor(skin['inverseBindMatrices']).reshape(-1, 4, 4).transpose(0, 2, 1)
            joint_mats = world[joints] @ ibm
            names = np.array([self.nodes[j].get('name') for j in joints])
            for prim in self.json['meshes'][node['mesh']]['primitives']:
                attrs = prim['attributes']
                if 'JOINTS_0' not in attrs:
                    continue
                pos = self.accessor(attrs['POSITION'])
                jidx = self.accessor(attrs['JOINTS_0']).astype(int)
                wts = self.accessor(attrs['WEIGHTS_0'])
                wts = wts / np.maximum(wts.sum(1, keepdims=True), 1e-12)
                homo = np.c_[pos, np.ones(len(pos))]
                verts = sum(wts[:, k:k + 1] * np.einsum('nij,nj->ni', joint_mats[jidx[:, k]], homo)[:, :3]
                            for k in range(jidx.shape[1]))
                faces = (self.accessor(prim['indices']).astype(int).reshape(-1, 3)
                         if 'indices' in prim else np.arange(len(pos)).reshape(-1, 3))
                tris.append(verts[faces].reshape(-1, 9))
                dominant.append(names[jidx[np.arange(len(jidx)), wts.argmax(1)]][faces])
        if not tris:
            return np.zeros((0, 9)), np.zeros((0, 3), dtype=object)
        return np.concatenate(tris), np.concatenate(dominant)
