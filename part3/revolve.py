import bpy
import mathutils
from mathutils import Vector
import bmesh
from bpy import context
def triangulate_object(obj):
    me = obj.data
    # Get a BMesh representation
    bm = bmesh.new()
    bm.from_mesh(me)

    bmesh.ops.triangulate(bm, faces=bm.faces[:], quad_method=0, ngon_method=0)

    # Finish up, write the bmesh back to the mesh
    bm.to_mesh(me)
    bm.free()
    return me
def createMeshFromData(name, origin, verts, faces):
    # Create mesh and object
    me = bpy.data.meshes.new(name+'Mesh')
    ob = bpy.data.objects.new(name, me)
    ob.location = origin
    ob.show_name = True
 
    # Link object to scene and make active
    scn = bpy.context.scene
    scn.objects.link(ob)
    scn.objects.active = ob
    ob.select = True
 
    # Create mesh from given verts, faces.
    me.from_pydata(verts, [], faces)
    # Update mesh with new data
    me=triangulate_object(ob)
    me.update()    
    return ob


def readVertices(fileName):
    vertices=[]
    for line in open(fileName,"r"):
        if line.startswith('#'):continue
        values=line.split()
        if not values:continue
        if values[0]=='v':
            vertex=[]
            vertex.append(10*float(values[1]))
            vertex.append(10*float(values[2]))
            vertex.append(10*float(values[3]))
            vertices.append(vertex)
    return vertices

def readSplit(fileName):
    for line in open(fileName,"r"):
        if line.startswith('#'):continue
        values=line.split()
        if not values:continue
        if values[0]!='v':
            sp=int(values[0])
    return sp

def duplicate(vertices):
    verticesReturn=[] 
    vertices2=[]
    verticesReturn.extend(vertices)
    for vertex in vertices:
        vertex2=[]
        vertex2.append(vertex[0])
        vertex2.append(vertex[1])
        vertex2.append(vertex[2]+1.2)
        vertices2.append(vertex2)
    verticesReturn.extend(vertices2)
    return verticesReturn


def applyModifier (mod,objA, objB):
    target = objA 
    bpy.context.scene.objects.active = target
    boo = target.modifiers.new('Booh', 'BOOLEAN')
    boo.object = objB
    boo.operation = mod
    bpy.ops.object.modifier_apply(apply_as='DATA', modifier="Booh")
    bpy.context.scene.objects.unlink(objB)
    return target


def makeFaces(verts):
    faces=[]
    index=[]
    half=int(len(verts)/2)
    for i in range(half):
        index.append(i)
    faces.append(index)
    index=[]
    for i in range(half):
        index.append(i+half)
    faces.append(index)
    for i in range(half-1):
        index=[]
        index.append(i)
        index.append(i+1)
        index.append(i+1+half)
        index.append(i+half)
        faces.append(index)
    index=[]
    index.append(half-1)
    index.append(0)
    index.append(half)
    index.append(len(verts)-1)
    faces.append(index)
    return faces

def export_obj(filepath,obj):
    mesh = obj.data
    with open(filepath, 'w') as f:
        f.write("# OBJ file\n")
        for v in mesh.vertices:
            f.write("v %.4f %.4f %.4f\n" % v.co[:])
        for p in mesh.polygons:
            f.write("f")
            for i in p.vertices:
                f.write(" %d" % (i + 1))
            f.write("\n")
    
def male_connector(ob,loc):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.503,location=(loc[0],loc[1],loc[2]+0.6),end_fill_type='TRIFAN',depth=1.2)
    objA=bpy.context.object
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2,location=(loc[0],loc[1],loc[2]),end_fill_type='TRIFAN', depth=20)

    objB=bpy.context.object

    target=applyModifier ('DIFFERENCE',objA, objB)

    bpy.ops.mesh.primitive_cylinder_add(radius=0.6,location=(loc[0],loc[1],loc[2]+0.6),end_fill_type='TRIFAN',depth=0.6)
    objA=bpy.context.object

    target=applyModifier ('DIFFERENCE',target, objA)


    bpy.ops.mesh.primitive_cylinder_add(radius=0.2,location=(loc[0],loc[1],loc[2]), depth=20)

    objB=bpy.context.object
    target2=applyModifier ('DIFFERENCE',ob, objB)
    
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5,location=(loc[0],loc[1],loc[2]+0.6),end_fill_type='TRIFAN',depth=20)
    objA=bpy.context.object
    target3=applyModifier ('DIFFERENCE',target2, objA)

    finalResult=applyModifier ('UNION',target, target3)
    triangulate_object(finalResult)
    return finalResult


def female_connector(ob,loc):
    bpy.ops.mesh.primitive_cylinder_add(radius=0.5,location=(loc[0],loc[1],loc[2]+0.6),end_fill_type='TRIFAN',depth=0.6)
    objA=bpy.context.object
    bpy.ops.mesh.primitive_cylinder_add(radius=0.2,location=(loc[0],loc[1],loc[2]),end_fill_type='TRIFAN', depth=20)

    objB=bpy.context.object

    target=applyModifier ('DIFFERENCE',objA, objB)

#    bpy.ops.mesh.primitive_cylinder_add(radius=0.2,location=(loc[0],loc[1],loc[2]), depth=20)

#    objB=bpy.context.object
    #target2=applyModifier ('DIFFERENCE',ob, objB)

    #finalResult=applyModifier ('UNION',target2, target
    bpy.ops.mesh.primitive_cylinder_add(radius=0.495,location=(loc[0],loc[1],loc[2]+0.6),end_fill_type='TRIFAN',depth=20)
    objA=bpy.context.object
    target2=applyModifier ('DIFFERENCE',ob, objA)
    
    finalResult=applyModifier ('UNION',target2, target)
    triangulate_object(finalResult)
    return finalResult
     
def make_ob_file(verts):
    faces=makeFaces(verts)
    ob=createMeshFromData("test",(0,0,0),verts,faces)
    return ob
def make_Verts(file_path):
    verts=readVertices(file_path)
    verts=duplicate(verts)
    return verts
  
     
file_path="C:\\Users\\amahdavi\\Desktop\\piece.txt"
verts=make_Verts(file_path)
sp=readSplit(file_path)
ob=make_ob_file(verts)
loc=[]
loc.append(verts[int(len(verts)/2)-1][0])
loc.append(verts[int(len(verts)/2)-1][1])
loc.append(verts[int(len(verts)/2)-1][2])
finalResult=male_connector(ob,loc)
loc=[]
loc.append(verts[sp][0])
loc.append(verts[sp][1])
loc.append(verts[sp][2])
finalResult=male_connector(finalResult,loc)
filepath = "C:\\Users\\amahdavi\\Desktop\\test4.obj"
obj = context.object
export_obj(filepath,obj)