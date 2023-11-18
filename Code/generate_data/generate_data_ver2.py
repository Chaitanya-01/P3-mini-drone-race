#import subprocess
#import sys
#import os
# 
## path to python.exe
#python_exe = os.path.join(sys.prefix, 'bin', 'python.exe')
# 
## upgrade pip
#subprocess.call([python_exe, "-m", "ensurepip"])
#subprocess.call([python_exe, "-m", "pip", "install", "--upgrade", "pip"])
# 
## install required packages
#subprocess.call([python_exe, "-m", "pip", "install", "scipy"])

import os
import sys
import random

import importlib
import numpy as np
import bpy


class generate_data:
    def __init__(self,limits,window_count):
        self.scene = bpy.data.scenes['Scene']
        
        self.camera = bpy.data.objects['Camera']
        self.axis = bpy.data.objects['Axis']
        
        self.light1 = bpy.data.objects['Light1']
        self.light2 = bpy.data.objects['Light2']
        
        #-----For environment 1--------------------------------
        self.frontwindow = bpy.data.objects['FrontWindow']
        self.window1 = bpy.data.objects['Window.001']
        self.window2 = bpy.data.objects['Window.002']
        self.window3 = bpy.data.objects['Window.003']
        self.window4 = bpy.data.objects['Window.004']
        self.window5 = bpy.data.objects['Window.005']
        
        self.limits = limits # A dict
        self.window_count = window_count
        #-----------------------------------------------------------
#        self.absolute_path = os.path.dirname(__file__)
#        # Filepaths
#        self.frame_path = os.path.join(self.absolute_path,'..','frames/')
#        self.segmask_path = os.path.join(self.absolute_path,'..','seg_masks/')
#        self.frontseg_path = os.path.join(self.absolute_path,'..','frontseg_masks/')
        
        # Compositing nodes
        bpy.context.scene.use_nodes = True
        self.tree = bpy.context.scene.node_tree
        
    
    def setRenderSettings(self):
        # get_devices() to let Blender detects GPU device
        bpy.context.preferences.addons["cycles"].preferences.get_devices()
        print(bpy.context.preferences.addons["cycles"].preferences.compute_device_type)
        for d in bpy.context.preferences.addons["cycles"].preferences.devices:
            d["use"] = 1 # Using all devices, include GPU and CPU
            print(d["name"], d["use"])
        bpy.context.scene.render.engine = 'CYCLES'
        bpy.context.scene.cycles.device = 'GPU'

        # Setting up file output paths
        curr_filepath = os.path.dirname(__file__)
        # curr_filepath = curr_filepath.replace("main.blend","") #for main.blend
        curr_filepath = curr_filepath.replace("main2.blend","")
        output_folders = ["newframes/","newfrontseg_masks/","newsegment_mask/"]
        idx = 0
        for scene in bpy.data.scenes:
            for node in scene.node_tree.nodes:
                if node.type == 'OUTPUT_FILE':
                    # print("output nodes are:",os.path.join(curr_filepath,output_folders[idx]))
                    node.base_path = os.path.join(curr_filepath,output_folders[idx])
                    idx+=1

        
    
    def init_setup(self):

        self.setRenderSettings()
        print("camera location",)
        # Place the camera and axis at the start location
        self.camera.location = self.limits["camerainit_loc"]
        self.camera.rotation_euler = ((np.pi/180)*self.limits["camerainit_rot"][0],(np.pi/180)*self.limits["camerainit_rot"][1],(np.pi/180)*self.limits["camerainit_rot"][2])
        
        self.axis.location = (0,0,0)
        self.axis.rotation_euler = (0,0,0)
        # Place the light at the start location
        self.light1.location = self.limits["light1init_loc"]
        self.light1.rotation_euler = self.limits["light1init_rot"]
        self.light2.location = self.limits["light2init_loc"]
        self.light2.rotation_euler = self.limits["light2init_rot"]
        
        # Reset Frame to 0
        self.scene.frame_set(0)
        # Deselect all objects
        for obj in bpy.data.objects:
            obj.select_set(False)
            obj.animation_data_clear()
            
    def render_imgs(self):
        depth_variation = ((self.limits["axis_depth"][1]-self.limits["axis_depth"][0])/self.limits["depth_step"])
        axisrotationx_variation = ((self.limits["axisrotd_x"][1]-self.limits["axisrotd_x"][0])/self.limits["rotd_step"])
        axisrotationy_variation = ((self.limits["axisrotd_y"][1]-self.limits["axisrotd_y"][0])/self.limits["rotd_step"])
        axisrotationz_variation = ((self.limits["axisrotd_z"][1]-self.limits["axisrotd_z"][0])/self.limits["rotd_step"])
        num_renders = depth_variation*axisrotationx_variation*axisrotationy_variation*axisrotationz_variation
        print("Total number of renders:",num_renders)

        render_count = 0
        # Get limits
        # depth = random.randint(1,8)
        # for depth in range(10*self.limits["axis_depth"][0],10*self.limits["axis_depth"][1],int(10*self.limits["depth_step"])):
        # for rotx in range(self.limits["axisrotd_x"][0],self.limits["axisrotd_x"][1],self.limits["rotd_step"]+8):
        #     for roty in range(self.limits["axisrotd_y"][0],self.limits["axisrotd_y"][1],self.limits["rotd_step"]+3):
        #         for rotz in range(self.limits["axisrotd_z"][0],self.limits["axisrotd_z"][1],self.limits["rotd_step"]+8):
        #             # self.axis.location = (depth/10,0,0)
        #             self.axis.location = (depth,0,0)
        #             self.axis.rotation_euler = (np.pi*rotx/180,np.pi*roty/180,np.pi*rotz/180)
        #             render_count += 1
        #             print("On render:", render_count)
        #             print("--> Location of the axis:")
        #             print("     position:", self.axis.location, "m")
        #             print("     angle:", self.axis.rotation_euler, "deg")
                    
        #             self.light1.data.energy = random.randint(self.limits["light_power"][0], self.limits["light_power"][1])
        #             self.light2.data.energy = random.randint(self.limits["light_power"][0], self.limits["light_power"][1])

        #             # render image resolution
        #             bpy.context.scene.cycles.samples = 25
        #             self.scene.render.resolution_x = 480
        #             self.scene.render.resolution_y = 360
        #             self.scene.render.resolution_percentage = 95
                    
        #             # Define the base paths from the compositor nodes
        #             path_dir1 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 1"].base_path
        #             path_dir2 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 2"].base_path
        #             path_dir3 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 3"].base_path
                    
        #             # Set filenames for outputs
        #             bpy.data.scenes["Scene"].node_tree.nodes["File Output 1"].file_slots[0].path = f'Frame{render_count:06d}'
        #             bpy.data.scenes["Scene"].node_tree.nodes["File Output 2"].file_slots[0].path = f'frontmask{render_count:06d}'
        #             bpy.data.scenes["Scene"].node_tree.nodes["File Output 3"].file_slots[0].path = f'segmask{render_count:06d}'

        #             # Render the image. All outputs should save at their respective locations.
        #             bpy.ops.render.render(write_still=False)
                    
        # for rotx in range(self.limits["axisrotd_x"][0],self.limits["axisrotd_x"][1],self.limits["rotd_step"]+8):
        #     for roty in range(self.limits["axisrotd_y"][0],self.limits["axisrotd_y"][1],self.limits["rotd_step"]+3):
        #         for rotz in range(self.limits["axisrotd_z"][0]-40,self.limits["axisrotd_z"][1]+45,self.limits["rotd_step"]+8):
        #             # self.axis.location = (depth/10,0,0)
        #             self.axis.location = (depth,0,0)
        #             self.axis.rotation_euler = (np.pi*rotx/180,np.pi*roty/180,np.pi*rotz/180)
        #             render_count += 1
        #             print("On render:", render_count)
        #             print("--> Location of the axis:")
        #             print("     position:", self.axis.location, "m")
        #             print("     angle:", self.axis.rotation_euler, "deg")
                    
        #             self.light1.data.energy = random.randint(self.limits["light_power"][0], self.limits["light_power"][1])
        #             self.light2.data.energy = random.randint(self.limits["light_power"][0], self.limits["light_power"][1])

        #             # render image resolution
        #             bpy.context.scene.cycles.samples = 25
        #             self.scene.render.resolution_x = 480
        #             self.scene.render.resolution_y = 360
        #             self.scene.render.resolution_percentage = 95
                    
        #             # Define the base paths from the compositor nodes
        #             path_dir1 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 1"].base_path
        #             path_dir2 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 2"].base_path
        #             path_dir3 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 3"].base_path
                    
        #             # Set filenames for outputs
        #             bpy.data.scenes["Scene"].node_tree.nodes["File Output 1"].file_slots[0].path = f'Frame{render_count:06d}'
        #             bpy.data.scenes["Scene"].node_tree.nodes["File Output 2"].file_slots[0].path = f'frontmask{render_count:06d}'
        #             bpy.data.scenes["Scene"].node_tree.nodes["File Output 3"].file_slots[0].path = f'segmask{render_count:06d}'

        #             # Render the image. All outputs should save at their respective locations.
        #             bpy.ops.render.render(write_still=False)                       
        # for single window
        depth = random.randint(0,4)
        for rotx in range(self.limits["axisrotd_x"][0]-10,self.limits["axisrotd_x"][1]+25,self.limits["rotd_step"]+13):
            for roty in range(self.limits["axisrotd_y"][0],self.limits["axisrotd_y"][1],self.limits["rotd_step"]+3):
                for rotz in range(self.limits["axisrotd_z"][0]-10,self.limits["axisrotd_z"][1]+25,self.limits["rotd_step"]+13):
                    # self.axis.location = (depth/10,0,0)
                    self.axis.location = (depth,0,0)
                    self.axis.rotation_euler = (np.pi*rotx/180,np.pi*roty/180,np.pi*rotz/180)
                    render_count += 1
                    print("On render:", render_count)
                    print("--> Location of the axis:")
                    print("     position:", self.axis.location, "m")
                    print("     angle:", self.axis.rotation_euler, "deg")
                    
                    self.light1.data.energy = random.randint(self.limits["light_power"][0], self.limits["light_power"][1])
                    self.light2.data.energy = random.randint(self.limits["light_power"][0], self.limits["light_power"][1])

                    # render image resolution
                    bpy.context.scene.cycles.samples = 25
                    self.scene.render.resolution_x = 480
                    self.scene.render.resolution_y = 360
                    self.scene.render.resolution_percentage = 95
                    
                    # Define the base paths from the compositor nodes
                    path_dir1 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 1"].base_path
                    path_dir2 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 2"].base_path
                    path_dir3 = bpy.data.scenes["Scene"].node_tree.nodes["File Output 3"].base_path
                    
                    # Set filenames for outputs
                    bpy.data.scenes["Scene"].node_tree.nodes["File Output 1"].file_slots[0].path = f'Frame{render_count:06d}'
                    bpy.data.scenes["Scene"].node_tree.nodes["File Output 2"].file_slots[0].path = f'frontmask{render_count:06d}'
                    bpy.data.scenes["Scene"].node_tree.nodes["File Output 3"].file_slots[0].path = f'segmask{render_count:06d}'

                    # Render the image. All outputs should save at their respective locations.
                    bpy.ops.render.render(write_still=False)  
             
        
        
        

def main():
    limits = {"axis_depth":(0,4), "axisrotd_x":(-20,20), "axisrotd_y":(-10,10), "axisrotd_z":(-20,20), "depth_step":0.2, "rotd_step":2, "light_power":(20,2000),"camerainit_loc":(5,0,2),"camerainit_rot":(90,0,90),"light1init_loc":(2,1.5,3),"light1init_rot":(0,0,0),"light2init_loc":(-5,-2,3),"light2init_rot":(0,0,0)}
    num_windows = 6
    render = generate_data(limits,num_windows)
    render.init_setup()
    render.render_imgs()
#    print(list(bpy.data.objects))
#    print(bpy.context.object)

        

if __name__=="__main__":
    # donot run main.py if imported as a module
    main()