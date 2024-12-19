import bpy
import os
import numpy as np

# Enable GPU rendering
bpy.context.preferences.addons['cycles'].preferences.compute_device_type = 'OPTIX'  # Use 'CUDA' or 'OPTIX' for NVIDIA GPUs, 'HIP' for AMD GPUs, or 'METAL' for macOS.
bpy.context.scene.render.engine = 'CYCLES'  # Use the Cycles rendering engine
bpy.context.scene.cycles.device = 'GPU'  # Set Cycles to use GPU rendering

# Select the GPU devices (optional, auto-selects all available if not set)
devices = bpy.context.preferences.addons['cycles'].preferences.get_devices()
if devices and devices[0]:  # Check if devices is not None and not empty
    for device in devices[0]:
        device.use = True  # Enable all available GPU devices

# Define the output directory
output_dir = "/home/seaclear/hamish/dome_port_sim/focus"
os.makedirs(output_dir, exist_ok=True)

# Get the active camera
camera = bpy.context.scene.camera

# Define the range of focus distances (in Blender units)
focus_distances = np.linspace(0.1, 10.0, 100)  # Adjust range as needed

def render_next_focus_distance(index=0):
    if index >= len(focus_distances):
        print('Rendering completed!')
        return None  # Stop the timer

    # Set the focus distance
    focus_distance = focus_distances[index]
    camera.data.dof.use_dof = True  # Enable depth of field
    camera.data.dof.focus_distance = focus_distance

    # Set output file path
    output_file = os.path.join(output_dir, f"render_focus_distance_{focus_distance:.2f}.png")
    bpy.context.scene.render.filepath = output_file

    # Render the image
    bpy.ops.render.render(write_still=True)
    print(f"Rendered Focus Distance: {focus_distance:.2f}")

    # Continue rendering the next focus distance
    bpy.app.timers.register(lambda: render_next_focus_distance(index + 1), first_interval=1.0)

# Register the timer to start rendering
bpy.app.timers.register(render_next_focus_distance)