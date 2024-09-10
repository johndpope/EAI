import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter
import numpy as np
import json
from classes_and_palettes import GOLIATH_KPTS_COLORS

SKELETON_CONNECTIONS = [
    # Body
    (0, 1), (0, 2),  # Nose to eyes
    (1, 3), (2, 4),  # Eyes to ears
    (0, 5), (0, 6),  # Nose to shoulders
    (5, 7), (7, 9),  # Left arm
    (6, 8), (8, 10),  # Right arm
    (5, 11), (6, 12),  # Shoulders to hips
    (11, 13), (13, 15),  # Left leg
    (12, 14), (14, 16),  # Right leg
    (15, 17), (16, 20),  # Ankles to heels
    (15, 18), (15, 19),  # Left foot
    (16, 21), (16, 22),  # Right foot

    # Additional body connections
    (5, 67), (6, 68),  # Shoulders to acromion
    (7, 63), (8, 64),  # Elbows to olecranon
    (7, 65), (8, 66),  # Elbows to cubital fossa
    (5, 6),  # Shoulder to shoulder
    (11, 12),  # Hip to hip
    (69, 5), (69, 6),  # Neck to shoulders

    # Left Hand
    (9, 62),  # Wrist to hand root
    (62, 42), (42, 43), (43, 44), (44, 45),  # Thumb
    (62, 46), (46, 47), (47, 48), (48, 49),  # Index finger
    (62, 50), (50, 51), (51, 52), (52, 53),  # Middle finger
    (62, 54), (54, 55), (55, 56), (56, 57),  # Ring finger
    (62, 58), (58, 59), (59, 60), (60, 61),  # Pinky finger

    # Right Hand
    (10, 41),  # Wrist to hand root
    (41, 21), (21, 22), (22, 23), (23, 24),  # Thumb
    (41, 25), (25, 26), (26, 27), (27, 28),  # Index finger
    (41, 29), (29, 30), (30, 31), (31, 32),  # Middle finger
    (41, 33), (33, 34), (34, 35), (35, 36),  # Ring finger
    (41, 37), (37, 38), (38, 39), (39, 40),  # Pinky finger

    # Face (simplified, you can add more detailed connections if needed)
    (70, 71), (71, 72),  # Glabella to nose bridge
    (73, 74), (74, 75),  # Nose bridge connections
    (76, 77),  # Labiomental groove to chin
    (78, 80), (87, 89),  # Eyebrows
    (96, 97), (120, 121),  # Upper eyelids
    (144, 145), (161, 162),  # Lower eyelids
    (178, 179),  # Nose tip to bottom
    (180, 181),  # Nose corners
    (188, 189),  # Mouth corners
    (190, 191),  # Upper and lower lip centers

    # Ears (simplified)
    (256, 257), (282, 283),  # Top of ears
    (277, 279), (303, 305),  # Bottom of ears
]


def load_keypoints(file_path):
    with open(file_path, 'r') as f:
        data = json.load(f)
    return data[0]

def extract_coordinates(keypoints):
    return np.array([[kp[0], kp[1]] for kp in keypoints.values()])

def interpolate_keypoints(start_kps, end_kps, num_frames):
    interpolated = []
    for i in range(num_frames):
        t = i / (num_frames - 1)
        frame = {}
        for key in start_kps.keys():
            if key in end_kps:
                start = np.array(start_kps[key][:2])
                end = np.array(end_kps[key][:2])
                interp = start * (1 - t) + end * t
                confidence = (start_kps[key][2] * (1 - t) + end_kps[key][2] * t)
                frame[key] = [interp[0], interp[1], confidence]
        interpolated.append(frame)
    return interpolated

def animate_keypoints(keypoints_sequence):
    # Calculate the exact figure size based on the keypoint coordinates
    coordinates = extract_coordinates(keypoints_sequence[0])
    x_min, y_min = coordinates.min(axis=0)
    x_max, y_max = coordinates.max(axis=0)
    width = x_max - x_min
    height = y_max - y_min
    
    # Add some padding
    padding = 20
    width += 2 * padding
    height += 2 * padding
    
    # Create a figure with exact pixel dimensions
    dpi = 100  # You can adjust this for higher resolution
    fig, ax = plt.subplots(figsize=(width/dpi, height/dpi), dpi=dpi)
    
    def update(frame):
        ax.clear()
        coordinates = extract_coordinates(keypoints_sequence[frame])
        
        # Plot keypoints
        for i, (x, y) in enumerate(coordinates):
            ax.plot(x, y, 'o', color=[c/255 for c in GOLIATH_KPTS_COLORS[i]], 
                    markersize=3, markeredgewidth=0)
        
        # Plot skeleton
        for connection in SKELETON_CONNECTIONS:
            if all(idx < len(coordinates) for idx in connection):
                x = [coordinates[connection[0]][0], coordinates[connection[1]][0]]
                y = [coordinates[connection[0]][1], coordinates[connection[1]][1]]
                ax.plot(x, y, color='gray', linewidth=1, alpha=0.7, solid_capstyle='round')
        
        ax.set_xlim(x_min - padding, x_max + padding)
        ax.set_ylim(y_max + padding, y_min - padding)  # Inverted y-axis
        ax.set_aspect('equal', adjustable='box')
        
        # Remove axes and margins
        ax.axis('off')
        plt.subplots_adjust(left=0, right=1, top=1, bottom=0)
        
        # Optional: add frame number
        ax.text(x_min, y_max + padding/2, f'Frame {frame}', fontsize=8, ha='left', va='top')
    
    return fig, update, len(keypoints_sequence)


# Main execution
start_kps = load_keypoints('body1.json')
end_kps = load_keypoints('body2.json')

num_frames = 60
interpolated_sequence = interpolate_keypoints(start_kps, end_kps, num_frames)

fig, update_func, frames = animate_keypoints(interpolated_sequence)

# Set up the writer
writer = FFMpegWriter(fps=30, metadata=dict(artist='Me'), bitrate=1800)

# Save the animation
with writer.saving(fig, "keypoints_animation.mp4", dpi=100):
    for i in range(frames):
        update_func(i)
        writer.grab_frame()

plt.close(fig)
print("Animation saved as 'keypoints_animation.mp4'")

# To display the animation in a Jupyter notebook, you can use:
# from IPython.display import HTML
# HTML(animation.to_jshtml())