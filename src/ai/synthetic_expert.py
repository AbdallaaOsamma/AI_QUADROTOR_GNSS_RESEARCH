import numpy as np
import airsim
import cv2
import csv
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import yaml

@dataclass
class Waypoint:
    """3D waypoint with orientation"""
    x: float
    y: float
    z: float
    yaw: float  # radians

class TrajectoryGenerator:
    """Generate complex navigation trajectories"""
    
    def __init__(self, start_pos=(0, 0, -4), altitude=-4):
        self.start_pos = start_pos
        self.altitude = altitude
        self.trajectories = []
    
    def straight_corridor(self, length=30, num_points=50):
        """Straight path with lateral wandering (realistic centering)"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            x = self.start_pos[0] + length * progress
            # Add realistic lateral drift + correction
            y = self.start_pos[1] + 2.0 * np.sin(progress * 4 * np.pi) * np.exp(-progress * 2)
            z = self.altitude
            yaw = 0.1 * np.sin(progress * 8 * np.pi)  # Small yaw corrections
            points.append(Waypoint(x, y, z, yaw))
        return points
    
    def curved_path(self, radius=15, angle_deg=90, num_points=40):
        """Smooth curve (like turning at intersection)"""
        points = []
        angle_rad = np.deg2rad(angle_deg)
        for i in range(num_points):
            progress = i / num_points
            theta = angle_rad * progress
            x = self.start_pos[0] + radius * np.sin(theta)
            y = self.start_pos[1] + radius * (1 - np.cos(theta))
            z = self.altitude
            yaw = theta
            points.append(Waypoint(x, y, z, yaw))
        return points
    
    def s_curve(self, length=40, amplitude=8, num_points=60):
        """S-shaped path (lane changes, obstacle avoidance)"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            x = self.start_pos[0] + length * progress
            y = self.start_pos[1] + amplitude * np.sin(progress * 2 * np.pi)
            z = self.altitude
            # Yaw follows curve tangent
            dy_dx = amplitude * 2 * np.pi / length * np.cos(progress * 2 * np.pi)
            yaw = np.arctan(dy_dx)
            points.append(Waypoint(x, y, z, yaw))
        return points
    
    def intersection_turn(self, turn_direction='right', approach_len=15, turn_radius=10):
        """Approach intersection and turn (90 degrees)"""
        points = []
        
        # Approach phase
        for i in range(20):
            progress = i / 20
            x = self.start_pos[0] + approach_len * progress
            y = self.start_pos[1]
            z = self.altitude
            yaw = 0
            points.append(Waypoint(x, y, z, yaw))
        
        # Turn phase
        turn_angle = np.pi/2 if turn_direction == 'right' else -np.pi/2
        for i in range(30):
            progress = i / 30
            theta = turn_angle * progress
            if turn_direction == 'right':
                x = self.start_pos[0] + approach_len + turn_radius * np.sin(theta)
                y = self.start_pos[1] - turn_radius * (1 - np.cos(theta))
            else:
                x = self.start_pos[0] + approach_len + turn_radius * np.sin(-theta)
                y = self.start_pos[1] + turn_radius * (1 - np.cos(-theta))
            z = self.altitude
            yaw = theta
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def obstacle_avoidance(self, obstacle_pos=(15, 0), avoidance_dist=5, num_points=50):
        """Path that avoids an obstacle"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            x = self.start_pos[0] + 30 * progress
            
            # Calculate lateral offset to avoid obstacle
            dist_to_obstacle = np.abs(x - obstacle_pos[0])
            if dist_to_obstacle < avoidance_dist:
                # Smooth avoidance curve
                avoidance_factor = (avoidance_dist - dist_to_obstacle) / avoidance_dist
                y = self.start_pos[1] + 4 * avoidance_factor
            else:
                y = self.start_pos[1]
            
            z = self.altitude
            
            # Yaw correction during avoidance
            if i > 0:
                dy = y - points[-1].y
                dx = x - points[-1].x
                yaw = np.arctan2(dy, dx)
            else:
                yaw = 0
            
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def figure_eight(self, radius=12, num_points=80):
        """Figure-8 pattern (complex maneuver)"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            theta = progress * 4 * np.pi
            x = self.start_pos[0] + radius * np.sin(theta)
            y = self.start_pos[1] + radius * np.sin(theta) * np.cos(theta)
            z = self.altitude
            
            # Tangent yaw
            dx = radius * np.cos(theta) * 4 * np.pi / num_points
            dy = radius * (np.cos(2*theta)) * 4 * np.pi / num_points
            yaw = np.arctan2(dy, dx)
            
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def narrow_corridor(self, length=25, width_variation=True, num_points=50):
        """Narrow corridor with width changes (parking aisle)"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            x = self.start_pos[0] + length * progress
            
            # Corridor centering with realistic drift
            if width_variation:
                # Simulate narrowing corridor requiring precise centering
                corridor_width = 4 - 2 * np.sin(progress * np.pi)
                y = self.start_pos[1] + 0.5 * np.sin(progress * 6 * np.pi) * (corridor_width / 4)
            else:
                y = self.start_pos[1]
            
            z = self.altitude
            yaw = 0.05 * np.sin(progress * 10 * np.pi)
            points.append(Waypoint(x, y, z, yaw))
        
        return points

class VelocityCommandGenerator:
    """Convert waypoints to velocity commands"""
    
    @staticmethod
    def waypoints_to_commands(waypoints: List[Waypoint], dt=0.1, max_vx=0.8, max_vy=0.4, max_rz=0.5):
        """Generate smooth velocity commands from waypoint path"""
        commands = []
        
        for i in range(len(waypoints) - 1):
            current = waypoints[i]
            next_wp = waypoints[i + 1]
            
            # Calculate desired velocities
            dx = next_wp.x - current.x
            dy = next_wp.y - current.y
            dyaw = next_wp.yaw - current.yaw
            
            # Normalize by time step
            vx = np.clip(dx / dt, -max_vx, max_vx)
            vy = np.clip(dy / dt, -max_vy, max_vy)
            rz = np.clip(dyaw / dt, -max_rz, max_rz)
            vz = 0.0  # Maintain altitude
            
            # Add realistic noise
            vx += np.random.normal(0, 0.02)
            vy += np.random.normal(0, 0.01)
            rz += np.random.normal(0, 0.03)
            
            commands.append({
                'waypoint': current,
                'vx': vx,
                'vy': vy,
                'vz': vz,
                'r_z_rad': rz
            })
        
        return commands

class SyntheticDataCollector:
    """Collect images and commands from synthetic trajectories"""
    
    def __init__(self, client, output_dir, img_size=(320, 180)):
        self.client = client
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.frame_counter = 0
        
        # Create directories
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.labels_file = self.output_dir / "labels.csv"
        self.csv_file = open(self.labels_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=['t', 'img_path', 'alt_m', 'vx', 'vy', 'vz', 'r_z_rad', 'lat_err', 'heading_err']
        )
        self.csv_writer.writeheader()
    
    def collect_trajectory(self, commands, trajectory_name="trajectory"):
        """Teleport to each waypoint and capture image"""
        print(f"[INFO] Collecting {len(commands)} samples for {trajectory_name}")
        
        for i, cmd in enumerate(commands):
            wp = cmd['waypoint']
            
            # Teleport drone to position
            pose = airsim.Pose(
                airsim.Vector3r(wp.x, wp.y, wp.z),
                airsim.to_quaternion(0, 0, wp.yaw)
            )
            self.client.simSetVehiclePose(pose, True)
            time.sleep(0.05)  # Brief stabilization
            
            # Capture image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3)
                
                # Resize
                img_resized = cv2.resize(img_bgr, self.img_size, interpolation=cv2.INTER_AREA)
                
                # Save image
                img_filename = f"frame_{self.frame_counter:05d}.png"
                img_path = self.frames_dir / img_filename
                cv2.imwrite(str(img_path), img_resized)
                
                # Save label
                self.csv_writer.writerow({
                    't': self.frame_counter * 0.1,
                    'img_path': f"frames/{img_filename}",
                    'alt_m': -wp.z,
                    'vx': cmd['vx'],
                    'vy': cmd['vy'],
                    'vz': cmd['vz'],
                    'r_z_rad': cmd['r_z_rad'],
                    'lat_err': 0.0,  # Synthetic data is "perfect"
                    'heading_err': 0.0
                })
                
                self.frame_counter += 1
            
            if (i + 1) % 50 == 0:
                print(f"  Progress: {i+1}/{len(commands)}")
        
        print(f"[OK] Collected {self.frame_counter} total frames")
    
    def close(self):
        self.csv_file.close()

def main():
    # Connect to AirSim
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    
    # Output directory
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/expert/synthetic/run_{timestamp}"
    
    print(f"[INFO] Generating synthetic expert data -> {output_dir}")
    
    # Initialize generators
    traj_gen = TrajectoryGenerator(start_pos=(0, 0, -4), altitude=-4)
    vel_gen = VelocityCommandGenerator()
    collector = SyntheticDataCollector(client, output_dir)
    
    # Generate diverse trajectories
    trajectories = {
        'straight_1': traj_gen.straight_corridor(length=30, num_points=60),
        'straight_2': traj_gen.straight_corridor(length=40, num_points=70),
        'curve_right': traj_gen.curved_path(radius=15, angle_deg=90, num_points=50),
        'curve_left': traj_gen.curved_path(radius=15, angle_deg=-90, num_points=50),
        's_curve_1': traj_gen.s_curve(length=35, amplitude=6, num_points=70),
        's_curve_2': traj_gen.s_curve(length=45, amplitude=8, num_points=80),
        'intersection_right': traj_gen.intersection_turn('right', approach_len=15, turn_radius=12),
        'intersection_left': traj_gen.intersection_turn('left', approach_len=15, turn_radius=12),
        'obstacle_avoid': traj_gen.obstacle_avoidance(obstacle_pos=(15, 0), avoidance_dist=6, num_points=60),
        'figure_eight': traj_gen.figure_eight(radius=10, num_points=100),
        'narrow_corridor_1': traj_gen.narrow_corridor(length=30, width_variation=True, num_points=60),
        'narrow_corridor_2': traj_gen.narrow_corridor(length=25, width_variation=False, num_points=50),
    }
    
    # Collect data for each trajectory
    for traj_name, waypoints in trajectories.items():
        # Reset to start position for each trajectory
        traj_gen.start_pos = (np.random.uniform(-10, 10), np.random.uniform(-10, 10), -4)
        
        # Generate commands
        commands = vel_gen.waypoints_to_commands(waypoints, dt=0.1)
        
        # Collect images
        collector.collect_trajectory(commands, trajectory_name=traj_name)
    
    collector.close()
    
    print(f"[OK] Synthetic data generation complete!")
    print(f"[OK] Total frames: {collector.frame_counter}")
    print(f"[OK] Saved to: {output_dir}")
    
    # Reset drone
    client.enableApiControl(False)

if __name__ == "__main__":
    main()