import numpy as np
import airsim
import cv2
import csv
import os
import time
from pathlib import Path
from dataclasses import dataclass
from typing import List, Tuple
import random

@dataclass
class Waypoint:
    """3D waypoint with orientation"""
    x: float
    y: float
    z: float
    yaw: float

class AdvancedTrajectoryGenerator:
    """Generate highly diverse navigation trajectories"""
    
    def __init__(self, altitude=-4):
        self.altitude = altitude
    
    def generate_all_trajectories(self, num_variations=5):
        """Generate comprehensive trajectory set"""
        all_trajectories = []
        
        print("[INFO] Generating comprehensive trajectory dataset...")
        
        # 1. Straight corridors with varying conditions
        for i in range(num_variations * 3):
            start = self._random_start()
            length = np.random.uniform(20, 50)
            drift_amplitude = np.random.uniform(0.5, 3.0)
            drift_frequency = np.random.uniform(2, 8)
            
            traj = self._straight_with_drift(start, length, drift_amplitude, drift_frequency)
            all_trajectories.append(('straight', traj))
        
        # 2. Sharp 90-degree turns (parking intersections)
        for i in range(num_variations * 2):
            start = self._random_start()
            direction = random.choice(['right', 'left'])
            approach = np.random.uniform(10, 20)
            radius = np.random.uniform(8, 15)
            
            traj = self._sharp_turn(start, direction, approach, radius)
            all_trajectories.append((f'turn_{direction}', traj))
        
        # 3. S-curves (lane changes, weaving)
        for i in range(num_variations * 2):
            start = self._random_start()
            length = np.random.uniform(30, 50)
            amplitude = np.random.uniform(4, 10)
            frequency = np.random.uniform(1.5, 3)
            
            traj = self._s_curve(start, length, amplitude, frequency)
            all_trajectories.append(('s_curve', traj))
        
        # 4. Gradual curves (gentle bends)
        for i in range(num_variations * 2):
            start = self._random_start()
            radius = np.random.uniform(20, 40)
            angle = np.random.uniform(30, 120)
            
            traj = self._gradual_curve(start, radius, angle)
            all_trajectories.append(('gradual_curve', traj))
        
        # 5. Obstacle avoidance maneuvers
        for i in range(num_variations * 3):
            start = self._random_start()
            obstacle_dist = np.random.uniform(15, 30)
            avoidance_amplitude = np.random.uniform(4, 8)
            
            traj = self._obstacle_avoidance(start, obstacle_dist, avoidance_amplitude)
            all_trajectories.append(('obstacle_avoid', traj))
        
        # 6. Narrow corridor navigation
        for i in range(num_variations * 2):
            start = self._random_start()
            length = np.random.uniform(20, 40)
            centering_precision = np.random.uniform(0.5, 2.0)
            
            traj = self._narrow_corridor(start, length, centering_precision)
            all_trajectories.append(('narrow_corridor', traj))
        
        # 7. U-turns (180-degree reversals)
        for i in range(num_variations):
            start = self._random_start()
            radius = np.random.uniform(8, 12)
            
            traj = self._u_turn(start, radius)
            all_trajectories.append(('u_turn', traj))
        
        # 8. Chicane patterns (alternating turns)
        for i in range(num_variations):
            start = self._random_start()
            num_turns = random.randint(3, 5)
            
            traj = self._chicane(start, num_turns)
            all_trajectories.append(('chicane', traj))
        
        # 9. Spiral patterns
        for i in range(num_variations):
            start = self._random_start()
            radius_change = np.random.uniform(0.5, 1.5)
            
            traj = self._spiral(start, radius_change)
            all_trajectories.append(('spiral', traj))
        
        # 10. Parallel parking maneuvers
        for i in range(num_variations):
            start = self._random_start()
            
            traj = self._parallel_parking(start)
            all_trajectories.append(('parallel_parking', traj))
        
        # 11. T-intersections
        for i in range(num_variations):
            start = self._random_start()
            direction = random.choice(['left', 'right'])
            
            traj = self._t_intersection(start, direction)
            all_trajectories.append(('t_intersection', traj))
        
        # 12. Multi-turn sequences
        for i in range(num_variations):
            start = self._random_start()
            
            traj = self._complex_path(start)
            all_trajectories.append(('complex_path', traj))
        
        # 13. Speed variation trajectories
        for i in range(num_variations * 2):
            start = self._random_start()
            
            traj = self._speed_variation_path(start)
            all_trajectories.append(('speed_variation', traj))
        
        # 14. Diagonal navigation
        for i in range(num_variations):
            start = self._random_start()
            angle = np.random.uniform(15, 75)
            
            traj = self._diagonal_path(start, angle)
            all_trajectories.append(('diagonal', traj))
        
        # 15. Figure-8 patterns
        for i in range(num_variations):
            start = self._random_start()
            radius = np.random.uniform(10, 15)
            
            traj = self._figure_eight(start, radius)
            all_trajectories.append(('figure_eight', traj))
        
        print(f"[OK] Generated {len(all_trajectories)} trajectory variations")
        return all_trajectories
    
    def _random_start(self):
        """Random starting position"""
        return (
            np.random.uniform(-20, 20),
            np.random.uniform(-20, 20),
            self.altitude
        )
    
    def _straight_with_drift(self, start, length, drift_amp, drift_freq, num_points=60):
        """Straight path with realistic lateral drift"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            x = start[0] + length * progress
            y = start[1] + drift_amp * np.sin(progress * drift_freq * np.pi) * np.exp(-progress * 1.5)
            z = start[2]
            yaw = 0.1 * np.cos(progress * drift_freq * 2 * np.pi)
            points.append(Waypoint(x, y, z, yaw))
        return points
    
    def _sharp_turn(self, start, direction, approach_len, turn_radius, num_points=60):
        """90-degree turn (parking intersection)"""
        points = []
        approach_points = int(num_points * 0.4)
        turn_points = num_points - approach_points
        
        # Approach
        for i in range(approach_points):
            progress = i / approach_points
            x = start[0] + approach_len * progress
            y = start[1]
            z = start[2]
            yaw = 0
            points.append(Waypoint(x, y, z, yaw))
        
        # Turn
        turn_angle = np.pi/2 if direction == 'right' else -np.pi/2
        for i in range(turn_points):
            progress = i / turn_points
            theta = turn_angle * progress
            
            if direction == 'right':
                x = start[0] + approach_len + turn_radius * np.sin(theta)
                y = start[1] - turn_radius * (1 - np.cos(theta))
            else:
                x = start[0] + approach_len + turn_radius * np.sin(-theta)
                y = start[1] + turn_radius * (1 - np.cos(-theta))
            
            z = start[2]
            yaw = theta
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _s_curve(self, start, length, amplitude, frequency, num_points=70):
        """S-shaped path"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            x = start[0] + length * progress
            y = start[1] + amplitude * np.sin(progress * frequency * np.pi)
            z = start[2]
            
            dy_dx = amplitude * frequency * np.pi / length * np.cos(progress * frequency * np.pi)
            yaw = np.arctan(dy_dx)
            
            points.append(Waypoint(x, y, z, yaw))
        return points
    
    def _gradual_curve(self, start, radius, angle_deg, num_points=50):
        """Gentle curve"""
        points = []
        angle_rad = np.deg2rad(angle_deg)
        
        for i in range(num_points):
            progress = i / num_points
            theta = angle_rad * progress
            x = start[0] + radius * np.sin(theta)
            y = start[1] + radius * (1 - np.cos(theta))
            z = start[2]
            yaw = theta
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _obstacle_avoidance(self, start, obstacle_dist, avoidance_amp, num_points=50):
        """Path avoiding an obstacle"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            x = start[0] + obstacle_dist * 2 * progress
            
            # Avoidance curve centered at obstacle
            dist_from_obstacle = abs(x - (start[0] + obstacle_dist))
            if dist_from_obstacle < 10:
                avoidance = avoidance_amp * (1 - (dist_from_obstacle / 10)**2)
                y = start[1] + avoidance
            else:
                y = start[1]
            
            z = start[2]
            
            if i > 0:
                dy = y - points[-1].y
                dx = x - points[-1].x
                yaw = np.arctan2(dy, dx)
            else:
                yaw = 0
            
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _narrow_corridor(self, start, length, centering_precision, num_points=60):
        """Narrow corridor with precise centering"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            x = start[0] + length * progress
            
            # Realistic centering behavior
            y = start[1] + centering_precision * np.sin(progress * 8 * np.pi) * np.exp(-progress * 2)
            
            z = start[2]
            yaw = 0.05 * np.sin(progress * 12 * np.pi)
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _u_turn(self, start, radius, num_points=50):
        """180-degree U-turn"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            theta = np.pi * progress
            x = start[0] + radius * np.sin(theta)
            y = start[1] + radius * (1 - np.cos(theta))
            z = start[2]
            yaw = theta
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _chicane(self, start, num_turns, num_points=80):
        """Alternating left-right turns"""
        points = []
        turn_length = 10
        
        for turn_idx in range(num_turns):
            direction = 1 if turn_idx % 2 == 0 else -1
            points_per_turn = num_points // num_turns
            
            for i in range(points_per_turn):
                progress = i / points_per_turn
                x = start[0] + (turn_idx + progress) * turn_length
                y = start[1] + direction * 5 * np.sin(progress * np.pi)
                z = start[2]
                
                dy_dx = direction * 5 * np.pi / turn_length * np.cos(progress * np.pi)
                yaw = np.arctan(dy_dx)
                
                points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _spiral(self, start, radius_change, num_points=100):
        """Spiral inward or outward"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            theta = progress * 4 * np.pi
            radius = 15 + radius_change * progress * 10
            
            x = start[0] + radius * np.cos(theta)
            y = start[1] + radius * np.sin(theta)
            z = start[2]
            yaw = theta + np.pi/2
            
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _parallel_parking(self, start, num_points=40):
        """Parallel parking maneuver"""
        points = []
        
        # Approach
        for i in range(15):
            progress = i / 15
            x = start[0] + 10 * progress
            y = start[1]
            z = start[2]
            yaw = 0
            points.append(Waypoint(x, y, z, yaw))
        
        # Back into space
        for i in range(15):
            progress = i / 15
            x = start[0] + 10 - 2 * progress
            y = start[1] - 3 * progress
            z = start[2]
            yaw = -np.pi/4 * progress
            points.append(Waypoint(x, y, z, yaw))
        
        # Straighten
        for i in range(10):
            progress = i / 10
            x = start[0] + 8 - progress
            y = start[1] - 3
            z = start[2]
            yaw = -np.pi/4 * (1 - progress)
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _t_intersection(self, start, direction, num_points=50):
        """T-intersection navigation"""
        points = []
        
        # Approach
        for i in range(int(num_points * 0.4)):
            progress = i / (num_points * 0.4)
            x = start[0] + 15 * progress
            y = start[1]
            z = start[2]
            yaw = 0
            points.append(Waypoint(x, y, z, yaw))
        
        # Turn
        turn_sign = 1 if direction == 'right' else -1
        for i in range(int(num_points * 0.6)):
            progress = i / (num_points * 0.6)
            theta = turn_sign * np.pi/2 * progress
            
            x = start[0] + 15 + 10 * np.sin(abs(theta))
            y = start[1] + turn_sign * 10 * (1 - np.cos(theta))
            z = start[2]
            yaw = theta
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _complex_path(self, start, num_points=100):
        """Complex multi-segment path"""
        points = []
        
        # Segment 1: Straight
        for i in range(20):
            progress = i / 20
            x = start[0] + 10 * progress
            y = start[1]
            z = start[2]
            yaw = 0
            points.append(Waypoint(x, y, z, yaw))
        
        # Segment 2: Right turn
        for i in range(20):
            progress = i / 20
            theta = np.pi/2 * progress
            x = start[0] + 10 + 8 * np.sin(theta)
            y = start[1] - 8 * (1 - np.cos(theta))
            z = start[2]
            yaw = theta
            points.append(Waypoint(x, y, z, yaw))
        
        # Segment 3: Straight
        for i in range(20):
            progress = i / 20
            x = start[0] + 18
            y = start[1] - 8 - 10 * progress
            z = start[2]
            yaw = np.pi/2
            points.append(Waypoint(x, y, z, yaw))
        
        # Segment 4: Left turn
        for i in range(20):
            progress = i / 20
            theta = np.pi/2 + np.pi/2 * progress
            x = start[0] + 18 - 8 * np.sin(theta - np.pi/2)
            y = start[1] - 18 + 8 * (1 - np.cos(theta - np.pi/2))
            z = start[2]
            yaw = theta
            points.append(Waypoint(x, y, z, yaw))
        
        # Segment 5: Final straight
        for i in range(20):
            progress = i / 20
            x = start[0] + 10 - 10 * progress
            y = start[1] - 26
            z = start[2]
            yaw = np.pi
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _speed_variation_path(self, start, num_points=70):
        """Path with speed changes (acceleration/deceleration)"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            
            # Variable speed affects spacing
            speed_factor = 0.5 + 0.5 * np.sin(progress * 2 * np.pi)
            x = start[0] + 30 * progress * speed_factor
            y = start[1] + 3 * np.sin(progress * 3 * np.pi)
            z = start[2]
            
            dy_dx = 3 * 3 * np.pi / 30 * np.cos(progress * 3 * np.pi)
            yaw = np.arctan(dy_dx)
            
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _diagonal_path(self, start, angle_deg, num_points=50):
        """Diagonal navigation across space"""
        points = []
        angle_rad = np.deg2rad(angle_deg)
        
        for i in range(num_points):
            progress = i / num_points
            distance = 30 * progress
            x = start[0] + distance * np.cos(angle_rad)
            y = start[1] + distance * np.sin(angle_rad)
            z = start[2]
            yaw = angle_rad
            points.append(Waypoint(x, y, z, yaw))
        
        return points
    
    def _figure_eight(self, start, radius, num_points=120):
        """Figure-8 pattern"""
        points = []
        for i in range(num_points):
            progress = i / num_points
            theta = progress * 4 * np.pi
            x = start[0] + radius * np.sin(theta)
            y = start[1] + radius * np.sin(theta) * np.cos(theta)
            z = start[2]
            
            dx = radius * np.cos(theta) * 4 * np.pi / num_points
            dy = radius * np.cos(2 * theta) * 4 * np.pi / num_points
            yaw = np.arctan2(dy, dx)
            
            points.append(Waypoint(x, y, z, yaw))
        
        return points

class VelocityCommandGenerator:
    """Convert waypoints to realistic velocity commands"""
    
    @staticmethod
    def waypoints_to_commands(waypoints, dt=0.1, velocity_profile='normal'):
        """Generate velocity commands with different profiles"""
        commands = []
        
        # Velocity limits based on profile
        if velocity_profile == 'slow':
            max_vx, max_vy, max_rz = 0.4, 0.2, 0.3
        elif velocity_profile == 'fast':
            max_vx, max_vy, max_rz = 1.0, 0.6, 0.6
        else:  # normal
            max_vx, max_vy, max_rz = 0.7, 0.4, 0.5
        
        for i in range(len(waypoints) - 1):
            current = waypoints[i]
            next_wp = waypoints[i + 1]
            
            dx = next_wp.x - current.x
            dy = next_wp.y - current.y
            dyaw = next_wp.yaw - current.yaw
            
            # Normalize angles
            while dyaw > np.pi:
                dyaw -= 2 * np.pi
            while dyaw < -np.pi:
                dyaw += 2 * np.pi
            
            # Calculate velocities
            vx = np.clip(dx / dt, -max_vx, max_vx)
            vy = np.clip(dy / dt, -max_vy, max_vy)
            rz = np.clip(dyaw / dt, -max_rz, max_rz)
            vz = 0.0
            
            # Add realistic sensor noise
            vx += np.random.normal(0, 0.02)
            vy += np.random.normal(0, 0.015)
            rz += np.random.normal(0, 0.025)
            
            # Re-clip after noise
            vx = np.clip(vx, -max_vx, max_vx)
            vy = np.clip(vy, -max_vy, max_vy)
            rz = np.clip(rz, -max_rz, max_rz)
            
            commands.append({
                'waypoint': current,
                'vx': float(vx),
                'vy': float(vy),
                'vz': float(vz),
                'r_z_rad': float(rz)
            })
        
        return commands

class SyntheticDataCollector:
    """Collect synthetic training data"""
    
    def __init__(self, client, output_dir, img_size=(320, 180)):
        self.client = client
        self.output_dir = Path(output_dir)
        self.img_size = img_size
        self.frame_counter = 0
        
        self.frames_dir = self.output_dir / "frames"
        self.frames_dir.mkdir(parents=True, exist_ok=True)
        
        self.labels_file = self.output_dir / "labels.csv"
        self.csv_file = open(self.labels_file, 'w', newline='')
        self.csv_writer = csv.DictWriter(
            self.csv_file,
            fieldnames=['t', 'img_path', 'alt_m', 'vx', 'vy', 'vz', 'r_z_rad', 'lat_err', 'heading_err']
        )
        self.csv_writer.writeheader()
    
    def collect_trajectory(self, commands, trajectory_name, augment=True):
        """Collect images with optional augmentation"""
        print(f"[INFO] Collecting {trajectory_name}: {len(commands)} samples")
        
        for i, cmd in enumerate(commands):
            wp = cmd['waypoint']
            
            # Teleport to waypoint
            pose = airsim.Pose(
                airsim.Vector3r(wp.x, wp.y, wp.z),
                airsim.to_quaternion(0, 0, wp.yaw)
            )
            self.client.simSetVehiclePose(pose, True)
            time.sleep(0.03)
            
            # Capture image
            responses = self.client.simGetImages([
                airsim.ImageRequest("0", airsim.ImageType.Scene, False, False)
            ])
            
            if responses and responses[0].height > 0:
                img1d = np.frombuffer(responses[0].image_data_uint8, dtype=np.uint8)
                img_bgr = img1d.reshape(responses[0].height, responses[0].width, 3)
                img_resized = cv2.resize(img_bgr, self.img_size, interpolation=cv2.INTER_AREA)
                
                # Save original
                self._save_sample(img_resized, cmd, wp)
                
                # Augmentation
                if augment and random.random() < 0.3:
                    # Horizontal flip
                    img_flip = cv2.flip(img_resized, 1)
                    cmd_flip = cmd.copy()
                    cmd_flip['vy'] = -cmd['vy']
                    cmd_flip['r_z_rad'] = -cmd['r_z_rad']
                    self._save_sample(img_flip, cmd_flip, wp)
                
                if augment and random.random() < 0.2:
                    # Brightness variation
                    brightness_factor = np.random.uniform(0.7, 1.3)
                    img_bright = np.clip(img_resized * brightness_factor, 0, 255).astype(np.uint8)
                    self._save_sample(img_bright, cmd, wp)
        
        print(f"  → Collected {self.frame_counter} total frames so far")
    
    def _save_sample(self, img, cmd, wp):
        """Save single image-command pair"""
        img_filename = f"frame_{self.frame_counter:05d}.png"
        img_path = self.frames_dir / img_filename
        cv2.imwrite(str(img_path), img)
        
        self.csv_writer.writerow({
            't': self.frame_counter * 0.1,
            'img_path': f"frames/{img_filename}",
            'alt_m': -wp.z,
            'vx': cmd['vx'],
            'vy': cmd['vy'],
            'vz': cmd['vz'],
            'r_z_rad': cmd['r_z_rad'],
            'lat_err': 0.0,
            'heading_err': 0.0
        })
        
        self.frame_counter += 1
    
    def close(self):
        self.csv_file.close()
        print(f"\n[OK] Total frames collected: {self.frame_counter}")

def main():
    print("="*70)
    print("COMPREHENSIVE SYNTHETIC EXPERT DATA GENERATOR v2")
    print("="*70)
    
    # Connect
    client = airsim.MultirotorClient()
    client.confirmConnection()
    client.enableApiControl(True)
    
    # Output
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    output_dir = f"data/expert/synthetic/run_{timestamp}"
    
    print(f"\n[INFO] Output directory: {output_dir}")
    print("[INFO] Generating diverse expert trajectories...")
    print()
    
    # Generate trajectories
    traj_gen = AdvancedTrajectoryGenerator(altitude=-4)
    all_trajectories = traj_gen.generate_all_trajectories(num_variations=5)
    
    # Collect data
    vel_gen = VelocityCommandGenerator()
    collector = SyntheticDataCollector(client, output_dir, img_size=(320, 180))
    
    total_trajectories = len(all_trajectories)
    
    for idx, (traj_name, waypoints) in enumerate(all_trajectories):
        # Generate commands with varying speed profiles
        speed_profile = random.choice(['slow', 'normal', 'fast'])
        commands = vel_gen.waypoints_to_commands(waypoints, dt=0.1, velocity_profile=speed_profile)
        
        # Collect data (with augmentation)
        collector.collect_trajectory(commands, f"{traj_name}_{idx}", augment=True)
        
        # Progress
        progress_pct = ((idx + 1) / total_trajectories) * 100
        print(f"[PROGRESS] {idx+1}/{total_trajectories} ({progress_pct:.1f}%) complete\n")
    
    collector.close()
    
    print("\n" + "="*70)
    print(f"[SUCCESS] Synthetic data generation complete!")
    print(f"[SUCCESS] Total samples: {collector.frame_counter}")
    print(f"[SUCCESS] Location: {output_dir}")
    print("="*70)
    
    # Reset
    client.enableApiControl(False)

if __name__ == "__main__":
    main()