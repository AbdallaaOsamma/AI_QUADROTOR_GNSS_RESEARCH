# src/ai/perception.py
"""
Multi-Modal Perception System for AirSim Quadrotor Navigation

This module provides multiple perception strategies that can work in different
environments, falling back to more robust methods when edge detection fails.

Perception Modes:
1. RGB Edge Detection (Traditional Canny/Hough)
2. Depth-Based Navigation (obstacle avoidance)
3. Segmentation-Based (ground vs obstacles)
4. Hybrid (combines multiple signals)
"""
import cv2
import numpy as np
import airsim
import math
from typing import Tuple, Optional, Dict, Any


class PerceptionResult:
    """Container for perception results."""
    
    def __init__(self):
        self.lateral_error = 0.0      # -1.0 to 1.0 (normalized)
        self.heading_error = 0.0      # radians
        self.obstacle_distance = 100.0  # meters to nearest obstacle
        self.confidence = 0.0         # 0.0 to 1.0
        self.mode_used = "none"
        self.lines_detected = 0
        self.debug_overlay = None
        
    def __repr__(self):
        return (f"PerceptionResult(lat={self.lateral_error:.2f}, head={self.heading_error:.2f}, "
                f"obs={self.obstacle_distance:.1f}m, conf={self.confidence:.2f}, mode={self.mode_used})")


class MultiModalPerception:
    """
    Multi-modal perception system that automatically selects the best
    available perception method based on image quality and detection success.
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """
        Initialize perception system.
        
        Args:
            config: Configuration dict with optional keys:
                - canny_low: Canny low threshold (default 40)
                - canny_high: Canny high threshold (default 120)
                - hough_thresh: Hough line threshold (default 20)
                - roi_ymin_frac: ROI top fraction (default 0.5)
                - min_lines_threshold: Min lines for confidence (default 4)
                - depth_near_thresh: Near obstacle threshold (default 2.0m)
                - depth_far_thresh: Far obstacle threshold (default 20.0m)
        """
        self.cfg = config or {}
        
        # Vision parameters with optimized defaults
        self.canny_low = self.cfg.get("canny_low", 40)
        self.canny_high = self.cfg.get("canny_high", 120)
        self.hough_thresh = self.cfg.get("hough_thresh", 20)
        self.roi_ymin_frac = self.cfg.get("roi_ymin_frac", 0.5)
        self.min_lines_threshold = self.cfg.get("min_lines_threshold", 4)
        
        # Depth parameters
        self.depth_near_thresh = self.cfg.get("depth_near_thresh", 2.0)
        self.depth_far_thresh = self.cfg.get("depth_far_thresh", 20.0)
        
        # State tracking
        self.edge_detection_failures = 0
        self.last_mode = "edge"
        
    def process(self, rgb_image: np.ndarray, 
                depth_image: Optional[np.ndarray] = None,
                seg_image: Optional[np.ndarray] = None) -> PerceptionResult:
        """
        Process images with multi-modal perception.
        
        Tries edge detection first, falls back to depth, then segmentation.
        
        Args:
            rgb_image: BGR image from camera
            depth_image: Optional depth image (float32, meters)
            seg_image: Optional segmentation image
            
        Returns:
            PerceptionResult with navigation signals
        """
        result = PerceptionResult()
        
        # Try edge detection first
        edge_result = self._edge_detection(rgb_image)
        
        if edge_result.confidence > 0.5:
            self.edge_detection_failures = 0
            result = edge_result
            result.mode_used = "edge"
        else:
            self.edge_detection_failures += 1
            
            # Try depth-based navigation if available
            if depth_image is not None:
                depth_result = self._depth_navigation(depth_image)
                if depth_result.confidence > edge_result.confidence:
                    result = depth_result
                    result.mode_used = "depth"
                    # Merge obstacle distance
                    result.obstacle_distance = depth_result.obstacle_distance
                else:
                    result = edge_result
                    result.mode_used = "edge_fallback"
            # Try segmentation if available
            elif seg_image is not None:
                seg_result = self._segmentation_navigation(seg_image)
                if seg_result.confidence > edge_result.confidence:
                    result = seg_result
                    result.mode_used = "segmentation"
                else:
                    result = edge_result
                    result.mode_used = "edge_fallback"
            else:
                result = edge_result
                result.mode_used = "edge_only"
                
        # If depth available, always compute obstacle distance
        if depth_image is not None and result.obstacle_distance == 100.0:
            result.obstacle_distance = self._compute_obstacle_distance(depth_image)
            
        self.last_mode = result.mode_used
        return result
        
    def _edge_detection(self, img: np.ndarray) -> PerceptionResult:
        """Traditional Canny/Hough edge detection for lane following."""
        result = PerceptionResult()
        h, w = img.shape[:2]
        
        # Extract ROI (lower portion where road should be)
        y0 = int(h * self.roi_ymin_frac)
        roi = img[y0:, :]
        roi_h, roi_w = roi.shape[:2]
        
        # Convert to grayscale and apply Gaussian blur
        gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Histogram equalization for better contrast
        equalized = cv2.equalizeHist(blurred)
        
        # Canny edge detection
        edges = cv2.Canny(equalized, self.canny_low, self.canny_high)
        
        # Hough line detection
        lines = cv2.HoughLinesP(edges, 1, np.pi/180,
                                threshold=self.hough_thresh,
                                minLineLength=25,
                                maxLineGap=15)
        
        # Create debug overlay
        overlay = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        cx = w // 2
        left_lines = []
        right_lines = []
        
        if lines is not None:
            result.lines_detected = len(lines)
            
            for line in lines:
                x1, y1, x2, y2 = line[0]
                
                # Calculate slope
                if abs(x2 - x1) < 1:
                    slope = 100.0  # Near vertical
                else:
                    slope = (y2 - y1) / (x2 - x1)
                    
                # Filter by slope - reject near-horizontal lines
                if abs(slope) < 0.3:
                    continue
                    
                if slope < 0:
                    left_lines.append((x1, y1, x2, y2))
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                else:
                    right_lines.append((x1, y1, x2, y2))
                    cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    
        # Compute lateral error and heading
        if left_lines or right_lines:
            # Estimate lane center by averaging bottom x of detected lines
            bottom_xs = []
            
            for lines_list in [left_lines, right_lines]:
                for (x1, y1, x2, y2) in lines_list:
                    # Extrapolate to bottom of ROI
                    if y2 != y1:
                        x_at_bottom = x1 + (roi_h - y1) * (x2 - x1) / (y2 - y1 + 1e-6)
                    else:
                        x_at_bottom = (x1 + x2) / 2
                    bottom_xs.append(x_at_bottom)
                    
            if bottom_xs:
                lane_center_x = np.mean(bottom_xs)
                result.lateral_error = float(np.clip((lane_center_x - cx) / (w * 0.5), -1.0, 1.0))
                
                # Draw lane center
                lane_center_int = int(lane_center_x)
                cv2.circle(overlay, (lane_center_int, roi_h - 10), 8, (255, 0, 255), -1)
                
            # Compute heading from average line angle
            if left_lines or right_lines:
                angles = []
                weights = []
                for lines_list in [left_lines, right_lines]:
                    for (x1, y1, x2, y2) in lines_list:
                        # Line length as weight
                        length = math.sqrt((x2-x1)**2 + (y2-y1)**2)
                        dx = x2 - x1
                        dy = y2 - y1
                        # Angle from vertical (0 = perfectly vertical)
                        if abs(dy) > 1:
                            angle_from_vertical = math.atan(dx / dy)
                        else:
                            angle_from_vertical = 0
                        angles.append(angle_from_vertical)
                        weights.append(length)
                        
                if angles and sum(weights) > 0:
                    avg_deviation = float(np.average(angles, weights=weights))
                    result.heading_error = np.clip(avg_deviation, -0.8, 0.8)
                    
        # Calculate confidence based on detection quality
        if result.lines_detected >= self.min_lines_threshold:
            balance = min(len(left_lines), len(right_lines)) / max(len(left_lines), len(right_lines), 1)
            result.confidence = min(1.0, 0.5 + 0.5 * balance)
        elif result.lines_detected > 0:
            result.confidence = result.lines_detected / self.min_lines_threshold * 0.5
        else:
            result.confidence = 0.0
            
        result.debug_overlay = overlay
        return result
        
    def _depth_navigation(self, depth: np.ndarray) -> PerceptionResult:
        """Depth-based navigation for obstacle avoidance."""
        result = PerceptionResult()
        h, w = depth.shape[:2]
        
        # Focus on lower central region (path ahead)
        roi_y = int(h * 0.5)
        roi = depth[roi_y:, :]
        
        # Create debug visualization
        depth_viz = np.clip(depth / 30.0, 0, 1)  # Normalize to 30m
        overlay = cv2.applyColorMap((depth_viz * 255).astype(np.uint8), cv2.COLORMAP_JET)
        
        # Divide into left, center, right zones
        third = w // 3
        zones = {
            "left": roi[:, :third],
            "center": roi[:, third:2*third],
            "right": roi[:, 2*third:]
        }
        
        # Compute mean depth for each zone (excluding very far values)
        zone_depths = {}
        for name, zone in zones.items():
            valid = zone[(zone > 0.5) & (zone < self.depth_far_thresh)]
            if len(valid) > 100:
                zone_depths[name] = np.mean(valid)
            else:
                zone_depths[name] = self.depth_far_thresh
                
        # Steering: move away from closer obstacles
        left_d = zone_depths.get("left", self.depth_far_thresh)
        right_d = zone_depths.get("right", self.depth_far_thresh)
        center_d = zone_depths.get("center", self.depth_far_thresh)
        
        # Lateral error: steer toward more open space
        if left_d > right_d + 1.0:  # Left is more open
            result.lateral_error = -0.3 * min(1.0, (left_d - right_d) / 10.0)
        elif right_d > left_d + 1.0:  # Right is more open
            result.lateral_error = 0.3 * min(1.0, (right_d - left_d) / 10.0)
            
        # Heading: turn toward more open area
        depth_diff = (right_d - left_d)
        result.heading_error = float(np.clip(depth_diff / 20.0, -0.5, 0.5))
        
        # Obstacle distance (minimum in center zone)
        result.obstacle_distance = center_d
        
        # Confidence based on valid depth readings
        total_valid = sum(1 for d in zone_depths.values() if d < self.depth_far_thresh)
        result.confidence = total_valid / 3.0
        
        # Add zone visualization
        cv2.line(overlay, (third, 0), (third, h), (255, 255, 255), 2)
        cv2.line(overlay, (2*third, 0), (2*third, h), (255, 255, 255), 2)
        for i, (name, d) in enumerate(zone_depths.items()):
            x = int((i + 0.5) * third)
            cv2.putText(overlay, f"{d:.1f}m", (x - 20, h - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
            
        result.debug_overlay = overlay
        return result
        
    def _segmentation_navigation(self, seg: np.ndarray) -> PerceptionResult:
        """Segmentation-based navigation using ground plane detection."""
        result = PerceptionResult()
        h, w = seg.shape[:2]
        
        # Focus on lower portion
        roi_y = int(h * 0.5)
        roi = seg[roi_y:, :]
        roi_h = roi.shape[0]
        
        # Convert to grayscale if needed
        if len(roi.shape) == 3:
            gray = cv2.cvtColor(roi, cv2.COLOR_BGR2GRAY)
        else:
            gray = roi
            
        # Find dominant ground color (assumed to be most common in lower center)
        center_sample = gray[roi_h//2:, w//3:2*w//3]
        ground_color = int(np.median(center_sample))
        
        # Create ground mask
        ground_mask = np.abs(gray.astype(int) - ground_color) < 30
        ground_mask = ground_mask.astype(np.uint8) * 255
        
        # Find ground centroid
        moments = cv2.moments(ground_mask)
        if moments["m00"] > 1000:
            cx_ground = moments["m10"] / moments["m00"]
            result.lateral_error = float(np.clip((cx_ground - w/2) / (w * 0.5), -1.0, 1.0))
            result.confidence = min(1.0, moments["m00"] / (roi_h * w * 0.5))
        else:
            result.confidence = 0.0
            
        # Create debug overlay
        overlay = cv2.cvtColor(ground_mask, cv2.COLOR_GRAY2BGR)
        result.debug_overlay = overlay
        return result
        
    def _compute_obstacle_distance(self, depth: np.ndarray) -> float:
        """Compute distance to nearest obstacle in central region."""
        h, w = depth.shape[:2]
        
        # Focus on central region
        center_x = w // 4
        center_y = h // 3
        center_roi = depth[center_y:2*center_y, center_x:3*center_x]
        
        # Find minimum valid depth
        valid = center_roi[(center_roi > 0.5) & (center_roi < 100)]
        if len(valid) > 50:
            # Use 10th percentile for robustness
            return float(np.percentile(valid, 10))
        return 100.0


def grab_all_frames(client: airsim.MultirotorClient, 
                    camera_name: str = "0") -> Dict[str, np.ndarray]:
    """
    Grab RGB, depth, and segmentation from a camera.
    
    Returns:
        Dict with keys: 'rgb', 'depth', 'seg' (may be None if not available)
    """
    result = {}
    
    # RGB
    try:
        resp = client.simGetImage(camera_name, airsim.ImageType.Scene)
        if resp:
            result['rgb'] = cv2.imdecode(np.frombuffer(resp, np.uint8), cv2.IMREAD_COLOR)
    except:
        result['rgb'] = None
        
    # Depth
    try:
        resp = client.simGetImages([
            airsim.ImageRequest(camera_name, airsim.ImageType.DepthPerspective, True, False)
        ])
        if resp and len(resp) > 0 and resp[0].width > 0:
            result['depth'] = airsim.list_to_2d_float_array(
                resp[0].image_data_float, resp[0].width, resp[0].height)
    except:
        result['depth'] = None
        
    # Segmentation
    try:
        resp = client.simGetImage(camera_name, airsim.ImageType.Segmentation)
        if resp:
            result['seg'] = cv2.imdecode(np.frombuffer(resp, np.uint8), cv2.IMREAD_COLOR)
    except:
        result['seg'] = None
        
    return result


# Convenience function for simple usage
def create_perception(config: Dict = None) -> MultiModalPerception:
    """Factory function to create perception system."""
    return MultiModalPerception(config)


if __name__ == "__main__":
    # Quick test
    print("Testing MultiModalPerception...")
    perception = create_perception()
    
    # Create dummy image
    test_img = np.zeros((480, 640, 3), dtype=np.uint8)
    cv2.line(test_img, (200, 480), (300, 240), (255, 255, 255), 3)
    cv2.line(test_img, (440, 480), (340, 240), (255, 255, 255), 3)
    
    result = perception.process(test_img)
    print(result)
