# src/utils/vision_diagnostic.py
"""
Vision Diagnostic Tool for AirSim Quadrotor Navigation

This script diagnoses vision pipeline issues by:
1. Capturing images from all cameras
2. Testing multiple vision algorithms
3. Displaying live debug visualization
4. Saving diagnostic reports

Run with: python -m src.utils.vision_diagnostic
"""
import os
import sys
import cv2
import time
import numpy as np
import airsim
from datetime import datetime


class VisionDiagnostic:
    """Comprehensive vision pipeline diagnostic tool."""
    
    def __init__(self):
        self.client = None
        self.output_dir = "data/diagnostic"
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Vision parameters to test
        self.canny_params = [
            {"low": 30, "high": 100, "name": "low_threshold"},
            {"low": 50, "high": 150, "name": "medium_threshold"},
            {"low": 80, "high": 200, "name": "high_threshold"},
        ]
        
        self.hough_params = [15, 25, 35, 50]
        
    def connect(self):
        """Connect to AirSim."""
        print("[INFO] Connecting to AirSim...")
        self.client = airsim.MultirotorClient()
        self.client.confirmConnection()
        print("[OK] Connected to AirSim")
        return True
        
    def get_available_cameras(self):
        """List available cameras."""
        test_cameras = ["0", "front_down", "bottom", "1", "2", "3", "4"]
        available = []
        for cam in test_cameras:
            try:
                resp = self.client.simGetImage(cam, airsim.ImageType.Scene)
                if resp and len(resp) > 100:
                    available.append(cam)
            except:
                pass
        return available
        
    def capture_all_cameras(self):
        """Capture from all available cameras."""
        cameras = self.get_available_cameras()
        print(f"[INFO] Found {len(cameras)} cameras: {cameras}")
        
        images = {}
        for cam in cameras:
            # RGB Scene
            try:
                resp = self.client.simGetImage(cam, airsim.ImageType.Scene)
                if resp:
                    img = cv2.imdecode(np.frombuffer(resp, np.uint8), cv2.IMREAD_COLOR)
                    if img is not None:
                        images[f"{cam}_rgb"] = img
            except Exception as e:
                print(f"[WARN] Camera {cam} RGB failed: {e}")
                
            # Depth
            try:
                resp = self.client.simGetImages([
                    airsim.ImageRequest(cam, airsim.ImageType.DepthPerspective, True, False)
                ])
                if resp and len(resp) > 0 and resp[0].width > 0:
                    depth = airsim.list_to_2d_float_array(resp[0].image_data_float, 
                                                          resp[0].width, resp[0].height)
                    depth_viz = np.clip(depth / 50.0, 0, 1)  # Normalize to 50m range
                    depth_viz = (depth_viz * 255).astype(np.uint8)
                    depth_viz = cv2.applyColorMap(depth_viz, cv2.COLORMAP_JET)
                    images[f"{cam}_depth"] = depth_viz
            except:
                pass
                
            # Segmentation
            try:
                resp = self.client.simGetImage(cam, airsim.ImageType.Segmentation)
                if resp:
                    seg = cv2.imdecode(np.frombuffer(resp, np.uint8), cv2.IMREAD_COLOR)
                    if seg is not None:
                        images[f"{cam}_seg"] = seg
            except:
                pass
                
        return images
        
    def analyze_image(self, img, name="image"):
        """Analyze an image for road/path features."""
        results = {
            "name": name,
            "shape": img.shape,
            "issues": [],
            "edge_counts": {},
            "recommendations": []
        }
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape
        
        # Check image brightness
        mean_brightness = np.mean(gray)
        if mean_brightness > 200:
            results["issues"].append(f"Image too bright (mean: {mean_brightness:.0f})")
        elif mean_brightness < 50:
            results["issues"].append(f"Image too dark (mean: {mean_brightness:.0f})")
        results["brightness"] = mean_brightness
        
        # Check contrast
        std_dev = np.std(gray)
        if std_dev < 30:
            results["issues"].append(f"Low contrast (std: {std_dev:.0f})")
        results["contrast"] = std_dev
        
        # Check for visible edges with different thresholds
        for params in self.canny_params:
            edges = cv2.Canny(gray, params["low"], params["high"])
            edge_pixels = np.sum(edges > 0)
            edge_percent = (edge_pixels / (h * w)) * 100
            results["edge_counts"][params["name"]] = {
                "count": edge_pixels,
                "percent": edge_percent
            }
            
        # Check for lines in lower half (road detection area)
        roi = gray[h//2:, :]
        best_lines = 0
        best_thresh = 0
        
        for canny in self.canny_params:
            edges = cv2.Canny(roi, canny["low"], canny["high"])
            for hough_thresh in self.hough_params:
                lines = cv2.HoughLinesP(edges, 1, np.pi/180, 
                                        threshold=hough_thresh,
                                        minLineLength=20,
                                        maxLineGap=10)
                if lines is not None and len(lines) > best_lines:
                    best_lines = len(lines)
                    best_thresh = hough_thresh
                    results["best_canny"] = canny
                    results["best_hough"] = hough_thresh
                    
        results["max_lines_detected"] = best_lines
        
        # Generate recommendations
        if best_lines == 0:
            results["recommendations"].append("NO LINES DETECTED - Check if road is visible in image")
            results["recommendations"].append("Consider using depth camera for obstacle detection")
            results["recommendations"].append("Try segmentation for ground/path detection")
        elif best_lines < 5:
            results["recommendations"].append("Few lines detected - reduce Canny/Hough thresholds")
        else:
            results["recommendations"].append(f"Lines detected! Best params: Canny {results.get('best_canny', {})} Hough {best_thresh}")
            
        return results
        
    def create_debug_overlay(self, img, params=None):
        """Create visualization overlay with all detection methods."""
        if params is None:
            params = {"canny_low": 40, "canny_high": 120, "hough_thresh": 20}
            
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Canny edge detection
        edges = cv2.Canny(blurred, params["canny_low"], params["canny_high"])
        
        # ROI - lower 50% of image
        roi_y = h // 2
        roi = edges[roi_y:, :]
        
        # Hough line detection
        lines = cv2.HoughLinesP(roi, 1, np.pi/180, 
                                threshold=params["hough_thresh"],
                                minLineLength=25, maxLineGap=15)
        
        # Create visualizations
        overlay = img.copy()
        
        # Draw ROI boundary
        cv2.line(overlay, (0, roi_y), (w, roi_y), (255, 255, 0), 2)
        cv2.putText(overlay, "ROI", (10, roi_y - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        
        # Draw detected lines
        line_count = 0
        left_lines = []
        right_lines = []
        
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                # Offset y coordinates to account for ROI
                y1 += roi_y
                y2 += roi_y
                
                # Calculate slope
                if x2 != x1:
                    slope = (y2 - y1) / (x2 - x1 + 1e-6)
                else:
                    slope = 1000
                    
                # Filter by slope (reject near-horizontal lines)
                if abs(slope) > 0.3:
                    if slope < 0:
                        left_lines.append((x1, y1, x2, y2))
                        cv2.line(overlay, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    else:
                        right_lines.append((x1, y1, x2, y2))
                        cv2.line(overlay, (x1, y1), (x2, y2), (0, 0, 255), 2)
                    line_count += 1
                    
        # Calculate centerline if we have both left and right lines
        cx = w // 2
        centerline_x = cx
        lateral_error = 0
        
        if left_lines and right_lines:
            # Average bottom x of left and right lines
            left_bottom = np.mean([l[2] if l[3] > l[1] else l[0] for l in left_lines])
            right_bottom = np.mean([l[2] if l[3] > l[1] else l[0] for l in right_lines])
            centerline_x = int((left_bottom + right_bottom) / 2)
            lateral_error = centerline_x - cx
            
            # Draw centerline
            cv2.circle(overlay, (centerline_x, h - 20), 8, (255, 0, 255), -1)
            cv2.line(overlay, (cx, h), (centerline_x, h - 40), (255, 0, 255), 2)
            
        # Add text overlay
        info_bg = overlay.copy()
        cv2.rectangle(info_bg, (0, 0), (250, 100), (0, 0, 0), -1)
        overlay = cv2.addWeighted(overlay, 0.7, info_bg, 0.3, 0)
        
        cv2.putText(overlay, f"Lines: {line_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
        cv2.putText(overlay, f"Left: {len(left_lines)} Right: {len(right_lines)}", (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
        cv2.putText(overlay, f"Lat Error: {lateral_error:+.0f}px", (10, 60),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 255), 1)
        cv2.putText(overlay, f"Canny: {params['canny_low']}/{params['canny_high']}", (10, 80),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
                    
        # Create edge visualization
        edges_color = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        return overlay, edges_color, line_count
        
    def run_live_diagnostic(self, camera="0", duration=30):
        """Run live diagnostic with visualization."""
        print(f"\n[INFO] Starting live diagnostic on camera '{camera}' for {duration}s")
        print("[INFO] Press 'q' to quit, '+/-' to adjust thresholds")
        
        params = {"canny_low": 40, "canny_high": 120, "hough_thresh": 20}
        
        start = time.time()
        frame_count = 0
        total_lines = 0
        
        while time.time() - start < duration:
            # Capture frame
            resp = self.client.simGetImage(camera, airsim.ImageType.Scene)
            if resp is None or len(resp) < 100:
                print(f"[WARN] No image from camera {camera}")
                time.sleep(0.5)
                continue
                
            img = cv2.imdecode(np.frombuffer(resp, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                continue
                
            # Create debug overlay
            overlay, edges, lines = self.create_debug_overlay(img, params)
            total_lines += lines
            frame_count += 1
            
            # Stack images for display
            h, w = img.shape[:2]
            display = np.hstack([
                cv2.resize(overlay, (w//2, h//2)),
                cv2.resize(edges, (w//2, h//2))
            ])
            
            cv2.imshow("Vision Diagnostic", display)
            key = cv2.waitKey(30) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('+') or key == ord('='):
                params["canny_low"] = min(params["canny_low"] + 5, 100)
                print(f"Canny low: {params['canny_low']}")
            elif key == ord('-'):
                params["canny_low"] = max(params["canny_low"] - 5, 10)
                print(f"Canny low: {params['canny_low']}")
            elif key == ord('['):
                params["hough_thresh"] = max(params["hough_thresh"] - 5, 5)
                print(f"Hough thresh: {params['hough_thresh']}")
            elif key == ord(']'):
                params["hough_thresh"] = min(params["hough_thresh"] + 5, 100)
                print(f"Hough thresh: {params['hough_thresh']}")
                
        cv2.destroyAllWindows()
        
        # Report
        avg_lines = total_lines / max(frame_count, 1)
        print(f"\n{'='*60}")
        print(f"DIAGNOSTIC REPORT")
        print(f"{'='*60}")
        print(f"Frames analyzed: {frame_count}")
        print(f"Average lines per frame: {avg_lines:.1f}")
        print(f"Final params: {params}")
        
        if avg_lines < 1:
            print("\n[CRITICAL] Vision pipeline NOT detecting lines!")
            print("Recommendations:")
            print("  1. Check camera points toward road/path")
            print("  2. Try 'front_down' camera for better road view")
            print("  3. Switch to depth-based navigation")
            print("  4. Environment may lack visible road edges")
        elif avg_lines < 5:
            print("\n[WARNING] Few lines detected - may be unstable")
        else:
            print("\n[OK] Vision pipeline detecting lines")
            
        return {"avg_lines": avg_lines, "params": params}
        
    def save_diagnostic_report(self, results):
        """Save diagnostic results to file."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = os.path.join(self.output_dir, f"diagnostic_{timestamp}.txt")
        
        with open(report_path, "w") as f:
            f.write("VISION DIAGNOSTIC REPORT\n")
            f.write(f"Generated: {timestamp}\n")
            f.write("=" * 60 + "\n\n")
            
            for name, data in results.items():
                f.write(f"\n--- {name} ---\n")
                for key, value in data.items():
                    f.write(f"  {key}: {value}\n")
                    
        print(f"[OK] Report saved to {report_path}")
        return report_path


def main():
    """Main diagnostic entry point."""
    diag = VisionDiagnostic()
    
    if not diag.connect():
        print("[ERROR] Could not connect to AirSim")
        return
        
    # Capture from all cameras
    print("\n[STEP 1] Capturing from all cameras...")
    images = diag.capture_all_cameras()
    print(f"Captured {len(images)} images")
    
    # Save and analyze each image
    print("\n[STEP 2] Analyzing images...")
    results = {}
    for name, img in images.items():
        # Save image
        path = os.path.join(diag.output_dir, f"{name}.png")
        cv2.imwrite(path, img)
        print(f"  Saved: {name}")
        
        # Analyze
        if "rgb" in name:
            analysis = diag.analyze_image(img, name)
            results[name] = analysis
            print(f"  {name}: brightness={analysis['brightness']:.0f}, "
                  f"contrast={analysis['contrast']:.0f}, "
                  f"max_lines={analysis['max_lines_detected']}")
            for issue in analysis["issues"]:
                print(f"    [ISSUE] {issue}")
            for rec in analysis["recommendations"]:
                print(f"    [REC] {rec}")
                
    # Run live diagnostic
    print("\n[STEP 3] Running live diagnostic...")
    try:
        cameras = diag.get_available_cameras()
        if cameras:
            # Try front_down first if available, else default
            cam = "front_down" if "front_down" in cameras else cameras[0]
            live_results = diag.run_live_diagnostic(camera=cam, duration=30)
            results["live"] = live_results
    except Exception as e:
        print(f"[WARN] Live diagnostic skipped: {e}")
        
    # Save report
    print("\n[STEP 4] Saving report...")
    diag.save_diagnostic_report(results)
    
    print("\n[DONE] Diagnostic complete!")
    print(f"Results saved to: {diag.output_dir}")


if __name__ == "__main__":
    main()
