#!/usr/bin/env python3
"""
Complete Workflow: Synthetic Mirror Ball Image Generation and Processing
This script runs the complete pipeline from Blender rendering to final stitched output
"""

import subprocess
import os
import sys
import argparse
from pathlib import Path

class WorkflowManager:
    def __init__(self, config):
        self.config = config
        self.blender_script = "generate_synthetic_images.py"
        self.unwrap_script = "unwrap_and_crop.py"
        
    def check_dependencies(self):
        """Check if required tools are available"""
        print("Checking dependencies...")
        
        # Check Blender
        blender_cmd = self.config.get('blender_executable', 'blender')
        try:
            result = subprocess.run([blender_cmd, '--version'], 
                                  capture_output=True, text=True, timeout=5)
            if result.returncode == 0:
                print(f"  ✓ Blender found: {result.stdout.split()[0]} {result.stdout.split()[1]}")
            else:
                print(f"  ✗ Blender not found at: {blender_cmd}")
                return False
        except (subprocess.TimeoutExpired, FileNotFoundError):
            print(f"  ✗ Blender not found. Please specify path with --blender")
            return False
        
        # Check Python packages
        try:
            import cv2
            import numpy
            print(f"  ✓ OpenCV version: {cv2.__version__}")
            print(f"  ✓ NumPy version: {numpy.__version__}")
        except ImportError as e:
            print(f"  ✗ Missing Python package: {e.name}")
            print(f"    Install with: pip install opencv-python numpy")
            return False
        
        return True
    
    def run_blender_generation(self):
        """Run Blender to generate synthetic images"""
        print("\n" + "="*70)
        print("STEP 1: GENERATING SYNTHETIC IMAGES IN BLENDER")
        print("="*70 + "\n")
        
        blender_cmd = self.config.get('blender_executable', 'blender')
        output_dir = self.config.get('render_output_dir', '/tmp/mirror_captures')
        
        # Update blender script with config
        blender_script_path = Path(__file__).parent / self.blender_script
        
        if not blender_script_path.exists():
            print(f"  ✗ Blender script not found: {blender_script_path}")
            return False
        
        # Run Blender in background mode
        cmd = [
            blender_cmd,
            '--background',
            '--python', str(blender_script_path)
        ]
        
        print(f"Running: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd, timeout=self.config.get('timeout', 3600))
            
            if result.returncode == 0:
                print("\n✓ Blender rendering complete!")
                return True
            else:
                print(f"\n✗ Blender failed with return code: {result.returncode}")
                return False
                
        except subprocess.TimeoutExpired:
            print("\n✗ Blender rendering timed out!")
            return False
        except Exception as e:
            print(f"\n✗ Error running Blender: {e}")
            return False
    
    def run_unwrapping(self):
        """Run unwrapping and cropping on generated images"""
        print("\n" + "="*70)
        print("STEP 2: UNWRAPPING AND CROPPING IMAGES")
        print("="*70 + "\n")
        
        input_dir = self.config.get('render_output_dir', '/tmp/mirror_captures')
        output_dir = self.config.get('unwrap_output_dir', '/tmp/mirror_unwrapped')
        
        unwrap_script_path = Path(__file__).parent / self.unwrap_script
        
        if not unwrap_script_path.exists():
            print(f"  ✗ Unwrap script not found: {unwrap_script_path}")
            return False
        
        # Build command
        cmd = [
            sys.executable,  # Use same Python interpreter
            str(unwrap_script_path),
            input_dir,
            '-o', output_dir,
            '--batch',
            '--width', str(self.config.get('unwrap_width', 3600)),
            '--height', str(self.config.get('unwrap_height', 900)),
            '--crop', str(self.config.get('crop_percentage', 0.4)),
            '--projection', self.config.get('projection', 'parabolic')
        ]
        
        if self.config.get('save_full_unwrap', False):
            cmd.append('--save-full')
        
        if self.config.get('no_stitch', False):
            cmd.append('--no-stitch')
        
        print(f"Running: {' '.join(cmd)}\n")
        
        try:
            result = subprocess.run(cmd)
            
            if result.returncode == 0:
                print("\n✓ Unwrapping complete!")
                return True
            else:
                print(f"\n✗ Unwrapping failed with return code: {result.returncode}")
                return False
                
        except Exception as e:
            print(f"\n✗ Error running unwrapping: {e}")
            return False
    
    def run_complete_pipeline(self):
        """Run the complete workflow"""
        print("\n" + "="*70)
        print("MIRROR BALL CAMERA - COMPLETE WORKFLOW")
        print("="*70)
        
        # Check dependencies
        if not self.check_dependencies():
            print("\n✗ Dependency check failed! Please install missing components.")
            return False
        
        # Step 1: Generate images
        if not self.config.get('skip_render', False):
            if not self.run_blender_generation():
                print("\n✗ Pipeline failed at rendering stage!")
                return False
        else:
            print("\nSkipping rendering (--skip-render)")
        
        # Step 2: Unwrap and crop
        if not self.run_unwrapping():
            print("\n✗ Pipeline failed at unwrapping stage!")
            return False
        
        # Success
        print("\n" + "="*70)
        print("✓ COMPLETE PIPELINE FINISHED SUCCESSFULLY")
        print("="*70)
        print(f"\nFinal outputs:")
        print(f"  - Individual bands: {self.config.get('unwrap_output_dir')}")
        print(f"  - Stitched map: {self.config.get('unwrap_output_dir')}/stitched_cylinder_map.png")
        print("="*70 + "\n")
        
        return True

def main():
    parser = argparse.ArgumentParser(
        description='Complete workflow for mirror ball camera simulation',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run complete pipeline
  python %(prog)s
  
  # Specify Blender path
  python %(prog)s --blender /usr/local/bin/blender
  
  # Skip rendering (process existing images)
  python %(prog)s --skip-render
  
  # Custom directories
  python %(prog)s --render-dir ./renders --output-dir ./processed
  
  # Adjust cropping
  python %(prog)s --crop 0.5
        """
    )
    
    parser.add_argument('--blender', dest='blender_executable', 
                       default='blender', help='Path to Blender executable')
    parser.add_argument('--render-dir', dest='render_output_dir',
                       default='/tmp/mirror_captures', help='Directory for rendered images')
    parser.add_argument('--output-dir', dest='unwrap_output_dir',
                       default='/tmp/mirror_unwrapped', help='Directory for processed images')
    parser.add_argument('--skip-render', action='store_true',
                       help='Skip Blender rendering (process existing images)')
    parser.add_argument('--unwrap-width', dest='unwrap_width', type=int, default=3600,
                       help='Unwrap panorama width')
    parser.add_argument('--unwrap-height', dest='unwrap_height', type=int, default=900,
                       help='Unwrap panorama height')
    parser.add_argument('--crop', dest='crop_percentage', type=float, default=0.4,
                       help='Crop percentage (0-1) for central band')
    parser.add_argument('--projection', choices=['linear', 'parabolic', 'equidistant'],
                       default='parabolic', help='Unwrap projection type')
    parser.add_argument('--save-full', dest='save_full_unwrap', action='store_true',
                       help='Save full unwrapped images')
    parser.add_argument('--no-stitch', action='store_true',
                       help='Disable automatic stitching')
    parser.add_argument('--timeout', type=int, default=3600,
                       help='Timeout for Blender rendering (seconds)')
    
    args = parser.parse_args()
    
    # Convert args to config dict
    config = vars(args)
    
    # Run workflow
    workflow = WorkflowManager(config)
    success = workflow.run_complete_pipeline()
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()
