# Mirror Ball Camera: Vertical Scanning System
## Synthetic Data Generation for Cylinder Inspection

---

## 🎯 What This Does

This system creates **synthetic mirror ball camera images** for inspecting engine cylinders. It simulates a hemispherical mirror that captures 360° views, then processes these images to create high-quality cylinder maps with minimal distortion.

### Key Innovation: Vertical Scanning Approach

Instead of capturing just ONE image (which would have heavy distortion at edges), we:
1. **Capture 10+ images** at different vertical positions 
2. **Unwrap each** circular image to panoramic view
3. **Crop only the central 40%** of each (least distorted part)
4. **Stitch all bands together** to create a complete, low-distortion map

**Result**: Professional-quality cylinder inspection data!

---

## 📁 Files Overview

### 🚀 Main Workflow Files (START HERE)

| File | Purpose |
|------|---------|
| **run_workflow.py** | Complete automated pipeline (Blender + processing) |
| **generate_synthetic_images.py** | Blender script to render at multiple heights |
| **unwrap_and_crop.py** | Process images: unwrap → crop → stitch |
| **WORKFLOW_GUIDE.md** | Complete documentation & examples |
| **workflow_visualization.html** | Interactive visual guide |

### 📚 Reference Files

| File | Purpose |
|------|---------|
| **requirements.txt** | Python dependencies |
| **mirror_ball_camera_enhanced.py** | Original single-capture Blender script |
| **system_diagram.html** | Optical system visualization |
| **README.md** | This file |

---

## ⚡ Quick Start (3 Commands)

### Option 1: Fully Automated
```bash
# Install dependencies
pip install -r requirements.txt

# Run everything (Blender rendering + image processing)
python run_workflow.py

# View result
open /tmp/mirror_unwrapped/stitched_cylinder_map.png
```

### Option 2: Step-by-Step
```bash
# Step 1: Generate synthetic images in Blender
blender --background --python generate_synthetic_images.py

# Step 2: Process the images
python unwrap_and_crop.py /tmp/mirror_captures/ -o /tmp/mirror_unwrapped/ --batch

# Step 3: View results
ls /tmp/mirror_unwrapped/
```

---

## 🔧 How It Works

### The Vertical Scanning Strategy

```
     Cylinder Interior
    ┌─────────────────┐
    │                 │
    │    Position 1   │ ← Capture image here
    │       ↓         │    Extract central band
    ├─────────────────┤
    │                 │
    │    Position 2   │ ← Capture image here
    │       ↓         │    Extract central band
    ├─────────────────┤
    │                 │
    │    Position 3   │ ← And so on...
    │       ↓         │
    └─────────────────┘

    Each position captures 360° around cylinder
    We take only the CENTRAL part (least distortion)
    Stack all bands → Complete cylinder map!
```

### Why This Approach?

**Problem with single capture:**
- Center of image: Good quality ✓
- Edges of image: Heavy distortion ✗

**Solution with vertical scanning:**
- Take multiple captures at different heights
- Each capture: use ONLY the good center part
- Stack them together
- **Every part** of final map has good quality ✓✓✓

---

## 📊 Configuration Guide

### Adjust Capture Parameters

Edit `generate_synthetic_images.py`:

```python
# How many vertical positions?
NUM_CAPTURES = 10        # More = better coverage, slower
                        # 5 = fast preview
                        # 10 = standard
                        # 20 = high quality

# Vertical range
START_HEIGHT = 0.1       # Start height (meters)
END_HEIGHT = 1.3         # End height (meters)

# Render quality
SAMPLES = 256           # 64 = fast, 256 = good, 512 = excellent

# Image size
RENDER_WIDTH = 2048     # 1024 = fast, 2048 = good, 4096 = excellent
RENDER_HEIGHT = 2048
```

### Adjust Processing Parameters

Command line options for `unwrap_and_crop.py`:

```bash
python unwrap_and_crop.py INPUT -o OUTPUT \
  --crop 0.4              # How much to keep? (0.3-0.5 recommended)
  --width 3600            # Unwrap panorama width
  --height 900            # Unwrap panorama height
  --projection parabolic  # linear, parabolic, or equidistant
  --save-full            # Save full unwrapped images (debug)
  --no-stitch            # Don't auto-stitch (manual control)
```

---

## 📈 Quality Settings Comparison

| Profile | Captures | Samples | Resolution | Crop | Time | Use Case |
|---------|----------|---------|------------|------|------|----------|
| **Fast** | 5 | 64 | 1024² | 0.5 | ~3 min | Testing |
| **Standard** | 10 | 256 | 2048² | 0.4 | ~17 min | Production |
| **High** | 20 | 512 | 4096² | 0.3 | ~95 min | Research |

---

## 🎨 Customization Examples

### Example 1: Quick Preview Run
```python
# In generate_synthetic_images.py:
NUM_CAPTURES = 5
SAMPLES = 64
RENDER_WIDTH = 1024
RENDER_HEIGHT = 1024

# Then run:
python run_workflow.py
```

### Example 2: High-Quality Production
```python
# In generate_synthetic_images.py:
NUM_CAPTURES = 15
SAMPLES = 384
RENDER_WIDTH = 3072
RENDER_HEIGHT = 3072

# Run with settings:
python unwrap_and_crop.py /tmp/mirror_captures/ -o ./output/ \
    --batch --crop 0.35 --width 5400 --height 1350
```

### Example 3: Focus on Specific Region
```python
# In generate_synthetic_images.py:
START_HEIGHT = 0.5      # Start at 50cm
END_HEIGHT = 1.0        # End at 100cm
NUM_CAPTURES = 20       # Dense sampling in this region
```

### Example 4: Process Existing Images
```bash
# Already have rendered images? Just process them:
python run_workflow.py --skip-render --render-dir ./my_images/
```

---

## 📤 Output Files

### After Blender Rendering
```
/tmp/mirror_captures/
├── capture_000_h0.100.png    # Lowest position
├── capture_001_h0.244.png
├── capture_002_h0.389.png
├── ...
└── capture_009_h1.300.png    # Highest position
```

### After Processing
```
/tmp/mirror_unwrapped/
├── capture_000_h0.100_band.png       # Cropped central bands
├── capture_001_h0.244_band.png
├── ...
├── capture_009_h1.300_band.png
└── stitched_cylinder_map.png         # ⭐ FINAL RESULT
```

The **stitched_cylinder_map.png** is your complete cylinder inspection map:
- Width = 360° circumference
- Height = Vertical extent of cylinder
- Red markers = Defects for inspection

---

## 🔍 Understanding the Output

The final stitched image represents the "unwrapped" cylinder interior:

```
┌──────────────────────────────────────────┐
│ 0°            180°            360°       │ ← Circumference
├──────────────────────────────────────────┤
│                                          │
│                                          │
│     Cylinder Interior (Unwrapped)        │ ← Height
│                                          │
│         🔴 ← Defect markers              │
│                                          │
└──────────────────────────────────────────┘

Width:  3600 pixels = 360° (10 px per degree)
Height: Varies based on NUM_CAPTURES and crop %
```

---

## 🛠️ Troubleshooting

### "Blender not found"
```bash
# Specify Blender path:
python run_workflow.py --blender /path/to/blender
```

### "Visible seams in stitched image"
```bash
# Use more captures and larger overlap:
# In generate_synthetic_images.py:
NUM_CAPTURES = 15  # or 20

# Seams should blend automatically
```

### "Rendering is too slow"
```python
# In generate_synthetic_images.py:
SAMPLES = 128           # Reduce from 256
RENDER_WIDTH = 1536     # Reduce from 2048
NUM_CAPTURES = 7        # Reduce from 10
```

### "Images still have distortion"
```bash
# Use more conservative cropping:
python unwrap_and_crop.py INPUT -o OUTPUT --crop 0.3

# And/or use more captures:
# Set NUM_CAPTURES = 15 or 20
```

### "Mirror not detected correctly"
```bash
# Inspect the unwrapped images:
python unwrap_and_crop.py INPUT -o OUTPUT --save-full

# Then manually check the full_unwrap images
```

---

## 🎓 Advanced Usage

### Custom Defect Generation

Edit `generate_synthetic_images.py`:

```python
# Add specific defect at known location
bpy.ops.mesh.primitive_cube_add(
    size=0.01,
    location=(0.48, 0.1, 0.6)  # Near cylinder wall
)
defect = bpy.context.object
defect.scale = (1, 0.2, 20)  # Vertical scratch
defect.data.materials.append(defect_material)
```

### Batch Processing Multiple Scenarios

```bash
# Create multiple cylinder configurations:
for height in 1.0 1.5 2.0; do
    # Edit CYLINDER_HEIGHT in script
    sed -i "s/CYLINDER_HEIGHT = .*/CYLINDER_HEIGHT = $height/" generate_synthetic_images.py
    
    # Run workflow
    python run_workflow.py --render-dir ./output_h${height}/
done
```

### Integration with ML Pipeline

```python
# Use generated data for training
import cv2
import glob

# Load all stitched maps
maps = [cv2.imread(f) for f in glob.glob('./output*/stitched_*.png')]

# Extract defect regions
# Train defect detector
# Validate on synthetic data
```

---

## 🌟 Key Benefits

✅ **High Quality**: Minimal distortion across entire cylinder map
✅ **Realistic**: Physically-based rendering in Blender
✅ **Flexible**: Easily adjust parameters for different scenarios
✅ **Complete Pipeline**: From rendering to final output
✅ **Reproducible**: Synthetic data with known ground truth
✅ **Scalable**: Generate thousands of training images

---

## 📖 Documentation

- **WORKFLOW_GUIDE.md** - Complete detailed guide
- **workflow_visualization.html** - Visual workflow diagram
- **system_diagram.html** - Optical system explanation

---

## 🚀 Next Steps

1. ✅ Run the quick start example
2. ✅ Adjust parameters for your use case
3. ✅ Add custom defects to the Blender scene
4. ✅ Integrate with your inspection pipeline
5. ✅ Generate training data for ML models

---

## 💡 Tips for Best Results

1. **Start small**: Use fast settings first, then scale up
2. **Validate early**: Check one capture before running full batch
3. **Use GPU**: Enable in Blender for 3-5x speedup
4. **Monitor RAM**: Large batches can use significant memory
5. **Save settings**: Document your configuration for reproducibility

---

## 📊 Performance Optimization

### Speed Up Rendering
- Enable GPU in Blender (Edit → Preferences → System)
- Reduce SAMPLES (256 → 128)
- Use smaller resolution (2048 → 1536)
- Fewer captures (10 → 7)

### Improve Quality
- Increase SAMPLES (256 → 512)
- More captures (10 → 20)
- Larger resolution (2048 → 4096)
- Smaller crop percentage (0.4 → 0.3)

---

## 🔗 Related Resources

- Blender: https://www.blender.org/
- OpenCV: https://opencv.org/
- Omnidirectional vision: Research papers on catadioptric imaging
- Industrial inspection: ISO standards for surface inspection

---

## ❓ FAQ

**Q: Can I use this with real camera data?**
A: Yes! Use `unwrap_and_crop.py` on real images. Adjust mirror parameters as needed.

**Q: How do I add more defects?**
A: Edit `generate_synthetic_images.py` and increase `NUM_DEFECTS` or add custom ones.

**Q: What if my cylinder is different size?**
A: Adjust `CYLINDER_RADIUS` and `CYLINDER_HEIGHT` in the Blender script.

**Q: Can I automate different scenarios?**
A: Yes! Use bash/python loops to run workflow with different parameters.

**Q: How accurate is the distortion correction?**
A: The parabolic projection is a good approximation. For higher accuracy, calibrate with real data.

---

**Ready to start?** → `python run_workflow.py`

For questions or issues, refer to **WORKFLOW_GUIDE.md** for detailed documentation.
