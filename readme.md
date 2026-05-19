# Smart Spatial Digital Twin for Construction Monitoring

> **Detecting construction deviations by comparing phone-captured 3D reconstructions against a BIM model**

A lightweight pipeline using photogrammetry, Open3D, and a Streamlit dashboard — no drones or LiDAR required.

---

## The Problem

Construction sites deviate from design. Catching those deviations late is expensive:

- Manual inspections are infrequent and subjective
- BIM models represent intent, not reality
- **Result:** Late detection of execution errors, contractor disputes, and costly rework

---

## The Solution

Reconstruct the physical site from ordinary phone photos, align it to the BIM model, and automatically surface deviations — giving engineers objective, visual feedback at any point in the project.

---

## Pipeline

```
Input              Reconstruction        Alignment             Analysis & Output
──────────────     ──────────────        ──────────────        ──────────────────
Phone images  →    Photogrammetry   →    BIM Registration  →   Deviation Heatmap
(20–100 imgs)      (COLMAP /             (Open3D ICP)          Streamlit Dashboard
                    Meshroom)                                   Alerts
BIM Model     ↗    Point Cloud /
(IFC / OBJ)        Mesh (.ply/.obj)
```

---

## Input Data

| Input | Format | Source |
|-------|--------|--------|
| Site photos | JPG / PNG | Phone camera (20–100 overlapping images) |
| BIM model | IFC / OBJ / PLY | Revit, ArchiCAD, or similar |

### Photo Capture Guidelines
- 60–80% overlap between consecutive frames
- Walk around the object methodically, not randomly
- Consistent lighting, no motion blur
- Cover the target from multiple heights and angles

---

## Technology Stack

| Step | Tool | Purpose |
|------|------|---------|
| Photogrammetry | COLMAP / Meshroom | Phone images → dense point cloud |
| BIM parsing | IfcOpenShell | IFC → mesh / sampled point cloud |
| Registration | Open3D (ICP) | Align reconstructed cloud to BIM |
| Deviation detection | Open3D | Cloud-to-mesh distance computation |
| Visualization | Streamlit + Plotly | Interactive heatmap and alerts dashboard |

---

## Pipeline Steps

1. **Capture** — photograph the site with a phone following overlap guidelines
2. **Reconstruct** — run photogrammetry (Meshroom recommended for beginners) to produce a `.ply` point cloud
3. **Parse BIM** — convert the IFC/BIM model to a comparable mesh or point cloud
4. **Register** — align the two using ICP in Open3D (coarse + fine registration)
5. **Detect deviations** — compute point-to-mesh distances; threshold to flag anomalies
6. **Visualize** — display a colored heatmap and deviation alerts on a Streamlit dashboard

---

## Applications

| Domain | Use Case |
|--------|----------|
| **Construction** | Scan-to-BIM comparison, detecting execution deviations |
| **Interior fit-out** | Verifying wall positions, openings, and installed elements |
| **Renovation** | As-built documentation from phone survey |

---

## Roadmap

```
MVP (Now)                    Near Future                    Long-Term
──────────────               ──────────────────────         ──────────────────────
Phone images +    →          Drone / LiDAR input       →    Real-time edge capture
BIM comparison               Higher accuracy clouds          Predictive alerts
Deviation heatmap            Time-series tracking            IoT integration
```

---

## Project Structure (planned)

```
tempi1/
├── data/
│   ├── images/          # raw phone photos
│   ├── pointclouds/     # reconstructed .ply files
│   └── bim/             # IFC / OBJ BIM exports
├── pipeline/
│   ├── reconstruct.py   # COLMAP wrapper or Meshroom guide
│   ├── parse_bim.py     # IFC → point cloud via IfcOpenShell
│   ├── register.py      # Open3D ICP registration
│   └── deviation.py     # cloud-to-mesh distance + thresholding
├── dashboard/
│   └── app.py           # Streamlit dashboard
└── readme.md
```

---

## Summary

> A ground-up pipeline that turns phone photos and a BIM model into an actionable deviation report — no specialized hardware required.

**Phone images + BIM → 3D reconstruction → registration → deviation map → decisions.**
