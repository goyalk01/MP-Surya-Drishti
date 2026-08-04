# MP Surya-Drishti Project Roadmap

This document outlines the multi-phase engineering roadmap for **MP Surya-Drishti: AI-Assisted Community Solar Advisory System** (EPICS Project at VIT Bhopal University).

---

## 🎯 Phase 1: Pre-Installation Rooftop Feasibility (Current Foundation)
- [x] Pluggable Computer Vision Segmentation Framework (`BaseSegmentationModel`)
- [x] SegFormer model implementation (`nvidia/mit-b2`)
- [x] Official Massachusetts Buildings Dataset loader integration
- [x] Postprocessing pipeline (Morphological cleaning, polygon extraction, scale-aware area estimation)
- [x] TensorBoard integration, smart checkpointing (`best_iou.pth`, `best_loss.pth`), experiment versioning
- [x] Standardized `prediction_report.json` data contract for downstream analytics

---

## 🌤️ Phase 2: Post-Installation Shadow & Obstacle Analytics
- [ ] Computer Vision obstacle detection (HVAC units, water tanks, chimneys, trees)
- [ ] Sun position & ray-tracing shadow prediction engine based on solar azimuth/elevation angles
- [ ] Usable rooftop area calculation deducting obstacles and shaded zones
- [ ] Intelligent Panel Placement Optimization algorithm (tilt angle, azimuth orientation, string layout)

---

## ⚡ Phase 3: Energy Generation & Financial Modeling
- [ ] Weather API integration (GHI, DNI, ambient temperature, cloud cover)
- [ ] PV generation estimation engine (hourly, monthly, annual kWh yields)
- [ ] Tariff structure & Net Metering financial model (DISCOM rates, ROI, payback period, CO2 offset)
- [ ] Panel degradation & lifecycle performance simulation

---

## 🔍 Phase 4: Post-Installation Panel Health Monitoring
- [ ] Drone / satellite infrared & RGB panel defect detection (hotspots, cracks, dust accumulation)
- [ ] Anomaly classification using fine-tuned YOLOv8 / ViT vision backbone
- [ ] Maintenance alert system & efficiency loss reporting

---

## 🌐 Phase 5: Production Deployment & Platform UI
- [ ] FastAPI backend microservices wrapping all inference and analytics modules
- [ ] Next.js + TailWind CSS interactive dashboard with Leaflet/MapLibre map visualization
- [ ] PDF Advisory Report generation service for householders & community planners
