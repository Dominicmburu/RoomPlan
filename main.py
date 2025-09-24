from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
import numpy as np
import cv2
import open3d as o3d
import json
import uuid
import os
from typing import List, Optional
import asyncio
from pathlib import Path
import tempfile
import zipfile
from pydantic import BaseModel
import base64

app = FastAPI(title="Room Scanner API", version="1.0.0")

# Enable CORS for web clients
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure appropriately for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Data models
class CameraFrame(BaseModel):
    image_data: str  # Base64 encoded image
    depth_data: Optional[str] = None  # Base64 encoded depth map
    camera_position: List[float]  # [x, y, z]
    camera_rotation: List[float]  # [qx, qy, qz, qw] quaternion
    timestamp: float
    intrinsics: List[float]  # Camera intrinsic parameters [fx, fy, cx, cy]

class ScanSession:
    def __init__(self, session_id: str):
        self.session_id = session_id
        self.frames = []
        self.point_clouds = []
        self.mesh = None
        self.temp_dir = tempfile.mkdtemp()
        
    def add_frame(self, frame: CameraFrame):
        self.frames.append(frame)
        
    def cleanup(self):
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)

# Global session storage (use Redis/database in production)
sessions = {}

class RoomReconstructor:
    @staticmethod
    def decode_image(base64_data: str) -> np.ndarray:
        """Decode base64 image data"""
        image_data = base64.b64decode(base64_data.split(',')[1])
        nparr = np.frombuffer(image_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    @staticmethod
    def decode_depth(base64_data: str) -> np.ndarray:
        """Decode base64 depth data"""
        depth_data = base64.b64decode(base64_data.split(',')[1])
        nparr = np.frombuffer(depth_data, np.uint8)
        return cv2.imdecode(nparr, cv2.IMREAD_UNCHANGED)
    
    @staticmethod
    def create_point_cloud_from_rgbd(rgb_image: np.ndarray, 
                                   depth_image: np.ndarray,
                                   intrinsics: List[float],
                                   camera_pose: np.ndarray) -> o3d.geometry.PointCloud:
        """Create point cloud from RGB-D data"""
        height, width = rgb_image.shape[:2]
        fx, fy, cx, cy = intrinsics
        
        # Create Open3D RGB-D image
        rgb_o3d = o3d.geometry.Image(cv2.cvtColor(rgb_image, cv2.COLOR_BGR2RGB))
        depth_o3d = o3d.geometry.Image(depth_image.astype(np.uint16))
        rgbd_image = o3d.geometry.RGBDImage.create_from_color_and_depth(rgb_o3d, depth_o3d)
        
        # Camera intrinsics
        intrinsic_matrix = o3d.camera.PinholeCameraIntrinsic(
            width, height, fx, fy, cx, cy
        )
        
        # Generate point cloud
        pcd = o3d.geometry.PointCloud.create_from_rgbd_image(
            rgbd_image, intrinsic_matrix
        )
        
        # Transform to world coordinates
        pcd.transform(camera_pose)
        
        return pcd
    
    @staticmethod
    def quaternion_to_matrix(position: List[float], rotation: List[float]) -> np.ndarray:
        """Convert position and quaternion to transformation matrix"""
        x, y, z = position
        qx, qy, qz, qw = rotation
        
        # Quaternion to rotation matrix
        R = np.array([
            [1-2*(qy*qy + qz*qz), 2*(qx*qy - qz*qw), 2*(qx*qz + qy*qw)],
            [2*(qx*qy + qz*qw), 1-2*(qx*qx + qz*qz), 2*(qy*qz - qx*qw)],
            [2*(qx*qz - qy*qw), 2*(qy*qz + qx*qw), 1-2*(qx*qx + qy*qy)]
        ])
        
        # Create transformation matrix
        T = np.eye(4)
        T[:3, :3] = R
        T[:3, 3] = [x, y, z]
        
        return T
    
    @staticmethod
    def process_frames(frames: List[CameraFrame]) -> o3d.geometry.PointCloud:
        """Process frames and create unified point cloud"""
        all_pcds = []
        
        for frame in frames:
            try:
                # Decode image data
                rgb_image = RoomReconstructor.decode_image(frame.image_data)
                
                # Skip if no depth data
                if not frame.depth_data:
                    continue
                    
                depth_image = RoomReconstructor.decode_depth(frame.depth_data)
                
                # Get camera pose
                camera_pose = RoomReconstructor.quaternion_to_matrix(
                    frame.camera_position, frame.camera_rotation
                )
                
                # Create point cloud
                pcd = RoomReconstructor.create_point_cloud_from_rgbd(
                    rgb_image, depth_image, frame.intrinsics, camera_pose
                )
                
                if len(pcd.points) > 0:
                    all_pcds.append(pcd)
                    
            except Exception as e:
                print(f"Error processing frame: {e}")
                continue
        
        if not all_pcds:
            return o3d.geometry.PointCloud()
        
        # Combine all point clouds
        combined_pcd = all_pcds[0]
        for pcd in all_pcds[1:]:
            combined_pcd += pcd
        
        # Remove statistical outliers
        combined_pcd, _ = combined_pcd.remove_statistical_outlier(
            nb_neighbors=20, std_ratio=2.0
        )
        
        # Downsample for performance
        combined_pcd = combined_pcd.voxel_down_sample(voxel_size=0.02)
        
        return combined_pcd
    
    @staticmethod
    def create_mesh_from_pointcloud(pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """Create mesh from point cloud using Poisson reconstruction"""
        if len(pcd.points) < 100:
            raise ValueError("Insufficient points for reconstruction")
        
        # Estimate normals
        pcd.estimate_normals()
        pcd.orient_normals_consistent_tangent_plane(30)
        
        # Poisson surface reconstruction
        mesh, _ = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(
            pcd, depth=9
        )
        
        # Remove low density vertices
        bbox = pcd.get_axis_aligned_bounding_box()
        mesh = mesh.crop(bbox)
        
        # Simplify mesh
        mesh = mesh.simplify_quadric_decimation(target_number_of_triangles=10000)
        
        # Smooth mesh
        mesh = mesh.filter_smooth_simple(number_of_iterations=1)
        
        # Compute vertex normals
        mesh.compute_vertex_normals()
        
        return mesh

class RoomSegmenter:
    """Simple rule-based room segmentation"""
    
    @staticmethod
    def segment_room_elements(mesh: o3d.geometry.TriangleMesh) -> dict:
        """Segment mesh into room elements (walls, floor, ceiling)"""
        vertices = np.asarray(mesh.vertices)
        
        if len(vertices) == 0:
            return {"walls": [], "floor": [], "ceiling": []}
        
        # Simple height-based segmentation
        min_y = vertices[:, 1].min()
        max_y = vertices[:, 1].max()
        height_range = max_y - min_y
        
        # Floor: bottom 10% of points
        floor_threshold = min_y + 0.1 * height_range
        # Ceiling: top 10% of points
        ceiling_threshold = max_y - 0.1 * height_range
        
        floor_indices = vertices[:, 1] < floor_threshold
        ceiling_indices = vertices[:, 1] > ceiling_threshold
        wall_indices = ~(floor_indices | ceiling_indices)
        
        return {
            "floor": np.where(floor_indices)[0].tolist(),
            "ceiling": np.where(ceiling_indices)[0].tolist(),
            "walls": np.where(wall_indices)[0].tolist(),
            "bounds": {
                "min": vertices.min(axis=0).tolist(),
                "max": vertices.max(axis=0).tolist()
            }
        }

@app.post("/api/scan/start")
async def start_scan():
    """Start a new scanning session"""
    session_id = str(uuid.uuid4())
    sessions[session_id] = ScanSession(session_id)
    return {"session_id": session_id, "status": "started"}

@app.post("/api/scan/{session_id}/frame")
async def add_frame(session_id: str, frame: CameraFrame):
    """Add a frame to the scanning session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    session.add_frame(frame)
    
    return {
        "status": "frame_added",
        "frame_count": len(session.frames)
    }

@app.post("/api/scan/{session_id}/process")
async def process_scan(session_id: str):
    """Process the scan and generate 3D model"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    
    if len(session.frames) < 3:
        raise HTTPException(status_code=400, detail="Need at least 3 frames for reconstruction")
    
    try:
        # Create point cloud from frames
        combined_pcd = RoomReconstructor.process_frames(session.frames)
        
        if len(combined_pcd.points) == 0:
            raise HTTPException(status_code=400, detail="No valid point cloud generated")
        
        # Create mesh
        mesh = RoomReconstructor.create_mesh_from_pointcloud(combined_pcd)
        session.mesh = mesh
        
        # Segment room elements
        segmentation = RoomSegmenter.segment_room_elements(mesh)
        
        # Save mesh as GLB
        mesh_path = os.path.join(session.temp_dir, f"{session_id}.glb")
        o3d.io.write_triangle_mesh(mesh_path, mesh)
        
        return {
            "status": "processed",
            "mesh_vertices": len(mesh.vertices),
            "mesh_faces": len(mesh.triangles),
            "segmentation": segmentation,
            "download_url": f"/api/scan/{session_id}/download"
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Processing failed: {str(e)}")

@app.get("/api/scan/{session_id}/download")
async def download_model(session_id: str):
    """Download the generated 3D model"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    mesh_path = os.path.join(session.temp_dir, f"{session_id}.glb")
    
    if not os.path.exists(mesh_path):
        raise HTTPException(status_code=404, detail="Model not found")
    
    return FileResponse(
        mesh_path,
        media_type="model/gltf-binary",
        filename=f"room_scan_{session_id}.glb"
    )

@app.get("/api/scan/{session_id}/status")
async def get_scan_status(session_id: str):
    """Get scanning session status"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    return {
        "session_id": session_id,
        "frame_count": len(session.frames),
        "has_mesh": session.mesh is not None
    }

@app.delete("/api/scan/{session_id}")
async def cleanup_session(session_id: str):
    """Clean up scanning session"""
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="Session not found")
    
    session = sessions[session_id]
    session.cleanup()
    del sessions[session_id]
    
    return {"status": "cleaned_up"}

@app.get("/")
async def root():
    """API health check"""
    return {
        "message": "Room Scanner API is running",
        "active_sessions": len(sessions)
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)