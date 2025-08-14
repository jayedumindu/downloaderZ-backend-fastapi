#!/usr/bin/env python3
"""
FastAPI backend for downloading and converting media files from URLs.
Supports multiple formats with automatic cleanup and bandwidth optimization.
"""

import asyncio
import os
import pathlib
import shutil
import tempfile
import time
import uuid
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

import yt_dlp
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from pydantic import BaseModel, validator
import uvicorn
from fastapi import APIRouter
from fastapi.staticfiles import StaticFiles
from enum import Enum

# ---------- Configuration ----------
MAX_FILE_SIZE_MB = 100  # Maximum file size in MB
MAX_CONCURRENT_DOWNLOADS = 3  # Limit concurrent downloads
CLEANUP_INTERVAL_SECONDS = 60  # Clean up files after 1 minute
SUPPORTED_FORMATS = ["mp3", "mp4", "wav", "m4a"]
# TEMP_DIR = pathlib.Path("temp_downloads")
MAX_URLS_PER_REQUEST = 5  # Limit URLs per request to save bandwidth

# Create temp directory
# TEMP_DIR.mkdir(exist_ok=True)

PUBLIC_DIR = pathlib.Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)

print("yt-dlp version:", yt_dlp.version.__version__)

# Global storage for download tracking
downloads_db = {}
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

app = FastAPI(title="Media Downloader API", version="1.0.0")
app.mount("/public", StaticFiles(directory="public"), name="public")
logging.basicConfig(level=logging.INFO)

# CORS middleware for NextJS frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:9002", "https://yourdomain.com"],  # Add your frontend URLs
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Pydantic Models ----------
class DownloadRequest(BaseModel):
    urls: List[str]
    format: str = "mp3"
    quality: Optional[str] = "192K"
    
    @validator('urls')
    def validate_urls(cls, v):
        if len(v) > MAX_URLS_PER_REQUEST:
            raise ValueError(f'Maximum {MAX_URLS_PER_REQUEST} URLs allowed per request')
        if not v:
            raise ValueError('At least one URL is required')
        return v
    
    @validator('format')
    def validate_format(cls, v):
        if v.lower() not in SUPPORTED_FORMATS:
            raise ValueError(f'Format must be one of: {", ".join(SUPPORTED_FORMATS)}')
        return v.lower()

class DownloadResponse(BaseModel):
    download_id: str
    status: str
    message: str
    files: List[dict] = []

class DownloadStatus(BaseModel):
    download_id: str
    status: str
    progress: str
    files: List[dict] = []
    error: Optional[str] = None

class InfoType(str, Enum):
    audio = "audio"
    video = "video"
    both = "both"

class InfoRequest(BaseModel):
    urls: List[str]
    type: InfoType = InfoType.both  # "audio", "video", or "both"


class FormatInfo(BaseModel):
    url: str
    title: str
    thumbnail: str
    formats: List[Dict[str, Any]]

class BatchDownloadItem(BaseModel):
    url: str
    format: str
    quality: Optional[str] = "192K"

class BatchDownloadRequest(BaseModel):
    files: List[BatchDownloadItem]

# ---------- Utility Functions ----------
def get_file_size_mb(file_path: pathlib.Path) -> float:
    """Get file size in MB."""
    return file_path.stat().st_size / (1024 * 1024)

def generate_download_id() -> str:
    """Generate unique download ID."""
    return str(uuid.uuid4())

def cleanup_old_files():
    """Clean up files older than CLEANUP_INTERVAL_SECONDS."""
    try:
        current_time = time.time()
        for item in PUBLIC_DIR.iterdir():
            if item.is_file():
                file_age = current_time - item.stat().st_mtime
                if file_age > CLEANUP_INTERVAL_SECONDS:
                    item.unlink()
                    print(f"Cleaned up old file: {item.name}")
            elif item.is_dir():
                # Clean up empty directories or old download directories
                try:
                    if not any(item.iterdir()):
                        item.rmdir()
                except OSError:
                    pass
    except Exception as e:
        print(f"Cleanup error: {e}")

async def precheck_file_sizes(urls_with_formats):
    """Check all files for size before downloading. Returns (ok, error_message)."""
    for item in urls_with_formats:
        url = item["url"]
        format = item["format"]
        quality = item.get("quality", "192K")
        with yt_dlp.YoutubeDL({
            "quiet": True,
            "no_warnings": True,
            "http_headers": {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
            }
        }) as ydl:
            info = ydl.extract_info(url, download=False)
            best_format = None
            for f in info.get("formats", []):
                ext = f.get("ext", "").lower()
                if ext == format and "storyboard" not in (f.get("format_note") or "").lower():
                    best_format = f
                    break
            if best_format:
                est_size = best_format.get("filesize") or best_format.get("filesize_approx") or 0
                est_size_mb = est_size / (1024 * 1024)
                if est_size > 0 and est_size_mb > MAX_FILE_SIZE_MB:
                    return False, f"File '{info.get('title', url)}' estimated size {est_size_mb:.1f}MB exceeds the {MAX_FILE_SIZE_MB}MB limit."
    return True, None

async def download_single_url(url: str, format: str, quality: str, download_dir: pathlib.Path, request: Request) -> dict:    
    """Download a single URL and return file info, with pre-download size check."""
    async with download_semaphore:
        try:
            # Pre-check: get estimated size for the requested format
            with yt_dlp.YoutubeDL({
                "quiet": True,
                "no_warnings": True,
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                }
            }) as ydl:
                info = ydl.extract_info(url, download=False)
                # Find the best matching format
                best_format = None
                for f in info.get("formats", []):
                    ext = f.get("ext", "").lower()
                    if ext == format and "storyboard" not in (f.get("format_note") or "").lower():
                        best_format = f
                        break
                if best_format:
                    est_size = best_format.get("filesize") or best_format.get("filesize_approx") or 0
                    est_size_mb = est_size / (1024 * 1024)
                    if est_size > 0 and est_size_mb > MAX_FILE_SIZE_MB:
                        return {
                            "url": url,
                            "title": info.get("title", "Unknown"),
                            "filename": None,
                            "size_mb": round(est_size_mb, 2),
                            "download_url": None,
                            "status": "error",
                            "error": f"Estimated file size too large: {est_size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)"
                        }

            # Configure yt-dlp options for actual download
            ydl_opts = {
                "format": "bestaudio/best" if format in ["mp3", "wav", "m4a"] else "bestvideo+bestaudio/best",
                "outtmpl": str(download_dir / "%(title)s.%(ext)s"),
                "quiet": True,
                "no_warnings": True,
                "noplaylist": True,
                "ignoreerrors": True,
                "no_color": True,
                "force_ipv4": True,
                "socket_timeout": 30,
                "extractaudio": format in ["mp3", "wav", "m4a"],
                "audioformat": format if format in ["mp3", "wav", "m4a"] else None,
                "audioquality": quality.rstrip("K") if format in ["mp3", "wav", "m4a"] else None,
                "postprocessors": [],
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36"
                }
            }

            if format in ["mp3", "wav", "m4a"]:
                ydl_opts["postprocessors"] = [
                    {
                        "key": "FFmpegExtractAudio",
                        "preferredcodec": format,
                        "preferredquality": quality.rstrip("K"),
                    }
                ]

            # Download the file
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                title = info.get('title', 'Unknown')

                # Find the downloaded file
                downloaded_files = list(download_dir.glob(f"{title}*"))
                if not downloaded_files:
                    # Try to find any new file in the directory
                    downloaded_files = [f for f in download_dir.iterdir() if f.is_file()]

                if not downloaded_files:
                    raise Exception("Downloaded file not found")

                file_path = downloaded_files[0]
                file_size_mb = get_file_size_mb(file_path)

                # Check file size limit after download (actual size)
                if file_size_mb > MAX_FILE_SIZE_MB:
                    file_path.unlink()  # Delete oversized file
                    raise Exception(f"File too large: {file_size_mb:.1f}MB (max: {MAX_FILE_SIZE_MB}MB)")

                return {
                    "url": url,
                    "title": title,
                    "filename": file_path.name,
                    "size_mb": round(file_size_mb, 2),
                    "download_url": str(request.base_url) + f"public/{download_dir.name}/{file_path.name}",                    "status": "success"
                }

        except Exception as e:
            return {
                "url": url,
                "title": "Failed",
                "filename": None,
                "size_mb": 0,
                "download_url": None,
                "status": "error",
                 "error": str(e)
            }

async def process_downloads(download_id: str, urls: List[str], format: str, quality: str, request: Request):
    """Process all downloads for a request."""
    download_dir = PUBLIC_DIR / download_id
    download_dir.mkdir(exist_ok=True)
    
    # Update status
    downloads_db[download_id]["status"] = "downloading"
    downloads_db[download_id]["progress"] = "Starting downloads..."
    
    files = []
    for i, url in enumerate(urls):
        downloads_db[download_id]["progress"] = f"Downloading {i+1}/{len(urls)}: {url[:50]}..."
        
        file_info = await download_single_url(url, format, quality, download_dir, request)        
        files.append(file_info)
        
        downloads_db[download_id]["files"] = files
    
    # Update final status
    successful_downloads = [f for f in files if f["status"] == "success"]
    failed_downloads = [f for f in files if f["status"] == "error"]
    
    if successful_downloads:
        downloads_db[download_id]["status"] = "completed"
        downloads_db[download_id]["progress"] = f"Completed: {len(successful_downloads)} successful, {len(failed_downloads)} failed"
    else:
        downloads_db[download_id]["status"] = "failed"
        downloads_db[download_id]["progress"] = "All downloads failed"
        downloads_db[download_id]["error"] = "No files were successfully downloaded"

# ---------- Background Tasks ----------
async def cleanup_task():
    """Background task to clean up old files."""
    while True:
        cleanup_old_files()
        # Also clean up old download records
        current_time = datetime.now()
        expired_downloads = [
            download_id for download_id, data in downloads_db.items()
            if current_time - data["created_at"] > timedelta(minutes=5)
        ]
        for download_id in expired_downloads:
            downloads_db.pop(download_id, None)
        
        await asyncio.sleep(30)  # Check every 30 seconds

# Start cleanup task
@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task())

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    return {"message": "Media Downloader API", "version": "1.0.0"}

@app.post("/info")
async def get_info(request: InfoRequest):
    """
    Get media info for provided URLs.
    - type: "audio" returns only audio formats
    - type: "video" returns only video formats
    - type: "both" returns all supported formats
    """
    logging.info(f"Received /info request: {request.urls} type={request.type}")
    results = []
    for url in request.urls:
        try:
            logging.info(f"Processing URL: {url}")
            with yt_dlp.YoutubeDL({
                "quiet": True,
                "no_warnings": True,
                # Example for yt-dlp usage in your /info endpoint and download logic
                "http_headers": {
                    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36",
                    "Accept-Language": "en-US,en;q=0.9",
                    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
                    "Accept-Encoding": "gzip, deflate, br",
                    "Connection": "keep-alive",
                    "DNT": "1",
                    "Upgrade-Insecure-Requests": "1",
                    "Sec-Fetch-Dest": "document",
                    "Sec-Fetch-Mode": "navigate",
                    "Sec-Fetch-Site": "none",
                    "Sec-Fetch-User": "?1"
                }
            }) as ydl:
                info = ydl.extract_info(url, download=False)
                logging.info(f"yt-dlp info fetched for {url}: {info.get('title')}")
                formats = []
                for f in info.get("formats", []):
                    ext = f.get("ext", "").lower()
                    acodec = (f.get("acodec") or "").lower()
                    vcodec = (f.get("vcodec") or "").lower()
                    # Only include supported formats and skip storyboard, images, etc.
                    if ext not in SUPPORTED_FORMATS:
                        continue
                    if "storyboard" in (f.get("format_note") or "").lower():
                        continue
                    # Filter by requested type
                    if request.type == InfoType.audio and (not acodec or acodec == "none"):
                        continue
                    if request.type == InfoType.video and (not vcodec or vcodec == "none"):
                        continue
                    filesize = f.get("filesize") or f.get("filesize_approx") or 0
                    formats.append({
                        "format_id": f.get("format_id"),
                        "ext": ext,
                        "format_note": f.get("format_note"),
                        "filesize": filesize,
                        "audio_channels": f.get("audio_channels"),
                        "acodec": f.get("acodec"),
                        "vcodec": f.get("vcodec"),
                        "resolution": f.get("resolution"),
                        "abr": f.get("abr"),
                        "fps": f.get("fps"),
                        "tbr": f.get("tbr"),
                        "url": f.get("url"),
                    })
                results.append({
                    "url": url,
                    "title": info.get("title"),
                    "thumbnail": info.get("thumbnail"),
                    "formats": formats,
                })
        except Exception as e:
            logging.error(f"Error processing {url}: {e}")
            results.append({
                "url": url,
                "title": None,
                "thumbnail": None,
                "formats": [],
                "error": str(e)
            })
    logging.info(f"/info results: {results}")
    return {"results": results}

@app.post("/download", response_model=DownloadResponse)
async def create_download(request: DownloadRequest, background_tasks: BackgroundTasks, fastapi_request: Request):
    """Create a new download request."""
    try:
        # Pre-check all files
        urls_with_formats = [{"url": url, "format": request.format, "quality": request.quality or "192K"} for url in request.urls]
        ok, error_message = await precheck_file_sizes(urls_with_formats)
        if not ok:
            raise HTTPException(status_code=400, detail=error_message)

        download_id = generate_download_id()
        downloads_db[download_id] = {
            "status": "queued",
            "progress": "Queued for download",
            "files": [],
            "created_at": datetime.now(),
            "error": None
        }
        background_tasks.add_task(
            process_downloads, 
            download_id, 
            request.urls, 
            request.format, 
            request.quality or "192K",
            fastapi_request
        )
        return DownloadResponse(
            download_id=download_id,
            status="queued",
            message="Download request queued successfully",
            files=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create download: {str(e)}")

@app.get("/status/{download_id}", response_model=DownloadStatus)
async def get_download_status(download_id: str):
    """Get download status and progress."""
    if download_id not in downloads_db:
        raise HTTPException(status_code=404, detail="Download not found")
    
    data = downloads_db[download_id]
    return DownloadStatus(
        download_id=download_id,
        status=data["status"],
        progress=data["progress"],
        files=data["files"],
        error=data["error"]
    )

@app.post("/download/batch", response_model=DownloadResponse)
async def batch_download(request: BatchDownloadRequest, background_tasks: BackgroundTasks):
    try:
        # Pre-check all files
        urls_with_formats = [{"url": item.url, "format": item.format, "quality": item.quality or "192K"} for item in request.files]
        ok, error_message = await precheck_file_sizes(urls_with_formats)
        if not ok:
            raise HTTPException(status_code=400, detail=error_message)

        download_id = generate_download_id()
        downloads_db[download_id] = {
            "status": "queued",
            "progress": "Queued for batch download",
            "files": [],
            "created_at": datetime.now(),
            "error": None
        }
        async def process_batch(download_id, files):
            download_dir = PUBLIC_DIR / download_id
            download_dir.mkdir(exist_ok=True)
            downloads_db[download_id]["status"] = "downloading"
            downloads_db[download_id]["progress"] = "Starting batch downloads..."
            results = []
            for i, item in enumerate(files):
                downloads_db[download_id]["progress"] = f"Downloading {i+1}/{len(files)}: {item.url[:50]}..."
                file_info = await download_single_url(item.url, item.format, item.quality or "192K", download_dir)
                results.append(file_info)
                downloads_db[download_id]["files"] = results
            successful = [f for f in results if f["status"] == "success"]
            failed = [f for f in results if f["status"] == "error"]
            if successful:
                downloads_db[download_id]["status"] = "completed"
                downloads_db[download_id]["progress"] = f"Completed: {len(successful)} successful, {len(failed)} failed"
            else:
                downloads_db[download_id]["status"] = "failed"
                downloads_db[download_id]["progress"] = "All downloads failed"
                downloads_db[download_id]["error"] = "No files were successfully downloaded"

        background_tasks.add_task(process_batch, download_id, request.files)
        return DownloadResponse(
            download_id=download_id,
            status="queued",
            message="Batch download request queued successfully",
            files=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create batch download: {str(e)}")

@app.get("/download/{filename}")
async def download_file(filename: str, request: Request):
    """Download a specific file."""
    # Find file in any subdirectory
    file_path = None
    for item in PUBLIC_DIR.rglob(filename):
        if item.is_file():
            file_path = item
            break
    
    if not file_path or not file_path.exists():
        raise HTTPException(status_code=404, detail="File not found")
    
    # Check if file is too old (shouldn't happen due to cleanup, but just in case)
    file_age = time.time() - file_path.stat().st_mtime
    if file_age > CLEANUP_INTERVAL_SECONDS * 2:  # Double the cleanup interval for safety
        raise HTTPException(status_code=410, detail="File has expired")
    
    return FileResponse(
        path=str(file_path),
        filename=filename,
        media_type='application/octet-stream'
    )

@app.get("/formats")
async def get_supported_formats():
    """Get list of supported formats."""
    return {"formats": SUPPORTED_FORMATS}

@app.delete("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup (for testing)."""
    cleanup_old_files()
    return {"message": "Cleanup completed"}

# ---------- Error Handlers ----------
@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    return JSONResponse(
        status_code=500,
        content={"detail": f"Internal server error: {str(exc)}"}
    )

if __name__ == "__main__":
    import os
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(
        "app:app",  # Replace "main" with your filename (without .py)
        host="0.0.0.0",
        port=port,
        reload=True
    )
