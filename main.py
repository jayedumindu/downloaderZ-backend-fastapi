#!/usr/bin/env python3
"""
FastAPI backend for downloading and converting media files from URLs.
Uses headless browser + HTTP downloads to avoid yt-dlp server detection issues.
"""

import asyncio
import os
import pathlib
import subprocess
import time
import uuid
import json
import httpx
from datetime import datetime, timedelta
from typing import List, Optional, Dict, Any
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import uvicorn
from enum import Enum
import shutil

try:
    from playwright.async_api import async_playwright
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: Playwright not available. Using subprocess method only.")

# ---------- Configuration ----------
MAX_FILE_SIZE_MB = 50
MAX_CONCURRENT_DOWNLOADS = 3
CLEANUP_INTERVAL_SECONDS = 300
SUPPORTED_FORMATS = ["mp3", "mp4", "wav", "m4a"]
MAX_URLS_PER_REQUEST = 5

PUBLIC_DIR = pathlib.Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)

# Global storage for download tracking
downloads_db = {}
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

app = FastAPI(title="Browser-Based Media Downloader API", version="2.0.0")
app.mount("/public", StaticFiles(directory="public"), name="public")
logging.basicConfig(level=logging.INFO)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URLs
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
    type: InfoType = InfoType.both

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
    """Clean up files and folders older than CLEANUP_INTERVAL_SECONDS."""
    try:
        current_time = time.time()
        for item in PUBLIC_DIR.iterdir():
            if item.is_file():
                file_age = current_time - item.stat().st_mtime
                if file_age > CLEANUP_INTERVAL_SECONDS:
                    item.unlink()
                    print(f"Cleaned up old file: {item.name}")
            elif item.is_dir():
                # Check the newest file in the directory
                latest_mtime = max((f.stat().st_mtime for f in item.glob("*")), default=item.stat().st_mtime)
                dir_age = current_time - latest_mtime
                if dir_age > CLEANUP_INTERVAL_SECONDS:
                    shutil.rmtree(item, ignore_errors=True)
                    print(f"Cleaned up old directory: {item.name}")
    except Exception as e:
        print(f"Cleanup error: {e}")

# ---------- Browser-based extraction ----------
async def extract_media_info_with_playwright(url: str) -> Dict[str, Any]:
    """Extract media info using headless browser."""
    if not PLAYWRIGHT_AVAILABLE:
        raise HTTPException(status_code=500, detail="Playwright not available")
    
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True, 
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"]
        )
        page = await browser.new_page()
        
        # Set realistic headers
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        media_urls = []
        title = "Unknown"
        thumbnail = ""
        
        # Capture network requests for media files
        def handle_request(request):
            if any(ext in request.url.lower() for ext in ['.mp4', '.m4a', '.webm', '.mp3']):
                media_urls.append({
                    "url": request.url,
                    "ext": request.url.split('.')[-1].split('?')[0],
                    "format_note": "direct_link"
                })
        
        page.on("request", handle_request)
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            # Extract title
            try:
                title_elem = await page.query_selector("title")
                if title_elem:
                    title = await title_elem.text_content()
            except:
                pass
            
            # Extract thumbnail
            try:
                thumb_elem = await page.query_selector('meta[property="og:image"]')
                if thumb_elem:
                    thumbnail = await thumb_elem.get_attribute("content")
            except:
                pass
            
            # Wait a bit more for lazy-loaded content
            await asyncio.sleep(2)
            
        except Exception as e:
            print(f"Error navigating to {url}: {e}")
        finally:
            await browser.close()
        
        return {
            "title": title,
            "thumbnail": thumbnail,
            "formats": media_urls
        }

# ---------- Subprocess-based extraction ----------
async def extract_info_with_subprocess(url: str) -> Dict[str, Any]:
    """Extract media info using yt-dlp subprocess."""
    try:
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--quiet",
            "--no-warnings",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            url
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            info = json.loads(stdout.decode())
            formats = []
            
            for f in info.get("formats", []):
                ext = f.get("ext", "").lower()
                if ext in SUPPORTED_FORMATS:
                    formats.append({
                        "format_id": f.get("format_id"),
                        "ext": ext,
                        "format_note": f.get("format_note"),
                        "filesize": f.get("filesize") or f.get("filesize_approx") or 0,
                        "url": f.get("url"),
                        "acodec": f.get("acodec"),
                        "vcodec": f.get("vcodec"),
                        "resolution": f.get("resolution"),
                        "abr": f.get("abr"),
                        "fps": f.get("fps"),
                        "tbr": f.get("tbr")
                    })
            
            return {
                "title": info.get("title", "Unknown"),
                "thumbnail": info.get("thumbnail", ""),
                "formats": formats
            }
        else:
            raise Exception(f"yt-dlp failed: {stderr.decode()}")
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to extract info: {str(e)}")

# ---------- Download functions ----------
async def download_with_http(url: str, filepath: pathlib.Path) -> None:
    """Download file using HTTP client."""
    async with httpx.AsyncClient(follow_redirects=True, timeout=300) as client:
        async with client.stream("GET", url) as response:
            if response.status_code != 200:
                raise Exception(f"HTTP download failed: {response.status_code}")
            
            with open(filepath, "wb") as f:
                async for chunk in response.aiter_bytes():
                    f.write(chunk)

# async def download_with_subprocess(url: str, format: str, quality: str, filepath: pathlib.Path) -> None:
#     """Download using yt-dlp subprocess."""
#     cmd = [
#         "yt-dlp",
#         "--output", str(filepath.parent / "%(title)s.%(ext)s"),
#         "--quiet",
#         "--no-warnings",
#         "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
#     ]
    
#     if format in ["mp3", "wav", "m4a"]:
#         cmd.extend([
#             "--extract-audio",
#             "--audio-format", format,
#             "--audio-quality", quality.rstrip("K")
#         ])
#     else:
#         cmd.extend([
#             "--format", "bestvideo+bestaudio/best"
#         ])
    
#     cmd.append(url)
    
#     process = await asyncio.create_subprocess_exec(
#         *cmd,
#         stdout=asyncio.subprocess.PIPE,
#         stderr=asyncio.subprocess.PIPE
#     )
    
#     stdout, stderr = await process.communicate()
    
#     if process.returncode != 0:
#         raise Exception(f"yt-dlp download failed: {stderr.decode()}")

async def download_single_url(url: str, format: str, quality: str, download_dir: pathlib.Path, request: Request) -> dict:
    """Download a single URL using available methods."""
    async with download_semaphore:
        try:
            # Try browser method first
            if PLAYWRIGHT_AVAILABLE:
                try:
                    info = await extract_media_info_with_playwright(url)
                    title = info.get("title", "Unknown")
                    
                    # Find best matching format
                    best_format = None
                    for f in info.get("formats", []):
                        if f.get("ext") == format:
                            best_format = f
                            break
                    
                    if best_format and best_format.get("url"):
                        filename = f"{title}.{format}"
                        filepath = download_dir / filename
                        
                        await download_with_http(best_format["url"], filepath)
                        
                        return {
                            "url": url,
                            "title": title,
                            "filename": filename,
                            "size_mb": round(get_file_size_mb(filepath), 2),
                            "download_url": f"{request.base_url}public/{download_dir.name}/{filename}",
                            "status": "success"
                        }
                except Exception as e:
                    print(f"Browser method failed for {url}: {e}")
            
            # Fallback to subprocess method
            filename = f"download_{int(time.time())}.{format}"
            filepath = download_dir / filename
            
            await download_with_subprocess(url, format, quality, filepath)
            
            # Find the actual downloaded file
            downloaded_files = [f for f in download_dir.iterdir() if f.is_file() and f.name != filename]
            if downloaded_files:
                actual_file = downloaded_files[0]
                return {
                    "url": url,
                    "title": actual_file.stem,
                    "filename": actual_file.name,
                    "size_mb": round(get_file_size_mb(actual_file), 2),
                    "download_url": f"{request.base_url}public/{download_dir.name}/{actual_file.name}",
                    "status": "success"
                }
            
            raise Exception("Downloaded file not found")
            
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
    
    downloads_db[download_id]["status"] = "downloading"
    downloads_db[download_id]["progress"] = "Starting downloads..."
    
    files = []
    for i, url in enumerate(urls):
        downloads_db[download_id]["progress"] = f"Downloading {i+1}/{len(urls)}: {url[:50]}..."
        
        file_info = await download_single_url(url, format, quality, download_dir, request)
        files.append(file_info)
        downloads_db[download_id]["files"] = files
    
    # Update final status
    successful = [f for f in files if f["status"] == "success"]
    failed = [f for f in files if f["status"] == "error"]
    
    if successful:
        downloads_db[download_id]["status"] = "completed"
        downloads_db[download_id]["progress"] = f"Completed: {len(successful)} successful, {len(failed)} failed"
    else:
        downloads_db[download_id]["status"] = "failed"
        downloads_db[download_id]["progress"] = "All downloads failed"
        downloads_db[download_id]["error"] = "No files were successfully downloaded"

# ---------- Background Tasks ----------
async def cleanup_task():
    """Background task to clean up old files."""
    while True:
        cleanup_old_files()
        current_time = datetime.now()
        expired_downloads = [
            download_id for download_id, data in downloads_db.items()
            if current_time - data["created_at"] > timedelta(minutes=5)
        ]
        for download_id in expired_downloads:
            downloads_db.pop(download_id, None)
        
        await asyncio.sleep(3000) 

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task())

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    return {
        "message": "Browser-Based Media Downloader API", 
        "version": "2.0.0",
        "methods": ["browser", "subprocess"],
        "playwright_available": PLAYWRIGHT_AVAILABLE
    }


async def should_use_subprocess(url: str) -> bool:
    """Determine if we should use subprocess method based on URL."""
    complex_sites = [
        "youtube.com", "youtu.be", "twitch.tv", "vimeo.com", 
        "dailymotion.com", "facebook.com", "instagram.com"
    ]
    return any(site in url.lower() for site in complex_sites)

async def extract_media_info_with_subprocess(url: str) -> Dict[str, Any]:
    """Extract media info using yt-dlp subprocess - RELIABLE method."""
    try:
        cmd = [
            "yt-dlp",
            "--dump-json",
            "--no-download",
            "--quiet",
            "--no-warnings",
            "--no-playlist",
            "--ignore-errors",
            "--no-check-certificate",
            "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            # Add these flags to handle age-restricted and other issues
            "--age-limit", "0",  # Try to bypass age restrictions
            "--extractor-retries", "3",
            "--fragment-retries", "3",
            url
        ]
        
        process = await asyncio.create_subprocess_exec(
            *cmd,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            info = json.loads(stdout.decode())
            formats = []
            
            for f in info.get("formats", []):
                ext = f.get("ext", "").lower()
                if ext in SUPPORTED_FORMATS and "storyboard" not in (f.get("format_note") or "").lower():
                    formats.append({
                        "format_id": f.get("format_id"),
                        "ext": ext,
                        "format_note": f.get("format_note"),
                        "filesize": f.get("filesize") or f.get("filesize_approx") or 0,
                        "acodec": f.get("acodec"),
                        "vcodec": f.get("vcodec"),
                        "resolution": f.get("resolution"),
                        "abr": f.get("abr"),
                        "fps": f.get("fps"),
                        "tbr": f.get("tbr")
                    })
            
            return {
                "title": info.get("title", "Unknown"),
                "thumbnail": info.get("thumbnail", ""),
                "formats": formats
            }
        else:
            error_msg = stderr.decode()
            # Handle specific error cases
            if "Sign in to confirm your age" in error_msg:
                raise Exception("Age-restricted video: Cannot access without authentication")
            elif "Video unavailable" in error_msg:
                raise Exception("Video is unavailable or private")
            elif "This video is not available" in error_msg:
                raise Exception("Video not available in this region")
            else:
                raise Exception(f"yt-dlp extraction failed: {error_msg}")
    
    except json.JSONDecodeError:
        raise Exception("Failed to parse video information")
    except Exception as e:
        if "Age-restricted" in str(e):
            raise HTTPException(status_code=403, detail=str(e))
        else:
            raise HTTPException(status_code=500, detail=f"Failed to extract info: {str(e)}")


async def download_with_subprocess(url: str, format: str, quality: str, filepath: pathlib.Path) -> None:
    """Download using yt-dlp subprocess with improved error handling."""
    cmd = [
        "yt-dlp",
        "--output", str(filepath.parent / "%(title)s.%(ext)s"),
        "--quiet",
        "--no-warnings",
        "--ignore-errors",
        "--no-check-certificate",
        "--age-limit", "0",  # Try to bypass age restrictions
        "--extractor-retries", "3",
        "--fragment-retries", "3",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"
    ]
    
    if format in ["mp3", "wav", "m4a"]:
        cmd.extend([
            "--extract-audio",
            "--audio-format", format,
            "--audio-quality", quality.rstrip("K")
        ])
    else:
        cmd.extend([
            "--format", "bestvideo+bestaudio/best"
        ])
    
    cmd.append(url)
    
    process = await asyncio.create_subprocess_exec(
        *cmd,
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE
    )
    
    stdout, stderr = await process.communicate()
    
    if process.returncode != 0:
        error_msg = stderr.decode()
        if "Sign in to confirm your age" in error_msg:
            raise Exception("Age-restricted video: Cannot download without authentication")
        elif "Video unavailable" in error_msg:
            raise Exception("Video is unavailable or private")
        else:
            raise Exception(f"Download failed: {error_msg}")


# Updated info endpoint with better error handling
@app.post("/info")
async def get_info(request: InfoRequest):
    """Get media info for provided URLs."""
    logging.info(f"Received /info request: {request.urls} type={request.type}")
    results = []
    
    for url in request.urls:
        try:
            logging.info(f"Processing URL: {url}")
            
            # Use subprocess method for complex sites like YouTube
            if await should_use_subprocess(url):
                info = await extract_media_info_with_subprocess(url)
            else:
                # Try browser method for simpler sites, fallback to subprocess
                if PLAYWRIGHT_AVAILABLE:
                    try:
                        info = await extract_media_info_with_playwright(url)
                    except Exception as e:
                        print(f"Browser extraction failed: {e}, trying subprocess")
                        info = await extract_media_info_with_subprocess(url)
                else:
                    info = await extract_media_info_with_subprocess(url)
            
            # Filter formats based on type
            filtered_formats = []
            for f in info.get("formats", []):
                ext = f.get("ext", "").lower()
                acodec = (f.get("acodec") or "").lower()
                vcodec = (f.get("vcodec") or "").lower()
                
                if ext not in SUPPORTED_FORMATS:
                    continue
                
                if request.type == InfoType.audio and (not acodec or acodec == "none"):
                    continue
                if request.type == InfoType.video and (not vcodec or vcodec == "none"):
                    continue
                
                filtered_formats.append(f)
            
            results.append({
                "url": url,
                "title": info.get("title"),
                "thumbnail": info.get("thumbnail"),
                "formats": filtered_formats
            })
            
        except HTTPException as he:
            # Handle HTTP exceptions (like 403 for age-restricted)
            results.append({
                "url": url,
                "title": None,
                "thumbnail": None,
                "formats": [],
                "error": he.detail,
                "error_code": he.status_code
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
    
    return {"results": results}

# Updated info endpoint
@app.post("/info")
async def get_info(request: InfoRequest):
    """Get media info for provided URLs."""
    logging.info(f"Received /info request: {request.urls} type={request.type}")
    results = []
    
    for url in request.urls:
        try:
            logging.info(f"Processing URL: {url}")
            
            # Use subprocess method for complex sites like YouTube
            if await should_use_subprocess(url):
                info = await extract_media_info_with_subprocess(url)
            else:
                # Try browser method for simpler sites, fallback to subprocess
                if PLAYWRIGHT_AVAILABLE:
                    try:
                        info = await extract_media_info_with_playwright(url)
                    except Exception as e:
                        print(f"Browser extraction failed: {e}, trying subprocess")
                        info = await extract_media_info_with_subprocess(url)
                else:
                    info = await extract_media_info_with_subprocess(url)
            
            # Filter formats based on type
            filtered_formats = []
            for f in info.get("formats", []):
                ext = f.get("ext", "").lower()
                acodec = (f.get("acodec") or "").lower()
                vcodec = (f.get("vcodec") or "").lower()
                
                if ext not in SUPPORTED_FORMATS:
                    continue
                
                if request.type == InfoType.audio and (not acodec or acodec == "none"):
                    continue
                if request.type == InfoType.video and (not vcodec or vcodec == "none"):
                    continue
                
                filtered_formats.append(f)
            
            results.append({
                "url": url,
                "title": info.get("title"),
                "thumbnail": info.get("thumbnail"),
                "formats": filtered_formats
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
    
    return {"results": results}


@app.post("/download/batch", response_model=DownloadResponse)
async def batch_download(request: BatchDownloadRequest, background_tasks: BackgroundTasks, fastapi_request: Request):
    """Batch download multiple files."""
    try:
        # Pre-check all files for size
        urls_with_formats = [{"url": item.url, "format": item.format, "quality": item.quality or "192K"} for item in request.files]
        # If you have a precheck_file_sizes function, call it here
        # ok, error_message = await precheck_file_sizes(urls_with_formats)
        # if not ok:
        #     raise HTTPException(status_code=400, detail=error_message)

        download_id = generate_download_id()
        downloads_db[download_id] = {
            "status": "queued",
            "progress": "Queued for batch download",
            "files": [],
            "created_at": datetime.now(),
            "error": None
        }

        async def process_batch(download_id, files, fastapi_request):
            download_dir = PUBLIC_DIR / download_id
            download_dir.mkdir(exist_ok=True)
            downloads_db[download_id]["status"] = "downloading"
            downloads_db[download_id]["progress"] = "Starting batch downloads..."
            results = []
            for i, item in enumerate(files):
                downloads_db[download_id]["progress"] = f"Downloading {i+1}/{len(files)}: {item.url[:50]}..."
                file_info = await download_single_url(item.url, item.format, item.quality or "192K", download_dir, fastapi_request)
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

        background_tasks.add_task(process_batch, download_id, request.files, fastapi_request)
        return DownloadResponse(
            download_id=download_id,
            status="queued",
            message="Batch download request queued successfully",
            files=[]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create batch download: {str(e)}")

@app.post("/download", response_model=DownloadResponse)
async def create_download(request: DownloadRequest, background_tasks: BackgroundTasks, fastapi_request: Request):
    """Create a new download request."""
    try:
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

@app.get("/formats")
async def get_supported_formats():
    """Get list of supported formats."""
    return {"formats": SUPPORTED_FORMATS}

@app.delete("/cleanup")
async def manual_cleanup():
    """Manually trigger cleanup."""
    cleanup_old_files()
    return {"message": "Cleanup completed"}

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
        "main:app",
        host="0.0.0.0",
        port=port,
        reload=False
    )
