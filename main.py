#!/usr/bin/env python3
"""
FastAPI backend for downloading and converting media files from URLs.
Supports browser cookies for bypassing age restrictions.
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

# Cookie file path
COOKIES_FILE = pathlib.Path("youtube_cookies.txt")

# Global storage for download tracking
downloads_db = {}
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

def sanitize_filename(filename: str) -> str:
    """Sanitize filename by removing/replacing problematic characters."""
    import re
    # Remove or replace problematic characters
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)  # Remove invalid filename chars
    filename = re.sub(r'\s+', '_', filename)  # Replace spaces with underscores
    filename = re.sub(r'[^\w\-_.]', '', filename)  # Keep only alphanumeric, hyphens, underscores, dots
    filename = filename.strip('._')  # Remove leading/trailing dots and underscores
    return filename or "download"  # Fallback if empty

def get_base_url(request: Request) -> str:
    """Get the correct base URL for the environment."""
    # Allow manual override via environment variable
    if os.environ.get('FORCE_HTTPS') == 'true':
        host = request.headers.get('x-forwarded-host') or request.headers.get('host', 'localhost')
        return f"https://{host}/"
    
    # Check if we're in production (Railway, Heroku, etc.)
    # These platforms typically set specific environment variables
    is_production = any([
        os.environ.get('RAILWAY_ENVIRONMENT') == 'production',
        os.environ.get('HEROKU_APP_NAME'),
        os.environ.get('DYNO'),  # Heroku
        os.environ.get('PORT'),  # Common in cloud platforms
        request.headers.get('x-forwarded-proto') == 'https',
        request.headers.get('x-forwarded-port') == '443'
    ])
    
    # Check for HTTPS headers that proxies set
    forwarded_proto = request.headers.get('x-forwarded-proto')
    forwarded_host = request.headers.get('x-forwarded-host')
    forwarded_port = request.headers.get('x-forwarded-port')
    
    if forwarded_proto == 'https' or (is_production and not forwarded_proto):
        # Use HTTPS in production
        host = forwarded_host or request.headers.get('host', 'localhost')
        return f"https://{host}/"
    else:
        # Use the original base_url for development
        return str(request.base_url)

app = FastAPI(title="Cookie-Enabled Media Downloader API", version="2.1.0")
app.mount("/public", StaticFiles(directory="public"), name="public")
logging.basicConfig(level=logging.INFO)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
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
                try:
                    latest_mtime = max((f.stat().st_mtime for f in item.glob("*")), default=item.stat().st_mtime)
                    dir_age = current_time - latest_mtime
                    if dir_age > CLEANUP_INTERVAL_SECONDS:
                        shutil.rmtree(item, ignore_errors=True)
                        print(f"Cleaned up old directory: {item.name}")
                except:
                    pass
    except Exception as e:
        print(f"Cleanup error: {e}")

def get_base_yt_dlp_cmd(use_cookies=True):
    """Get base yt-dlp command with optional cookies."""
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--ignore-errors",
        "--user-agent", "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ]
    
    # Add cookies if file exists and use_cookies is True
    if use_cookies and COOKIES_FILE.exists():
        cmd.extend(["--cookies", str(COOKIES_FILE)])
        print("Using cookies for authentication")
    
    return cmd

# ---------- Cookie-enabled extraction ----------
async def extract_media_info_with_subprocess(url: str) -> Dict[str, Any]:
    """Extract media info using yt-dlp subprocess with cookie support."""
    try:
        # First attempt with cookies if available
        cmd = get_base_yt_dlp_cmd(use_cookies=True)
        cmd.extend(["--dump-json", "--no-download", url])
        
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
            
            # If cookies failed or not available, try without cookies
            if COOKIES_FILE.exists() and "Sign in to confirm your age" in error_msg:
                print("Cookies may be expired, trying without cookies...")
                cmd_no_cookies = get_base_yt_dlp_cmd(use_cookies=False)
                cmd_no_cookies.extend(["--dump-json", "--no-download", url])
                
                process = await asyncio.create_subprocess_exec(
                    *cmd_no_cookies,
                    stdout=asyncio.subprocess.PIPE,
                    stderr=asyncio.subprocess.PIPE
                )
                
                stdout, stderr = await process.communicate()
                
                if process.returncode == 0:
                    info = json.loads(stdout.decode())
                    # Process formats same as above...
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
            
            # Handle specific errors
            if "Sign in to confirm your age" in error_msg:
                return {
                    "title": "Age-Restricted Video",
                    "thumbnail": "",
                    "formats": [],
                    "error": "This video is age-restricted. Please add valid YouTube cookies to access it.",
                    "error_type": "age_restricted"
                }
            elif "Video unavailable" in error_msg:
                return {
                    "title": "Unavailable Video",
                    "thumbnail": "",
                    "formats": [],
                    "error": "This video is unavailable or private",
                    "error_type": "unavailable"
                }
            else:
                raise HTTPException(status_code=500, detail=f"Failed to extract video info: {error_msg}")
    
    except json.JSONDecodeError:
        raise HTTPException(status_code=500, detail="Failed to parse video information")
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Unexpected error: {str(e)}")

# ---------- Cookie-enabled download ----------
async def download_with_subprocess(url: str, format: str, quality: str, filepath: pathlib.Path) -> None:
    """Download using yt-dlp subprocess with cookie support."""
    cmd = get_base_yt_dlp_cmd(use_cookies=True)
    cmd.extend([
        "--output", str(filepath.parent / "%(title)s.%(ext)s"),
        "--no-check-certificate",
        "--extractor-retries", "3",
        "--fragment-retries", "3"
    ])
    
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
        
        # Try without cookies if the first attempt failed
        if COOKIES_FILE.exists() and any(err in error_msg for err in ["Sign in", "age", "authentication"]):
            print("Download with cookies failed, trying without cookies...")
            cmd_no_cookies = get_base_yt_dlp_cmd(use_cookies=False)
            cmd_no_cookies.extend([
                "--output", str(filepath.parent / "%(title)s.%(ext)s"),
                "--no-check-certificate",
                "--extractor-retries", "3",
                "--fragment-retries", "3"
            ])
            
            if format in ["mp3", "wav", "m4a"]:
                cmd_no_cookies.extend([
                    "--extract-audio",
                    "--audio-format", format,
                    "--audio-quality", quality.rstrip("K")
                ])
            else:
                cmd_no_cookies.extend([
                    "--format", "bestvideo+bestaudio/best"
                ])
            
            cmd_no_cookies.append(url)
            
            process = await asyncio.create_subprocess_exec(
                *cmd_no_cookies,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode()
        
        # Final error handling
        if process.returncode != 0:
            if "Sign in to confirm your age" in error_msg:
                raise Exception("Age-restricted video: Add valid YouTube cookies to download this video")
            elif "Video unavailable" in error_msg:
                raise Exception("Video is unavailable or private")
            else:
                raise Exception(f"Download failed: {error_msg}")

# ---------- Browser-based extraction (fallback) ----------
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
        
        await page.set_extra_http_headers({
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
        })
        
        media_urls = []
        title = "Unknown"
        thumbnail = ""
        
        def handle_request(request):
            if any(ext in request.url.lower() for ext in ['.mp4', '.m4a', '.webm', '.mp3']):
                media_urls.append({
                    "format_id": f"direct_{len(media_urls)}",
                    "ext": request.url.split('.')[-1].split('?')[0],
                    "format_note": "direct_link",
                    "url": request.url
                })
        
        page.on("request", handle_request)
        
        try:
            await page.goto(url, wait_until="networkidle", timeout=30000)
            
            try:
                title_elem = await page.query_selector("title")
                if title_elem:
                    title = await title_elem.text_content()
            except:
                pass
            
            try:
                thumb_elem = await page.query_selector('meta[property="og:image"]')
                if thumb_elem:
                    thumbnail = await thumb_elem.get_attribute("content")
            except:
                pass
            
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

async def should_use_subprocess(url: str) -> bool:
    """Determine if we should use subprocess method based on URL."""
    complex_sites = [
        "youtube.com", "youtu.be", "twitch.tv", "vimeo.com", 
        "dailymotion.com", "facebook.com", "instagram.com"
    ]
    return any(site in url.lower() for site in complex_sites)

async def download_single_url(url: str, format: str, quality: str,
                               download_dir: pathlib.Path,
                               request: Request) -> dict:
    """Download a single URL using available methods, with proper logging."""

    async with download_semaphore:
        try:
            # For complex sites, use subprocess method directly
            if await should_use_subprocess(url):
                temp_filename = f"download_{int(time.time())}.{format}"
                temp_filepath = download_dir / temp_filename

                await download_with_subprocess(url, format, quality, temp_filepath)

                # Log all files in the directory after download
                all_files = list(download_dir.glob("*"))
                print(f"[DEBUG] Files in {download_dir} after download:")
                for f in all_files:
                    print(f" - {f} ({f.stat().st_size} bytes)")

                # Try to pick the largest (final) file
                if all_files:
                    # Pick file that is not the temp placeholder
                    final_file = max(
                        [f for f in all_files if f.is_file() and f.name != temp_filename],
                        key=lambda x: x.stat().st_size,
                        default=None
                    )

                    if final_file:
                        # Sanitize the filename
                        original_name = final_file.stem
                        sanitized_name = sanitize_filename(original_name)
                        new_filename = f"{sanitized_name}.{final_file.suffix}"
                        new_filepath = final_file.parent / new_filename
                        
                        # Rename the file if the name is different
                        if final_file.name != new_filename:
                            try:
                                final_file.rename(new_filepath)
                                final_file = new_filepath
                                print(f"[DEBUG] Renamed file from {original_name} to {sanitized_name}")
                            except Exception as e:
                                print(f"[WARN] Could not rename file: {e}")
                                # Continue with original filename
                        
                        rel_path = final_file.relative_to(PUBLIC_DIR)  # relative inside /public
                        base_url = get_base_url(request)
                        download_url = f"{base_url}public/{rel_path}"
                        print(f"[DEBUG] Serving download URL: {download_url}")

                        return {
                            "url": url,
                            "title": sanitized_name,
                            "filename": final_file.name,
                            "size_mb": round(get_file_size_mb(final_file), 2),
                            "download_url": download_url,
                            "status": "success"
                        }

                raise Exception("Downloaded file not found in expected directory")

            # For simpler sites (browser method first)
            elif PLAYWRIGHT_AVAILABLE:
                try:
                    info = await extract_media_info_with_playwright(url)
                    title = info.get("title", "Unknown")

                    best_format = next(
                        (f for f in info.get("formats", []) if f.get("ext") == format),
                        None
                    )
                    if best_format and best_format.get("url"):
                        final_filename = f"{title}.{format}"
                        filepath = download_dir / final_filename
                        await download_with_http(best_format["url"], filepath)

                        print(f"[DEBUG] Saved file: {filepath} ({filepath.stat().st_size} bytes)")

                        rel_path = filepath.relative_to(PUBLIC_DIR)
                        base_url = get_base_url(request)
                        download_url = f"{base_url}public/{rel_path}"
                        print(f"[DEBUG] Serving download URL: {download_url}")

                        return {
                            "url": url,
                            "title": title,
                            "filename": final_filename,
                            "size_mb": round(get_file_size_mb(filepath), 2),
                            "download_url": download_url,
                            "status": "success"
                        }
                except Exception as e:
                    print(f"[WARN] Browser method failed for {url}: {e}")
                    await download_with_subprocess(url, format, quality, download_dir / f"fallback.{format}")

            raise Exception("All download methods failed")

        except Exception as e:
            print(f"[ERROR] Download failed: {e}")
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
    # Log cookie status
    if COOKIES_FILE.exists():
        print(f"✅ Found cookies file: {COOKIES_FILE}")
        print("Age-restricted videos should be accessible")
    else:
        print(f"⚠️  No cookies file found at: {COOKIES_FILE}")
        print("Age-restricted videos will not be accessible")

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    cookies_available = COOKIES_FILE.exists()
    return {
        "message": "Cookie-Enabled Media Downloader API", 
        "version": "2.1.0",
        "cookies_available": cookies_available,
        "cookies_status": "Active" if cookies_available else "Not available - age-restricted content will fail",
        "playwright_available": PLAYWRIGHT_AVAILABLE
    }

@app.get("/cookies/status")
async def get_cookies_status():
    """Check if cookies are available and when they were last modified."""
    if COOKIES_FILE.exists():
        stat = COOKIES_FILE.stat()
        return {
            "available": True,
            "file_size": stat.st_size,
            "last_modified": datetime.fromtimestamp(stat.st_mtime).isoformat(),
            "message": "Cookies are available for age-restricted content"
        }
    else:
        return {
            "available": False,
            "message": "No cookies file found. Age-restricted content will not be accessible.",
            "instructions": "Export cookies from your browser to 'youtube_cookies.txt' file"
        }

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
                
                # Handle graceful error responses
                if "error_type" in info:
                    results.append({
                        "url": url,
                        "title": info.get("title"),
                        "thumbnail": info.get("thumbnail"),
                        "formats": [],
                        "error": info.get("error"),
                        "error_type": info.get("error_type")
                    })
                    continue
                    
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

@app.post("/download/batch", response_model=DownloadResponse)
async def batch_download(request: BatchDownloadRequest, background_tasks: BackgroundTasks, fastapi_request: Request):
    """Batch download multiple files."""
    try:
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
