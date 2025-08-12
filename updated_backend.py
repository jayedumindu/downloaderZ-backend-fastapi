#!/usr/bin/env python3
"""
FastAPI backend for downloading and converting media files from URLs.
Supports browser cookies for bypassing age restrictions.
Enhanced with proxy rotation and anti-detection measures for server environments.
"""

import asyncio
import os
import pathlib
import subprocess
import time
import uuid
import json
import httpx
import random
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

# Proxy configuration
PROXY_LIST_FILE = pathlib.Path("proxies.txt")
current_proxy_index = 0
proxy_list = []

# User agents for rotation
USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.2 Safari/605.1.15",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

# Global storage for download tracking
downloads_db = {}
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

app = FastAPI(title="Enhanced Cookie-Enabled Media Downloader API", version="2.2.0")
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

class ProxyConfig(BaseModel):
    proxies: List[str]
    rotation_enabled: bool = True

# ---------- Proxy Management ----------
def load_proxies():
    """Load proxies from file if it exists."""
    global proxy_list
    if PROXY_LIST_FILE.exists():
        try:
            with open(PROXY_LIST_FILE, 'r') as f:
                proxy_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"✅ Loaded {len(proxy_list)} proxies from {PROXY_LIST_FILE}")
        except Exception as e:
            print(f"⚠️ Error loading proxies: {e}")
            proxy_list = []
    else:
        print(f"⚠️ No proxy file found at {PROXY_LIST_FILE}")
        proxy_list = []

def get_next_proxy():
    """Get the next proxy in rotation."""
    global current_proxy_index
    if not proxy_list:
        return None
    
    proxy = proxy_list[current_proxy_index]
    current_proxy_index = (current_proxy_index + 1) % len(proxy_list)
    return proxy

def get_random_user_agent():
    """Get a random user agent."""
    return random.choice(USER_AGENTS)

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

def get_base_yt_dlp_cmd(use_cookies=True, use_proxy=True):
    """Get base yt-dlp command with optional cookies and proxy."""
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--ignore-errors",
        "--user-agent", get_random_user_agent(),
        "--sleep-interval", str(random.randint(1, 3)),  # Random sleep 1-3 seconds
        "--throttled-rate", "500K",  # Limit download speed to appear less suspicious
        "--extractor-retries", "3",
        "--fragment-retries", "3",
        "--no-check-certificate"
    ]
    
    # Add cookies if file exists and use_cookies is True
    if use_cookies and COOKIES_FILE.exists():
        cmd.extend(["--cookies", str(COOKIES_FILE)])
        print("Using cookies for authentication")
    
    # Add proxy if available and use_proxy is True
    if use_proxy:
        proxy = get_next_proxy()
        if proxy:
            cmd.extend(["--proxy", proxy])
            print(f"Using proxy: {proxy}")
    
    return cmd

# ---------- Cookie-enabled extraction ----------
async def extract_media_info_with_subprocess(url: str) -> Dict[str, Any]:
    """Extract media info using yt-dlp subprocess with enhanced anti-detection."""
    try:
        # First attempt with cookies and proxy if available
        cmd = get_base_yt_dlp_cmd(use_cookies=True, use_proxy=True)
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
            
            # Try different combinations if first attempt failed
            retry_configs = [
                {"use_cookies": False, "use_proxy": True},   # No cookies, with proxy
                {"use_cookies": True, "use_proxy": False},   # With cookies, no proxy
                {"use_cookies": False, "use_proxy": False}   # No cookies, no proxy
            ]
            
            for config in retry_configs:
                print(f"Retrying with config: {config}")
                cmd_retry = get_base_yt_dlp_cmd(**config)
                cmd_retry.extend(["--dump-json", "--no-download", url])
                
                process = await asyncio.create_subprocess_exec(
                    *cmd_retry,
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

# ---------- Enhanced download function ----------
async def download_with_subprocess(url: str, format: str, quality: str, filepath: pathlib.Path) -> None:
    """Download using yt-dlp subprocess with enhanced anti-detection."""
    cmd = get_base_yt_dlp_cmd(use_cookies=True, use_proxy=True)
    cmd.extend([
        "--output", str(filepath.parent / "%(title)s.%(ext)s"),
        "--min-sleep-interval", "1",
        "--max-sleep-interval", "5"  # Random sleep between 1-5 seconds
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
        
        # Try different configurations if first attempt failed
        retry_configs = [
            {"use_cookies": False, "use_proxy": True},   # No cookies, with proxy
            {"use_cookies": True, "use_proxy": False},   # With cookies, no proxy
            {"use_cookies": False, "use_proxy": False}   # No cookies, no proxy
        ]
        
        for config in retry_configs:
            print(f"Download retry with config: {config}")
            cmd_retry = get_base_yt_dlp_cmd(**config)
            cmd_retry.extend([
                "--output", str(filepath.parent / "%(title)s.%(ext)s"),
                "--min-sleep-interval", "1",
                "--max-sleep-interval", "5"
            ])
            
            if format in ["mp3", "wav", "m4a"]:
                cmd_retry.extend([
                    "--extract-audio",
                    "--audio-format", format,
                    "--audio-quality", quality.rstrip("K")
                ])
            else:
                cmd_retry.extend([
                    "--format", "bestvideo+bestaudio/best"
                ])
            
            cmd_retry.append(url)
            
            process = await asyncio.create_subprocess_exec(
                *cmd_retry,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode == 0:
                return  # Success
            else:
                error_msg = stderr.decode()
        
        # Final error handling
        if "Sign in to confirm your age" in error_msg:
            raise Exception("Age-restricted video: Add valid YouTube cookies to download this video")
        elif "Video unavailable" in error_msg:
            raise Exception("Video is unavailable or private")
        else:
            raise Exception(f"Download failed: {error_msg}")

# ---------- Browser-based extraction (fallback) ----------
async def extract_media_info_with_playwright(url: str) -> Dict[str, Any]:
    """Extract media info using headless browser with enhanced anti-detection."""
    if not PLAYWRIGHT_AVAILABLE:
        raise HTTPException(status_code=500, detail="Playwright not available")
    
    async with async_playwright() as p:
        # Use a random proxy if available
        proxy = get_next_proxy() if proxy_list else None
        proxy_config = None
        if proxy:
            if proxy.startswith('http://'):
                proxy_config = {"server": proxy}
            elif proxy.startswith('socks5://'):
                proxy_config = {"server": proxy}
        
        browser = await p.chromium.launch(
            headless=True, 
            args=["--no-sandbox", "--disable-setuid-sandbox", "--disable-dev-shm-usage"],
            proxy=proxy_config
        )
        page = await browser.new_page()
        
        await page.set_extra_http_headers({
            "User-Agent": get_random_user_agent()
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
            
            await asyncio.sleep(random.randint(2, 5))  # Random delay
            
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
    """Download file using HTTP client with proxy support."""
    proxy = get_next_proxy() if proxy_list else None
    proxies = {"http://": proxy, "https://": proxy} if proxy else None
    
    async with httpx.AsyncClient(
        follow_redirects=True, 
        timeout=300,
        proxies=proxies,
        headers={"User-Agent": get_random_user_agent()}
    ) as client:
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

async def download_single_url(url: str, format: str, quality: str, download_dir: pathlib.Path, request: Request) -> dict:
    """Download a single URL using available methods with enhanced anti-detection."""
    async with download_semaphore:
        try:
            # Add random delay before starting download
            await asyncio.sleep(random.uniform(0.5, 2.0))
            
            # For complex sites, use subprocess method directly
            if await should_use_subprocess(url):
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
            
            # For simpler sites, try browser method first
            elif PLAYWRIGHT_AVAILABLE:
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
                    # Fallback to subprocess
                    await download_with_subprocess(url, format, quality, download_dir / f"fallback.{format}")
            
            raise Exception("All download methods failed")
            
        except Exception as e:
            print(f"ERROR: {e}")
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
    """Process all downloads for a request with enhanced anti-detection."""
    download_dir = PUBLIC_DIR / download_id
    download_dir.mkdir(exist_ok=True)
    
    downloads_db[download_id]["status"] = "downloading"
    downloads_db[download_id]["progress"] = "Starting downloads..."
    
    files = []
    for i, url in enumerate(urls):
        downloads_db[download_id]["progress"] = f"Downloading {i+1}/{len(urls)}: {url[:50]}..."
        
        # Add random delay between downloads
        if i > 0:
            delay = random.uniform(2, 8)
            await asyncio.sleep(delay)
        
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
    
    # Load proxies
    load_proxies()
    
    # Log status
    if COOKIES_FILE.exists():
        print(f"✅ Found cookies file: {COOKIES_FILE}")
        print("Age-restricted videos should be accessible")
    else:
        print(f"⚠️  No cookies file found at: {COOKIES_FILE}")
        print("Age-restricted videos will not be accessible")
    
    if proxy_list:
        print(f"✅ Loaded {len(proxy_list)} proxies for rotation")
    else:
        print("⚠️  No proxies loaded. Running without proxy rotation.")

# ---------- API Endpoints ----------
@app.get("/")
async def root():
    cookies_available = COOKIES_FILE.exists()
    proxies_available = len(proxy_list) > 0
    return {
        "message": "Enhanced Cookie-Enabled Media Downloader API", 
        "version": "2.2.0",
        "cookies_available": cookies_available,
        "cookies_status": "Active" if cookies_available else "Not available - age-restricted content will fail",
        "proxies_available": proxies_available,
        "proxy_count": len(proxy_list),
        "playwright_available": PLAYWRIGHT_AVAILABLE,
        "enhancements": [
            "Rotating proxy support",
            "Random sleep intervals",
            "Throttled download rates",
            "User-agent rotation",
            "Enhanced retry mechanisms"
        ]
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

@app.get("/proxies/status")
async def get_proxies_status():
    """Check proxy configuration status."""
    return {
        "available": len(proxy_list) > 0,
        "count": len(proxy_list),
        "current_index": current_proxy_index,
        "message": f"{'Proxy rotation active' if proxy_list else 'No proxies configured'}",
        "instructions": "Add proxies to 'proxies.txt' file, one per line (format: http://user:pass@host:port or socks5://user:pass@host:port)"
    }

@app.post("/proxies/configure")
async def configure_proxies(config: ProxyConfig):
    """Configure proxy list dynamically."""
    global proxy_list, current_proxy_index
    proxy_list = config.proxies
    current_proxy_index = 0
    
    # Optionally save to file
    if config.proxies:
        with open(PROXY_LIST_FILE, 'w') as f:
            for proxy in config.proxies:
                f.write(f"{proxy}\n")
    
    return {
        "message": f"Configured {len(proxy_list)} proxies",
        "proxies_count": len(proxy_list),
        "rotation_enabled": config.rotation_enabled
    }

@app.post("/info")
async def get_info(request: InfoRequest):
    """Get media info for provided URLs with enhanced anti-detection."""
    logging.info(f"Received /info request: {request.urls} type={request.type}")
    results = []
    
    for i, url in enumerate(request.urls):
        try:
            logging.info(f"Processing URL: {url}")
            
            # Add random delay between requests
            if i > 0:
                delay = random.uniform(1, 4)
                await asyncio.sleep(delay)
            
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
    """Create a new download request with enhanced anti-detection."""
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
    """Batch download multiple files with enhanced anti-detection."""
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
                
                # Add random delay between downloads
                if i > 0:
                    delay = random.uniform(3, 10)
                    await asyncio.sleep(delay)
                
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

