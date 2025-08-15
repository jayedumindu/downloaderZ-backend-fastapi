#!/usr/bin/env python3
"""
FastAPI backend for downloading Instagram media files from URLs.
Supports browser cookies for private/login-required content.
Supports proxy rotation and anti-detection behavior.
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
from fastapi.responses import JSONResponse
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
MAX_FILE_SIZE_MB = 100
MAX_CONCURRENT_DOWNLOADS = 3
CLEANUP_INTERVAL_SECONDS = 300
SUPPORTED_FORMATS = ["mp4", "mp3"]
MAX_URLS_PER_REQUEST = 5

PUBLIC_DIR = pathlib.Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)

# Cookie file path (Instagram)
COOKIES_FILE = pathlib.Path("instagram_cookies.txt")

# Proxy configuration
PROXY_LIST_FILE = pathlib.Path("proxies.txt")
current_proxy_index = 0
proxy_list = []

USER_AGENTS = [
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/119.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:109.0) Gecko/20100101 Firefox/121.0",
    "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
]

downloads_db = {}
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

app = FastAPI(title="Instagram Media Downloader API", version="1.0.0")
app.mount("/public", StaticFiles(directory="public"), name="public")

logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------- Models ----------
class DownloadRequest(BaseModel):
    urls: List[str]
    format: str = "mp4"
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

# ---------- Proxy Utils ----------
def load_proxies():
    global proxy_list
    if PROXY_LIST_FILE.exists():
        try:
            with open(PROXY_LIST_FILE, 'r') as f:
                proxy_list = [line.strip() for line in f if line.strip() and not line.startswith('#')]
            print(f"✅ Loaded {len(proxy_list)} proxies")
        except Exception as e:
            print(f"⚠️ Error loading proxies: {e}")
    else:
        proxy_list = []

def get_next_proxy():
    global current_proxy_index
    if not proxy_list:
        return None
    proxy = proxy_list[current_proxy_index]
    current_proxy_index = (current_proxy_index + 1) % len(proxy_list)
    return proxy

def get_random_user_agent():
    return random.choice(USER_AGENTS)

# ---------- Utility ----------
def get_file_size_mb(file_path: pathlib.Path) -> float:
    return file_path.stat().st_size / (1024 * 1024)

def generate_download_id() -> str:
    return str(uuid.uuid4())

def cleanup_old_files():
    try:
        current_time = time.time()
        for item in PUBLIC_DIR.iterdir():
            if current_time - item.stat().st_mtime > CLEANUP_INTERVAL_SECONDS:
                if item.is_file():
                    item.unlink()
                elif item.is_dir():
                    shutil.rmtree(item, ignore_errors=True)
    except Exception as e:
        print(f"Cleanup error: {e}")

# ---------- yt-dlp command for Instagram ----------
def get_base_yt_dlp_cmd(use_cookies=True, use_proxy=False):
    """Build yt-dlp command with comprehensive anti-detection for Instagram."""
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--ignore-errors",
        # Essential anti-detection headers
        "--user-agent", get_random_user_agent(),
        "--add-header", "Accept:text/html,application/xhtml+xml,application/xml;q=0.9,image/webp,*/*;q=0.8",
        "--add-header", "Accept-Language:en-US,en;q=0.5",
        "--add-header", "Accept-Encoding:gzip, deflate",
        "--add-header", "Connection:keep-alive",
        "--add-header", "Upgrade-Insecure-Requests:1",
        # Rate limiting (crucial for Instagram)
        "--sleep-interval", str(random.randint(3, 8)),
        "--max-sleep-interval", str(random.randint(10, 20)),
        "--throttled-rate", "200K",  # Lower rate for Instagram
        # Enhanced retry logic
        "--extractor-retries", "5",
        "--fragment-retries", "5",
        "--file-access-retries", "3",
        "--socket-timeout", "30",
        # Network optimizations
        "--no-check-certificate",
        "--prefer-insecure",
        "--concurrent-fragments", "1",  # Single thread for Instagram
        # Format selection
        "-f", "best[height<=1080]/best"
    ]
    
    # Cookies are CRITICAL for Instagram
    if use_cookies and COOKIES_FILE.exists():
        cmd.extend(["--cookies", str(COOKIES_FILE)])
        print("[INFO] ✅ Using Instagram cookies")
        
        # Verify cookies file has content
        try:
            with open(COOKIES_FILE, 'r') as f:
                content = f.read().strip()
                if not content or len(content) < 100:
                    print("[WARNING] ⚠️ Cookies file seems empty or too small")
                else:
                    print(f"[INFO] Cookies file size: {len(content)} chars")
        except Exception as e:
            print(f"[ERROR] Cannot read cookies file: {e}")
    else:
        print("[WARNING] ⚠️ No Instagram cookies - WILL LIKELY FAIL in production")
    
    # Proxy rotation for production
    if use_proxy and proxy_list:
        proxy = get_next_proxy()
        if proxy:
            cmd.extend(["--proxy", proxy])
            print(f"[INFO] Using proxy: {proxy}")
    elif use_proxy:
        print("[WARNING] ⚠️ No proxies available")
    
    return cmd

async def extract_media_info_with_subprocess(url: str) -> Dict[str, Any]:
    """Extract Instagram media info using anti-detection measures."""
    try:
        # USE YOUR SOPHISTICATED COMMAND BUILDER (this was missing!)
        cmd = get_base_yt_dlp_cmd(use_cookies=True, use_proxy=True)
        cmd.extend([
            "--dump-json",
            "--no-download",
            "--no-playlist",
            # Instagram-specific optimizations
            "--sleep-requests", str(random.randint(1, 3)),
            "--sleep-subtitles", str(random.randint(1, 2)),
            url
        ])
        
        print(f"[DEBUG] Using command: {' '.join(cmd)}")
        
        process = await asyncio.create_subprocess_exec(
            *cmd, 
            stdout=asyncio.subprocess.PIPE, 
            stderr=asyncio.subprocess.PIPE
        )
        
        stdout, stderr = await process.communicate()
        
        if process.returncode == 0:
            info = json.loads(stdout.decode())
            print(f"[SUCCESS] Extracted: {info.get('title', 'Unknown')}")

            # Your existing format processing code...
            all_formats = []
            if "formats" in info:
                all_formats.extend(info["formats"])
            if "requested_formats" in info and isinstance(info["requested_formats"], list):
                all_formats.extend(info["requested_formats"])

            seen_resolutions = set()
            unique_formats = []
            for f in all_formats:
                url_val = f.get("url")
                res_val = f.get("resolution")
                ext_val = f.get("ext")
                key = (res_val, ext_val)
                if (
                    ext_val in SUPPORTED_FORMATS
                    and url_val
                    and key not in seen_resolutions
                ):
                    seen_resolutions.add(key)
                    unique_formats.append({
                        k: f.get(k)
                        for k in [
                            'url', 'format_id', 'ext', 'format_note', 'filesize', 'acodec', 'vcodec',
                            'resolution', 'abr', 'fps', 'tbr'
                        ]
                    })

            return {
                "title": info.get("title", "Unknown"),
                "thumbnail": info.get("thumbnail", ""),
                "formats": unique_formats
            }
        else:
            error_msg = stderr.decode()
            print(f"[ERROR] yt-dlp failed: {error_msg}")
            
            # Enhanced error handling for production
            error_lower = error_msg.lower()
            if any(phrase in error_lower for phrase in ["login required", "sign in", "authentication", "private account"]):
                return {
                    "title": "Login Required",
                    "formats": [],
                    "error": "Instagram login required. Cookies may be expired or invalid.",
                    "error_type": "login_required"
                }
            elif "429" in error_msg or "rate limit" in error_lower or "too many requests" in error_lower:
                return {
                    "title": "Rate Limited",
                    "formats": [],
                    "error": "Instagram rate limiting. Try again later or use different proxy.",
                    "error_type": "rate_limited"
                }
            else:
                raise HTTPException(status_code=500, detail=f"yt-dlp error: {error_msg}")
                
    except json.JSONDecodeError as e:
        print(f"[ERROR] JSON parsing failed: {e}")
        raise HTTPException(status_code=500, detail="Failed to parse media information")
    except Exception as e:
        print(f"[ERROR] Exception: {e}")
        raise HTTPException(status_code=500, detail=str(e))


# ---------- Download ----------
async def download_with_subprocess(url: str, fmt: str, quality: str, filepath: pathlib.Path):
    cmd = get_base_yt_dlp_cmd()
    cmd += ["--output", str(filepath.parent / "%(title)s.%(ext)s")]
    if fmt == "mp3":
        cmd += ["--extract-audio", "--audio-format", "mp3", "--audio-quality", quality.rstrip("K")]
    cmd.append(url)
    proc = await asyncio.create_subprocess_exec(
        *cmd, stdout=asyncio.subprocess.PIPE, stderr=asyncio.subprocess.PIPE
    )
    _, stderr = await proc.communicate()
    if proc.returncode != 0:
        raise Exception(stderr.decode())

async def download_single_url(url: str, fmt: str, quality: str, download_dir: pathlib.Path, request: Request) -> dict:
    async with download_semaphore:
        try:
            filename = f"{int(time.time())}.{fmt}"
            filepath = download_dir / filename
            await download_with_subprocess(url, fmt, quality, filepath)
            downloaded_files = list(download_dir.glob("*"))
            if not downloaded_files:
                raise Exception("No file downloaded")
            actual = max(downloaded_files, key=lambda f: f.stat().st_mtime)
            return {
                "url": url,
                "title": actual.stem,
                "filename": actual.name,
                "size_mb": round(get_file_size_mb(actual), 2),
                "download_url": f"{request.base_url}public/{download_dir.name}/{actual.name}",
                "status": "success"
            }
        except Exception as e:
            return {"url": url, "status": "error", "error": str(e)}

async def process_downloads(download_id: str, urls: List[str], fmt: str, quality: str, request: Request):
    download_dir = PUBLIC_DIR / download_id
    download_dir.mkdir(exist_ok=True)
    files = []
    for url in urls:
        files.append(await download_single_url(url, fmt, quality, download_dir, request))
    downloads_db[download_id]["files"] = files
    if any(f["status"] == "success" for f in files):
        downloads_db[download_id]["status"] = "completed"
    else:
        downloads_db[download_id]["status"] = "failed"

# ---------- Background ----------
async def cleanup_task():
    while True:
        cleanup_old_files()
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    load_proxies()
    asyncio.create_task(cleanup_task())
    if COOKIES_FILE.exists():
        print(f"✅ Cookies loaded: {COOKIES_FILE}")
    else:
        print("⚠️ No Instagram cookies found.")

# ---------- API ----------
@app.get("/")
async def root():
    return {
        "message": "Instagram Downloader API",
        "cookies_available": COOKIES_FILE.exists(),
        "proxies_available": len(proxy_list) > 0,
        "playwright_available": PLAYWRIGHT_AVAILABLE
    }

@app.post("/info")
async def get_info(req: InfoRequest):
    results = []
    for url in req.urls:
        info = await extract_media_info_with_subprocess(url)
        results.append(info)
    return {"results": results}

@app.post("/download", response_model=DownloadResponse)
async def create_download(request: DownloadRequest, bg: BackgroundTasks, req: Request):
    download_id = generate_download_id()
    downloads_db[download_id] = {"status": "queued", "files": [], "created_at": datetime.now()}
    bg.add_task(process_downloads, download_id, request.urls, request.format, request.quality, req)
    return DownloadResponse(download_id=download_id, status="queued", message="Queued", files=[])

@app.get("/status/{download_id}", response_model=DownloadStatus)
async def get_status(download_id: str):
    if download_id not in downloads_db:
        raise HTTPException(404)
    d = downloads_db[download_id]
    return DownloadStatus(download_id=download_id, status=d["status"], progress="", files=d["files"])

@app.get("/formats")
async def formats():
    return {"formats": SUPPORTED_FORMATS}

if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=int(os.getenv("PORT",8000)))


