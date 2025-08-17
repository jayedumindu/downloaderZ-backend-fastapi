#!/usr/bin/env python3
"""
FastAPI backend for downloading and converting media files from URLs.
Uses yt-dlp with a list of HTTP proxies from working_proxies.txt,
tries each proxy in order until one succeeds.
"""

import asyncio
import os
import pathlib
import time
import uuid
import json
from datetime import datetime
from typing import List, Optional, Dict, Any
import logging

from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, validator
import uvicorn
import shutil

PROXIES_FILE = "working_proxies.txt"

MAX_FILE_SIZE_MB = 50
MAX_CONCURRENT_DOWNLOADS = 3
CLEANUP_INTERVAL_SECONDS = 300
SUPPORTED_FORMATS = ["mp3", "mp4", "wav", "m4a"]
MAX_URLS_PER_REQUEST = 5

PUBLIC_DIR = pathlib.Path("public")
PUBLIC_DIR.mkdir(exist_ok=True)

downloads_db = {}
download_semaphore = asyncio.Semaphore(MAX_CONCURRENT_DOWNLOADS)

def load_proxies(filepath):
    with open(filepath, "r") as f:
        return [line.strip() for line in f if line.strip()]

def sanitize_filename(filename: str) -> str:
    import re
    filename = re.sub(r'[<>:"/\\|?*]', '', filename)
    filename = re.sub(r'\s+', '_', filename)
    filename = re.sub(r'[^\w\-\_.]', '', filename)
    filename = filename.strip('._')
    return filename or "download"

def get_base_url(request: Request) -> str:
    if os.environ.get('FORCE_HTTPS') == 'true':
        host = request.headers.get('x-forwarded-host') or request.headers.get('host', 'localhost')
        return f"https://{host}/"
    is_production = any([
        os.environ.get('RAILWAY_ENVIRONMENT') == 'production',
        os.environ.get('HEROKU_APP_NAME'),
        os.environ.get('DYNO'),
        os.environ.get('PORT'),
        request.headers.get('x-forwarded-proto') == 'https',
        request.headers.get('x-forwarded-port') == '443'
    ])
    forwarded_proto = request.headers.get('x-forwarded-proto')
    forwarded_host = request.headers.get('x-forwarded-host')
    if forwarded_proto == 'https' or (is_production and not forwarded_proto):
        host = forwarded_host or request.headers.get('host', 'localhost')
        return f"https://{host}/"
    else:
        return str(request.base_url)

app = FastAPI(title="yt-dlp Multi Proxy Media Downloader API", version="4.1.0")
app.mount("/public", StaticFiles(directory="public"), name="public")
logging.basicConfig(level=logging.INFO)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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

class BatchDownloadItem(BaseModel):
    url: str
    format: str
    quality: Optional[str] = "192K"

class BatchDownloadRequest(BaseModel):
    files: List[BatchDownloadItem]

def get_base_yt_dlp_cmd(proxy_url):
    cmd = [
        "yt-dlp",
        "--quiet",
        "--no-warnings",
        "--no-playlist",
        "--ignore-errors",
        f"--proxy=http://{proxy_url}",
        "--user-agent", "Mozilla/5.0"
    ]
    if pathlib.Path("youtube_cookies.txt").exists():
        cmd.extend(["--cookies", "youtube_cookies.txt"])
    logging.info(f"Using yt-dlp base command with proxy: http://{proxy_url}")
    return cmd

async def extract_media_info_with_proxies(url: str) -> Dict[str, Any]:
    proxies = load_proxies(PROXIES_FILE)
    last_err = None
    for proxy in proxies:
        try:
            cmd = get_base_yt_dlp_cmd(proxy)
            cmd.extend(["--dump-json", "--no-download", url])
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
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
                            "filesize": f.get("filesize") or f.get("filesize_approx") or 0,
                            "acodec": f.get("acodec"),
                            "vcodec": f.get("vcodec"),
                            "resolution": f.get("resolution"),
                            "abr": f.get("abr"),
                            "fps": f.get("fps"),
                            "tbr": f.get("tbr"),
                        })
                return {
                    "title": info.get("title", "Unknown"),
                    "thumbnail": info.get("thumbnail", ""),
                    "formats": formats,
                    "proxy_used": proxy
                }
            else:
                last_err = stderr.decode()
                logging.warning(f"Proxy {proxy} failed: {last_err}")
        except Exception as err:
            last_err = str(err)
            logging.warning(f"Proxy {proxy} error: {last_err}")
    raise HTTPException(status_code=500, detail=f"Failed to extract info with all proxies. Last error: {last_err}")

async def download_with_proxies(url: str, format: str, quality: str, filepath: pathlib.Path) -> None:
    proxies = load_proxies(PROXIES_FILE)
    last_err = None
    for proxy in proxies:
        try:
            cmd = get_base_yt_dlp_cmd(proxy)
            cmd.extend([
                "--output", str(filepath.parent / "%(title)s.%(ext)s"),
                "--no-check-certificate",
                "--extractor-retries", "3",
                "--fragment-retries", "3",
            ])
            if format in ["mp3", "wav", "m4a"]:
                cmd.extend([
                    "--extract-audio",
                    "--audio-format", format,
                    "--audio-quality", quality.rstrip("K"),
                ])
            else:
                cmd.extend([
                    "--format", "bestvideo+bestaudio/best",
                ])
            cmd.append(url)
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )
            stdout, stderr = await process.communicate()
            if process.returncode == 0:
                return
            else:
                last_err = stderr.decode()
                logging.warning(f"Proxy {proxy} failed: {last_err}")
        except Exception as err:
            last_err = str(err)
            logging.warning(f"Proxy {proxy} error: {last_err}")
    raise Exception(f"Download failed with all proxies. Last error: {last_err}")

async def download_single_url(url: str, format: str, quality: str, download_dir: pathlib.Path, request: Request) -> dict:
    async with download_semaphore:
        try:
            temp_filename = f"download_{int(time.time())}.{format}"
            temp_filepath = download_dir / temp_filename
            await download_with_proxies(url, format, quality, temp_filepath)
            all_files = list(download_dir.glob("*"))
            final_file = max(
                [f for f in all_files if f.is_file() and f != temp_filepath],
                key=lambda x: x.stat().st_size,
                default=None,
            )
            if final_file:
                sanitized_name = sanitize_filename(final_file.stem)
                new_path = final_file.parent / f"{sanitized_name}{final_file.suffix}"
                if final_file != new_path:
                    final_file.rename(new_path)
                    final_file = new_path
                base_url = get_base_url(request)
                download_url = f"{base_url}public/{download_dir.name}/{final_file.name}"
                return {
                    "url": url,
                    "title": sanitized_name,
                    "filename": final_file.name,
                    "size_mb": round(final_file.stat().st_size / (1024*1024), 2),
                    "download_url": download_url,
                    "status": "success",
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
                "error": str(e),
            }

async def process_downloads(download_id: str, urls: List[str], format: str, quality: str, request: Request):
    download_dir = PUBLIC_DIR / download_id
    download_dir.mkdir(exist_ok=True)
    downloads_db[download_id]["status"] = "downloading"
    files = []
    for i, url in enumerate(urls):
        downloads_db[download_id]["progress"] = f"Downloading {i+1}/{len(urls)}: {url[:50]}"
        file_info = await download_single_url(url, format, quality, download_dir, request)
        files.append(file_info)
        downloads_db[download_id]["files"] = files
    successful = [f for f in files if f["status"] == "success"]
    if successful:
        downloads_db[download_id]["status"] = "completed"
        downloads_db[download_id]["progress"] = f"Completed: {len(successful)} succeeded, {len(files) - len(successful)} failed"
    else:
        downloads_db[download_id]["status"] = "failed"
        downloads_db[download_id]["progress"] = "All downloads failed"
        downloads_db[download_id]["error"] = "No files were successfully downloaded"

async def cleanup_task():
    while True:
        now = time.time()
        for item in PUBLIC_DIR.iterdir():
            if now - item.stat().st_mtime > CLEANUP_INTERVAL_SECONDS:
                if item.is_file():
                    item.unlink()
                else:
                    shutil.rmtree(item, ignore_errors=True)
        await asyncio.sleep(300)

@app.on_event("startup")
async def startup_event():
    asyncio.create_task(cleanup_task())
    print(f"yt-dlp Multi Proxy Downloader started. Loaded proxies from {PROXIES_FILE}")

@app.get("/")
async def root():
    return {
        "message": "yt-dlp Multi Proxy Media Downloader API",
        "version": "4.1.0",
        "proxies_file": PROXIES_FILE,
        "n_proxies": len(load_proxies(PROXIES_FILE))
    }

@app.post("/info")
async def get_info(request: DownloadRequest):
    results = []
    for url in request.urls:
        try:
            info = await extract_media_info_with_proxies(url)
            results.append({"url": url, **info})
        except Exception as e:
            results.append({"url": url, "error": str(e)})
    return {"results": results}

@app.post("/download", response_model=DownloadResponse)
async def create_download(request: DownloadRequest, background_tasks: BackgroundTasks, fastapi_request: Request):
    download_id = str(uuid.uuid4())
    downloads_db[download_id] = {
        "status": "queued",
        "files": [],
        "created_at": datetime.now(),
        "progress": "",
        "error": None,
    }
    background_tasks.add_task(process_downloads, download_id, request.urls, request.format, request.quality or "192K", fastapi_request)
    return DownloadResponse(download_id=download_id, status="queued", message="Download queued", files=[])

@app.get("/status/{download_id}", response_model=DownloadStatus)
async def get_status(download_id: str):
    if download_id not in downloads_db:
        raise HTTPException(status_code=404, detail="Download not found")
    data = downloads_db[download_id]
    return DownloadStatus(download_id=download_id, status=data["status"], progress=data.get("progress", ""), files=data.get("files", []), error=data.get("error"))

@app.delete("/cleanup")
async def manual_cleanup():
    now = time.time()
    for item in PUBLIC_DIR.iterdir():
        if now - item.stat().st_mtime > CLEANUP_INTERVAL_SECONDS:
            if item.is_file():
                item.unlink()
            else:
                shutil.rmtree(item, ignore_errors=True)
    return {"message": "Cleanup completed"}

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    logging.error(f"Unhandled error: {exc}")
    return JSONResponse(status_code=500, content={"detail": f"Internal server error: {str(exc)}"})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port)
