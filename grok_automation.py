"""
Grok Imagine Browser Automation
Automates image-to-video generation using Playwright
"""

import asyncio
import os
import time
from pathlib import Path
from typing import Optional
import json

try:
    from playwright.async_api import async_playwright, Page, Browser
    PLAYWRIGHT_AVAILABLE = True
except ImportError:
    PLAYWRIGHT_AVAILABLE = False
    print("Warning: playwright not installed. Run: pip install playwright && playwright install chromium")


# Configuration
GROK_URL = "https://grok.com"
GROK_IMAGINE_URL = "https://grok.com/imagine"
USER_DATA_DIR = Path(__file__).parent / ".grok_browser_data"
DOWNLOAD_DIR = Path(__file__).parent / "clips"


async def setup_browser(headless: bool = False):
    """
    Launch browser with persistent context to maintain login session.
    First run: login manually, session will be saved.
    Subsequent runs: session is reused.
    """
    if not PLAYWRIGHT_AVAILABLE:
        raise Exception("Playwright not installed. Run: pip install playwright && playwright install chromium")

    playwright = await async_playwright().start()

    # Use persistent context to save login state
    USER_DATA_DIR.mkdir(parents=True, exist_ok=True)
    DOWNLOAD_DIR.mkdir(parents=True, exist_ok=True)

    browser = await playwright.chromium.launch_persistent_context(
        user_data_dir=str(USER_DATA_DIR),
        headless=headless,
        viewport={"width": 1280, "height": 900},
        accept_downloads=True,
    )

    return playwright, browser


async def wait_for_login(page: Page, timeout: int = 300):
    """
    Wait for user to login manually if not already logged in.
    Checks for presence of 'Imagine' in sidebar indicating logged-in state.
    """
    print("[GROK] Checking login status...")

    try:
        # Check if we're on the imagine page and logged in
        await page.wait_for_selector('text=Imagine', timeout=5000)
        print("[GROK] Already logged in!")
        return True
    except:
        pass

    print("[GROK] Please login to Grok in the browser window...")
    print(f"[GROK] Waiting up to {timeout} seconds for login...")

    try:
        await page.wait_for_selector('text=Imagine', timeout=timeout * 1000)
        print("[GROK] Login successful!")
        return True
    except:
        print("[GROK] Login timeout - please try again")
        return False


async def navigate_to_imagine(page: Page):
    """Navigate to the Imagine section."""
    print("[GROK] Navigating to Imagine...")

    # Try clicking Imagine in sidebar first
    try:
        imagine_link = page.locator('text=Imagine').first
        await imagine_link.click()
        await page.wait_for_timeout(2000)
    except:
        # Direct navigation
        await page.goto(GROK_IMAGINE_URL)
        await page.wait_for_timeout(3000)


async def upload_image_and_generate(
    page: Page,
    image_path: str,
    prompt: Optional[str] = None,
    output_path: Optional[str] = None,
    timeout: int = 180
) -> Optional[str]:
    """
    Upload an image and generate a video.

    Args:
        page: Playwright page
        image_path: Path to the image file
        prompt: Optional motion prompt (if None, uses auto-generate)
        output_path: Where to save the video (if None, auto-generates name)
        timeout: Max seconds to wait for generation

    Returns:
        Path to downloaded video or None if failed
    """
    print(f"[GROK] Processing: {image_path}")

    if not os.path.exists(image_path):
        print(f"[GROK] ERROR: Image not found: {image_path}")
        return None

    try:
        # Click "Upload image" button
        print("[GROK] Clicking Upload image button...")
        upload_btn = page.locator('button:has-text("Upload image"), [aria-label*="Upload"]').first

        # Set up file chooser before clicking
        async with page.expect_file_chooser() as fc_info:
            await upload_btn.click()

        file_chooser = await fc_info.value
        await file_chooser.set_files(image_path)
        print("[GROK] Image uploaded")

        # Wait for image to load
        await page.wait_for_timeout(3000)

        # Check if auto-generate is happening or we need to trigger manually
        # Look for progress indicator or "Make video" button

        if prompt:
            # Enter custom prompt
            print(f"[GROK] Entering prompt: {prompt[:50]}...")
            prompt_input = page.locator('input[placeholder*="customize"], textarea[placeholder*="customize"]').first
            await prompt_input.fill(prompt)
            await page.wait_for_timeout(500)

            # Click Make video button
            make_video_btn = page.locator('button:has-text("Make video")').first
            await make_video_btn.click()

        # Wait for generation to complete
        print("[GROK] Waiting for video generation...")
        start_time = time.time()

        while time.time() - start_time < timeout:
            # Check for download button (indicates completion)
            try:
                download_btn = page.locator('[aria-label*="download"], [aria-label*="Download"], button:has(svg[data-icon="download"])').first
                if await download_btn.is_visible():
                    print("[GROK] Generation complete!")
                    break
            except:
                pass

            # Check for "Redo" button (also indicates completion)
            try:
                redo_btn = page.locator('button:has-text("Redo")').first
                if await redo_btn.is_visible():
                    print("[GROK] Generation complete (Redo visible)!")
                    break
            except:
                pass

            # Check progress
            try:
                progress_text = await page.locator('text=/\\d+%/').first.text_content()
                print(f"[GROK] Progress: {progress_text}")
            except:
                pass

            await page.wait_for_timeout(3000)

        # Download the video
        print("[GROK] Downloading video...")

        # Determine output path
        if not output_path:
            base_name = Path(image_path).stem
            output_path = str(DOWNLOAD_DIR / f"{base_name}_video.mp4")

        # Click download and wait for download
        async with page.expect_download() as download_info:
            # Try multiple selectors for download button
            download_selectors = [
                '[aria-label*="ownload"]',
                'button:has(svg):near(button:has-text("Redo"))',
                '[data-testid="download"]',
            ]

            for selector in download_selectors:
                try:
                    btn = page.locator(selector).first
                    if await btn.is_visible():
                        await btn.click()
                        break
                except:
                    continue

        download = await download_info.value
        await download.save_as(output_path)
        print(f"[GROK] Video saved to: {output_path}")

        # Go back to main Imagine page for next upload
        await navigate_to_imagine(page)
        await page.wait_for_timeout(2000)

        return output_path

    except Exception as e:
        print(f"[GROK] ERROR: {e}")
        return None


async def process_batch(
    image_paths: list,
    prompts: Optional[list] = None,
    output_dir: Optional[str] = None,
    headless: bool = False,
    delay_between: int = 5
):
    """
    Process multiple images into videos.

    Args:
        image_paths: List of image file paths
        prompts: Optional list of prompts (same length as image_paths)
        output_dir: Directory to save videos
        headless: Run browser in headless mode (requires prior login)
        delay_between: Seconds to wait between generations

    Returns:
        List of output video paths (None for failed ones)
    """
    if not PLAYWRIGHT_AVAILABLE:
        print("ERROR: Playwright not installed")
        print("Run: pip install playwright && playwright install chromium")
        return []

    if output_dir:
        Path(output_dir).mkdir(parents=True, exist_ok=True)

    playwright, browser = await setup_browser(headless=headless)

    try:
        page = await browser.new_page()
        await page.goto(GROK_URL)

        # Wait for login
        if not await wait_for_login(page):
            print("[GROK] Login failed - aborting")
            return []

        # Navigate to Imagine
        await navigate_to_imagine(page)

        results = []
        total = len(image_paths)

        for i, image_path in enumerate(image_paths):
            print(f"\n[GROK] Processing {i+1}/{total}: {image_path}")

            prompt = prompts[i] if prompts and i < len(prompts) else None

            if output_dir:
                base_name = Path(image_path).stem
                output_path = str(Path(output_dir) / f"{base_name}_video.mp4")
            else:
                output_path = None

            result = await upload_image_and_generate(page, image_path, prompt, output_path)
            results.append(result)

            if i < total - 1:
                print(f"[GROK] Waiting {delay_between}s before next...")
                await page.wait_for_timeout(delay_between * 1000)

        return results

    finally:
        await browser.close()
        await playwright.stop()


def generate_video_sync(image_path: str, prompt: Optional[str] = None, output_path: Optional[str] = None) -> Optional[str]:
    """
    Synchronous wrapper for single video generation.
    """
    return asyncio.run(process_batch([image_path], [prompt] if prompt else None))[0]


# CLI interface
if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage: python grok_automation.py <image_path> [prompt]")
        print("       python grok_automation.py --batch <image_dir> [output_dir]")
        print("")
        print("First run will open browser for login. Session is saved for future runs.")
        sys.exit(1)

    if sys.argv[1] == "--batch":
        # Batch mode
        image_dir = sys.argv[2] if len(sys.argv) > 2 else "."
        output_dir = sys.argv[3] if len(sys.argv) > 3 else "./clips"

        image_paths = list(Path(image_dir).glob("*.png")) + list(Path(image_dir).glob("*.jpg"))
        image_paths = [str(p) for p in sorted(image_paths)]

        print(f"Found {len(image_paths)} images to process")
        results = asyncio.run(process_batch(image_paths, output_dir=output_dir))

        successful = sum(1 for r in results if r)
        print(f"\nComplete: {successful}/{len(image_paths)} videos generated")

    else:
        # Single image mode
        image_path = sys.argv[1]
        prompt = sys.argv[2] if len(sys.argv) > 2 else None

        result = generate_video_sync(image_path, prompt)
        if result:
            print(f"Success: {result}")
        else:
            print("Failed to generate video")
