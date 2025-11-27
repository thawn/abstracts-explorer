#!/usr/bin/env python3
"""
Demonstration script for E2E test browser selection.

This script shows how to check which browsers are available and run E2E tests.
"""

import subprocess
import sys
import os


def check_browser_available(browser_name):
    """Check if a browser is available for testing."""
    print(f"\nChecking {browser_name} availability...")

    try:
        if browser_name.lower() == "chrome":
            result = subprocess.run(
                [
                    "python",
                    "-c",
                    "from tests.test_web_e2e import _check_chrome_available; " "print(_check_chrome_available())",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        elif browser_name.lower() == "firefox":
            result = subprocess.run(
                [
                    "python",
                    "-c",
                    "from tests.test_web_e2e import _check_firefox_available; " "print(_check_firefox_available())",
                ],
                capture_output=True,
                text=True,
                timeout=10,
            )
        else:
            return False

        available = "True" in result.stdout
        status = "✓ Available" if available else "✗ Not available"
        print(f"{browser_name}: {status}")
        return available

    except Exception as e:
        print(f"{browser_name}: ✗ Error checking - {e}")
        return False


def main():
    """Main function to demonstrate E2E test browser selection."""

    print("=" * 60)
    print("E2E Test Browser Configuration Demo")
    print("=" * 60)

    # Check available browsers
    chrome_available = check_browser_available("Chrome")
    firefox_available = check_browser_available("Firefox")

    print("\n" + "=" * 60)
    print("Browser Availability Summary")
    print("=" * 60)

    if chrome_available and firefox_available:
        print("✓ Both Chrome and Firefox are available")
        print("\nYou can run E2E tests with:")
        print("  - Default (auto):  pytest tests/test_web_e2e.py -m e2e")
        print("  - Chrome only:     E2E_BROWSER=chrome pytest tests/test_web_e2e.py -m e2e")
        print("  - Firefox only:    E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e")
    elif chrome_available:
        print("✓ Chrome is available")
        print("✗ Firefox is not available")
        print("\nYou can run E2E tests with:")
        print("  - pytest tests/test_web_e2e.py -m e2e")
        print("  - E2E_BROWSER=chrome pytest tests/test_web_e2e.py -m e2e")
    elif firefox_available:
        print("✗ Chrome is not available")
        print("✓ Firefox is available")
        print("\nYou can run E2E tests with:")
        print("  - pytest tests/test_web_e2e.py -m e2e")
        print("  - E2E_BROWSER=firefox pytest tests/test_web_e2e.py -m e2e")
    else:
        print("✗ Neither Chrome nor Firefox is available")
        print("\nPlease install at least one browser:")
        print("\n  Chrome (macOS):   brew install --cask google-chrome")
        print("  Firefox (macOS):  brew install --cask firefox")
        print("\n  Chrome (Linux):   sudo apt-get install google-chrome-stable")
        print("  Firefox (Linux):  sudo apt-get install firefox")
        return 1

    # Show current environment setting
    print("\n" + "=" * 60)
    print("Current Configuration")
    print("=" * 60)

    browser_env = os.environ.get("E2E_BROWSER", "auto")
    print(f"E2E_BROWSER environment variable: {browser_env}")

    if browser_env == "auto":
        print("Behavior: Will try Chrome first, then Firefox")
    elif browser_env == "chrome":
        print("Behavior: Will only use Chrome")
    elif browser_env == "firefox":
        print("Behavior: Will only use Firefox")

    print("\n" + "=" * 60)
    print("Example Test Commands")
    print("=" * 60)
    print("\n# Run a single test with auto browser selection:")
    print("pytest tests/test_web_e2e.py::TestWebUIE2E::test_page_loads -v -m e2e")

    print("\n# Run all E2E tests with Firefox:")
    print("E2E_BROWSER=firefox pytest tests/test_web_e2e.py -v -m e2e")

    print("\n# Run all E2E tests with Chrome:")
    print("E2E_BROWSER=chrome pytest tests/test_web_e2e.py -v -m e2e")

    print("\n# Run accessibility tests only:")
    print("pytest tests/test_web_e2e.py::TestWebUIAccessibility -v -m e2e")

    print("\n" + "=" * 60)

    return 0


if __name__ == "__main__":
    sys.exit(main())
