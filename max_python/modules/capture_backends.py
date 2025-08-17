from dataclasses import dataclass
from typing import List, Optional
import threading
import time

import numpy as np
import psutil
import win32gui
import win32process

# Optional libs
try:
    import dxcam
    HAS_DXCAM = True
except Exception:
    dxcam = None
    HAS_DXCAM = False

try:
    from windows_capture import WindowsCapture  # WGC wrapper (optional)
    HAS_WGC = True
except Exception:
    WindowsCapture = None
    HAS_WGC = False


# ---------- Window enumeration ----------

@dataclass
class WinInfo:
    hwnd: int
    title: str
    cls: str
    pid: int
    exe: str  # lowercased process name

def list_windows() -> List["WinInfo"]:
    out: List[WinInfo] = []

    def cb(hwnd, _):
        if not win32gui.IsWindowVisible(hwnd):
            return
        title = win32gui.GetWindowText(hwnd)
        if not title:
            return
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        if r - l < 80 or b - t < 80:
            return
        try:
            cls = win32gui.GetClassName(hwnd)
        except Exception:
            cls = ""
        pid = 0
        exe = ""
        try:
            _, pid = win32process.GetWindowThreadProcessId(hwnd)
            if pid:
                try:
                    exe = (psutil.Process(pid).name() or "").lower()
                except Exception:
                    exe = ""
        except Exception:
            pass
        out.append(WinInfo(hwnd=hwnd, title=title, cls=cls, pid=pid, exe=exe))

    win32gui.EnumWindows(cb, None)
    # Prefer Java windows first (graphwar), then title
    out.sort(key=lambda w: (w.exe != "javaw.exe", w.title.lower()))
    return out


# ---------- DXCAM (display region) backend ----------

class DXRegionCapture:
    """
    Low-latency display-region capture cropped to the selected HWND rect.
    Tuned ring buffer to keep freshest frames (max_buffer_len=8).
    """
    def __init__(self):
        self._dx = None
        self._running = False
        self._last = None
        self._lock = threading.Lock()

    def start(self, hwnd: int):
        if not HAS_DXCAM:
            raise RuntimeError("dxcam not installed. pip install dxcam")
        l, t, r, b = win32gui.GetWindowRect(hwnd)
        self._dx = dxcam.create(output_idx=0, max_buffer_len=8)  # keep latency low
        self._dx.start(target_fps=60, region=(l, t, r, b))
        self._running = True
        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while self._running and self._dx is not None:
            try:
                frm = self._dx.get_latest_frame()  # RGB ndarray (H,W,3)
                if frm is not None:
                    with self._lock:
                        self._last = frm
            except Exception:
                pass
            time.sleep(1/60.0)

    def stop(self):
        self._running = False
        if self._dx is not None:
            try:
                self._dx.stop()
            except Exception:
                pass
        self._dx = None
        with self._lock:
            self._last = None

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last is None:
                return None
            return self._last.copy()


# ---------- WGC (Windows Graphics Capture) backend (optional) ----------

class WGCCapture:
    """
    WGC via windows-capture (title-based).
    NOTE: WGC won't deliver frames if the window is minimized (OS behavior).
    """
    def __init__(self):
        self._cap = None
        self._running = False
        self._last = None
        self._lock = threading.Lock()
        self._thread = None
        self._title = None

    def start(self, title: str):
        if not HAS_WGC:
            raise RuntimeError("windows-capture not installed. pip install windows-capture")
        self.stop()
        self._title = title
        self._running = True
        self._thread = threading.Thread(target=self._worker, name="WGCWorker", daemon=True)
        self._thread.start()

    def _worker(self):
        try:
            cap = WindowsCapture(window_name=self._title,
                                 cursor_capture=False,
                                 draw_border=True,
                                 monitor_index=None)
            @cap.event
            def on_frame_arrived(frame, control):
                try:
                    buf = frame.frame_buffer  # BGRA
                    rgb = buf[:, :, :3][:, :, ::-1]  # -> RGB
                    # Light downscale if huge
                    h, w, _ = rgb.shape
                    if w > 1280:
                        import cv2
                        nh = int(h * (1280 / w))
                        rgb = cv2.resize(rgb, (1280, nh), interpolation=cv2.INTER_LINEAR)
                    with self._lock:
                        self._last = rgb
                except Exception:
                    pass

            @cap.event
            def on_closed():
                self._running = False
                self._cap = None

            cap.start()
            self._cap = cap
            while self._running and self._cap is not None:
                time.sleep(0.05)
        except Exception as e:
            import numpy as np
            img = np.zeros((240, 800, 3), dtype=np.uint8)
            try:
                import cv2
                cv2.putText(img, f"WGC error: {e}", (10, 140),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            except Exception:
                pass
            with self._lock:
                self._last = img

    def stop(self):
        self._running = False
        if self._cap is not None:
            try:
                self._cap.stop()
            except Exception:
                pass
        self._cap = None
        with self._lock:
            self._last = None

    def latest(self) -> Optional[np.ndarray]:
        with self._lock:
            if self._last is None:
                return None
            return self._last.copy()


# ---------- Facade ----------

class CaptureController:
    """
    Simple facade the UI uses. You pick backend + window; it exposes latest() frame.
    """
    def __init__(self):
        self.backend = "DXCAM"
        self.dx = DXRegionCapture()
        self.wgc = WGCCapture()

    def set_backend(self, name: str):
        name = (name or "").upper()
        if name not in ("DXCAM", "WGC"):
            raise ValueError("Backend must be 'DXCAM' or 'WGC'")
        self.backend = name

    def start(self, win: WinInfo):
        self.stop()
        if self.backend == "DXCAM":
            self.dx.start(win.hwnd)
        else:
            self.wgc.start(win.title)

    def stop(self):
        try: self.dx.stop()
        except Exception: pass
        try: self.wgc.stop()
        except Exception: pass

    def latest(self) -> Optional[np.ndarray]:
        if self.backend == "DXCAM":
            return self.dx.latest()
        else:
            return self.wgc.latest()
