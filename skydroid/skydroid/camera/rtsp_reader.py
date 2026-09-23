#!/usr/bin/env python3
"""Xem 2 stream RTSP cua gimbal Skydroid C12 qua Ethernet — KHONG dung ROS.

    Visible (RGB)  : rtsp://192.168.144.108:554/stream=1
    Thermal        : rtsp://192.168.144.108:555/stream=2

PHAI chay bang python he thong (co GStreamer):

    /usr/bin/python3 rtsp_reader.py

Trong conda robot_env (cv2 pip khong co GStreamer) thi phai them --backend ffmpeg.

Cach dung:
    /usr/bin/python3 rtsp_reader.py                  # ca 2 stream
    /usr/bin/python3 rtsp_reader.py --only rgb       # chi RGB, nhe nhat
    /usr/bin/python3 rtsp_reader.py --scale 0.5      # thu nho khung hien thi
    /usr/bin/python3 rtsp_reader.py --hw             # giai ma bang NVDEC
    /usr/bin/python3 rtsp_reader.py --stats          # in FPS + do tre moi giay



    /usr/bin/python3 rtsp_reader.py --only rgb --codec h265 --hw --stats

Nhan ESC hoac q de thoat.
"""

import argparse
import os
import sys
import threading
import time

import cv2

HOST = "192.168.144.108"

STREAMS = {
    "rgb":     {"port": 554, "path": "stream=1", "title": "Skydroid RGB"},
    # "thermal": {"port": 555, "path": "stream=2", "title": "Skydroid Thermal"},
}

# Bao hieu chung: capture thread danh thuc main thread khi co frame moi
NEW_FRAME = threading.Event()


def build_source(name, args):
    """Tra ve chuoi truyen cho cv2.VideoCapture tuy backend."""
    cfg = STREAMS[name]
    rtsp = f"rtsp://{HOST}:{cfg['port']}/{cfg['path']}"
    print(type(rtsp))
    if args.backend == "ffmpeg":
        return rtsp

    src = (f"rtspsrc location={rtsp} latency={args.latency} "
           f"protocols={args.protocol} drop-on-latency=true")

    if args.codec == "auto":
        # decodebin tu do codec (H.264 hay H.265) va tu chon bo giai ma
        chain = "decodebin"
    else:
        # Duong ro rang: khoi dong nhanh hon vi khong phai do codec
        n = "264" if args.codec == "h264" else "265"
        dec = (f"nv{'h' + n}dec ! cudadownload" if args.hw else f"avdec_h{n}")
        chain = f"rtph{n}depay ! h{n}parse ! {dec}"

    return (
        f"{src} ! {chain} ! "
        "videoconvert n-threads=4 ! video/x-raw,format=BGR ! "
        "appsink sync=false drop=true max-buffers=1"
    )


class RTSPStream:
    """Mot stream = mot thread doc lap, tu reconnect khi mat ket noi."""

    def __init__(self, name, args):
        self.name = name
        self.title = STREAMS[name]["title"]
        self.url = build_source(name, args)
       
        self.api = cv2.CAP_FFMPEG if args.backend == "ffmpeg" else cv2.CAP_GSTREAMER

        self.fail_threshold = args.fail_threshold
        self.reconnect_cooldown = args.reconnect_cooldown

        self.cap = None
        self.fail_count = 0
        self.was_opened = False
        self.last_attempt = None

        # Frame moi nhat + so thu tu de main thread biet co frame moi hay chua.
        # Khong copy: cap.read() cap phat mang moi moi lan goi nen truyen
        # thang tham chieu la an toan.
        self._frame = None
        self._seq = 0
        self._lock = threading.Lock()

        self._running = False
        self._thread = None

        self.fps = 0.0
        self._fps_t0 = time.monotonic()
        self._fps_n0 = 0

    # ---------------------------------------------------------------- API
    def start(self):
        self._running = True
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

    def stop(self):
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=3.0)
        if self.cap is not None:
            self.cap.release()
            self.cap = None

    def take(self, last_seq):
        """Tra ve (seq, frame) neu co frame moi hon last_seq, nguoc lai None."""
        with self._lock:
            if self._seq == last_seq or self._frame is None:
                return None
            return self._seq, self._frame

    @property
    def connected(self):
        return self.cap is not None and self.cap.isOpened()

    # ------------------------------------------------------------ internal
    def _open(self):
        self.last_attempt = time.monotonic()
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.fail_count = 0


        # if not os.path.exists(disk):
        #     os.makedirs(disk)
        #     print(f"Folder created: {disk}")
        print("url",self.url)
        print("=======================")
        cap = cv2.VideoCapture(self.url, self.api)

        if cap.isOpened():
            if self.api == cv2.CAP_FFMPEG:
                cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
            self.cap = cap
            print(f"[{self.name}] {'Reconnect OK' if self.was_opened else 'Stream opened'}")
            self.was_opened = True
        else:
            cap.release()
            print(f"[{self.name}] Khong mo duoc stream, thu lai sau "
                  f"{self.reconnect_cooldown:.0f}s...")
            


    def _cooldown_elapsed(self):
        if self.last_attempt is None:
            return True
        return (time.monotonic() - self.last_attempt) >= self.reconnect_cooldown

    def _loop(self):
        last_warn = 0.0
        frames = 0

        while self._running:
            if not self.connected:
                if self._cooldown_elapsed():
                    self._open()
                if not self.connected:
                    time.sleep(0.2)
                    continue

            ok, frame = self.cap.read()

            if ok and frame is not None:
                self.fail_count = 0
                frames += 1

                with self._lock:
                    self._frame = frame
                    self._seq += 1
                NEW_FRAME.set()

                now = time.monotonic()
                dt = now - self._fps_t0
                if dt >= 1.0:
                    self.fps = (frames - self._fps_n0) / dt
                    self._fps_t0 = now
                    self._fps_n0 = frames
            else:
                self.fail_count += 1

                now = time.monotonic()
                if now - last_warn >= 2.0:
                    print(f"[{self.name}] Mat frame "
                          f"({self.fail_count}/{self.fail_threshold})")
                    last_warn = now

                if self.fail_count >= self.fail_threshold:
                    print(f"[{self.name}] Stream mat ket noi, dang reconnect...")
                    self._open()


def main():
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", choices=["rgb", "thermal"],
                    help="chi mo 1 stream (mac dinh mo ca 2)")
    ap.add_argument("--backend", choices=["gstreamer", "ffmpeg"], default="gstreamer")
    ap.add_argument("--protocol", choices=["tcp", "udp"], default="tcp",
                    help="lower transport RTSP; udp tre thap hon nhung de mat goi")
    ap.add_argument("--latency", type=int, default=0,
                    help="rtspsrc jitterbuffer, ms (0 = thap nhat)")
    ap.add_argument("--codec", choices=["auto", "h264", "h265"], default="auto",
                    help="codec cua stream; auto = de decodebin tu do")
    ap.add_argument("--hw", action="store_true",
                    help="giai ma bang NVDEC thay vi CPU (can --codec h264/h265)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="ty le thu nho khung hien thi, vd 0.5")
    ap.add_argument("--fail-threshold", type=int, default=90)
    ap.add_argument("--reconnect-cooldown", type=float, default=5.0)
    ap.add_argument("--stats", action="store_true", help="in FPS moi giay")
    args = ap.parse_args()

    # cv2 cai bang pip (vd trong conda) khong co backend GStreamer -> bao ro
    # thay vi de VideoCapture im lang fail va bao "khong mo duoc stream".
    if args.backend == "gstreamer" and "GStreamer:                   YES" \
            not in cv2.getBuildInformation():
        print(f"LOI: cv2 {cv2.__version__} tai\n      {cv2.__file__}\n"
              "      duoc build KHONG co GStreamer.\n\n"
              "Chay bang python he thong:\n"
              f"      /usr/bin/python3 {os.path.basename(sys.argv[0])} "
              f"{' '.join(sys.argv[1:])}\n"
              "hoac dung backend khac:  --backend ffmpeg", file=sys.stderr)
        return 1

    if args.backend == "ffmpeg":
        os.environ.setdefault(
            "OPENCV_FFMPEG_CAPTURE_OPTIONS",
            f"rtsp_transport;{args.protocol}|fflags;nobuffer|flags;low_delay")

    names = [args.only] if args.only else list(STREAMS)
    streams = [RTSPStream(n, args) for n in names]

    for s in streams:
        cv2.namedWindow(s.title, cv2.WINDOW_AUTOSIZE)
        s.start()

    seen = {s.name: 0 for s in streams}
    last_stats = time.monotonic()
    img_count = 1
    try:
        while True:
            # Ngu cho den khi co frame moi, khong quay vong vo ich
            NEW_FRAME.wait(timeout=0.02)
            NEW_FRAME.clear()

            for s in streams:
                got = s.take(seen[s.name])
                if got is None:
                    continue            # chua co frame moi -> khong ve lai
                seq, frame = got
                seen[s.name] = seq

                if args.scale != 1.0:
                    frame = cv2.resize(frame, None, fx=args.scale, fy=args.scale,
                                       interpolation=cv2.INTER_NEAREST)
                cv2.imshow(s.title, frame)
                ##############
            disk = "/home/ngoc/drone_ws/src/skydroid/skydroid/camera"
            
            max_images = 40



            key = cv2.waitKey(1) & 0xFF
            if key == ord('s'):
                img_name = os.path.join(disk, f"{img_count:02d}.jpg")
                cv2.imwrite(img_name, frame)



                # captured = cv2.imread(img_name)
                # if captured is not None:
                #     cv2.imshow("Captured Image", captured)
                #     cv2.waitKey(1)



                print(f"[{img_count}/{max_images}] successfully save: {img_name}")
                
                img_count += 1
                
                if img_count > max_images:
                    print("\nFinished.")
                    return
                    
            elif key == ord('q'):
                return

            # got.release()
            
            if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                break

            if args.stats:
                now = time.monotonic()
                if now - last_stats >= 1.0:
                    print(" | ".join(
                        f"{s.name}: {s.fps:5.1f} fps"
                        f"{'' if s.connected else ' [DISCONNECTED]'}"
                        for s in streams))
                    last_stats = now
        cv2.destroyAllWindows()
    except KeyboardInterrupt:
        pass
    finally:
        for s in streams:
            s.stop()
        cv2.destroyAllWindows()
        print("Da dong tat ca stream.")


if __name__ == "__main__":
    sys.exit(main() or 0)
