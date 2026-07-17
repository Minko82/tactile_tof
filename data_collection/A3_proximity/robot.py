import socket, re, math, sys, time, csv, threading, os

# ===========================================================================
#  CONNECTION  (static IPs — leave alone)
# ===========================================================================
ROBOT_IP = "192.168.1.10"     # robot
HOST_IP  = "192.168.1.20"     # Mac
CB_PORT  = 50002              # port robot reports pose back on

TABLE_Z_MM       = -143.30    #sensor-just-touching-table height  (python3 robot.py table)
WRIST_OFFSET_DEG = 45.0       #deg to make the sensor straight     (python3 robot.py wrist N)
CLEARANCE_MM     = 12.5       #stop this far above the table (~half inch lower than 25)
HOME_Z_MM        = 475.0      #safe perpendicular "init" height to rise to before descending

# motion tuning (accel, vel)
ALIGN_ACC, ALIGN_VEL = 0.30, 0.15    # perpendicular reorient
WRIST_ACC, WRIST_VEL = 0.30, 0.30    # wrist straighten
DOWN_ACC,  DOWN_VEL  = 0.25, 0.05    # down & up sweep speed (a bit faster, still smooth)

# recording
HOLD_SECONDS = 3.0                   # hold at the top (start) and at the bottom, seconds
STREAM_HZ    = 60                    # robot pose refresh rate (kept > sensor rate so each
                                     # logged row uses a fresh pose; the CSVs are frame-locked
                                     # to the sensor, so their actual rate == the sensor's)
RESULTS_DIR   = os.path.dirname(os.path.abspath(__file__))   # this script's folder;
                                     # runs saved under <here>/<profile>/<n>/{robot_log.csv, tof_log.csv}
ROBOT_CSV     = "robot_log.csv"
TOF_CSV       = "tof_log.csv"
BATCH_PAUSE_S = 3.0                  # pause between rounds in a batch
N_STEPS       = 4                    # 'steps' mode: number of equal steps each way (down, then up)
STEP_DWELL_S  = 3.0                  # 'steps' mode: hold 3 s at each step

# 'filtertest' mode: four-phase live Kalman-filter accuracy test
LIN_PHASE_S    = 10.0                # phase 1: constant-velocity linear descent, seconds
STATIC_PHASE_S = 10.0                # phase 2: static hold at the bottom, seconds
RAND_PHASE_S   = 10.0                # phase 4: rapid random movels, ~seconds (estimated)
RAND_ACC, RAND_VEL = 1.5, 0.5        # rapid phase accel (m/s^2) / vel (m/s) — bounded for safety
RAND2_PHASE_S  = 10.0                # phase 3: slow random movels, ~seconds (estimated)
RAND2_ACC, RAND2_VEL = 0.4, 0.12     # slow random phase accel (m/s^2) / vel (m/s)
RAND_Z_MAX_MM  = 250.0               # random targets stay within [clearance, this] above table
RAND_XY_MM     = 25.0                # random XY jitter about the start column, +/- mm
FT_CENTRAL     = (27, 28, 35, 36)    # central 2x2 zones of the 8x8 grid (boresight)
# Per-surface constant offsets (mount + reflectivity bias + sheet thickness,
# all lumped together so the UR5 table height never needs re-measuring).
# To add/update a surface: run one filtertest round on it, then
#   python3 robot.py offset <run_dir>     and paste the printed constant here.
# NOTE: the vl53l5cx_stream firmware (2026-07) shifted readings ~-4.5 mm vs the
# old firmware. matte_black is re-measured; the others are STALE old-firmware
# values — re-measure each with `python3 robot.py offset <run_dir>` after the
# first new-firmware run on that surface.
SURFACE_OFFSETS_MM = {
    "white": 29.0,                   # 2026-07-17 remount epoch (measured
                                     # +28.52..+29.50; pre-remount was +19.8)
    "wood":  22.1,                   # 2026-07-17 remount epoch (measured
                                     # +21.87..+22.46; pre-remount was +5.7)
    "black_shiny": 21.8,             # 2026-07-17 remount epoch (measured
                                     # +21.36..+22.71; pre-remount was +12.3)
    "matte_black": 26.7,             # 2026-07-17 remount epoch (measured +26.73,
                                     # single run; pre-remount was +16.1)
}
DEFAULT_SURFACE = "white"
MOUNT_OFFSET_MM = SURFACE_OFFSETS_MM[DEFAULT_SURFACE]   # back-compat alias
FT_ACCEL_PSD   = 3000.0              # LiveFilter process_accel_psd: raise (2000-5000) to
                                     # track the rapid phase tighter, at the cost of
                                     # passing more noise in the linear/static phases


def send(urscript):
    s = socket.create_connection((ROBOT_IP, 30002), timeout=5)
    s.sendall(urscript.encode()); s.close()

def collect(urscript, n_lines, timeout=90):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", CB_PORT)); srv.listen(1); srv.settimeout(timeout)
    send(urscript)
    conn, _ = srv.accept()
    buf = b""
    while buf.count(b"\n") < n_lines:
        chunk = conn.recv(4096)
        if not chunk:
            break
        buf += chunk
    conn.close(); srv.close()
    return [ln for ln in buf.decode().splitlines() if ln.strip()]

def report(expr):
    return '  socket_send_string(to_str(%s))\n  socket_send_byte(10)\n' % expr

def parse(line):
    return [float(v) for v in re.findall(r"-?\d+\.?\d*", line)][:6]

def show(label, p):
    x, y, z, rx, ry, rz = p
    print(f"\n{label}")
    print(f"  Position:    X={x*1000:8.2f} mm   Y={y*1000:8.2f} mm   Z={z*1000:8.2f} mm")
    print(f"  Orientation: Rx={rx:7.4f}   Ry={ry:7.4f}   Rz={rz:7.4f}   rad")

def get_pose():
    us = ('def rpt():\n'
          '  socket_open("%s", %d)\n' % (HOST_IP, CB_PORT) +
          report("get_actual_tcp_pose()") +
          '  socket_close()\nend\nrpt()\n')
    return parse(collect(us, 1, timeout=15)[0])

def dashboard(cmd):
    s = socket.create_connection((ROBOT_IP, 29999), timeout=5)
    s.recv(4096); s.sendall((cmd + "\n").encode())
    resp = s.recv(4096).decode().strip(); s.close()
    return resp


# Commands
def _init_target_urscript(tilt_deg=0.0):
    """URScript snippet that builds pose `t` = (current X,Y at HOME_Z) with the
    perpendicular + sensor-straight orientation (the 45 deg is baked in here, so
    the arm reaches the top already correctly oriented). tilt_deg rotates the
    tool off perpendicular about its X axis — incidence-angle test runs."""
    if abs(tilt_deg) > 20.0:
        raise SystemExit(f"tilt {tilt_deg} deg refused: keep within +/-20 deg "
                         f"(mount clearance above the table)")
    home_z = HOME_Z_MM / 1000.0
    woff   = WRIST_OFFSET_DEG * math.pi / 180.0
    s = (
        '  t = get_actual_tcp_pose()\n'
        '  t[2] = %.5f\n' % home_z +                                # init height
        '  t[3] = 3.14159265\n  t[4] = 0.0\n  t[5] = 0.0\n'          # perpendicular (tool down)
        '  t = pose_trans(t, p[0,0,0,0,0,%.6f])\n' % woff           # +45 deg about tool Z -> sensor straight
    )
    if tilt_deg:
        s += '  t = pose_trans(t, p[0,0,0,%.6f,0,0])\n' % (tilt_deg * math.pi / 180.0)
    return s


def _home_urscript(with_descent):
    """Rise to the perpendicular + sensor-straight init spot, optionally descend."""
    target_z = (TABLE_Z_MM + CLEARANCE_MM) / 1000.0
    s = (
        'def home():\n'
        '  socket_open("%s", %d)\n' % (HOST_IP, CB_PORT) +
        report("get_actual_tcp_pose()") +                          # start pose
        _init_target_urscript() +                                  # build oriented target `t`
        '  movel(t, a=%.4f, v=%.4f)\n' % (ALIGN_ACC, ALIGN_VEL)     # go up + orient in one move
    )
    if with_descent:
        s += (
            '  t[2] = %.5f\n' % target_z +                          # lower Z, keep orientation
            '  movel(t, a=%.4f, v=%.4f)\n' % (DOWN_ACC, DOWN_VEL)
        )
    s += report("get_actual_tcp_pose()") + '  socket_close()\nend\nhome()\n'
    return s


def cmd_run():
    """Rise to perpendicular init spot, then descend to table+clearance."""
    lines = collect(_home_urscript(with_descent=True), 2)
    start, final = parse(lines[0]), parse(lines[-1])
    show("START pose:", start)
    print(f"\n>> up to init ({HOME_Z_MM:.0f} mm) + perpendicular + wrist "
          f"{WRIST_OFFSET_DEG:+.1f} deg + descend to {CLEARANCE_MM:.0f} mm above table")
    show("FINAL pose:", final)
    print(f"\n  Clearance above table: {final[2]*1000 - TABLE_Z_MM:6.2f} mm "
          f"(target {CLEARANCE_MM:.0f} mm)")


def cmd_up():
    """Rise to the perpendicular init spot only (no descent)."""
    lines = collect(_home_urscript(with_descent=False), 2)
    show("START pose:", parse(lines[0]))
    print(f"\n>> raised to init height {HOME_Z_MM:.0f} mm, perpendicular, sensor straight")
    show("INIT pose:", parse(lines[-1]))


def _record_urscript():
    """Position at the init spot (NOT recorded), then stream the TCP pose while the
    arm goes DOWN to the table and back UP, holds 5 s, then cuts the round."""
    home_z   = HOME_Z_MM / 1000.0
    target_z = (TABLE_Z_MM + CLEARANCE_MM) / 1000.0
    dt       = 1.0 / STREAM_HZ
    return (
        'def record():\n'
        '  socket_open("%s", %d)\n' % (HOST_IP, CB_PORT) +
        # 1) position at perpendicular + sensor-straight init spot (positioning is NOT streamed)
        _init_target_urscript() +
        '  movel(t, a=%.4f, v=%.4f)\n' % (ALIGN_ACC, ALIGN_VEL) +
        # 2) start streaming the pose now that we're at the top
        '  thread streamer():\n'
        '    while (True):\n'
        '      socket_send_string(to_str(get_actual_tcp_pose()))\n'
        '      socket_send_byte(10)\n'
        '      sleep(%.4f)\n' % dt +
        '    end\n'
        '  end\n'
        '  gh = run streamer()\n'
        '  sleep(0.3)\n' +
        # 3) hold at the top (recording), then go DOWN to the table
        '  sleep(%.2f)\n' % HOLD_SECONDS +
        '  t[2] = %.5f\n' % target_z +
        '  movel(t, a=%.4f, v=%.4f)\n' % (DOWN_ACC, DOWN_VEL) +
        # 4) hold at the bottom, then come back UP to init
        '  sleep(%.2f)\n' % HOLD_SECONDS +
        '  t[2] = %.5f\n' % home_z +
        '  movel(t, a=%.4f, v=%.4f)\n' % (DOWN_ACC, DOWN_VEL) +
        # 5) back at the top -> cut the round
        '  kill gh\n'
        '  socket_close()\n'
        'end\n'
        'record()\n'
    )


def _record_steps_urscript():
    """Like _record_urscript, but the arm descends and ascends in N_STEPS equal steps,
    pausing STEP_DWELL_S at each step (a discrete stair-step scan). Same holds."""
    home_z   = HOME_Z_MM / 1000.0
    target_z = (TABLE_Z_MM + CLEARANCE_MM) / 1000.0
    n_steps  = N_STEPS
    step     = (home_z - target_z) / n_steps
    dt       = 1.0 / STREAM_HZ
    L = [
        "def record():",
        f'  socket_open("{HOST_IP}", {CB_PORT})',
        _init_target_urscript().rstrip("\n"),                 # position at init (NOT recorded)
        f"  movel(t, a={ALIGN_ACC:.4f}, v={ALIGN_VEL:.4f})",
        "  thread streamer():",
        "    while (True):",
        "      socket_send_string(to_str(get_actual_tcp_pose()))",
        "      socket_send_byte(10)",
        f"      sleep({dt:.4f})",
        "    end",
        "  end",
        "  gh = run streamer()",
        "  sleep(0.3)",
        f"  sleep({HOLD_SECONDS:.2f})",                        # hold at the top
        "  i = 0",                                             # down in steps
        f"  while i < {n_steps}:",
        "    i = i + 1",
        f"    z = {home_z:.5f} - i * {step:.5f}",
        f"    if z < {target_z:.5f}:",
        f"      z = {target_z:.5f}",
        "    end",
        "    t[2] = z",
        f"    movel(t, a={DOWN_ACC:.4f}, v={DOWN_VEL:.4f})",
        f"    sleep({STEP_DWELL_S:.2f})",
        "  end",
        f"  sleep({HOLD_SECONDS:.2f})",                        # hold at the bottom
        "  i = 0",                                             # up in steps
        f"  while i < {n_steps}:",
        "    i = i + 1",
        f"    z = {target_z:.5f} + i * {step:.5f}",
        f"    if z > {home_z:.5f}:",
        f"      z = {home_z:.5f}",
        "    end",
        "    t[2] = z",
        f"    movel(t, a={DOWN_ACC:.4f}, v={DOWN_VEL:.4f})",
        f"    sleep({STEP_DWELL_S:.2f})",
        "  end",
        "  kill gh",
        "  socket_close()",
        "end",
        "record()",
        "",
    ]
    return "\n".join(L)


def _next_run_dir(profile):
    """Create and return (n, path) for the next numbered run folder under
    A3/<profile>/ (or under `profile` itself if it is an absolute path)."""
    base = profile if os.path.isabs(profile) else os.path.join(RESULTS_DIR, profile)
    os.makedirs(base, exist_ok=True)
    used = [int(d) for d in os.listdir(base)
            if d.isdigit() and os.path.isdir(os.path.join(base, d))]
    n = (max(used) + 1) if used else 1
    path = os.path.join(base, str(n))
    os.makedirs(path, exist_ok=True)
    return n, path


def cmd_record(stepped=False):
    """One round: position at init (not recorded), then log the arm going DOWN to the
    table and back UP with 3 s holds at top and bottom, then cut. Motion is smooth,
    or in N_STEPS equal steps if stepped=True. Robot pose and ToF sensor are frame-locked
    (one row per sensor frame, shared timestamp). Saved to
    A3/<profile>/<n>/robot_log.csv and tof_log.csv  (profile = steps or smooth)."""
    import tof_logger

    profile = "steps" if stepped else "smooth"
    run_n, run_dir = _next_run_dir(profile)
    robot_path = os.path.join(run_dir, ROBOT_CSV)
    tof_path   = os.path.join(run_dir, TOF_CSV)

    # open the sensor FIRST so it's booted and streaming before the robot moves
    try:
        dev = tof_logger.open_sensor()
    except Exception as e:
        print("Sensor open failed:", e)
        return False

    # start motion + a continuous robot pose stream
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", CB_PORT)); srv.listen(1); srv.settimeout(60)
    send(_record_steps_urscript() if stepped else _record_urscript())
    conn, _ = srv.accept(); srv.close()

    state = {"pose": None}
    stop_event = threading.Event()

    def robot_reader():
        """Keep the latest robot pose fresh; stop everything when the stream ends."""
        conn.settimeout(30)
        buf = b""
        while True:
            try:
                chunk = conn.recv(4096)
            except socket.timeout:
                break
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf:
                raw, buf = buf.split(b"\n", 1)
                p = parse(raw.decode(errors="ignore").strip())
                if len(p) >= 6:
                    state["pose"] = p
        stop_event.set()

    reader = threading.Thread(target=robot_reader, daemon=True)
    reader.start()

    # wait for the first streamed pose (robot finishes positioning, then streams from the top)
    while state["pose"] is None and not stop_event.is_set():
        time.sleep(0.005)
    dev.reset_input_buffer()   # drop frames buffered during positioning; start logging fresh

    print(f"Recording run {run_n}  ->  {run_dir}/")

    n = 0
    t0 = time.perf_counter()
    with open(robot_path, "w", newline="") as rf, open(tof_path, "w", newline="") as tf:
        rw = csv.writer(rf)
        rw.writerow(["time_s", "x_mm", "y_mm", "z_mm", "rx_rad", "ry_rad", "rz_rad"])
        tw = csv.writer(tf)
        tw.writerow(["time_s"] + [f"z{i}" for i in range(tof_logger.N_ZONES)])
        while not stop_event.is_set():
            frame = tof_logger.read_frame(dev)
            if frame is None:
                continue
            p = state["pose"]
            if p is None:
                continue
            t = time.perf_counter() - t0
            tw.writerow([f"{t:.4f}"] + frame)                       # sensor row
            rw.writerow([f"{t:.4f}", f"{p[0]*1000:.3f}", f"{p[1]*1000:.3f}",
                         f"{p[2]*1000:.3f}", f"{p[3]:.5f}", f"{p[4]:.5f}", f"{p[5]:.5f}"])
            n += 1

    dev.close()
    conn.close()
    reader.join(timeout=3)

    print(f"Done (run {run_n}).  {n} frame-locked rows each ->  {robot_path} , {tof_path}")
    return n > 0


def cmd_record_batch(count, stepped=False):
    """Run `count` rounds back to back, each into A3/<profile>/<n>/."""
    profile = "steps" if stepped else "smooth"
    print(f"Batch: {count} {profile} round(s), ~{BATCH_PAUSE_S:.0f}s pause between each.\n")
    done = 0
    for i in range(count):
        print(f"========== round {i + 1} / {count} ({profile}) ==========")
        ok = cmd_record(stepped=stepped)
        if not ok:
            print(f"\nStopping batch: round {i + 1} did not record. "
                  f"{done} of {count} completed.")
            return
        done += 1
        if i < count - 1:
            print(f"... pausing {BATCH_PAUSE_S:.0f}s before next round ...\n")
            time.sleep(BATCH_PAUSE_S)
    print(f"\nBatch complete: {done} round(s) saved under "
          f"{os.path.join(RESULTS_DIR, profile)}/.")

def _gen_random_waypoints(rng, duration, vel, acc):
    """Pre-generate one random phase as (dx, dy, z_abs) movel targets whose
    estimated total duration ~= `duration` at the given vel/acc. URScript has no
    RNG, so the random trajectory is baked in here — which also means the
    commanded path is known exactly and gets saved alongside the logs."""
    z_lo = (TABLE_Z_MM + CLEARANCE_MM) / 1000.0
    z_hi = (TABLE_Z_MM + RAND_Z_MAX_MM) / 1000.0
    xy   = RAND_XY_MM / 1000.0
    pts, t_est = [], 0.0
    x = y = 0.0
    z_prev = z_lo                       # both phases start near the bottom of the box
    while t_est < duration:
        nx, ny = rng.uniform(-xy, xy), rng.uniform(-xy, xy)
        nz     = rng.uniform(z_lo, z_hi)
        dist   = math.sqrt((nx - x)**2 + (ny - y)**2 + (nz - z_prev)**2)
        t_est += dist / vel + vel / acc                     # trapezoidal estimate
        pts.append((nx, ny, nz))
        x, y, z_prev = nx, ny, nz
    return pts


def _filtertest_urscript(fast_pts, slow_pts, tilt_deg=0.0):
    """Position at init (NOT recorded), then stream the pose through four phases:
      PH,1  linear:      constant-velocity descent to table+clearance over LIN_PHASE_S
      PH,2  static:      hold STATIC_PHASE_S at the bottom
      PH,3  random slow: slow pre-randomized movels (bounded box above the table)
      PH,4  random fast: same box, at RAND accel/vel
      PH,0  done:        gentle return to init height, cut the round
    Phase markers are sent on the same socket as the pose stream."""
    home_z   = HOME_Z_MM / 1000.0
    target_z = (TABLE_Z_MM + CLEARANCE_MM) / 1000.0
    lin_vel  = (home_z - target_z) / LIN_PHASE_S
    dt       = 1.0 / STREAM_HZ

    def mark(ph):
        return [f'  socket_send_string("PH,{ph}")', '  socket_send_byte(10)']

    L = [
        'def ftest():',
        f'  socket_open("{HOST_IP}", {CB_PORT})',
        _init_target_urscript(tilt_deg).rstrip("\n"),          # position at init (NOT recorded)
        f'  movel(t, a={ALIGN_ACC:.4f}, v={ALIGN_VEL:.4f})',
        '  thread streamer():',
        '    while (True):',
        '      socket_send_string(to_str(get_actual_tcp_pose()))',
        '      socket_send_byte(10)',
        f'      sleep({dt:.4f})',
        '    end',
        '  end',
        '  gh = run streamer()',
        '  sleep(0.3)',
    ]
    L += mark(1)                                               # phase 1: linear descent
    L += [f'  t[2] = {target_z:.5f}',
          f'  movel(t, a={DOWN_ACC:.4f}, v={lin_vel:.5f})']
    L += mark(2)                                               # phase 2: static hold
    L += [f'  sleep({STATIC_PHASE_S:.2f})']
    def moves(pts, acc, vel):
        # offsets are parenthesized: URScript rejects "a + -b" as a syntax error
        out = []
        for dx, dy, z in pts:
            out += [f'  t[0] = base[0] + ({dx:.5f})',
                    f'  t[1] = base[1] + ({dy:.5f})',
                    f'  t[2] = {z:.5f}',
                    f'  movel(t, a={acc:.4f}, v={vel:.4f})']
        return out

    L += mark(3)                                               # phase 3: slow random
    L += ['  base = get_actual_tcp_pose()']
    L += moves(slow_pts, RAND2_ACC, RAND2_VEL)
    L += mark(4)                                               # phase 4: rapid random
    L += moves(fast_pts, RAND_ACC, RAND_VEL)
    L += mark(0)                                               # done: gentle return, cut
    L += ['  t[0] = base[0]',
          '  t[1] = base[1]',
          f'  t[2] = {home_z:.5f}',
          f'  movel(t, a={DOWN_ACC:.4f}, v={DOWN_VEL:.4f})',
          '  kill gh',
          '  socket_close()',
          'end',
          'ftest()',
          '']
    return "\n".join(L)


def _ftest_live_plot(live, stop_event):
    """Live viewer for cmd_filtertest: raw vs filtered vs robot-Z truth forming in
    real time, phase boundaries marked, velocity estimate below. `live` is the
    append-only dict the acquisition thread fills; list appends are atomic under
    the GIL, so reading a consistent prefix needs no lock. Redraws are driven by a
    plt.pause() polling loop — FuncAnimation on the macosx backend dies with a
    native SIGTRAP (py3.14 / mpl3.11). Blocks until the window is closed (the run
    itself keeps going regardless)."""
    import matplotlib.pyplot as plt

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, sharex=True, figsize=(11, 8.5),
                                        height_ratios=[3, 1, 1])
    try:
        fig.canvas.manager.set_window_title("filtertest — live")
    except AttributeError:
        pass                                            # headless backends
    names = list(live["est"].keys())               # one trace set per pipeline
    multi = len(names) > 1
    COLORS = ["tab:blue", "tab:red", "tab:cyan", "tab:olive"]
    ln_raw, = ax1.plot([], [], ".", ms=3, color="0.6", label="raw (central 2x2 median)")
    ln_tru, = ax1.plot([], [], "--", lw=1.2, color="tab:green", label="robot Z truth")
    ln_est, ln_vel, ln_err = {}, {}, {}
    for i, nm in enumerate(names):
        c = COLORS[i % len(COLORS)]
        ln_est[nm], = ax1.plot([], [], "-", lw=1.5, color=c,
                               label=f"filtered ({nm})" if multi else "filtered")
        ln_vel[nm], = ax2.plot([], [], "-", lw=1.2, color=c,
                               label=nm if multi else None)
        ln_err[nm], = ax3.plot([], [], "-", lw=1.2, color=c,
                               label=nm if multi else None)
    ax1.set_ylabel("distance above table (mm)")
    ax1.legend(loc="upper right", fontsize=8)
    ax1.grid(alpha=0.3)
    ax2.axhline(0, color="0.8", lw=0.8)
    ax2.set_ylabel("velocity est (mm/s)")
    ax2.grid(alpha=0.3)
    ax3.axhline(0, color="0.8", lw=0.8)
    ax3.set_ylabel("filtered − truth (mm)")
    ax3.set_xlabel("time (s)")
    ax3.grid(alpha=0.3)
    if multi:
        ax3.legend(loc="upper right", fontsize=8)
    fig.suptitle("filtertest — waiting for robot ...")

    PHASES = {1: ("linear", "tab:blue"), 2: ("static", "tab:orange"),
              3: ("random slow", "tab:purple"), 4: ("random fast", "tab:red")}
    marked = set()

    def _redraw():
        m = min([len(live[k]) for k in ("t", "raw", "truth", "phase")]
                + [len(v) for v in live["est"].values()]
                + [len(v) for v in live["vel"].values()])
        if m == 0:
            return
        t = live["t"][:m]
        raw, tru = live["raw"][:m], live["truth"][:m]
        ph = live["phase"][:m]
        ln_raw.set_data([x for x, r in zip(t, raw) if r is not None],
                        [r for r in raw if r is not None])
        ln_tru.set_data(t, tru)
        for nm in names:
            est = live["est"][nm][:m]                  # NaN before init draws as a gap
            ln_est[nm].set_data(t, est)
            ln_vel[nm].set_data(t, live["vel"][nm][:m])
            ln_err[nm].set_data(t, [e - g for e, g in zip(est, tru)])
        for p in set(ph) - marked:                     # mark each phase start once
            if p in PHASES:
                name, c = PHASES[p]
                x0 = t[ph.index(p)]
                for ax in (ax1, ax2, ax3):
                    ax.axvline(x0, color=c, lw=1, ls=":")
                ax1.annotate(f" {name}", xy=(x0, 0.98),
                             xycoords=("data", "axes fraction"),
                             color=c, fontsize=9, va="top")
                marked.add(p)
        for ax in (ax1, ax2, ax3):
            ax.relim()
            ax.autoscale_view()
        fig.suptitle("filtertest — DONE (close window for the summary)"
                     if stop_event.is_set() else "filtertest — live")

    plt.show(block=False)
    if plt.get_backend().lower() in ("agg", "pdf", "ps", "svg", "template"):
        _redraw()                       # headless (tests): draw once and return
        return fig
    while plt.fignum_exists(fig.number):
        _redraw()
        plt.pause(0.1)                  # runs the GUI event loop between redraws
    return fig


def cmd_offset(run_dir):
    """Compute a surface's constant offset from a recorded run: mean of
    (raw central-2x2 median − robot-Z truth) over slow samples only. Reads the
    raw tof_log/robot_log pair, so it works regardless of what offset (if any)
    the run itself used. Paste the result into SURFACE_OFFSETS_MM."""
    import statistics
    tof = list(csv.reader(open(os.path.join(run_dir, TOF_CSV))))[1:]
    rob = list(csv.reader(open(os.path.join(run_dir, ROBOT_CSV))))[1:]
    rows = []
    for trow, rrow in zip(tof, rob):
        vals = [float(trow[1 + i]) for i in FT_CENTRAL if float(trow[1 + i]) > 0]
        if vals:
            rows.append((statistics.median(vals),                # raw, no offset
                         float(rrow[3]) - TABLE_Z_MM,            # truth
                         float(trow[0])))
    diffs = []
    for i in range(1, len(rows) - 1):        # slow samples only (sync slop)
        dt = rows[i + 1][2] - rows[i - 1][2]
        if dt > 0 and abs((rows[i + 1][1] - rows[i - 1][1]) / dt) <= 100.0:
            diffs.append(rows[i][0] - rows[i][1])
    if len(diffs) < 30:
        print(f"{run_dir}: only {len(diffs)} slow samples — record a longer round")
        return None
    m = statistics.fmean(diffs)
    print(f"{run_dir}: {len(diffs)} slow samples, "
          f"mean(raw − truth) = {m:+.2f} mm (std {statistics.pstdev(diffs):.2f})")
    print(f"  -> SURFACE_OFFSETS_MM entry: {m:.1f}")
    return m


def cmd_filtertest(seed=None, show_script=False, viz=False,
                   make_filter=None, transform=None, out_base="filtertest",
                   pipelines=None, surface=None, tilt_deg=0.0):
    """Four-phase live-filter accuracy test with the Kalman filter running in
    REAL TIME on the ToF stream while the UR5 moves:
      1) 10 s constant-velocity linear descent
      2) 10 s static hold at table+clearance
      3) ~10 s slow random movels in a bounded box above the table
      4) ~10 s rapid random movels in the same box
    Each sensor frame -> central-2x2 median - MOUNT_OFFSET_MM
    -> live_filter.LiveFilter.update(z, t). Robot Z (minus TABLE_Z_MM) is ground
    truth. Saves A3/filtertest/<n>/{robot_log,tof_log,filter_log,waypoints}.csv
    and prints per-phase raw-vs-filtered error stats. viz=True opens a live
    matplotlib window with the traces forming in real time (logging unaffected).

    Pipeline injection (used by A2_learned_calibration/run_test.py):
      transform   : callable central-median-mm -> measurement fed to the filter
                    (default: subtract MOUNT_OFFSET_MM)
      make_filter : zero-arg callable returning a LiveFilter-compatible object
                    (default: LiveFilter(FT_ACCEL_PSD, adaptive R))
      out_base    : results folder name under A3, or an absolute path
      pipelines   : list of (name, transform, make_filter) to run SEVERAL
                    pipelines side by side on the same frame stream (overrides
                    transform/make_filter; wide filter_log.csv, one trace set
                    per pipeline in the live plot, one summary table each)
      surface     : SURFACE_OFFSETS_MM key selecting the constant used by the
                    default transform (default: DEFAULT_SURFACE)"""
    import random as _random
    import statistics
    import tof_logger
    sys.path.insert(0, os.path.join(os.path.dirname(RESULTS_DIR), "A2_data_filter"))
    from live_filter import LiveFilter

    surface = surface or DEFAULT_SURFACE
    offset = SURFACE_OFFSETS_MM.get(surface)
    if offset is None:
        offset = SURFACE_OFFSETS_MM[DEFAULT_SURFACE]
        print(f"NOTE: no constant stored for surface '{surface}' — default pipeline "
              f"uses the '{DEFAULT_SURFACE}' constant ({offset} mm). After the run:\n"
              f"  python3 robot.py offset <run_dir>   then add '{surface}' to "
              f"SURFACE_OFFSETS_MM.")

    def _waypoints(s):
        rng = _random.Random(s)
        return (_gen_random_waypoints(rng, RAND_PHASE_S, RAND_VEL, RAND_ACC),
                _gen_random_waypoints(rng, RAND2_PHASE_S, RAND2_VEL, RAND2_ACC))

    if show_script:
        print(_filtertest_urscript(*_waypoints(seed or 0), tilt_deg=tilt_deg))
        return True

    run_n, run_dir = _next_run_dir(out_base)
    seed = run_n if seed is None else seed
    fast_pts, slow_pts = _waypoints(seed)
    tilt_note = f", tilt {tilt_deg:g} deg" if tilt_deg else ""
    print(f"filtertest run {run_n}  (seed {seed}, surface '{surface}' offset "
          f"{offset} mm{tilt_note}, {len(fast_pts)} fast + {len(slow_pts)} slow "
          f"random waypoints)  ->  {run_dir}/")

    with open(os.path.join(run_dir, "waypoints.csv"), "w", newline="") as wf:
        ww = csv.writer(wf)
        ww.writerow(["phase", "dx_m", "dy_m", "z_abs_m", "seed"])
        for ph, pts in ((3, slow_pts), (4, fast_pts)):
            for i, (dx, dy, z) in enumerate(pts):
                ww.writerow([ph, f"{dx:.5f}", f"{dy:.5f}", f"{z:.5f}",
                             seed if ph == 3 and i == 0 else ""])

    # open the sensor FIRST so it's booted and streaming before the robot moves
    try:
        dev = tof_logger.open_sensor()
    except Exception as e:
        print("Sensor open failed:", e)
        return False

    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind(("0.0.0.0", CB_PORT)); srv.listen(1); srv.settimeout(60)
    send(_filtertest_urscript(fast_pts, slow_pts, tilt_deg=tilt_deg))
    print("URScript sent; waiting for the robot to connect back (60 s max) ...")

    state = {"pose": None, "phase": None}
    stop_event = threading.Event()
    pipes = pipelines or [("filter", transform, make_filter)]
    multi = len(pipes) > 1
    rows = []                          # (phase, [(z, est) per pipeline], truth)
    live = {"t": [], "raw": [], "truth": [], "phase": [],
            "est": {nm: [] for nm, _, _ in pipes},   # append-only, read by the
            "vel": {nm: [] for nm, _, _ in pipes}}   # live plot

    def run_loop():
        """Robot handshake + acquire + filter + log until the robot cuts the stream.
        Runs inline normally, or in a worker thread under viz (so the window opens
        immediately instead of blocking on the robot connection)."""
        try:
            conn, _ = srv.accept()
        except socket.timeout:
            print("ERROR: the robot never connected back. Check the pendant for a "
                  "program/compile error popup, that the robot is in Remote Control "
                  f"with no protective stop, and that this Mac is {HOST_IP}. "
                  "(Try:  python3 robot.py status  /  python3 robot.py filtertest print)")
            stop_event.set()
            srv.close(); dev.close()
            return
        srv.close()
        print("robot connected; recording ...")

        pose_hist = []          # (t_abs, pose) at the full 60 Hz stream rate —
                                # dumped to pose_log.csv so analysis can
                                # INTERPOLATE truth to each frame's timestamp
                                # instead of pairing with a stale latest-pose

        def robot_reader():
            """Keep the latest pose + phase fresh; stop everything when the stream ends."""
            conn.settimeout(30)
            buf = b""
            while True:
                try:
                    chunk = conn.recv(4096)
                except socket.timeout:
                    break
                if not chunk:
                    break
                buf += chunk
                while b"\n" in buf:
                    raw, buf = buf.split(b"\n", 1)
                    line = raw.decode(errors="ignore").strip()
                    if line.startswith("PH,"):
                        try:
                            state["phase"] = int(line[3:])
                        except ValueError:
                            pass
                        continue
                    p = parse(line)
                    if len(p) >= 6:
                        state["pose"] = p
                        pose_hist.append((time.perf_counter(), p))
            stop_event.set()

        reader = threading.Thread(target=robot_reader, daemon=True)
        reader.start()

        # wait for the first streamed pose (robot finishes positioning, then streams)
        while state["pose"] is None and not stop_event.is_set():
            time.sleep(0.005)
        dev.reset_input_buffer()   # drop frames buffered during positioning

        def _default_filter():
            return LiveFilter(process_accel_psd=FT_ACCEL_PSD,
                              adapt_measurement_var=True)  # learn sensor noise online
        filts = [(nm, tr, (mk() if mk else _default_filter()))
                 for nm, tr, mk in pipes]
        t0 = time.perf_counter()
        robot_path  = os.path.join(run_dir, ROBOT_CSV)
        tof_path    = os.path.join(run_dir, TOF_CSV)
        filter_path = os.path.join(run_dir, "filter_log.csv")
        with open(robot_path, "w", newline="") as rf, open(tof_path, "w", newline="") as tf, \
             open(filter_path, "w", newline="") as ff:
            rw = csv.writer(rf)
            rw.writerow(["time_s", "x_mm", "y_mm", "z_mm", "rx_rad", "ry_rad", "rz_rad"])
            tw = csv.writer(tf)
            # z* columns stay in positions 1..64 (analysis code indexes them);
            # signal-rate / sigma columns append after, blank on old firmware
            tw.writerow(["time_s"] + [f"z{i}" for i in range(tof_logger.N_ZONES)]
                        + [f"s{i}" for i in range(tof_logger.N_ZONES)]
                        + [f"q{i}" for i in range(tof_logger.N_ZONES)]
                        + [f"a{i}" for i in range(tof_logger.N_ZONES)])
            fw = csv.writer(ff)
            if multi:      # wide format: one raw/filt/vel column set per pipeline
                fw.writerow(["time_s", "phase", "robot_z_mm", "truth_mm"]
                            + [c for nm, _, _ in pipes
                               for c in (f"raw_{nm}", f"filt_{nm}", f"vel_{nm}")])
            else:
                fw.writerow(["time_s", "phase", "raw_mm", "filtered_mm", "vel_mm_s",
                             "robot_z_mm", "truth_mm", "downweighted", "maneuver"])
            while not stop_event.is_set():
                full = tof_logger.read_frame_full(dev)
                if full is None:
                    continue
                frame, sig, sigma, amb = full     # extras None on old firmware
                p, ph = state["pose"], state["phase"]
                if p is None or ph is None or ph == 0:
                    continue                                  # positioning / return leg: skip
                t = time.perf_counter() - t0
                vals = [frame[i] for i in FT_CENTRAL if frame[i] > 0]
                med = statistics.median(vals) if vals else None   # None = dropped frame
                svals = ([sig[i] for i in FT_CENTRAL if sig[i] > 0] if sig else [])
                sig_med = statistics.median(svals) if svals else None
                truth = p[2] * 1000.0 - TABLE_Z_MM
                ests = []
                for nm, tr, f in filts:
                    # per-pipeline measurement (default: surface offset), then
                    # filter; transforms take (median_mm, signal) — per-surface
                    # models ignore the signal, the generic model uses it
                    z = None if med is None else (tr(med, sig_med) if tr
                                                  else med - offset)
                    ests.append((nm, z, f.update(z, t), f))
                tw.writerow([f"{t:.4f}"] + frame + (sig or [""] * len(frame))
                            + (sigma or [""] * len(frame))
                            + (amb or [""] * len(frame)))
                rw.writerow([f"{t:.4f}", f"{p[0]*1000:.3f}", f"{p[1]*1000:.3f}",
                             f"{p[2]*1000:.3f}", f"{p[3]:.5f}", f"{p[4]:.5f}", f"{p[5]:.5f}"])
                if multi:
                    fw.writerow([f"{t:.4f}", ph, f"{p[2]*1000:.3f}", f"{truth:.2f}"]
                                + [c for nm, z, est, f in ests
                                   for c in ("" if z is None else f"{z:.1f}",
                                             f"{est:.2f}", f"{f.velocity:.1f}")])
                else:
                    nm, z, est, f = ests[0]
                    fw.writerow([f"{t:.4f}", ph, "" if z is None else f"{z:.1f}",
                                 f"{est:.2f}", f"{f.velocity:.1f}", f"{p[2]*1000:.3f}",
                                 f"{truth:.2f}", int(f.downweighted), int(f.maneuver)])
                rows.append((ph, [(z, est) for nm, z, est, f in ests], truth))
                live["t"].append(t)
                live["raw"].append(ests[0][1])
                live["truth"].append(truth)
                for nm, z, est, f in ests:
                    live["est"][nm].append(est)
                    live["vel"][nm].append(f.velocity)
                live["phase"].append(ph)

        dev.close()
        conn.close()
        reader.join(timeout=3)

        # full-rate pose stream, timestamps on the same clock as the frame rows
        # (negative t = poses received while waiting at the top, kept so the
        # first frames still have a bracketing sample to interpolate against)
        with open(os.path.join(run_dir, "pose_log.csv"), "w", newline="") as pf:
            pw = csv.writer(pf)
            pw.writerow(["time_s", "x_mm", "y_mm", "z_mm", "rx_rad", "ry_rad", "rz_rad"])
            for t_abs, p in pose_hist:
                pw.writerow([f"{t_abs - t0:.4f}", f"{p[0]*1000:.3f}", f"{p[1]*1000:.3f}",
                             f"{p[2]*1000:.3f}", f"{p[3]:.5f}", f"{p[4]:.5f}", f"{p[5]:.5f}"])

    if viz:
        # matplotlib must own the main thread (hard requirement on macOS), so the
        # acquisition loop moves to a worker; logging is identical either way.
        worker = threading.Thread(target=run_loop, daemon=True)
        worker.start()
        _ftest_live_plot(live, stop_event)   # blocks until the window is closed
        worker.join()                        # run continues even if window closed early
    else:
        run_loop()

    print(f"Done (run {run_n}).  {len(rows)} frame-locked rows  ->  {run_dir}/")

    # per-phase accuracy summary (RMSE includes any calibration bias; the
    # bias-removed std shows pure noise suppression)
    def _stats(errs):
        m = statistics.fmean(errs)
        rmse = math.sqrt(statistics.fmean([e * e for e in errs]))
        std = statistics.pstdev(errs)
        return rmse, m, std

    for pi, (pname, _, _) in enumerate(pipes):
        title = f"  [{pname}]" if multi else ""
        print(f"\n{'phase':<10}{'n':>6}{'raw RMSE':>10}{'filt RMSE':>11}"
              f"{'raw std':>9}{'filt std':>9}   (mm, vs robot Z truth){title}")
        for ph, name in ((1, "linear"), (2, "static"), (3, "rand-slow"), (4, "rand-fast")):
            sel = [(pe[pi][0], pe[pi][1], g) for (pp, pe, g) in rows
                   if pp == ph and pe[pi][0] is not None and math.isfinite(pe[pi][1])]
            if not sel:
                print(f"{name:<10}{0:>6}   (no samples)")
                continue
            raw_rmse, _, raw_std = _stats([r - g for r, e, g in sel])
            flt_rmse, _, flt_std = _stats([e - g for r, e, g in sel])
            print(f"{name:<10}{len(sel):>6}{raw_rmse:>10.2f}{flt_rmse:>11.2f}"
                  f"{raw_std:>9.2f}{flt_std:>9.2f}")
    return len(rows) > 0


def cmd_pose():
    show("Current pose (base frame):", get_pose())

def cmd_status():
    print("Robotmode: ", dashboard("robotmode"))
    print("Safety:    ", dashboard("safetystatus"))

def cmd_table():
    p = get_pose()
    print(f"\nTABLE_Z_MM = {p[2]*1000:.2f}   <- paste into SETTINGS at top of this file")

def cmd_wrist(deg):
    us = ('def r():\n'
          '  q = get_actual_joint_positions()\n'
          '  q[5] = q[5] + %.6f\n' % (deg * math.pi / 180.0) +
          '  movej(q, a=%.4f, v=%.4f)\n' % (WRIST_ACC, WRIST_VEL) +
          'end\nr()\n')
    send(us)
    print(f"Rotated wrist 3 by {deg:+.1f} deg.")


USAGE = """\
Usage:
  python3 robot.py            full routine: rise to init spot, then descend to clearance
  python3 robot.py record [N] run N smooth rounds -> A3/smooth/<n>/{robot_log,tof_log}.csv
  python3 robot.py steps  [N] run N rounds, DOWN/UP in N_STEPS equal steps -> A3/steps/<n>/
  python3 robot.py filtertest [seed]   live Kalman-filter test: 10s linear + 10s static
                              + ~10s slow random + ~10s rapid random motion, filter
                              running on the ToF stream in real time -> A3/filtertest/<n>/
  python3 robot.py filtertest viz [seed]   same, plus a live matplotlib window with
                              raw / filtered / truth traces forming in real time
  python3 robot.py filtertest <surface>    use that surface's constant from
                              SURFACE_OFFSETS_MM (e.g. white, wood); combinable
  python3 robot.py filtertest print    print the generated URScript (no motion)
  python3 robot.py offset <run_dir>    compute a surface's constant offset from a
                              recorded run (raw central median vs robot truth)
  python3 robot.py up         rise to perpendicular init spot only (no descent)
  python3 robot.py pose       print current end-effector pose (base frame)
  python3 robot.py status     robot mode + safety status
  python3 robot.py table      print TABLE_Z_MM (freedrive sensor to touch table first)
  python3 robot.py wrist 45   spin wrist 3 by N degrees (repeat to straighten sensor)
"""

if __name__ == "__main__":
    a = sys.argv[1:]
    if   not a:                             cmd_run()
    elif a[0] == "record":
        nums = [int(x) for x in a[1:] if x.isdigit()]
        cmd_record_batch(nums[0] if nums else 1)
    elif a[0] == "steps":
        nums = [int(x) for x in a[1:] if x.isdigit()]
        cmd_record_batch(nums[0] if nums else 1, stepped=True)
    elif a[0] == "filtertest":
        if "print" in a[1:]:
            cmd_filtertest(show_script=True)
        else:
            nums = [int(x) for x in a[1:] if x.isdigit()]
            words = [x for x in a[1:] if not x.isdigit() and x not in ("viz", "print")]
            cmd_filtertest(nums[0] if nums else None, viz="viz" in a[1:],
                           surface=words[0] if words else None)
    elif a[0] == "offset" and len(a) > 1:
        cmd_offset(a[1])
    elif a[0] == "up":                      cmd_up()
    elif a[0] == "pose":                    cmd_pose()
    elif a[0] == "status":                  cmd_status()
    elif a[0] == "table":                   cmd_table()
    elif a[0] == "wrist" and len(a) > 1:    cmd_wrist(float(a[1]))
    else:                                   print(USAGE)
