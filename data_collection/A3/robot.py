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


# ---------------------------------------------------------------------------
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


# ---------------------------------------------------------------------------
#  COMMANDS
# ---------------------------------------------------------------------------
def _init_target_urscript():
    """URScript snippet that builds pose `t` = (current X,Y at HOME_Z) with the
    perpendicular + sensor-straight orientation (the 45 deg is baked in here, so
    the arm reaches the top already correctly oriented)."""
    home_z = HOME_Z_MM / 1000.0
    woff   = WRIST_OFFSET_DEG * math.pi / 180.0
    return (
        '  t = get_actual_tcp_pose()\n'
        '  t[2] = %.5f\n' % home_z +                                # init height
        '  t[3] = 3.14159265\n  t[4] = 0.0\n  t[5] = 0.0\n'          # perpendicular (tool down)
        '  t = pose_trans(t, p[0,0,0,0,0,%.6f])\n' % woff           # +45 deg about tool Z -> sensor straight
    )


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
        "  i = 0",                                             # --- DOWN in steps ---
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
        "  i = 0",                                             # --- UP in steps ---
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
    """Create and return (n, path) for the next numbered run folder under A3/<profile>/."""
    base = os.path.join(RESULTS_DIR, profile)
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
    elif a[0] == "up":                      cmd_up()
    elif a[0] == "pose":                    cmd_pose()
    elif a[0] == "status":                  cmd_status()
    elif a[0] == "table":                   cmd_table()
    elif a[0] == "wrist" and len(a) > 1:    cmd_wrist(float(a[1]))
    else:                                   print(USAGE)
