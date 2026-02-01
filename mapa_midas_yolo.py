import cv2
import torch
import time
import math
import numpy as np
from pathlib import Path



# CUDA / diagnostyka

FORCE_CUDA = True          
USE_MIDAS_AMP = True
PRINT_DEVICE_INFO = True


USE_OPENCV_CUDA_FOR_YOLO = False



# YOLOv4-tiny (OpenCV DNN)

YOLO_CFG = "yolov4-tiny.cfg"
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_NAMES = "coco.names"

CONF_THRESH = 0.4
NMS_THRESH = 0.4

TARGET_LONG_SIDE = 256



# MiDaS

MIDAS_WEIGHTS = "midas_v21_small_256.pt"

POLY_COEFFS = [-1299.9, 4778, -6772.8, 4690.1, -1658.6, 313.91]

INNER_BOX_FRACTION_W = 0.15
INNER_BOX_FRACTION_H = 0.15

DYNAMIC_MARGIN = 1.5


# Undystorcja
CALIB_NPZ = "calibration_data.npz"  # zawiera: K (3x3), dist (1xN)


HFOV_DEG = 105.0

# mapa
MAP_SIZE_M = 4.0
MAP_RES_M = 0.02          
MAP_BG = 25
GRID_STEP_M = 0.5
MAP_VIEW_SCALE = 3

MIN_OBJ_THICK_M = 0.08
MAX_OBJ_THICK_M = 0.40
THICK_FROM_WIDTH_GAIN = 0.35

SHOW_WIDTH_IN_CM = True


def load_calibration_npz(npz_path: str):
    data = np.load(npz_path)
    K = data["K"].astype(np.float64)
    dist = data["dist"].astype(np.float64).reshape(-1)
    return K, dist


def rel_to_cm(x: float) -> float:
    a5, a4, a3, a2, a1, a0 = POLY_COEFFS
    return (((((a5 * x + a4) * x + a3) * x + a2) * x + a1) * x + a0)


def normalize01(depth: np.ndarray, invert: bool = False) -> np.ndarray:
    dmin = float(depth.min())
    dmax = float(depth.max())
    scale = dmax - dmin
    if scale < 1e-8:
        d01 = np.zeros_like(depth, dtype=np.float32)
    else:
        d01 = (depth - dmin) / (scale + 1e-8)
    return 1.0 - d01 if invert else d01


def scale_camera_matrix(K: np.ndarray, sx: float, sy: float) -> np.ndarray:
    Ks = K.copy().astype(np.float64)
    Ks[0, 0] *= sx
    Ks[1, 1] *= sy
    Ks[0, 2] *= sx
    Ks[1, 2] *= sy
    return Ks

def fx_from_hfov(frame_w: int, hfov_deg: float) -> float:
    hfov = math.radians(hfov_deg)
    return (frame_w / 2.0) / math.tan(hfov / 2.0)


def pixel_to_x_at_depth_hfov(u_px: float, z_m: float, frame_w: int, hfov_deg: float) -> float:
    """
    x = ((u - cx)/fx) * Z
    gdzie cx = srodek obrazu, fx z HFOV.
    """
    cx = (frame_w - 1) / 2.0
    fx = fx_from_hfov(frame_w, hfov_deg)
    return ((u_px - cx) / fx) * z_m


def compute_global_motion(prev_gray: np.ndarray, gray: np.ndarray) -> float:
    surf = None
    try:
        surf = cv2.xfeatures2d.SURF_create(hessianThreshold=400)
    except Exception:
        pass

    if surf is not None:
        kp1, des1 = surf.detectAndCompute(prev_gray, None)
        kp2, des2 = surf.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
    else:
        orb = cv2.ORB_create(500)
        kp1, des1 = orb.detectAndCompute(prev_gray, None)
        kp2, des2 = orb.detectAndCompute(gray, None)
        if des1 is None or des2 is None or len(kp1) < 10 or len(kp2) < 10:
            return 0.0
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    matches = bf.knnMatch(des1, des2, k=2)
    good = []
    for pair in matches:
        if len(pair) < 2:
            continue
        m, n = pair
        if m.distance < 0.6 * n.distance:
            good.append(m)

    if len(good) < 10:
        return 0.0

    pts1 = np.float32([kp1[m.queryIdx].pt for m in good])
    pts2 = np.float32([kp2[m.trainIdx].pt for m in good])

    M, _ = cv2.estimateAffinePartial2D(
        pts1, pts2, method=cv2.RANSAC, ransacReprojThreshold=3.0
    )
    if M is None:
        return 0.0

    ones = np.ones((pts1.shape[0], 1), dtype=np.float32)
    pts1_h = np.hstack([pts1, ones])
    pts1_trans = (M @ pts1_h.T).T
    flow_comp = pts1_trans - pts1
    return float(np.mean(np.linalg.norm(flow_comp, axis=1)))


def run_yolo_v4_tiny_dynamic_static(frame_bgr, prev_gray, net, output_layers, class_names):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1 / 255.0, (416, 416), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes, confidences, class_ids = [], [], []

    for out in outs:
        for det in out:
            scores = det[5:]
            class_id = int(np.argmax(scores))
            conf = float(scores[class_id])
            if conf < CONF_THRESH:
                continue

            cx, cy, bw, bh = det[0:4]
            cx *= w
            cy *= h
            bw *= w
            bh *= h
            x = int(cx - bw / 2)
            y = int(cy - bh / 2)

            boxes.append([x, y, int(bw), int(bh)])
            confidences.append(conf)
            class_ids.append(class_id)

    idxs = cv2.dnn.NMSBoxes(boxes, confidences, CONF_THRESH, NMS_THRESH)
    annotated = frame_bgr.copy()

    if len(idxs) == 0:
        return [], [], [], annotated

    idxs = idxs.flatten()
    gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

    global_motion = 0.0
    if prev_gray is not None:
        global_motion = compute_global_motion(prev_gray, gray)

    final_boxes, final_labels, dyn_flags = [], [], []

    for i in idxs:
        x, y, bw, bh = boxes[i]
        x1 = max(0, x)
        y1 = max(0, y)
        x2 = min(w - 1, x1 + bw)
        y2 = min(h - 1, y1 + bh)
        if x2 <= x1 or y2 <= y1:
            continue

        label = class_names[class_ids[i]] if 0 <= class_ids[i] < len(class_names) else str(class_ids[i])

        mean_obj_motion = 0.0
        is_dynamic = False

        if prev_gray is not None:
            roi_prev = prev_gray[y1:y2, x1:x2]
            pts_obj = cv2.goodFeaturesToTrack(roi_prev, maxCorners=200, qualityLevel=0.01, minDistance=3)
            if pts_obj is not None:
                pts_obj = pts_obj.reshape(-1, 2)
                pts_obj_full = pts_obj + np.array([x1, y1], dtype=np.float32)

                pts_curr, st2, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, pts_obj_full, None,
                    winSize=(21, 21), maxLevel=3,
                    criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
                )

                st2 = st2.reshape(-1)
                good_prev = pts_obj_full[st2 == 1]
                good_curr = pts_curr[st2 == 1]
                if len(good_prev) >= 5:
                    disp = np.linalg.norm(good_curr - good_prev, axis=1)
                    mean_obj_motion = float(np.mean(disp))

            is_dynamic = mean_obj_motion > (global_motion + DYNAMIC_MARGIN)

        color = (0, 0, 255) if is_dynamic else (0, 255, 0)
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, 2)
        cv2.putText(
            annotated, label, (x1, max(0, y1 - 5)),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA
        )

        final_boxes.append((x1, y1, x2, y2))
        final_labels.append(label)
        dyn_flags.append(is_dynamic)

    return final_boxes, final_labels, dyn_flags, annotated


def run_midas(frame_bgr, midas, transform, device, invert=False):
    h, w = frame_bgr.shape[:2]
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

    inp = transform(frame_rgb).to(device, non_blocking=True)

    with torch.no_grad():
        if device.type == "cuda" and USE_MIDAS_AMP:
            with torch.amp.autocast(device_type="cuda"):
                pred = midas(inp)
        else:
            pred = midas(inp)

        pred = torch.nn.functional.interpolate(
            pred.unsqueeze(1), size=(h, w),
            mode="bicubic", align_corners=False
        ).squeeze(1)

    depth_raw = pred[0].detach().float().cpu().numpy().astype(np.float32)
    depth_rel = normalize01(depth_raw, invert=invert)
    depth_u8 = (depth_rel * 255.0).astype(np.uint8)
    depth_color = cv2.applyColorMap(depth_u8, cv2.COLORMAP_MAGMA)
    return depth_rel, depth_color


def world_to_map_px(x_m: float, y_m: float, half_m: float, res_m: float):
    size_px = int(round((2.0 * half_m) / res_m))
    px = int(round((x_m + half_m) / res_m))
    py = int(round((half_m - y_m) / res_m))
    px = max(0, min(size_px - 1, px))
    py = max(0, min(size_px - 1, py))
    return px, py


def draw_occupancy_base(hfov_deg: float) -> np.ndarray:
    half_m = MAP_SIZE_M / 2.0
    size_px = int(round(MAP_SIZE_M / MAP_RES_M))
    img = np.full((size_px, size_px, 3), MAP_BG, dtype=np.uint8)

    cam_px = size_px // 2
    cam_py = size_px // 2

    # siatka
    step_px = int(round(GRID_STEP_M / MAP_RES_M))
    if step_px > 0:
        for x in range(0, size_px, step_px):
            cv2.line(img, (x, 0), (x, size_px - 1), (35, 35, 35), 1)
        for y in range(0, size_px, step_px):
            cv2.line(img, (0, y), (size_px - 1, y), (35, 35, 35), 1)

    # klin
    max_r = half_m
    a = math.radians(hfov_deg / 2.0)
    left_x = -math.tan(a) * max_r
    right_x = math.tan(a) * max_r
    left_pt = world_to_map_px(left_x, max_r, half_m, MAP_RES_M)
    right_pt = world_to_map_px(right_x, max_r, half_m, MAP_RES_M)
    fov_poly = np.array([[cam_px, cam_py], list(left_pt), list(right_pt)], dtype=np.int32)
    cv2.fillPoly(img, [fov_poly], (45, 45, 45))

    # k_i_k
    cv2.circle(img, (cam_px, cam_py), 4, (200, 200, 200), -1)
    cv2.line(img, (cam_px, cam_py), (cam_px, max(0, cam_py - int(0.5 / MAP_RES_M))), (200, 200, 200), 2)

    cv2.putText(img, "Occupancy 4x4m", (8, 18),
                cv2.FONT_HERSHEY_SIMPLEX, 0.55, (220, 220, 220), 1, cv2.LINE_AA)
    return img


def add_object_footprint(map_img: np.ndarray, xL: float, xR: float, z: float, label: str, is_dyn: bool):
    half_m = MAP_SIZE_M / 2.0

    if z <= 0.0 or z > half_m:
        return

    xL = max(-half_m, min(half_m, xL))
    xR = max(-half_m, min(half_m, xR))
    if xR <= xL:
        return

    width = xR - xL

    thick = THICK_FROM_WIDTH_GAIN * width
    thick = max(MIN_OBJ_THICK_M, min(MAX_OBJ_THICK_M, thick))

    y1 = max(-half_m, min(half_m, z - thick / 2.0))
    y2 = max(-half_m, min(half_m, z + thick / 2.0))
    if y2 <= y1:
        return

    p1 = world_to_map_px(xL, y1, half_m, MAP_RES_M)
    p2 = world_to_map_px(xR, y1, half_m, MAP_RES_M)
    p3 = world_to_map_px(xR, y2, half_m, MAP_RES_M)
    p4 = world_to_map_px(xL, y2, half_m, MAP_RES_M)
    poly = np.array([p1, p2, p3, p4], dtype=np.int32)

    color = (0, 0, 255) if is_dyn else (0, 255, 0)
    cv2.fillPoly(map_img, [poly], color)
    cv2.polylines(map_img, [poly], True, (0, 0, 0), 1)

    # label
    xc = (xL + xR) / 2.0
    yc = (y1 + y2) / 2.0
    tx, ty = world_to_map_px(xc, yc, half_m, MAP_RES_M)
    cv2.putText(map_img, label, (tx + 3, ty - 3),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (240, 240, 240), 1, cv2.LINE_AA)

    # szer
    top_x, top_y = world_to_map_px(xc, y2, half_m, MAP_RES_M)
    top_y = max(12, top_y - 6)

    if SHOW_WIDTH_IN_CM:
        w_txt = f"w={width * 100.0:.0f}cm"
    else:
        w_txt = f"w={width:.2f}m"

    cv2.putText(map_img, w_txt, (top_x - 12, top_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 0), 2, cv2.LINE_AA)
    cv2.putText(map_img, w_txt, (top_x - 12, top_y),
                cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)


def _assert_required_files():
    needed = [YOLO_CFG, YOLO_WEIGHTS, YOLO_NAMES, MIDAS_WEIGHTS, CALIB_NPZ]
    missing = [p for p in needed if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Brakuje plików: " + ", ".join(missing))


def main():
    torch.set_grad_enabled(False)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if FORCE_CUDA and device.type != "cuda":
        raise RuntimeError("CUDA nie jest dostępne. Uruchom w środowisku z torch CUDA.")

    if PRINT_DEVICE_INFO:
        print("TORCH DEVICE:", device)
        print("CUDA available:", torch.cuda.is_available())
        if torch.cuda.is_available():
            print("GPU:", torch.cuda.get_device_name(0))
            print("torch:", torch.__version__)

    _assert_required_files()

    K_base, dist = load_calibration_npz(CALIB_NPZ)

    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)

    if USE_OPENCV_CUDA_FOR_YOLO:
        try:
            net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
            net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)
            if PRINT_DEVICE_INFO:
                print("YOLO: ustawiono OpenCV DNN CUDA (FP16).")
        except Exception as e:
            print("YOLO: nie udało się włączyć CUDA w OpenCV DNN. Zostaje CPU.")
            print("Powód:", repr(e))

    ln = net.getLayerNames()
    output_layers = [ln[i - 1] for i in net.getUnconnectedOutLayers().flatten()]

    with open(YOLO_NAMES, "r", encoding="utf-8") as f:
        class_names = [c.strip() for c in f.readlines()]


    weights_path = Path(MIDAS_WEIGHTS).resolve()

    midas = torch.hub.load("intel-isl/MiDaS", "MiDaS_small", pretrained=False)

    sd = torch.load(str(weights_path), map_location=device)
    if isinstance(sd, dict) and "state_dict" in sd:
        sd = sd["state_dict"]
    sd = {k.replace("module.", "").replace("model.", ""): v for k, v in sd.items()}

    midas.load_state_dict(sd, strict=False)
    midas.to(device).eval()

    if PRINT_DEVICE_INFO:
        print("MiDaS param device:", next(midas.parameters()).device)
        print("MiDaS AMP:", USE_MIDAS_AMP and device.type == "cuda")

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Nie mogę otworzyć kamery.")


    ok, frame0 = cap.read()
    if not ok:
        raise RuntimeError("Nie mogę pobrać pierwszej klatki z kamery.")
    h0, w0 = frame0.shape[:2]

    scale = 1.0
    if max(h0, w0) > TARGET_LONG_SIDE:
        scale = TARGET_LONG_SIDE / float(max(h0, w0))

    new_w = int(round(w0 * scale))
    new_h = int(round(h0 * scale))
    sx = new_w / float(w0)
    sy = new_h / float(h0)

    def resize_frame(fr):
        if scale == 1.0:
            return fr
        return cv2.resize(fr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)

    K_scaled = scale_camera_matrix(K_base, sx, sy)

    newK, _ = cv2.getOptimalNewCameraMatrix(K_scaled, dist, (new_w, new_h), alpha=0.0, newImgSize=(new_w, new_h))
    map1, map2 = cv2.initUndistortRectifyMap(
        K_scaled, dist, R=None, newCameraMatrix=newK,
        size=(new_w, new_h), m1type=cv2.CV_16SC2
    )

    prev_gray = None
    prev_time = time.time()
    fps = 0.0

    while True:
        ok, frame_bgr = cap.read()
        if not ok:
            continue

        now = time.time()
        dt = now - prev_time
        if dt > 0:
            fps = 1.0 / dt
        prev_time = now

        frame_bgr = resize_frame(frame_bgr)

        frame_bgr = cv2.remap(frame_bgr, map1, map2, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

        depth_rel, depth_color = run_midas(frame_bgr, midas, transform, device, invert=False)

        bboxes, labels, dyn_flags, annotated_rgb = run_yolo_v4_tiny_dynamic_static(
            frame_bgr, prev_gray, net, output_layers, class_names
        )

        h, w = frame_bgr.shape[:2]
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)

        occ_map = draw_occupancy_base(HFOV_DEG)

        for (x1, y1, x2, y2), lab, is_dyn in zip(bboxes, labels, dyn_flags):
            inner_w = max(8, int((x2 - x1) * INNER_BOX_FRACTION_W))
            inner_h = max(8, int((y2 - y1) * INNER_BOX_FRACTION_H))
            cx_i = (x1 + x2) // 2
            cy_i = (y1 + y2) // 2

            ix1 = int(np.clip(cx_i - inner_w // 2, 0, w - 1))
            ix2 = int(np.clip(cx_i + inner_w // 2, 0, w - 1))
            iy1 = int(np.clip(cy_i - inner_h // 2, 0, h - 1))
            iy2 = int(np.clip(cy_i + inner_h // 2, 0, h - 1))

            patch = depth_rel[iy1:iy2, ix1:ix2]
            if patch.size == 0:
                continue

            rel_mean = float(np.mean(patch))
            dist_cm = float(rel_to_cm(rel_mean))
            if not np.isfinite(dist_cm) or dist_cm <= 0.0 or dist_cm > 2000.0:
                continue

            z_m = dist_cm / 100.0

            x_left_m = pixel_to_x_at_depth_hfov(float(x1), z_m, w, HFOV_DEG)
            x_right_m = pixel_to_x_at_depth_hfov(float(x2), z_m, w, HFOV_DEG)
            if x_right_m < x_left_m:
                x_left_m, x_right_m = x_right_m, x_left_m

            add_object_footprint(occ_map, x_left_m, x_right_m, z_m, lab, is_dyn)

            color = (0, 0, 255) if is_dyn else (0, 255, 0)
            yellow = (0, 255, 255)

            cv2.rectangle(depth_color, (x1, y1), (x2, y2), color, 2)
            cv2.rectangle(annotated_rgb, (ix1, iy1), (ix2, iy2), yellow, 1)
            cv2.rectangle(depth_color, (ix1, iy1), (ix2, iy2), yellow, 1)

            y_rel = max(0, y1 - 30)
            y_dist = max(0, y1 - 12)
            cv2.putText(annotated_rgb, f"rel={rel_mean:.3f}", (x1, y_rel),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
            cv2.putText(annotated_rgb, f"{dist_cm:.1f} cm", (x1, y_dist),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2, cv2.LINE_AA)

        combo = np.hstack((annotated_rgb, depth_color))
        cv2.putText(combo, f"FPS: {fps:.1f}", (10, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2, cv2.LINE_AA)
        cv2.imshow("RGB | Depth (undistort + YOLO + MiDaS)", combo)

        occ_vis = cv2.resize(
            occ_map,
            (occ_map.shape[1] * MAP_VIEW_SCALE, occ_map.shape[0] * MAP_VIEW_SCALE),
            interpolation=cv2.INTER_NEAREST
        )
        cv2.imshow("Occupancy Map 4x4m", occ_vis)

        prev_gray = gray.copy()

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord('q'), ord('Q')):
            break

    cap.release()
    print("new (W,H) =", new_w, new_h)
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()