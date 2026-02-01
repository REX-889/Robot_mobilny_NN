import cv2
import time
import argparse
import numpy as np
from pathlib import Path


YOLO_CFG = "yolov4-tiny.cfg"
YOLO_WEIGHTS = "yolov4-tiny.weights"
YOLO_NAMES = "coco.names"


CONF_THRESH = 0.4
NMS_THRESH = 0.4


INPUT_SIZE = 416


def assert_files():
    needed = [YOLO_CFG, YOLO_WEIGHTS, YOLO_NAMES]
    missing = [p for p in needed if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("Missing files: " + ", ".join(missing))


def load_class_names(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f.readlines()]


def get_output_layers(net):
    ln = net.getLayerNames()
    out_ids = net.getUnconnectedOutLayers()
    out_ids = out_ids.flatten() if len(out_ids.shape) > 1 else out_ids
    return [ln[i - 1] for i in out_ids]


def iou_xyxy(a, b):
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    inter_x1 = max(ax1, bx1)
    inter_y1 = max(ay1, by1)
    inter_x2 = min(ax2, bx2)
    inter_y2 = min(ay2, by2)
    iw = max(0, inter_x2 - inter_x1)
    ih = max(0, inter_y2 - inter_y1)
    inter = iw * ih
    area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
    area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
    union = area_a + area_b - inter + 1e-9
    return inter / union


def run_yolo_raw(frame_bgr, net, output_layers):
    h, w = frame_bgr.shape[:2]
    blob = cv2.dnn.blobFromImage(frame_bgr, 1 / 255.0, (INPUT_SIZE, INPUT_SIZE), swapRB=True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    boxes_xywh = []
    boxes_xyxy = []
    confs = []
    class_ids = []

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

            x = int(cx - bw / 2.0)
            y = int(cy - bh / 2.0)
            bw_i = int(bw)
            bh_i = int(bh)

            x1 = max(0, x)
            y1 = max(0, y)
            x2 = min(w - 1, x1 + bw_i)
            y2 = min(h - 1, y1 + bh_i)
            if x2 <= x1 or y2 <= y1:
                continue

            boxes_xywh.append([x1, y1, x2 - x1, y2 - y1])
            boxes_xyxy.append((x1, y1, x2, y2))
            confs.append(conf)
            class_ids.append(class_id)

    return boxes_xywh, boxes_xyxy, confs, class_ids


def apply_nms(boxes_xywh, confs):
    if not boxes_xywh:
        return []
    idxs = cv2.dnn.NMSBoxes(boxes_xywh, confs, CONF_THRESH, NMS_THRESH)
    if idxs is None or len(idxs) == 0:
        return []
    idxs = idxs.flatten().tolist()
    return idxs


def draw_boxes(img, boxes_xyxy, confs, class_ids, class_names, color, thickness=2):
    out = img.copy()
    for (x1, y1, x2, y2), c, cid in zip(boxes_xyxy, confs, class_ids):
        label = class_names[cid] if 0 <= cid < len(class_names) else str(cid)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, thickness)
        cv2.putText(out, f"{label} {c:.2f}", (x1, max(0, y1 - 6)),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", default="0", help="camera index (0) or path to video file")
    parser.add_argument("--show_iou_stats", action="store_true", help="print simple duplication stats")
    args = parser.parse_args()

    assert_files()

    class_names = load_class_names(YOLO_NAMES)

    net = cv2.dnn.readNetFromDarknet(YOLO_CFG, YOLO_WEIGHTS)
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    output_layers = get_output_layers(net)

    src = int(args.src) if args.src.isdigit() else args.src
    cap = cv2.VideoCapture(src)
    if not cap.isOpened():
        raise RuntimeError("Cannot open video source: " + str(args.src))

    prev_t = time.time()
    fps = 0.0
    paused = False

    while True:
        if not paused:
            ok, frame = cap.read()
            if not ok:
                break

            now = time.time()
            dt = now - prev_t
            prev_t = now
            if dt > 0:
                fps = 1.0 / dt

            boxes_xywh, boxes_xyxy, confs, class_ids = run_yolo_raw(frame, net, output_layers)
            idxs_nms = apply_nms(boxes_xywh, confs)

            raw_count = len(boxes_xyxy)
            nms_count = len(idxs_nms)
            suppressed = raw_count - nms_count

            view_raw = draw_boxes(frame, boxes_xyxy, confs, class_ids, class_names, color=(255, 0, 0), thickness=2)

            if nms_count > 0:
                kept_xyxy = [boxes_xyxy[i] for i in idxs_nms]
                kept_confs = [confs[i] for i in idxs_nms]
                kept_cids = [class_ids[i] for i in idxs_nms]
            else:
                kept_xyxy, kept_confs, kept_cids = [], [], []
            view_nms = draw_boxes(frame, kept_xyxy, kept_confs, kept_cids, class_names, color=(0, 255, 0), thickness=2)
es)
            dup_hits = 0
            if args.show_iou_stats and nms_count > 0 and raw_count > 0:
                for ki, kbox in zip(idxs_nms, kept_xyxy):
                    kcid = class_ids[ki]
                    for j, (rbox, rcid) in enumerate(zip(boxes_xyxy, class_ids)):
                        if j == ki:
                            continue
                        if rcid != kcid:
                            continue
                        if iou_xyxy(kbox, rbox) >= 0.5:
                            dup_hits += 1

            cv2.putText(view_raw, f"BEZ NMS | raw={raw_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(view_nms, f"NMS | nms={nms_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            cv2.putText(view_nms, f"FPS: {fps:.1f}", (10, 55),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)

            if args.show_iou_stats:
                cv2.putText(view_nms, f"dup_iou>=0.5(same class): {dup_hits}", (10, 85),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2, cv2.LINE_AA)

            h1, w1 = view_raw.shape[:2]
            h2, w2 = view_nms.shape[:2]
            if (h1 != h2) or (w1 != w2):
                view_nms = cv2.resize(view_nms, (w1, h1), interpolation=cv2.INTER_LINEAR)

            combo = np.hstack((view_raw, view_nms))
            cv2.imshow("YOLO compare: no NMS (left) vs NMS (right)", combo)

        key = cv2.waitKey(1) & 0xFF
        if key in (27, ord("q"), ord("Q")):
            break
        if key in (ord("p"), ord("P")):
            paused = not paused
        if key in (ord("s"), ord("S")) and "combo" in locals():
            ts = int(time.time())
            cv2.imwrite(f"compare_{ts}.png", combo)
            print("Saved:", f"compare_{ts}.png")

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
