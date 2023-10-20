# Ultralytics YOLO 🚀, GPL-3.0 license


import hydra
import torch
import easyocr
import cv2
from traitlets import Int
from ultralytics.yolo.engine.predictor import BasePredictor
from ultralytics.yolo.utils import DEFAULT_CONFIG, ROOT, ops
from ultralytics.yolo.utils.checks import check_imgsz
from ultralytics.yolo.utils.plotting import Annotator, colors, save_one_box
from datetime import datetime
import sqlite3
import re
import pytz

def process_license_plate(ocr):
    unique = set()

    # Check if ocr has more than 6 characters and contains only alphanumeric characters
    if len(ocr) >= 6 and ocr.isalnum():
        if ocr not in unique:
            # Remove spaces and special characters from ocr using regular expressions
            ocr = re.sub(r'\W+', '', ocr)
            unique.add(ocr)
            return ocr
    return ''


def getOCR(im, coors):
    x, y, w, h = int(coors[0]), int(coors[1]), int(coors[2]), int(coors[3])
    im = im[y:h, x:w]
    conf = 0.2

    gray = cv2.cvtColor(im, cv2.COLOR_RGB2GRAY)
    results = reader.readtext(gray)
    ocr = ""

    for result in results:
        if len(results) == 1:
            ocr = result[1]
        if len(results) > 1 and len(results[1]) > 6 and results[2] > conf:
            ocr = result[1]

    return str(process_license_plate(ocr))


class DetectionPredictor(BasePredictor):
    def get_annotator(self, img):
        return Annotator(img, line_width=self.args.line_thickness, example=str(self.model.names))

    def preprocess(self, img):
        img = torch.from_numpy(img).to(self.model.device)
        img = img.half() if self.model.fp16 else img.float()  # uint8 to fp16/32
        img /= 255  # 0 - 255 to 0.0 - 1.0
        return img

    def postprocess(self, preds, img, orig_img):
        preds = ops.non_max_suppression(preds,
                                        self.args.conf,
                                        self.args.iou,
                                        agnostic=self.args.agnostic_nms,
                                        max_det=self.args.max_det)

        for i, pred in enumerate(preds):
            shape = orig_img[i].shape if self.webcam else orig_img.shape
            pred[:, :4] = ops.scale_boxes(
                img.shape[2:], pred[:, :4], shape).round()

        return preds

    def write_results(self, idx, preds, batch):
        p, im, im0 = batch
        log_string = ""
        if len(im.shape) == 3:
            im = im[None]  # expand for batch dim
        self.seen += 1
        im0 = im0.copy()
        if self.webcam:  # batch_size >= 1
            log_string += f'{idx}: '
            frame = self.dataset.count
        else:
            frame = getattr(self.dataset, 'frame', 0)

        self.data_path = p
        # save_path = str(self.save_dir / p.name)  # im.jpg
        self.txt_path = str(self.save_dir / 'labels' / p.stem) + \
            ('' if self.dataset.mode == 'image' else f'_{frame}')
        log_string += '%gx%g ' % im.shape[2:]  # print string
        self.annotator = self.get_annotator(im0)

        det = preds[idx]
        self.all_outputs.append(det)
        if len(det) == 0:
            return log_string
        for c in det[:, 5].unique():
            n = (det[:, 5] == c).sum()  # detections per class
            log_string += f"{n} {self.model.names[int(c)]}{'s' * (n > 1)}, "
        # write
        gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        for *xyxy, conf, cls in reversed(det):
            if self.args.save_txt:  # Write to file
                xywh = (ops.xyxy2xywh(torch.tensor(xyxy).view(1, 4)) /
                        gn).view(-1).tolist()  # normalized xywh
                # label format
                line = (cls, *xywh, conf) if self.args.save_conf else (cls, *xywh)
                with open(f'{self.txt_path}.txt', 'a') as f:
                    f.write(('%g ' * len(line)).rstrip() % line + '\n')

            if self.args.save or self.args.save_crop or self.args.show:  # Add bbox to image
                c = int(cls)  # integer class
                label = None if self.args.hide_labels else (
                    self.model.names[c] if self.args.hide_conf else f'{self.model.names[c]} {conf:.2f}')
                ocr = getOCR(im0, xyxy)
                ocr = process_license_plate(ocr)
                if ocr != "":
                    date_time = datetime.now()
                    # Set the timezone to Indian Standard Time (IST)
                    indian_tz = pytz.timezone('Asia/Kolkata')
                    indian_date_time = date_time.astimezone(indian_tz)

                    # Format the Indian date and time
                    indian_formatted_date_time = indian_date_time.strftime('%Y-%m-%d %H:%M:%S %Z')

                    # connecting to the database
                    connection = sqlite3.connect("NGT2.db")
                    #  cursor
                    crsr = connection.cursor()

                    # Check if the table has been created already
                    global table_created
                    if not table_created:
                        # If not created, create the table in the database
                        sql_command = """CREATE TABLE IF NOT EXISTS vehiclesLicense (vehicle_number VARCHAR(20), timestamp DATE);"""
                        crsr.execute(sql_command)
                        table_created = True  # Set the flag to indicate that the table has been created

                    insert_query = "INSERT INTO vehiclesLicense(vehicle_number, timestamp) VALUES (?, ?)"
                    crsr.execute(insert_query, (ocr, indian_formatted_date_time))

                    sqlite_select_query = """SELECT * from vehiclesLicense"""
                    crsr.execute(sqlite_select_query)
                    records = crsr.fetchall()
                    for row in records:
                        with open('sqlite.txt', 'a') as f:
                            f.write(f'{row[0]}                                  {row[1]} \n')
                    crsr.close()
                    connection.close()
                    label = ocr
                self.annotator.box_label(xyxy, label, color=colors(c, True))
            if self.args.save_crop:
                imc = im0.copy()
                save_one_box(xyxy,
                             imc,
                             file=self.save_dir / 'crops' /
                             self.model.model.names[c] /
                             f'{self.data_path.stem}.jpg',
                             BGR=True)

        return log_string


table_created = False


@hydra.main(version_base=None, config_path=str(DEFAULT_CONFIG.parent), config_name=DEFAULT_CONFIG.name)
def predict(cfg):
    cfg.model = cfg.model or "yolov8n.pt"
    cfg.imgsz = check_imgsz(cfg.imgsz, min_dim=2)  # check image size
    cfg.source = cfg.source if cfg.source is not None else ROOT / "assets"
    predictor = DetectionPredictor(cfg)
    predictor()


if __name__ == "__main__":
    reader = easyocr.Reader(['en'])
    predict()
