import os
import cv2

# 📂 โฟลเดอร์ภาพและ labels
images_folder = "datasets/train/images"
labels_folder = "datasets/train/labels"

# ถ้าไม่มีโฟลเดอร์ labels จะสร้างอัตโนมัติ
os.makedirs(labels_folder, exist_ok=True)

# ขนาดกล่องรอบสิว (pixel)
box_size = 40  # เช่น 40x40 พิกเซล (ปรับได้)

def click_event(event, x, y, flags, params):
    if event == cv2.EVENT_LBUTTONDOWN:
        img_name = params["img_name"]
        img = params["img"]
        h, w, _ = img.shape

        # คำนวณกล่อง
        x_min = max(x - box_size // 2, 0)
        y_min = max(y - box_size // 2, 0)
        x_max = min(x + box_size // 2, w)
        y_max = min(y + box_size // 2, h)

        # เปลี่ยนเป็นอัตราส่วน (0-1)
        x_center = (x_min + x_max) / 2 / w
        y_center = (y_min + y_max) / 2 / h
        bbox_width = (x_max - x_min) / w
        bbox_height = (y_max - y_min) / h

        # เขียนไฟล์ label
        label_line = f"0 {x_center:.6f} {y_center:.6f} {bbox_width:.6f} {bbox_height:.6f}\n"
        label_path = os.path.join(labels_folder, img_name.replace(".jpg", ".txt"))
        with open(label_path, "a") as f:
            f.write(label_line)

        # วาดกรอบ preview
        cv2.rectangle(img, (x_min, y_min), (x_max, y_max), (0, 255, 0), 2)
        cv2.imshow("Labeling", img)

def main():
    images = [img for img in os.listdir(images_folder) if img.lower().endswith(".jpg")]

    for img_name in images:
        img_path = os.path.join(images_folder, img_name)
        img = cv2.imread(img_path)

        print(f"🖱 กำลังทำ labeling รูป: {img_name} (คลิกที่สิวแต่ละเม็ด)")
        cv2.imshow("Labeling", img)
        cv2.setMouseCallback("Labeling", click_event, {"img": img, "img_name": img_name})

        # คลิกหลายครั้งได้ พอเสร็จ กด ESC ไปต่อ
        while True:
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break

        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
