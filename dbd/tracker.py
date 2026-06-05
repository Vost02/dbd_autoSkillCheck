import cv2
import numpy as np
from mss import mss

class QTETracker:
    SCALES = [1.0]
    MATCH_THRESHOLD = 0.4
    SEARCH_INTERVAL = 0
    NONE_THRESHOLD = 5
    CROP_SIZE = 224
    CROP_IOR = 0.70
    
    def __init__(self, template_path, monitor_id=1):
        self.mss = mss()
        self.monitor = self.mss.monitors[monitor_id]

        # 1. 2K 自动分辨率自适应：计算屏幕中心 75% 的区域
        screen_w = self.monitor["width"]
        screen_h = self.monitor["height"]
        crop_w = int(screen_w * self.CROP_IOR)
        crop_h = int(screen_h * self.CROP_IOR)
        
        # 确保宽高为偶数
        if crop_w % 2 != 0: crop_w += 1
        if crop_h % 2 != 0: crop_h += 1
        
        # 构建 75% 居中区域的 mss 字典
        self.search_zone = {
            "top": self.monitor["top"] + (screen_h - crop_h) // 2,
            "left": self.monitor["left"] + (screen_w - crop_w) // 2,
            "width": crop_w,
            "height": crop_h
        }

        # 2. 预处理模板
        template_gray = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
        _, template_bin = cv2.threshold(
            template_gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        self.template_edges = cv2.Canny(template_bin, 50, 150)
        self.template_h, self.template_w = self.template_edges.shape

        self.locked_x = None
        self.locked_y = None
        self.search_countdown = 0
        self.none_count = 0

    def _preprocess(self, gray_img):
        _, thresh = cv2.threshold(gray_img, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return cv2.Canny(thresh, 50, 150)

    def _search_fullscreen(self):
        # 仅在 75% 的中心区域内截屏搜索，大幅提升 2K 下的性能
        frame = self.mss.grab(self.search_zone)
        img = np.array(frame)
        gray = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        edges = self._preprocess(gray)

        best_val = 0.0
        best_center = None

        for scale in self.SCALES:
            w = int(self.template_w * scale)
            h = int(self.template_h * scale)
            if w > edges.shape[1] or h > edges.shape[0]:
                continue
            tmpl = cv2.resize(self.template_edges, (w, h))
            result = cv2.matchTemplate(edges, tmpl, cv2.TM_CCOEFF_NORMED)
            _, max_val, _, max_loc = cv2.minMaxLoc(result)

            if max_val > best_val:
                best_val = max_val
                # 结合 search_zone 的相对坐标，换算出绝对屏幕坐标
                best_center = (
                    self.search_zone["left"] + max_loc[0] + w // 2,
                    self.search_zone["top"] + max_loc[1] + h // 2,
                )

        if best_val >= self.MATCH_THRESHOLD and best_center is not None:
            return best_center
        return None

    def _get_crop_region(self):
        half = self.CROP_SIZE // 2
        return {
            "left": int(self.locked_x - half),
            "top": int(self.locked_y - half),
            "width": self.CROP_SIZE,
            "height": self.CROP_SIZE,
        }

    def _release(self):
        self.locked_x = None
        self.locked_y = None
        self.none_count = 0
        self.search_countdown = 0

    def update(self, last_pred_was_none):
        # 情况 A：已经处于锁定状态，直接精确截取 224x224 区域
        if self.locked_x is not None:
            if last_pred_was_none:
                self.none_count += 1
            else:
                self.none_count = 0
                
            if self.none_count >= self.NONE_THRESHOLD:
                self._release()
                # 🌟 核心优化：无情风暴高频连续触发时，释放锁定的当前帧立刻进行一次全屏检索，不浪费任何一帧
                pos = self._search_fullscreen()
                if pos is not None:
                    self.locked_x, self.locked_y = pos
                    self.none_count = 0
                    return self._get_crop_region()
                return None
                
            return self._get_crop_region()

        # 情况 B：未锁定状态
        if self.search_countdown <= 0:
            self.search_countdown = self.SEARCH_INTERVAL
            pos = self._search_fullscreen()
            if pos is not None:
                self.locked_x, self.locked_y = pos
                self.none_count = 0
                return self._get_crop_region()
        else:
            self.search_countdown -= 1
            
        return None


    def is_locked(self):
        return self.locked_x is not None
