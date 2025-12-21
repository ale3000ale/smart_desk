from pkg import *
from pkg import config as const
class Gui:
    def __init__(self, width=const.WINDOW_DEFAULT_WIDTH, height=const.WINDOW_DEFAULT_WIDTH,
                 btn_x=50, btn_y=50, btn_w=200, btn_h=80,
                 btn_text="CLICK ME"):
        self.width = width
        self.height = height

        # Definisci area del pulsante
        self.btn_x = btn_x
        self.btn_y = btn_y
        self.btn_w = btn_w
        self.btn_h = btn_h
        self.btn_text = btn_text

        # Stato del pulsante
        self.button_pressed = False

        # Nome della finestra (deve coincidere con quella usata da Window)
        self.window_name = "Camera Viewer"

        # Registra callback mouse su questa finestra
        cv2.setMouseCallback(self.window_name, self._mouse_callback)

    def _mouse_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            if (self.btn_x <= x <= self.btn_x + self.btn_w and
                    self.btn_y <= y <= self.btn_y + self.btn_h):
                self.button_pressed = True

    def consume_button_press(self):
        """
        Ritorna True se il pulsante è stato premuto da quando
        è stato letto l’ultima volta e resetta lo stato.
        """
        if self.button_pressed:
            self.button_pressed = False
            return True
        return False

    def render(self, frame):
        """
        Prende il frame della camera, lo porta a (self.width, self.height)
        e disegna sopra il pulsante. Ritorna il nuovo frame.
        """
        if frame is None:
            # Se non c'è frame, crea un canvas vuoto
            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
        else:
            h, w = frame.shape[:2]

            # Ridimensiona mantenendo proporzioni e centrando nel canvas
            scale = min(self.width / w, self.height / h)
            new_w = int(w * scale)
            new_h = int(h * scale)
            resized = cv2.resize(frame, (new_w, new_h))

            canvas = np.zeros((self.height, self.width, 3), dtype=np.uint8)
            x_offset = (self.width - new_w) // 2
            y_offset = (self.height - new_h) // 2
            canvas[y_offset:y_offset + new_h, x_offset:x_offset + new_w] = resized

        # Disegna il pulsante
        pt1 = (self.btn_x, self.btn_y)
        pt2 = (self.btn_x + self.btn_w, self.btn_y + self.btn_h)
        cv2.rectangle(canvas, pt1, pt2, (0, 255, 0), thickness=-1)

        # Testo nel pulsante
        font = cv2.FONT_HERSHEY_SIMPLEX
        text_scale = 0.7
        text_thickness = 2
        (text_w, text_h), _ = cv2.getTextSize(self.btn_text, font, text_scale, text_thickness)
        text_x = self.btn_x + (self.btn_w - text_w) // 2
        text_y = self.btn_y + (self.btn_h + text_h) // 2
        cv2.putText(canvas, self.btn_text, (text_x, text_y),
                    font, text_scale, (0, 0, 0), text_thickness, cv2.LINE_AA)

        return canvas
