import cv2
import numpy as np
from deepface import DeepFace

logo = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
logo = cv2.resize(logo, (0, 0), fx=0.5, fy=0.5)
rows, cols, channels = logo.shape

# capture frames from a camera
cam = cv2.VideoCapture(0)
frame_width = cam.get(cv2.CAP_PROP_FRAME_WIDTH)
frame_height = cam.get(cv2.CAP_PROP_FRAME_HEIGHT)



x = 5
y = 5

def add_transparent_image(background, foreground, x_offset=None, y_offset=None):
    bg_h, bg_w, bg_channels = background.shape
    fg_h, fg_w, fg_channels = foreground.shape

    assert bg_channels == 3, f'background image should have exactly 3 channels (RGB). found:{bg_channels}'
    assert fg_channels == 4, f'foreground image should have exactly 4 channels (RGBA). found:{fg_channels}'

    # center by default
    if x_offset is None: x_offset = (bg_w - fg_w) // 2
    if y_offset is None: y_offset = (bg_h - fg_h) // 2

    w = min(fg_w, bg_w, fg_w + x_offset, bg_w - x_offset)
    h = min(fg_h, bg_h, fg_h + y_offset, bg_h - y_offset)

    if w < 1 or h < 1: return

    # clip foreground and background images to the overlapping regions
    bg_x = max(0, x_offset)
    bg_y = max(0, y_offset)
    fg_x = max(0, x_offset * -1)
    fg_y = max(0, y_offset * -1)
    foreground = foreground[fg_y:fg_y + h, fg_x:fg_x + w]
    background_subsection = background[bg_y:bg_y + h, bg_x:bg_x + w]

    # separate alpha and color channels from the foreground image
    foreground_colors = foreground[:, :, :3]
    alpha_channel = foreground[:, :, 3] / 255  # 0-255 => 0.0-1.0

    # construct an alpha_mask that matches the image shape
    alpha_mask = np.dstack((alpha_channel, alpha_channel, alpha_channel))

    # combine the background with the overlay image weighted by alpha
    composite = background_subsection * (1 - alpha_mask) + foreground_colors * alpha_mask

    # overwrite the section of the background image that has been updated
    background[bg_y:bg_y + h, bg_x:bg_x + w] = composite

while True:
    ret, img = cam.read()
    result = DeepFace.analyze(img, actions=['emotion'], enforce_detection=False)
    # check if region x and region y from result is not zero
    if result['region']['x'] != 0 and result['region']['y'] != 0:
        # draw a rectangle around the face
        cv2.rectangle(img, (result['region']['x'], result['region']['y']), (result['region']['x'] + result['region']['w'], result['region']['y'] + result['region']['h']), (0, 255, 0), 2)
        # insert emoji image to right of the face
        if result['dominant_emotion'] == 'angry':
            emoji = cv2.imread('angry.png', cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji, (0, 0), fx=0.05, fy=0.05)
            add_transparent_image(img, emoji, result['region']['x'] + result['region']['w'], result['region']['y'])
        elif result['dominant_emotion'] == 'happy':
            emoji = cv2.imread('happy.png', cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji, (0, 0), fx=0.05, fy=0.05)
            add_transparent_image(img, emoji, result['region']['x'] + result['region']['w'], result['region']['y'])
        elif result['dominant_emotion'] == 'surprise':
            emoji = cv2.imread('surprise.png', cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji, (0, 0), fx=0.05, fy=0.05)
            add_transparent_image(img, emoji, result['region']['x'] + result['region']['w'], result['region']['y'])
        elif result['dominant_emotion'] == 'neutral':
            emoji = cv2.imread('neutral.png', cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji, (0, 0), fx=0.05, fy=0.05)
            add_transparent_image(img, emoji, result['region']['x'] + result['region']['w'], result['region']['y'])
        elif result['dominant_emotion'] == 'sad':
            emoji = cv2.imread('sad.png', cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji, (0, 0), fx=0.05, fy=0.05)
            add_transparent_image(img, emoji, result['region']['x'] + result['region']['w'], result['region']['y'])
        elif result['dominant_emotion'] == 'fear':
            emoji = cv2.imread('fear.png', cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji, (0, 0), fx=0.05, fy=0.05)
            add_transparent_image(img, emoji, result['region']['x'] + result['region']['w'], result['region']['y'])
        elif result['dominant_emotion'] == 'disgust':
            emoji = cv2.imread('disgust.png', cv2.IMREAD_UNCHANGED)
            emoji = cv2.resize(emoji, (0, 0), fx=0.05, fy=0.05)
            add_transparent_image(img, emoji, result['region']['x'] + result['region']['w'], result['region']['y'])
         
        # write the emotion text
        cv2.putText(img, result['dominant_emotion'], (result['region']['x'], result['region']['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)        
    add_transparent_image(img, logo, x, y)
    # resize image to 720p
    img = cv2.resize(img, (1280, 720))
    cv2.imshow('img', img)
    # show emotion
    print(result)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cam.release()

cv2.destroyAllWindows()
    
