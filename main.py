import mediapipe as mp
import cv2
import pygame 
import threading
import math

# --- Initialize webcam capture ---
cap = cv2.VideoCapture(0)
window_name = "Webcam"
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# --- hand setup ---
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(model_complexity=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

# --- finger setup ---
# user finger position
def finger_point(hand_landmarks, result):
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]  # first hand
        landmark8 = hand_landmarks.landmark[8]  # Index fingertip
        
        x = int(landmark8.x * width)
        y = int(landmark8.y * height)
        
        return (x, y)
    return None

# check if user can grab object
def grab_object(hand_landmarks, result, threshold=30):
    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]  # first hand
        landmark8 = hand_landmarks.landmark[8]  # Index fingertip
        landmark4 = hand_landmarks.landmark[4]  # Thumb tip

        x8 = int(landmark8.x * width)
        y8 = int(landmark8.y * height)
        x4 = int(landmark4.x * width)
        y4 = int(landmark4.y * height)

        distance = math.sqrt((x8 - x4) ** 2 + (y8 - y4) ** 2) # calulate distance

        if threshold >= distance:
            return True
        return False
    return None

# --- Image recognition setup ---
def recognize_image():
    global coordinates, can_grab, running
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
        result = hands.process(frame)

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS)

                new_cord = finger_point(hand_landmarks, result)
                if new_cord:
                    with cord_lock:
                        coordinates = new_cord

                can_grab = grab_object(hand_landmarks, result)

        cv2.imshow(window_name, frame)

        if cv2.waitKey(1) & 0xFF == ord("q") or cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            cap.release()
            cv2.destroyAllWindows()
            running = False
            pygame.quit()
            break


# --- pygame setup ---
pygame.init()
window = pygame.display.set_mode((width, height))
pygame.display.set_caption("Finger Tracker")
clock = pygame.time.Clock()

# user
user_size = 5
coordinates = (width // 2, height // 2)
user_rect = pygame.Rect(coordinates[0], coordinates[1], user_size, user_size)
currently_grabbing = False
can_grab = False

# moveable object
rect_coordinates = (None, None)
move_rect = pygame.Rect(100, 100, 100, 100)
running = True

# Start image recognition thread
image_thread = threading.Thread(target=recognize_image, daemon=True)
image_thread.start()
cord_lock = threading.Lock()


# Main loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    with cord_lock:
        user_rect.topleft = coordinates
    
    window.fill((255, 255, 255)) 

    pygame.draw.rect(window, (0, 0, 0), move_rect)

    if move_rect.colliderect(user_rect) and can_grab or currently_grabbing:  # If colliding and can grab, move the object
        move_rect.center = user_rect.center
        currently_grabbing = True
        pygame.draw.rect(window, (0, 255, 0), user_rect)

    
    if not can_grab:
        currently_grabbing = False
        pygame.draw.rect(window, (255, 0, 0), user_rect)

    pygame.display.flip()
    clock.tick(30)

cap.release()
cv2.destroyAllWindows()
pygame.quit()