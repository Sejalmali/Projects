from flask import Flask, request, render_template, redirect, url_for, session, Response, send_file, jsonify
from flask_bcrypt import Bcrypt
from datetime import datetime
import mysql.connector
import threading
import cv2
import mediapipe as mp
import numpy as np
import os
import math
import base64


app = Flask(__name__)
app.secret_key = 'AKASIDSIDSEJ2003'
bcrypt = Bcrypt(app)


# Replace these database configurations with your own
db_config = {
    'host': 'localhost',
    'user': 'root',
    'password': 'root@123',
    'database': 'MINIPROJECT',  # Change to your database name
}


conn = mysql.connector.connect(**db_config)
cursor = conn.cursor()


# Create users table if not exists
cursor.execute('''
    CREATE TABLE IF NOT EXISTS users (
        id INT AUTO_INCREMENT PRIMARY KEY,
        username VARCHAR(255) NOT NULL,
        email VARCHAR(255) NOT NULL,
        password VARCHAR(255) NOT NULL
    );
''')


# Initialize the video capture for the model
cap = cv2.VideoCapture(0)
save_frame = False  # Flag to determine whether to save the current frame
def generate_frames():
    global save_frame

    while True:
        # read the camera frame
        success, frame = cap.read()
        if not success:
            break
        else:
            ret, buffer = cv2.imencode('.jpg', frame)
            frame = buffer.tobytes()

            if save_frame:
                # Save the current frame as .jpg
                filename = "saved_frame.jpg"
                directory = os.path.join("static", "saved_frames")
                os.makedirs(directory, exist_ok=True)  # Create the directory if it doesn't exist
                filepath = os.path.join(directory, filename)
                with open(filepath, 'wb') as f:
                    f.write(buffer)

                # Reset the flag after saving
                save_frame = False

                yield send_file(filepath, as_attachment=True)  # Serve the saved file for download

            else:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def home():
    return render_template('index.html') 

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/contact')
def contact():
    return render_template('contact.html')

@app.route('/save', methods=['POST'])
def save():
    try:
        frame_data = request.get_json()['frame']
        save_frame(frame_data)
        return 'Frame saved successfully!'
    except Exception as e:
        print(f"Error saving frame: {str(e)}")
        return 'Error saving frame.'

def save_frame(frame_data):
    try:
        # Split the frame_data if it contains a comma
        if ',' in frame_data:
            frame_bytes = frame_data.split(',')[1].encode('utf-8')
        else:
            frame_bytes = frame_data.encode('utf-8')

        # Decode base64 and save the image with a unique filename
        imgdata = base64.b64decode(frame_bytes)
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        filename = os.path.join(os.getcwd(), 'static', 'images', f'saved_frame_{timestamp}.jpg')
        with open(filename, 'wb') as f:
            f.write(imgdata)
        print(f"Frame saved to: {filename}")
    except Exception as e:
        print(f"Error saving frame: {str(e)}")



@app.route('/login', methods=['POST'])
def login():
    email = request.form['email']
    password = request.form['password']

    # Fetch user from the database
    cursor.execute('SELECT * FROM users WHERE email = %s', (email,))
    user = cursor.fetchone()

    if user and bcrypt.check_password_hash(user[3], password):
        # Valid credentials, store user information in the session
        session['user_id'] = user[0]
        session['username'] = user[1]
        
        # Render a template with a success message
        #return render_template('success_login.html', username=user[1])
        return redirect(url_for('profile'))
    else:
        # Invalid credentials, handle accordingly (e.g., redirect to login page with a message)
        return redirect(url_for('home'))


@app.route('/register', methods=['POST'])
def register():
    username = request.form['username']
    email = request.form['email']
    password = request.form['password']

    # Hash the password
    hashed_password = bcrypt.generate_password_hash(password).decode('utf-8')

    # Insert user into the database with the hashed password
    cursor.execute('INSERT INTO users (username, email, password) VALUES (%s, %s, %s)', (username, email, hashed_password))
    conn.commit()

    # Set a session variable to indicate successful registration
    session['registration_success'] = True

    return redirect(url_for('home'))


@app.route('/welcome')
def welcome():
    username = request.args.get('username')
    return f'Welcome, {username}!'


@app.route('/success_login')
def profile():
    if 'username' in session:
        return render_template('success_login.html')
    return 'You are not logged in'


@app.route('/video_feed')
def video_feed():
    def generate():
        # Your video capture and processing code here
        
        # Model Code Starts Here...

        mp_drawing = mp.solutions.drawing_utils
        mp_hands = mp.solutions.hands

        # For webcam input:
        cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
        cap.set(cv2.CAP_PROP_FPS, 5)
        width, height = 1280, 720
        cap.set(3, width)
        cap.set(4, height)

        # Image that will contain the drawing and then passed to the camera image
        # Initialize variables
        canvas_list = [np.zeros((height, width, 3), np.uint8)]  # Start with one empty canvas
        current_page_index = 0  # Track the index of the current page

        # Initialize flags for gestures
        new_canvas_gesture_done = False
        previous_canvas_gesture_done = False

        # Image that will contain the drawing and then passed to the camera image
        # imgCanvas = np.zeros((height, width, 3), np.uint8)

        # Getting all header images in a list
        folderPath = 'C:/Users/User/Desktop/design_mp5/static/Header_V1'
        myList = os.listdir(folderPath)
        overlayList = []
        for imPath in myList:
            image = cv2.imread(f'{folderPath}/{imPath}')
            overlayList.append(image)

        # Presettings:
        header = overlayList[0]
        drawColor = (0, 0, 255)
        thickness = 20 # Thickness of the painting
        tipIds = [4, 8, 12, 16, 20] # Fingertips indexes
        xp, yp = [0, 0] # Coordinates that will keep track of the last position of the index finger

        with mp_hands.Hands(min_detection_confidence=0.85, min_tracking_confidence=0.5, max_num_hands=1) as hands:
            while cap.isOpened():
                
                success, image = cap.read()
                if not success:
                    print("Ignoring empty camera frame.")
                    continue

                # Flip the image horizontally for a later selfie-view display, and convert
                # the BGR image to RGB.
                image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)
                # To improve performance, optionally mark the image as not writeable to
                # pass by reference.
                image.flags.writeable = False
                results = hands.process(image)

                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Getting all hand points coordinates
                        points = []
                        for lm in hand_landmarks.landmark:
                            points.append([int(lm.x * width), int(lm.y * height)])

                        # Only go through the code when a hand is detected
                        if len(points) != 0:
                            x1, y1 = points[8]  # Index finger
                            x2, y2 = points[12] # Middle finger
                            x3, y3 = points[4]  # Thumb
                            x4, y4 = points[20] # Pinky


                            # Check thumb direction and other fingers closed
                            thumb_direction = "Right" if x3 > x4 else "Left"
                            all_fingers_closed = all(points[i][1] > points[i - 2][1] for i in tipIds[1:])
                            all_fingers_open = all(points[i][1] < points[i - 2][1] for i in tipIds[1:])

                            # Create a new page if thumb points right and all other fingers are closed
                            if thumb_direction == "Right" and all_fingers_closed and not new_canvas_gesture_done:
                                new_page = np.zeros((height, width, 3), np.uint8)
                                canvas_list.append(new_page)
                                current_page_index += 1
                                new_canvas_gesture_done = True  # Set the flag to True after executing the gesture

                            # Switch to the previous page if thumb points left and all other fingers are closed
                            elif thumb_direction == "Left" and all_fingers_closed and not previous_canvas_gesture_done:
                                current_page_index = max(0, current_page_index - 1)
                                previous_canvas_gesture_done = True  # Set the flag to True after executing the gesture
                            
                            
                            ## Checking which fingers are up
                            fingers = []
                            # Checking the thumb
                            if points[tipIds[0]][0] < points[tipIds[0] - 1][0]:
                                fingers.append(1)
                            else:
                                fingers.append(0)

                            # The rest of the fingers
                            for id in range(1, 5):
                                if points[tipIds[id]][1] < points[tipIds[id] - 2][1]:
                                    fingers.append(1)
                                else:
                                    fingers.append(0)

                            ## Selection Mode - Two fingers are up
                            nonSel = [0, 3, 4] # indexes of the fingers that need to be down in the Selection Mode
                            if (fingers[1] and fingers[2]) and all(fingers[i] == 0 for i in nonSel):
                                xp, yp = [x1, y1]

                                # Selecting the colors and the eraser on the screen
                                if(y1 < 125):
                                    if(85 < x1 < 210):
                                        header = overlayList[0]
                                        drawColor = (0, 0, 255)  #red
                                    elif(327 < x1 < 452):
                                        header = overlayList[1]
                                        drawColor = (255, 0, 0)  #blue
                                    elif(436 < x1 < 561):
                                        header = overlayList[2]
                                        drawColor = (0, 255, 0)  #green
                                    elif(700 < x1 < 825):
                                        header = overlayList[3]
                                        drawColor = (29, 211, 247)  #yellow
                                    elif(900 < x1 < 925):
                                        header = overlayList[4]
                                        drawColor = (32, 32, 32)  #black
                                    elif(980 < x1 < 1105):
                                        header = overlayList[5]
                                        drawColor = (0, 0, 0)  #black(eraser)

                                cv2.rectangle(image, (x1-10, y1-15), (x2+10, y2+23), drawColor, cv2.FILLED)

                            ## Stand by Mode - Checking when the index and the pinky fingers are open and dont draw
                            nonStand = [0, 2, 3] # indexes of the fingers that need to be down in the Stand Mode
                            if (fingers[1] and fingers[4]) and all(fingers[i] == 0 for i in nonStand):
                                # The line between the index and the pinky indicates the Stand by Mode
                                cv2.line(image, (xp, yp), (x4, y4), drawColor, 5) 
                                xp, yp = [x1, y1]

                            ## Draw Mode - One finger is up
                            nonDraw = [0, 2, 3, 4]
                            if fingers[1] and all(fingers[i] == 0 for i in nonDraw):
                                # The circle in the index finger indicates the Draw Mode
                                cv2.circle(image, (x1, y1), int(thickness/2), drawColor, cv2.FILLED) 
                                if xp==0 and yp==0:
                                    xp, yp = [x1, y1]
                                # Draw a line between the current position and the last position of the index finger
                                cv2.line(imgCanvas, (xp, yp), (x1, y1), drawColor, thickness)
                                # Update the last position
                                xp, yp = [x1, y1]

                            ## Clear the canvas when the hand is closed
                            '''if all(fingers[i] == 0 for i in range(0, 5)):
                                imgCanvas = np.zeros((height, width, 3), np.uint8)
                                xp, yp = [x1, y1]'''

                            ## Adjust the thickness of the line using the index finger and thumb
                            selecting = [1, 1, 0, 0, 0] # Selecting the thickness of the line
                            setting = [1, 1, 0, 0, 1]   # Setting the thickness chosen
                            if all(fingers[i] == j for i, j in zip(range(0, 5), selecting)) or all(fingers[i] == j for i, j in zip(range(0, 5), setting)):

                                # Getting the radius of the circle that will represent the thickness of the draw
                                # using the distance between the index finger and the thumb.
                                r = int(math.sqrt((x1-x3)**2 + (y1-y3)**2)/3)
                                
                                # Getting the middle point between these two fingers
                                x0, y0 = [(x1+x3)/2, (y1+y3)/2]
                                
                                # Getting the vector that is orthogonal to the line formed between
                                # these two fingers
                                v1, v2 = [x1 - x3, y1 - y3]
                                v1, v2 = [-v2, v1]

                                # Normalizing it 
                                mod_v = math.sqrt(v1**2 + v2**2)
                                v1, v2 = [v1/mod_v, v2/mod_v]
                                
                                # Draw the circle that represents the draw thickness in (x0, y0) and orthogonaly 
                                # translated c units
                                c = 3 + r
                                x0, y0 = [int(x0 - v1*c), int(y0 - v2*c)]
                                cv2.circle(image, (x0, y0), int(r/2), drawColor, -1)

                                # Setting the thickness chosen when the pinky finger is up
                                if fingers[4]:                        
                                    thickness = r
                                    cv2.putText(image, 'Check', (x4-25, y4-8), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (0,0,0), 1)

                                xp, yp = [x1, y1]
                else:
                    # Reset the gesture flags when no hand is detected
                    new_canvas_gesture_done = False
                    previous_canvas_gesture_done = False

                # Display the current page
                imgCanvas = canvas_list[current_page_index]

                # Adding page number to the canvas
                cv2.putText(imgCanvas, f"Page {current_page_index + 1}", (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
                # Setting the header in the video
                image[0:125, 0:width] = header

                # The image processing to produce the image of the camera with the draw made in imgCanvas
                imgGray = cv2.cvtColor(imgCanvas, cv2.COLOR_BGR2GRAY)
                _, imgInv = cv2.threshold(imgGray, 5, 255, cv2.THRESH_BINARY_INV)
                imgInv = cv2.cvtColor(imgInv, cv2.COLOR_GRAY2BGR)
                img = cv2.bitwise_and(image, imgInv)
                img = cv2.bitwise_or(img, imgCanvas)

                # Convert the processed image to JPEG format
                _, jpeg_frame = cv2.imencode('.JPEG', img)

                yield (b'--frame\r\n'
                        b'Content-Type: image/jpeg\r\n\r\n' + jpeg_frame.tobytes() + b'\r\n')

                # Release the video capture and close OpenCV windows when 'q' key is pressed
                if cv2.waitKey(3) & 0xFF == ord('q'):
                    # break
                    cap.release()
                    cv2.destroyAllWindows()
                    print("Video generation complete.")
                    break

    # Model Code Ends Here...

    if 'username' in session:
        return Response(generate(), mimetype='multipart/x-mixed-replace; boundary=frame')
        

@app.route('/logout')
def logout():
    session.pop('username', None)
    return redirect(url_for('home'))


if __name__ == '__main__':
    app.run(debug=True)
