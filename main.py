import torch, os, glob, numpy as np, cv2, argparse
import torch
import cv2
import os
import glob
import numpy as np
import argparse
from typing import Union, Tuple, List, Optional
from numpy import ndarray
from core.detection import RetinaDetector, CascadeDetector
from core.recognition import ArcRecognizer
from numpy.linalg import norm
@torch.no_grad()
class Handler():
    def __init__(self, database_path, backend) -> None:
        self.arcface = ArcRecognizer()
        self.detector = None
        self.backend = backend
        if backend == 'retina':
            self.detector = RetinaDetector()
        elif backend == 'opencv':
            self.detector = CascadeDetector()
        # self.retina = RetinaDetector()
        # self.haar_cascade = CascadeDetector()
        self.mean_face_database = []
        self.face_database = []
        self.image_size = (112, 112) # for arcface
        # True to use mean-feature verification, False for single-feature verification (take more time)
        self.verify_mode = False   
        self.database_state = False    

        # initialize database automatically
        self.init_identity_database(database_path)

    def init_identity_database(self, parent_folder_path='database'):
        # check if database needs update or not
        if self.database_state == True:
            return
        
        # reset database
        self.mean_face_database = []
        self.face_database = []

        for identity_folder in os.listdir(parent_folder_path):
            
            
            # calculating features of one identity
            feat = []
            blobs = []
            print('Reading folder: ' + identity_folder)
            for file in glob.glob(os.path.join(parent_folder_path, identity_folder) + '/*.png') + \
                glob.glob(os.path.join(parent_folder_path, identity_folder) + '/*.jpg'):
                
                img = cv2.imread(file, cv2.IMREAD_COLOR)
                if img is None or self.detector is None:
                    continue
                faces, landms = self.detector.detect(img)
                if faces is None or len(faces) == 0:
                    continue
                # fool proof for many faces detected in one image (registration will denied this)
                for idx in range(len(faces)):
                    # extract face bounding box
                    x1, y1, x2, y2 = faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3]
                    # crop face from image with face bbox
                    face_frame = img[y1:y2, x1:x2]
                    # resize face to desired size
                    if face_frame.size == 0: continue
                    face_frame = cv2.resize(face_frame, self.image_size, interpolation=cv2.INTER_AREA)
                    # process landmark points from 'scaled' to actual 'coordinates'
                    if self.backend == 'retina':
                        landmk = [landm * self.image_size[0] for landm in landms[idx]]
                    elif self.backend == 'opencv':
                        landmk = []
                    # get image blob
                    blob = self.arcface.get_image(face_frame, landmk)
                    blobs.append(blob)
                    # # single image forwarding
                    # mean_feat.append(self.arcface.forward(blob))
            if len(blobs) != 0:
                # extract feature vectors
                feat = self.arcface.forward_many(blobs, len(blobs))
            if len(feat) != 0:
                # create holder vector for face feature (has 1 dimension, 1024 rows)
                mean_feat = np.zeros(shape=(1, 1024))
                # calculate 'mean/average' feature vector (sum/ n of vector on cols)
                mean_feat[0] = np.mean(feat, axis=0)
                print(f'Extracted feature for identity {identity_folder}: ', mean_feat)
                # save mean feature vector to mean face database
                self.mean_face_database.append((identity_folder, mean_feat))
                # save all feature vectors to (single) face database
                self.face_database.append((identity_folder, feat))
        print(self.mean_face_database)
        # print(self.face_database)
        self.database_state = True
    
    
    """
        This function take in an image, process any face detected within and return 
        a frame with face's bounding boxes, name and confidence score
    """
    def recognize(self, img: Optional[ndarray] = None) -> Optional[ndarray]:
        if img is None:
            return None
        if self.detector is None:
            print("No detector initialized")
            return img
        # detect face from image
        faces, landms = self.detector.detect(img)
        # iterate through each image detected in frame
        for idx in range(len(faces)):
            # getting face bounding box coordinates
            x1, y1, x2, y2 = faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3]
            # cut face from image
            face_frame = img[y1:y2, x1:x2]
            # resize image to desired size
            if face_frame.size == 0: continue
            face_frame = cv2.resize(face_frame, (112, 112), interpolation=cv2.INTER_AREA)
            # get 'this' face landmarks
            if self.backend == 'retina':
                landmk = [landm * self.image_size[0] for landm in landms[idx]]
            elif self.backend == 'opencv':
                landmk = []
            # get input blob
            blob = self.arcface.get_image(face_frame, landmk)
            # get face feature
            feat = self.arcface.forward(blob)

            """
                NOTE:
                This step specify for checking multiple feature vector from ONE person.
                Which mean we can either check with all single-feature vector (come from single image) or
                average-feature vector (normalize from all the single-feature vector)
                USAGE:
                Change self.verify_mode to  True to verify using the mean identity face-feature,
                                            False to verify using all the identity face-feature
            """
            # find match person
            match_name = ''
            match_score = 0
            database = None
            if self.verify_mode == True:
                database = self.mean_face_database
            else:
                database = self.face_database
            for identity in database:
                # identity
                name = identity[0]
                # feature vector(s)
                match_feats = identity[1]
                # print('Calculating for: ' + name)
                for f in match_feats:
                    a = np.squeeze(feat)
                    b = np.squeeze(f)
                    cosine_similarity = np.dot(a, b) / (norm(a)*norm(b))
                    # find best match
                    if cosine_similarity > match_score:
                        match_name = name
                        match_score = cosine_similarity
                # print()
            
            # draw processed result
            cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,255,255), 2)
            cv2.putText(img, 'Name: ' + match_name, (int(x1), int(y1 - 10)), cv2.FONT_HERSHEY_COMPLEX, .4, (255,255,255), 1)
            cv2.putText(img, 'Confidence: ' + str(match_score), (int(x1), int(y1 - 25)), cv2.FONT_HERSHEY_COMPLEX, .4, (255,255,255), 1)
        return img
        

    def cal_and_app_feature(self, features, identity_name):
        if len(features) == 0:
            return
        for f in features:
            self.face_database.append((identity_name, f))
        mean_f = np.zeros((1, 1024))
        mean_f = np.mean(features, axis=0)
        self.mean_face_database.append((identity_name, mean_f))

    def register_identity(self, img: Optional[ndarray] = None, identity: str = '') -> Tuple[List, Optional[ndarray]]:
        if img is None:
            return [], None
        if self.detector is None:
            print("No detector initialized")
            return [], img
        # detect face from image
        faces, landms = self.detector.detect(img)
        if faces is None or landms is None:
            return [], img
        # initialize parameters
        msg = ''
        # check if only one face detected in frame
        if len(faces) > 1:
            msg = 'More than one face detected!'
            print(msg)
            # print(type([]), type(img))
            return [], img
        else:
            # this for loop is only fool-proof, program logic will only add ONE person in ONE frame at a time.
            for idx in range(len(faces)):
                # getting face bounding box coordinates
                x1, y1, x2, y2 = faces[idx][0], faces[idx][1], faces[idx][2], faces[idx][3]
                # cut face from image
                face_frame = img[y1:y2, x1:x2]
                # resize image to desired size
                face_frame = cv2.resize(face_frame, (112, 112), interpolation=cv2.INTER_AREA)
                # get 'this' face landmarks
                landmk = landms[idx]
                # get input blob
                blob = self.arcface.get_image(face_frame, landmk)
                # get face feature
                feat = self.arcface.forward(blob)

                cv2.rectangle(img, (int(x1), int(y1)), (int(x2), int(y2)), (0,0,255), 2)
                if len(faces[idx]) > 4:  # Check if confidence exists
                    cv2.putText(img, str(faces[idx][4]), (int(x1), int(y1 + 10)), cv2.FONT_HERSHEY_COMPLEX, 0.5, (0,0,0))
                # print(type(feat), type(img))
                self.database_state = False
                return feat, img
        return [], img
    
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Facial Recognition Application')
    parser.add_argument('--mode', '-m', type=int, default=0, help='0 - register new face into system, 1 - face recognition')
    parser.add_argument('--imgs', '-i', type=str, default='', help='if empty, auto search for webcam, else - path to image file')
    parser.add_argument('--database', '-dp', type=str, default='database', help='Path to database directory containing identity folders')
    parser.add_argument('--backend', '-dbe', type=str, default='retina', help='Backend for face detection')
    args = vars(parser.parse_args())
    
    # Initialize handler
    handler = Handler(args['database'], args['backend'])

    # Check if input is image file or video
    if args['imgs'] and os.path.isfile(args['imgs']):
        # Process single image
        img = cv2.imread(args['imgs'])
        if img is None:
            print(f"Could not read image: {args['imgs']}")
            exit(1)
        
        if args['mode'] == 0:
            # Registration mode for single image
            identity_name = input('Input identity\'s name: ')
            identity_path = os.path.normpath(os.path.join(args['database'], identity_name))
            if not os.path.exists(identity_path):
                os.makedirs(identity_path)
            
            # Register the face
            features, processed_img = handler.register_identity(img, identity_name)
            if len(features) > 0:
                output_path = os.path.join(identity_path, f"{identity_name}_0.jpg")
                cv2.imwrite(output_path, img)
                print(f"Successfully registered {identity_name}")
            else:
                print("No valid face detected in image")
        
        else:
            # Recognition mode for single image
            processed_img = handler.recognize(img)
            if processed_img is not None:
                cv2.imshow('Recognition Result', processed_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
    
    else:
        # Video processing mode
        STATE = True
        FACE_NUM_THRESH = 20
        
        while STATE:
            cam = cv2.VideoCapture(0)
            if args['mode'] == 0:
                # Registration mode with video
                database_path = args['database']
                identity_name = input('Input identity\'s name: ')
                identity_path = os.path.normpath(os.path.join(database_path, identity_name))
                if not os.path.exists(identity_path):
                    os.makedirs(identity_path)
                
                face_count = len(glob.glob(os.path.join(identity_path, '*.png')) + \
                            glob.glob(os.path.join(identity_path, '*.jpg')))
                register_count = 0
                features = []
                
                while True:
                    ret, frame = cam.read()
                    if not ret:
                        break
                    
                    feature, processed_frame = handler.register_identity(frame, identity_name)
                    if len(feature) > 0:
                        features.extend([feature])
                        cv2.imwrite(os.path.join(identity_path, f"{identity_name}_{face_count}.jpg"), frame)
                        face_count += 1
                        register_count += 1
                    
                    if processed_frame is not None:
                        cv2.imshow('Registration', processed_frame)
                    if cv2.waitKey(1) == ord('q') or register_count >= FACE_NUM_THRESH:
                        break
                
                cv2.destroyAllWindows()
                cam.release()
                args['mode'] = 1
            
            elif args['mode'] == 1:
                # Recognition mode with video
                handler.init_identity_database(args['database'])
                while True:
                    ret, frame = cam.read()
                    if not ret:
                        break
                    
                    processed_frame = handler.recognize(frame)
                    if processed_frame is not None:
                        cv2.imshow('Recognition', processed_frame)
                    
                    if cv2.waitKey(1) == ord('q'):
                        break
                
                cv2.destroyAllWindows()
                cam.release()
                STATE = False