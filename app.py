import argparse, logging, shelve, os, cv2
import numpy as np

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter('%(asctime)s : %(levelname)s  : %(message)s')

file_handler = logging.FileHandler('app.log')
file_handler.setFormatter(formatter)

stream_handler = logging.StreamHandler()
stream_handler.setFormatter(formatter)

logger.addHandler(file_handler)
logger.addHandler(stream_handler)

shelfFile = shelve.open('data')
classifier_path = 'lbpcascade_facerecogniser.xml'
haarcascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

parser = argparse.ArgumentParser(description="Facial Recognition System")
parser.add_argument('-a', '--add-face', metavar='name', help='name of the person to recognise')
parser.add_argument('-c', '--configure', metavar='path', help='path to store cascade')
parser.add_argument('-l', '--list', action='store_true', help='list out the faces the app recognises')
parser.add_argument('-r', '--recognise', action='store_true', help='recognise the face')
parser.add_argument('-x', '--reset', action='store_true', help='reset the application')
args = parser.parse_args()

logger.info(args)

# Configure
if args.add_face is None and args.configure is not None and args.list == False and args.recognise == False and args.reset == False:
    logger.info('Configuring')
    if not os.path.exists('tmp'):
        os.mkdir('tmp')
    shelfFile['path'] = str(args.configure)
    logger.debug(f'Contents of shelve: {list(shelfFile.items())}')
    logger.info('Configured')

# Add face
elif args.add_face is not None and args.configure is None and args.list == False and args.recognise == False and args.reset == False:
    if 'faces' not in shelfFile.keys():
        shelfFile['faces'] = []
    logger.info('Adding Face')
    logger.info('Starting to capture images')
    cap = cv2.VideoCapture(0)
    count = 0
    while True:
        ret, frame = cap.read()
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        rects = haarcascade.detectMultiScale(gray) 
    
        for (x,y,w,h) in rects: 
            cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
        cv2.imshow('frame', frame)
        
        if len(rects) !=0:
            (x,y,w,h) = rects[0]
            count += 1
            logger.debug(f'saved{count}')
            cv2.imwrite(f'tmp/{count}.png', gray[y:y+w, x:x+h])
                
        
        if cv2.waitKey(1) & count == 250:
            break

    cap.release()
    cv2.destroyAllWindows()
    logger.info('images captured')
    
    logger.info('training started')
    images = []
    labels = []
    for folderName, subfolders, filenames in os.walk('tmp'):
        for filename in filenames:
            images.append(cv2.imread(folderName + '/' + filename, 0))
            labels.append(len(shelfFile['faces']))
    
    if not os.path.exists(shelfFile['path']+classifier_path):
        classifier = cv2.face.LBPHFaceRecognizer_create()
        classifier.train(images , np.array(labels))
        classifier.save(shelfFile['path']+classifier_path)
    else:
        classifier = cv2.CascadeClassifier(shelfFile['path']+classifier_path)
        classifier.update(images , np.array(labels))
        classifier.save(shelfFile['path']+classifier_path)
        
    shelfFile['faces'] = shelfFile['faces']+[str(args.add_face)]
    logger.debug(shelfFile['faces'])
    logger.info('training completed')
    
# List
elif args.add_face is None and args.configure is None and args.list == True and args.recognise == False and args.reset == False:
    print(shelfFile['faces'])
    logger.info('Face List')

# Recognise
elif args.add_face is None and args.configure is None and args.list == False and args.recognise == True and args.reset == False:
    logger.info('Recognising Face')

elif args.add_face is None and args.configure is None and args.list == False and args.recognise == False and args.reset == True:
    logger.info('reseting application')
    shelfFile['faces'] = []
    os.unlink(shelfFile['path']+classifier_path)
    shelfFile['path'] = ''
    logger.info('application reset')
    
    
else:
    logger.info('Wrong Combination of Arguments')

logger.debug(f'ShelfFile {list(shelfFile.items())}')
shelfFile.close()
logger.info('Application is exiting')